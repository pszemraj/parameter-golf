"""Patch W&B's PyTorch watch hooks so they stay eager under torch.compile.

The point of this helper is to keep using `wandb.watch(...)` instead of a
trainer-side histogram imitation. It monkey-patches W&B's hook creators so the
actual callback bodies are wrapped with `torch.compiler.disable` (or the older
`torch._dynamo.disable` fallback).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional

import torch


def _get_disable_decorator() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return the best available decorator for excluding code from Dynamo.

    :return Callable[[Callable[..., Any]], Callable[..., Any]]: Disable decorator.
    """
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
        return torch.compiler.disable  # type: ignore[return-value]
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
        return torch._dynamo.disable  # type: ignore[return-value]

    def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a callable unchanged when no disable decorator exists.

        :param Callable[..., Any] fn: Callable to leave untouched.
        :return Callable[..., Any]: The original callable.
        """
        return fn

    return identity


def _resolve_wandb_torch_module() -> Optional[Any]:
    """Import the W&B torch integration module if available.

    :return Optional[Any]: Imported module or ``None`` when unavailable.
    """
    candidates = (
        "wandb.integration.torch.wandb_torch",
        "wandb.wandb_torch",
    )
    for module_name in candidates:
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


def patch_wandb_watch_for_torch_compile(
    log_fn: Optional[Callable[[str], None]] = None,
) -> bool:
    """Patch W&B watch internals to keep hook callbacks out of Dynamo.

    :param Optional[Callable[[str], None]] log_fn: Optional logger for patch status.
    :return bool: ``True`` when the patch is active, else ``False``.
    """
    module = _resolve_wandb_torch_module()
    if module is None:
        if log_fn is not None:
            log_fn(
                "wandb_watch_patch status=skipped reason=wandb_torch_module_unavailable"
            )
        return False

    TorchHistory = getattr(module, "TorchHistory", None)
    TorchGraph = getattr(module, "TorchGraph", None)
    log_track_init = getattr(module, "log_track_init", None)
    log_track_update = getattr(module, "log_track_update", None)
    wandb_pkg = getattr(module, "wandb", None)
    torch_mod = getattr(module, "torch", None)
    disable = _get_disable_decorator()

    if (
        TorchHistory is None
        or log_track_init is None
        or log_track_update is None
        or wandb_pkg is None
        or torch_mod is None
    ):
        if log_fn is not None:
            log_fn("wandb_watch_patch status=skipped reason=unexpected_wandb_layout")
        return False

    if getattr(TorchHistory, "_compile_safe_watch_patched", False):
        return True

    orig_log_tensor_stats = TorchHistory.log_tensor_stats
    orig_torch_hook_handle_is_valid = TorchHistory._torch_hook_handle_is_valid

    @disable
    def compile_safe_log_tensor_stats(self: Any, tensor: Any, name: str) -> None:
        """Forward tensor-stat logging through the original eager implementation.

        :param Any self: W&B TorchHistory instance.
        :param Any tensor: Tensor-like object to summarize.
        :param str name: Metric namespace.
        :return None: Delegates to W&B logging.
        """
        return orig_log_tensor_stats(self, tensor, name)

    def add_log_parameters_hook(
        self: Any,
        module_obj: Any,
        name: str = "",
        prefix: str = "",
        log_freq: int = 0,
    ) -> None:
        """Register a compile-safe forward hook for parameter histograms.

        :param Any self: W&B TorchHistory instance.
        :param Any module_obj: Module receiving the hook.
        :param str name: Relative module name.
        :param str prefix: Parent module-name prefix.
        :param int log_freq: Histogram logging cadence.
        :return None: Registers hooks in-place.
        """
        if not hasattr(module_obj, "_wandb_hook_names"):
            module_obj._wandb_hook_names = []
        prefix_full = prefix + name

        @disable
        def parameter_log_hook(
            mod: Any, _input: Any, _output: Any, log_track: Any
        ) -> None:
            """Log module parameters from a forward hook without entering Dynamo.

            :param Any mod: Module whose parameters are logged.
            :param Any _input: Ignored hook input.
            :param Any _output: Ignored hook output.
            :param Any log_track: W&B logging cadence tracker.
            :return None: Emits histogram stats when due.
            """
            if not log_track_update(log_track):
                return
            for param_name, parameter in mod.named_parameters():
                if isinstance(parameter, torch_mod.autograd.Variable):
                    data = parameter.data
                else:
                    data = parameter
                self.log_tensor_stats(
                    data.cpu(), "parameters/" + prefix_full + param_name
                )

        log_track_params = log_track_init(log_freq)
        try:
            hook = module_obj.register_forward_hook(
                lambda mod, inp, outp: parameter_log_hook(
                    mod, inp, outp, log_track_params
                )
            )
            self._hook_handles["parameters/" + prefix_full] = hook
            module_obj._wandb_hook_names.append("parameters/" + prefix_full)
        except RuntimeError as exc:
            wandb_pkg.termwarn(
                f"Trying to register forward_hook failed ({exc}) - skipping parameter tracking."
            )

    def hook_variable_gradient_stats(
        self: Any, var: Any, name: str, log_track: Any
    ) -> Any:
        """Register a compile-safe gradient hook for a tracked variable.

        :param Any self: W&B TorchHistory instance.
        :param Any var: Variable receiving the gradient hook.
        :param str name: Histogram namespace.
        :param Any log_track: W&B logging cadence tracker.
        :return Any: Registered hook handle.
        """
        if not isinstance(var, torch_mod.autograd.Variable):
            cls = type(var)
            raise TypeError(
                f"Expected torch.Variable, not {cls.__module__}.{cls.__name__}"
            )
        handle = self._hook_handles.get(name)
        if handle is not None and orig_torch_hook_handle_is_valid(self, handle):
            raise ValueError(f'A hook has already been set under name "{name}"')

        @disable
        def callback(grad: Any, log_track_inner: Any) -> None:
            """Emit gradient stats through the eager-only W&B logging path.

            :param Any grad: Gradient tensor.
            :param Any log_track_inner: W&B logging cadence tracker.
            :return None: Emits histogram stats when due.
            """
            if not log_track_update(log_track_inner):
                return
            self.log_tensor_stats(grad.data, name)

        handle = var.register_hook(lambda grad: callback(grad, log_track))
        self._hook_handles[name] = handle
        return handle

    TorchHistory.log_tensor_stats = compile_safe_log_tensor_stats
    TorchHistory.add_log_parameters_hook = add_log_parameters_hook
    TorchHistory._hook_variable_gradient_stats = hook_variable_gradient_stats

    if TorchGraph is not None and hasattr(TorchGraph, "create_forward_hook"):
        orig_create_forward_hook = TorchGraph.create_forward_hook

        def create_forward_hook(self: Any, name: str, graph_idx: int) -> Any:
            """Wrap TorchGraph forward hooks with the disable decorator.

            :param Any self: W&B TorchGraph instance.
            :param str name: Hook namespace.
            :param int graph_idx: Graph identifier.
            :return Any: Disabled eager-only forward hook.
            """
            hook = orig_create_forward_hook(self, name, graph_idx)
            return disable(hook)

        TorchGraph.create_forward_hook = create_forward_hook

    TorchHistory._compile_safe_watch_patched = True
    if TorchGraph is not None:
        TorchGraph._compile_safe_watch_patched = True

    if log_fn is not None:
        log_fn("wandb_watch_patch status=applied")
    return True
