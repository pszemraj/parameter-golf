"""HGDN CUDA fused-op wrappers with safe PyTorch fallbacks."""

from __future__ import annotations

import importlib
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from .reference import packed_qkv_frontend_reference, rmsnorm_silu_gate_reference

_EXTENSION_NAME = "hgdn_cuda_ext"
_WARNED_KEYS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Emit one process-local warning once.

    :param str key: Process-local warning key.
    :param str message: Warning message to emit.
    """
    if key in _WARNED_KEYS:
        return
    warnings.warn(message)
    _WARNED_KEYS.add(key)


@lru_cache(maxsize=1)
def _load_extension() -> Any | None:
    """Load the optional HGDN CUDA extension if it is available.

    Import order:

    1. Reuse an in-place build from ``setup_hgdn_cuda.py build_ext --inplace``.
    2. Optionally JIT-build from sources when ``GDN_CUDA_ALLOW_JIT_BUILD=1``.

    The JIT path is opt-in because extension compilation is too expensive for the
    default short training/eval contract.

    :return Any | None: Imported extension module, or `None` when unavailable.
    """
    try:
        return importlib.import_module(_EXTENSION_NAME)
    except Exception:
        pass

    if not bool(int(os.environ.get("GDN_CUDA_ALLOW_JIT_BUILD", "0"))):
        return None
    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - import failure is rare
        _warn_once(
            "hgdn_cuda_import",
            f"Unable to import torch cpp_extension for HGDN CUDA JIT build: {exc}",
        )
        return None

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "csrc" / "binding.cpp"),
        str(root / "csrc" / "frontend_kernel.cu"),
        str(root / "csrc" / "output_kernel.cu"),
    ]
    verbose = bool(int(os.environ.get("GDN_CUDA_BUILD_VERBOSE", "0")))
    try:
        return load(
            name=_EXTENSION_NAME,
            sources=sources,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            with_cuda=True,
            verbose=verbose,
        )
    except Exception as exc:  # pragma: no cover - requires CUDA toolchain
        _warn_once(
            "hgdn_cuda_build",
            "HGDN CUDA JIT build failed; falling back to PyTorch reference ops. "
            f"Details: {exc}",
        )
        return None


def _dynamo_disable(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Disable TorchDynamo tracing for a function when available.

    :param Callable[..., Any] fn: Function to wrap.
    :return Callable[..., Any]: Wrapped function.
    """
    disable = getattr(torch._dynamo, "disable", None)
    if disable is None:
        return fn
    return disable(fn)


class _PackedQKVFrontendFunction(torch.autograd.Function):
    """CUDA autograd wrapper for the fused packed HGDN front-end."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        qkv: Tensor,
        weight: Tensor,
        n_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        eps: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the fused frontend forward pass.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor qkv: Packed q/k/v activations.
        :param Tensor weight: Packed depthwise conv weights.
        :param int n_heads: HGDN head count.
        :param int head_k_dim: Per-head q/k width.
        :param int head_v_dim: Per-head value width.
        :param float eps: L2-normalization epsilon.
        :raises RuntimeError: If the HGDN CUDA extension is unavailable.
        :return tuple[Tensor, Tensor, Tensor]: Normalized q/k and reshaped v.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN CUDA extension is not available")
        qkv_c = qkv.contiguous()
        weight_c = weight.contiguous()
        q, k, v, preact, inv_q, inv_k = ext.packed_qkv_frontend_forward(
            qkv_c,
            weight_c,
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            float(eps),
        )
        ctx.save_for_backward(qkv_c, weight_c, preact, q, k, inv_q, inv_k)
        return q, k, v

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_q: Tensor,
        grad_k: Tensor,
        grad_v: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, None, None, None, None]:
        """Run the fused frontend backward pass.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor grad_q: Query gradients.
        :param Tensor grad_k: Key gradients.
        :param Tensor grad_v: Value gradients.
        :raises RuntimeError: If the HGDN CUDA extension is unavailable.
        :return tuple[Tensor | None, Tensor | None, None, None, None, None]: Input and weight gradients.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN CUDA extension is not available")
        qkv, weight, preact, q, k, inv_q, inv_k = ctx.saved_tensors
        grad_input, grad_weight = ext.packed_qkv_frontend_backward(
            grad_q.contiguous(),
            grad_k.contiguous(),
            grad_v.contiguous(),
            qkv,
            weight,
            preact,
            q,
            k,
            inv_q,
            inv_k,
        )
        return grad_input, grad_weight, None, None, None, None


class _RMSNormSiluGateFunction(torch.autograd.Function):
    """CUDA autograd wrapper for fused RMSNorm * SiLU(gate)."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        o: Tensor,
        gate: Tensor,
        eps: float,
        fp32_accum: bool,
    ) -> Tensor:
        """Run the fused output forward pass.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor o: Recurrence outputs.
        :param Tensor gate: Output-gate preactivations.
        :param float eps: RMSNorm epsilon.
        :param bool fp32_accum: Whether RMS reduction should accumulate in fp32.
        :raises RuntimeError: If the HGDN CUDA extension is unavailable.
        :return Tensor: Gated normalized outputs.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN CUDA extension is not available")
        o_c = o.contiguous()
        gate_c = gate.contiguous()
        out, normalized, inv_rms = ext.rmsnorm_silu_gate_forward(
            o_c, gate_c, float(eps), bool(fp32_accum)
        )
        ctx.save_for_backward(normalized, gate_c, inv_rms)
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, None, None]:
        """Run the fused output backward pass.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor grad_out: Output gradients.
        :raises RuntimeError: If the HGDN CUDA extension is unavailable.
        :return tuple[Tensor | None, Tensor | None, None, None]: Gradients for ``o`` and ``gate``.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN CUDA extension is not available")
        normalized, gate, inv_rms = ctx.saved_tensors
        grad_o, grad_gate = ext.rmsnorm_silu_gate_backward(
            grad_out.contiguous(), normalized, gate, inv_rms
        )
        return grad_o, grad_gate, None, None


@_dynamo_disable
def extension_loaded() -> bool:
    """Return whether the HGDN CUDA extension is available in this process.

    :return bool: Whether the extension is loaded.
    """
    return _load_extension() is not None


@_dynamo_disable
def extension_status() -> dict[str, object]:
    """Return a small status payload for logging/preflight output.

    :return dict[str, object]: Extension availability and JIT-build status.
    """
    return {
        "loaded": extension_loaded(),
        "allow_jit_build": bool(int(os.environ.get("GDN_CUDA_ALLOW_JIT_BUILD", "0"))),
    }


@_dynamo_disable
def fused_packed_qkv_frontend(
    qkv: Tensor,
    weight: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
    enabled: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the packed HGDN front-end through CUDA or the reference path.

    :param Tensor qkv: Packed q/k/v projections shaped ``(batch, seq, channels)``.
    :param Tensor weight: Depthwise conv weights shaped ``(channels, kernel)``.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :param bool enabled: Whether the caller wants the CUDA path when available.
    :return tuple[Tensor, Tensor, Tensor]: Normalized q/k and reshaped v.
    """
    use_cuda_ext = enabled and qkv.is_cuda and weight.is_cuda and extension_loaded()
    if use_cuda_ext:
        return _PackedQKVFrontendFunction.apply(
            qkv,
            weight,
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            float(eps),
        )
    if enabled and qkv.is_cuda and weight.is_cuda:
        _warn_once(
            "hgdn_cuda_frontend_fallback",
            "GDN fused CUDA front-end requested but the extension is unavailable; "
            "using the PyTorch reference path instead.",
        )
    return packed_qkv_frontend_reference(
        qkv,
        weight,
        n_heads=n_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        eps=eps,
    )


@_dynamo_disable
def fused_rmsnorm_silu_gate(
    o: Tensor,
    gate: Tensor,
    *,
    eps: float = 1e-6,
    fp32_accum: bool = True,
    enabled: bool = True,
) -> Tensor:
    """Run the post-recurrence output fuse through CUDA or the reference path.

    :param Tensor o: Recurrence outputs.
    :param Tensor gate: Output-gate preactivations.
    :param float eps: RMSNorm epsilon.
    :param bool fp32_accum: Whether RMS reduction should accumulate in fp32.
    :param bool enabled: Whether the caller wants the CUDA path when available.
    :return Tensor: Gated normalized outputs.
    """
    use_cuda_ext = enabled and o.is_cuda and gate.is_cuda and extension_loaded()
    if use_cuda_ext:
        return _RMSNormSiluGateFunction.apply(o, gate, float(eps), bool(fp32_accum))
    if enabled and o.is_cuda and gate.is_cuda:
        _warn_once(
            "hgdn_cuda_output_fallback",
            "GDN fused CUDA output path requested but the extension is unavailable; "
            "using the PyTorch reference path instead.",
        )
    return rmsnorm_silu_gate_reference(o, gate, eps=eps, fp32_accum=fp32_accum)
