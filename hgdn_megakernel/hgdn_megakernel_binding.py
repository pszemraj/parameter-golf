"""Python wrapper for the optional HGDN megakernel extension."""

from __future__ import annotations

import importlib
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_EXTENSION_NAME = "hgdn_megakernel_ext"
_WARNED_KEYS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Emit one process-local warning once.

    :param str key: Warning dedupe key.
    :param str message: Warning message.
    """
    if key in _WARNED_KEYS:
        return
    warnings.warn(message)
    _WARNED_KEYS.add(key)


def _arch_list_from_env_or_device() -> str | None:
    """Return the requested `TORCH_CUDA_ARCH_LIST` for megakernel builds.

    :return str | None: Architecture list to export, or `None` when unchanged.
    """
    existing = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if existing:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    arches = {f"{major}.{minor}"}
    if bool(int(os.environ.get("HGDN_MEGAKERNEL_INCLUDE_H100", "0"))):
        arches.add("9.0")
    if bool(int(os.environ.get("HGDN_MEGAKERNEL_INCLUDE_H100A", "0"))):
        arches.add("9.0a")
    return ";".join(sorted(arches))


@lru_cache(maxsize=1)
def _load_extension() -> Any | None:
    """Load the optional HGDN megakernel extension when available.

    Import order:

    1. Reuse an in-place build from `setup_hgdn_megakernel.py build_ext --inplace`.
    2. Optionally JIT-build from source when `GDN_MEGAKERNEL_ALLOW_JIT_BUILD=1`.

    :return Any | None: Imported extension module, or `None`.
    """
    try:
        return importlib.import_module(_EXTENSION_NAME)
    except Exception:
        pass

    if not bool(int(os.environ.get("GDN_MEGAKERNEL_ALLOW_JIT_BUILD", "0"))):
        return None
    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - import failure is rare
        _warn_once(
            "hgdn_megakernel_import",
            "Unable to import torch cpp_extension for HGDN megakernel JIT build: "
            f"{exc}",
        )
        return None

    arch_list = _arch_list_from_env_or_device()
    if arch_list is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    root = Path(__file__).resolve().parent
    verbose = bool(int(os.environ.get("GDN_MEGAKERNEL_BUILD_VERBOSE", "0")))
    try:
        return load(
            name=_EXTENSION_NAME,
            sources=[str(root / "hgdn_megakernel.cu")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-lineinfo",
            ],
            with_cuda=True,
            verbose=verbose,
        )
    except Exception as exc:  # pragma: no cover - requires CUDA toolchain
        _warn_once(
            "hgdn_megakernel_build",
            "HGDN megakernel JIT build failed; using the eager HGDN path instead. "
            f"Details: {exc}",
        )
        return None


def extension_loaded() -> bool:
    """Return whether the megakernel extension is available.

    :return bool: Whether the extension was loaded successfully.
    """
    return _load_extension() is not None


def extension_status() -> dict[str, object]:
    """Return a small status payload for logging and preflight checks.

    :return dict[str, object]: Extension availability and build settings.
    """
    return {
        "loaded": extension_loaded(),
        "allow_jit_build": bool(
            int(os.environ.get("GDN_MEGAKERNEL_ALLOW_JIT_BUILD", "0"))
        ),
        "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }


def device_report() -> str:
    """Return the active CUDA device report from the megakernel extension.

    :raises RuntimeError: If the extension is unavailable.
    :return str: Multi-line device report string.
    """
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("HGDN megakernel extension is not available")
    return str(ext.device_report())


class HGDNMegakernelFunction(torch.autograd.Function):
    """Autograd shim for the HGDN megakernel extension."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        w_qkv: Tensor,
        w_a: Tensor,
        w_b: Tensor,
        w_g: Tensor,
        w_out: Tensor,
        conv_w: Tensor,
        A_log: Tensor,
        dt_bias: Tensor,
        n_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_size: int,
        allow_neg_eigval: bool,
    ) -> Tensor:
        """Run the extension forward path and save tensors for backward.

        :param Any ctx: Autograd context.
        :param Tensor x: Input activations.
        :param Tensor w_qkv: Packed qkv projection weight.
        :param Tensor w_a: Decay projection weight.
        :param Tensor w_b: Beta projection weight.
        :param Tensor w_g: Output-gate projection weight.
        :param Tensor w_out: Dense output projection weight.
        :param Tensor conv_w: Packed depthwise conv weights flattened to `(channels, kernel)`.
        :param Tensor A_log: Learnable decay magnitudes.
        :param Tensor dt_bias: Learnable decay biases.
        :param int n_heads: Number of HGDN heads.
        :param int head_k_dim: Per-head key width.
        :param int head_v_dim: Per-head value width.
        :param int conv_size: Causal conv width.
        :param bool allow_neg_eigval: Whether to scale beta by `2.0`.
        :return Tensor: HGDN block output shaped like `x`.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN megakernel extension is not available")
        outputs = ext.forward(
            x.contiguous(),
            w_qkv.contiguous(),
            w_a.contiguous(),
            w_b.contiguous(),
            w_g.contiguous(),
            w_out.contiguous(),
            conv_w.contiguous(),
            A_log.contiguous().float(),
            dt_bias.contiguous().float(),
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            int(conv_size),
            bool(allow_neg_eigval),
        )
        y, *saved = outputs
        ctx.save_for_backward(
            x,
            w_qkv,
            w_a,
            w_b,
            w_g,
            w_out,
            conv_w,
            A_log,
            dt_bias,
            *saved,
        )
        ctx.meta = (
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            int(conv_size),
            bool(allow_neg_eigval),
        )
        return y

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor) -> tuple[Tensor | None, ...]:
        """Run the extension backward path.

        :param Any ctx: Autograd context.
        :param Tensor grad_y: Gradient for the block output.
        :return tuple[Tensor | None, ...]: Gradients for the differentiable inputs.
        """
        (
            x,
            w_qkv,
            w_a,
            w_b,
            w_g,
            w_out,
            conv_w,
            A_log,
            dt_bias,
            qkv,
            pre,
            q_norm,
            k_norm,
            v_post,
            inv_q,
            inv_k,
            g_pre,
            beta_pre,
            g_log,
            beta,
            g_out,
            o_raw,
            o_norm,
            z,
            state_ckpt,
        ) = ctx.saved_tensors
        n_heads, head_k_dim, head_v_dim, conv_size, allow_neg_eigval = ctx.meta
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN megakernel extension is not available")
        grads = ext.backward(
            grad_y.contiguous().to(dtype=torch.bfloat16),
            x.contiguous(),
            w_qkv.contiguous(),
            w_a.contiguous(),
            w_b.contiguous(),
            w_g.contiguous(),
            w_out.contiguous(),
            conv_w.contiguous(),
            A_log.contiguous().float(),
            dt_bias.contiguous().float(),
            qkv,
            pre,
            q_norm,
            k_norm,
            v_post,
            inv_q,
            inv_k,
            g_pre,
            beta_pre,
            g_log,
            beta,
            g_out,
            o_raw,
            o_norm,
            z,
            state_ckpt,
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            int(conv_size),
            bool(allow_neg_eigval),
        )
        dx, dw_qkv, dw_a, dw_b, dw_g, dw_out, dconv_w, dA_log, ddt_bias = grads
        return (
            dx,
            dw_qkv,
            dw_a,
            dw_b,
            dw_g,
            dw_out,
            dconv_w,
            dA_log,
            ddt_bias,
            None,
            None,
            None,
            None,
            None,
        )


def hgdn_megakernel(
    x: Tensor,
    w_qkv: Tensor,
    w_a: Tensor,
    w_b: Tensor,
    w_g: Tensor,
    w_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    allow_neg_eigval: bool = True,
) -> Tensor:
    """Run the HGDN megakernel custom autograd path.

    :param Tensor x: Input activations shaped `(batch, seq, d_model)`.
    :param Tensor w_qkv: Packed qkv weight shaped `(channels, d_model)`.
    :param Tensor w_a: Decay projection weight shaped `(heads, d_model)`.
    :param Tensor w_b: Beta projection weight shaped `(heads, d_model)`.
    :param Tensor w_g: Output-gate projection weight shaped `(heads * d_v, d_model)`.
    :param Tensor w_out: Dense output projection weight shaped `(d_model, heads * d_v)`.
    :param Tensor conv_w: Packed depthwise conv weights shaped `(channels, kernel)`.
    :param Tensor A_log: Learnable decay magnitudes shaped `(heads,)`.
    :param Tensor dt_bias: Learnable decay biases shaped `(heads,)`.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param bool allow_neg_eigval: Whether to scale beta by `2.0`, defaults to True.
    :return Tensor: HGDN block output shaped `(batch, seq, d_model)`.
    """
    return HGDNMegakernelFunction.apply(
        x,
        w_qkv,
        w_a,
        w_b,
        w_g,
        w_out,
        conv_w,
        A_log,
        dt_bias,
        int(n_heads),
        int(head_k_dim),
        int(head_v_dim),
        int(conv_size),
        bool(allow_neg_eigval),
    )


def run_from_gated_delta_net(module: torch.nn.Module, x: Tensor) -> Tensor:
    """Run the HGDN megakernel from a configured `GatedDeltaNet` module.

    :param torch.nn.Module module: Configured packed HGDN module.
    :param Tensor x: Input activations.
    :raises RuntimeError: If the extension is unavailable.
    :raises ValueError: If the module config is incompatible with the megakernel.
    :return Tensor: HGDN block output.
    """
    if not extension_loaded():
        raise RuntimeError("HGDN megakernel extension is not available")
    if not bool(getattr(module, "use_packed_qkv_proj", False)):
        raise ValueError("HGDN megakernel requires use_packed_qkv_proj=True")
    if not bool(getattr(module, "use_packed_qkv_conv", False)):
        raise ValueError("HGDN megakernel requires use_packed_qkv_conv=True")
    if not bool(getattr(module, "gates_fp32", True)):
        raise ValueError("HGDN megakernel requires gates_fp32=True")
    if not bool(getattr(module, "output_norm_fp32", True)):
        raise ValueError("HGDN megakernel requires output_norm_fp32=True")
    if bool(getattr(module, "use_cuda_fused_frontend", False)):
        raise ValueError("HGDN megakernel is incompatible with use_cuda_fused_frontend")
    if bool(getattr(module, "use_cuda_fused_frontend_lib", False)):
        raise ValueError(
            "HGDN megakernel is incompatible with use_cuda_fused_frontend_lib"
        )
    if bool(getattr(module, "use_cuda_fused_output", False)):
        raise ValueError("HGDN megakernel is incompatible with use_cuda_fused_output")
    if bool(getattr(module, "use_cuda_packed_conv", False)):
        raise ValueError("HGDN megakernel is incompatible with use_cuda_packed_conv")
    if bool(getattr(module, "use_cuda_packed_conv_aten_backward", False)):
        raise ValueError(
            "HGDN megakernel is incompatible with use_cuda_packed_conv_aten_backward"
        )
    if bool(getattr(module, "use_cuda_packed_conv_aten_weight_backward", False)):
        raise ValueError(
            "HGDN megakernel is incompatible with "
            "use_cuda_packed_conv_aten_weight_backward"
        )
    if getattr(module, "qkv_conv", None) is None:
        raise ValueError("HGDN megakernel requires an active packed qkv conv module")
    if getattr(module, "w_qkv", None) is None:
        raise ValueError(
            "HGDN megakernel requires an active packed qkv projection module"
        )
    conv_w = module.qkv_conv.conv.weight.view(
        module.qkv_conv.conv.weight.shape[0],
        module.qkv_conv.conv.weight.shape[-1],
    )
    return hgdn_megakernel(
        x,
        module.w_qkv.weight,
        module.w_a.weight,
        module.w_b.weight,
        module.w_g.weight,
        module.w_out.weight,
        conv_w,
        module.A_log,
        module.dt_bias,
        n_heads=int(module.n_heads),
        head_k_dim=int(module.head_k_dim),
        head_v_dim=int(module.head_v_dim),
        conv_size=int(conv_w.shape[-1]),
        allow_neg_eigval=bool(module.allow_neg_eigval),
    )
