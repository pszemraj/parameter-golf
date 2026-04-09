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
from torch.nn import functional as F
from torch.nn import grad as nn_grad

from .reference import (
    packed_qkv_frontend_backward_reference,
    packed_qkv_frontend_reference_with_ctx,
    packed_qkv_conv_reference,
    packed_qkv_frontend_reference,
    packed_qkv_split_l2norm_backward_reference,
    packed_qkv_split_l2norm_reference,
    preact_silu_split_l2norm_nct_backward_reference,
    preact_silu_split_l2norm_nct_reference,
    rmsnorm_silu_gate_reference,
)

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
    )
except ImportError:
    _fla_chunk_gated_delta_rule = None

_EXTENSION_NAME = "hgdn_cuda_ext"
_WARNED_KEYS: set[str] = set()
HAS_FLA_GATED_DELTA_RULE = _fla_chunk_gated_delta_rule is not None


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


_FRONTEND_V2_LIB = torch.library.Library("hgdn_cuda_v2", "DEF")
_FRONTEND_V2_LIB.define(
    "preact_silu_split_l2norm_nct(Tensor preact_nct, int n_heads, int head_k_dim, int head_v_dim, float eps) "
    "-> (Tensor q, Tensor k, Tensor v, Tensor inv_q, Tensor inv_k)"
)
_FRONTEND_V2_LIB.define(
    "preact_silu_split_l2norm_nct_backward("
    "Tensor grad_q, Tensor grad_k, Tensor grad_v, "
    "Tensor preact_nct, Tensor q, Tensor k, Tensor inv_q, Tensor inv_k"
    ") -> Tensor"
)

_FRONTEND_V3_LIB = torch.library.Library("hgdn_cuda_v3", "DEF")
_FRONTEND_V3_LIB.define(
    "packed_qkv_frontend("
    "Tensor qkv, Tensor weight, int n_heads, int head_k_dim, int head_v_dim, float eps"
    ") -> (Tensor q, Tensor k, Tensor v, Tensor preact, Tensor inv_q, Tensor inv_k)"
)
_FRONTEND_V3_LIB.define(
    "packed_qkv_frontend_backward("
    "Tensor grad_q, Tensor grad_k, Tensor grad_v, "
    "Tensor qkv, Tensor weight, Tensor preact, Tensor q, Tensor k, Tensor inv_q, Tensor inv_k, "
    "int n_heads, int head_k_dim, int head_v_dim, float eps"
    ") -> (Tensor grad_qkv, Tensor grad_weight)"
)

_FRONTEND_V4_LIB = torch.library.Library("hgdn_cuda_v4", "DEF")
_FRONTEND_V4_LIB.define(
    "packed_qkv_split_l2norm("
    "Tensor packed, int n_heads, int head_k_dim, int head_v_dim, float eps"
    ") -> (Tensor q, Tensor k, Tensor v, Tensor inv_q, Tensor inv_k)"
)
_FRONTEND_V4_LIB.define(
    "packed_qkv_split_l2norm_backward("
    "Tensor grad_q, Tensor grad_k, Tensor grad_v, "
    "Tensor packed, Tensor q, Tensor k, Tensor inv_q, Tensor inv_k, "
    "int n_heads, int head_k_dim, int head_v_dim, float eps"
    ") -> Tensor"
)

_FLA_V1_LIB = torch.library.Library("hgdn_fla_v1", "DEF")
_FLA_V1_LIB.define(
    "chunk_gated_delta_rule(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta) -> Tensor"
)


def _run_fla_chunk_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tensor:
    """Run the upstream FLA gated-delta recurrence and return only the output.

    This keeps the recurrence behind one compile-visible operator boundary so
    Dynamo does not step into FLA backend dispatch internals.

    :param Tensor q: Query tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor k: Key tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor v: Value tensor shaped ``(batch, seq, heads, head_v)``.
    :param Tensor g: Log-space gate tensor shaped ``(batch, seq, heads)``.
    :param Tensor beta: Beta tensor shaped ``(batch, seq, heads)``.
    :raises RuntimeError: If FLA is unavailable.
    :return Tensor: Recurrence output shaped like ``v``.
    """
    if _fla_chunk_gated_delta_rule is None:
        raise RuntimeError(
            "FLA chunk_gated_delta_rule is unavailable; the compile-visible "
            "HGDN recurrence op cannot run."
        )
    out, _final_state = _fla_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=1.0,
        output_final_state=False,
    )
    return out


@torch.library.impl("hgdn_fla_v1::chunk_gated_delta_rule", "CPU")
def _fla_chunk_gated_delta_rule_cpu(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tensor:
    """CPU implementation for the compile-visible FLA recurrence op.

    :param Tensor q: Query tensor.
    :param Tensor k: Key tensor.
    :param Tensor v: Value tensor.
    :param Tensor g: Log-space gate tensor.
    :param Tensor beta: Beta tensor.
    :return Tensor: Recurrence output.
    """
    return _run_fla_chunk_gated_delta_rule(q, k, v, g, beta)


@torch.library.impl("hgdn_fla_v1::chunk_gated_delta_rule", "CUDA")
def _fla_chunk_gated_delta_rule_cuda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tensor:
    """CUDA implementation for the compile-visible FLA recurrence op.

    :param Tensor q: Query tensor.
    :param Tensor k: Key tensor.
    :param Tensor v: Value tensor.
    :param Tensor g: Log-space gate tensor.
    :param Tensor beta: Beta tensor.
    :return Tensor: Recurrence output.
    """
    return _run_fla_chunk_gated_delta_rule(q, k, v, g, beta)


@torch.library.impl("hgdn_cuda_v2::preact_silu_split_l2norm_nct", "CPU")
def _preact_silu_split_l2norm_nct_cpu(
    preact_nct: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CPU/reference implementation for the compile-visible NCT frontend op.

    :param Tensor preact_nct: Packed pre-activation tensor in ``(batch, channels, seq)`` layout.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Normalized q/k, reshaped v, and cached inverse norms.
    """
    return preact_silu_split_l2norm_nct_reference(
        preact_nct,
        n_heads=int(n_heads),
        head_k_dim=int(head_k_dim),
        head_v_dim=int(head_v_dim),
        eps=float(eps),
    )


@torch.library.impl("hgdn_cuda_v2::preact_silu_split_l2norm_nct_backward", "CPU")
def _preact_silu_split_l2norm_nct_backward_cpu(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    preact_nct: Tensor,
    q: Tensor,
    k: Tensor,
    inv_q: Tensor,
    inv_k: Tensor,
) -> Tensor:
    """CPU/reference implementation for the NCT frontend backward op.

    :param Tensor grad_q: Query gradients.
    :param Tensor grad_k: Key gradients.
    :param Tensor grad_v: Value gradients.
    :param Tensor preact_nct: Saved packed pre-activation tensor.
    :param Tensor q: Saved normalized q output.
    :param Tensor k: Saved normalized k output.
    :param Tensor inv_q: Saved inverse q norms.
    :param Tensor inv_k: Saved inverse k norms.
    :return Tensor: Gradient for ``preact_nct``.
    """
    return preact_silu_split_l2norm_nct_backward_reference(
        grad_q,
        grad_k,
        grad_v,
        preact_nct,
        q,
        k,
        inv_q,
        inv_k,
    )


@torch.library.impl("hgdn_cuda_v2::preact_silu_split_l2norm_nct", "CUDA")
def _preact_silu_split_l2norm_nct_cuda(
    preact_nct: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CUDA implementation for the compile-visible NCT frontend op.

    :param Tensor preact_nct: Packed pre-activation tensor in ``(batch, channels, seq)`` layout.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Normalized q/k, reshaped v, and cached inverse norms.
    """
    ext = _load_extension()
    if ext is None:
        _warn_once(
            "hgdn_cuda_frontend_nct_fallback",
            "GDN CUDA NCT frontend path requested but the extension is unavailable; "
            "using the PyTorch reference path instead.",
        )
        return preact_silu_split_l2norm_nct_reference(
            preact_nct,
            n_heads=int(n_heads),
            head_k_dim=int(head_k_dim),
            head_v_dim=int(head_v_dim),
            eps=float(eps),
        )
    return ext.preact_silu_split_l2norm_nct_forward(
        preact_nct.contiguous(),
        int(n_heads),
        int(head_k_dim),
        int(head_v_dim),
        float(eps),
    )


@torch.library.impl("hgdn_cuda_v2::preact_silu_split_l2norm_nct_backward", "CUDA")
def _preact_silu_split_l2norm_nct_backward_cuda(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    preact_nct: Tensor,
    q: Tensor,
    k: Tensor,
    inv_q: Tensor,
    inv_k: Tensor,
) -> Tensor:
    """CUDA implementation for the NCT frontend backward op.

    :param Tensor grad_q: Query gradients.
    :param Tensor grad_k: Key gradients.
    :param Tensor grad_v: Value gradients.
    :param Tensor preact_nct: Saved packed pre-activation tensor.
    :param Tensor q: Saved normalized q output.
    :param Tensor k: Saved normalized k output.
    :param Tensor inv_q: Saved inverse q norms.
    :param Tensor inv_k: Saved inverse k norms.
    :return Tensor: Gradient for ``preact_nct``.
    """
    ext = _load_extension()
    if ext is None:
        _warn_once(
            "hgdn_cuda_frontend_nct_bwd_fallback",
            "GDN CUDA NCT frontend backward requested but the extension is unavailable; "
            "using the PyTorch reference path instead.",
        )
        return preact_silu_split_l2norm_nct_backward_reference(
            grad_q,
            grad_k,
            grad_v,
            preact_nct,
            q,
            k,
            inv_q,
            inv_k,
        )
    return ext.preact_silu_split_l2norm_nct_backward(
        grad_q.contiguous(),
        grad_k.contiguous(),
        grad_v.contiguous(),
        preact_nct.contiguous(),
        q.contiguous(),
        k.contiguous(),
        inv_q.contiguous(),
        inv_k.contiguous(),
    )


@torch.library.impl("hgdn_cuda_v3::packed_qkv_frontend", "CPU")
def _packed_qkv_frontend_cpu(
    qkv: Tensor,
    weight: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CPU/reference implementation for the compile-visible packed frontend op.

    :param Tensor qkv: Packed q/k/v projections in ``(batch, seq, channels)`` layout.
    :param Tensor weight: Packed depthwise conv weights.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: Normalized q/k, reshaped v, packed preactivation, and inverse norms.
    """
    return packed_qkv_frontend_reference_with_ctx(
        qkv,
        weight,
        n_heads=int(n_heads),
        head_k_dim=int(head_k_dim),
        head_v_dim=int(head_v_dim),
        eps=float(eps),
    )


@torch.library.impl("hgdn_cuda_v3::packed_qkv_frontend", "CUDA")
def _packed_qkv_frontend_cuda(
    qkv: Tensor,
    weight: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CUDA implementation for the compile-visible packed frontend op.

    :param Tensor qkv: Packed q/k/v projections in ``(batch, seq, channels)`` layout.
    :param Tensor weight: Packed depthwise conv weights.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: Normalized q/k, reshaped v, packed preactivation, and inverse norms.
    """
    ext = _load_extension()
    if ext is None:
        _warn_once(
            "hgdn_cuda_frontend_v3_fallback",
            "GDN compile-visible packed frontend path requested but the extension is unavailable; "
            "using the PyTorch reference path instead.",
        )
        return packed_qkv_frontend_reference_with_ctx(
            qkv,
            weight,
            n_heads=int(n_heads),
            head_k_dim=int(head_k_dim),
            head_v_dim=int(head_v_dim),
            eps=float(eps),
        )
    return ext.packed_qkv_frontend_forward(
        qkv.contiguous(),
        weight.contiguous(),
        int(n_heads),
        int(head_k_dim),
        int(head_v_dim),
        float(eps),
    )


@torch.library.impl("hgdn_cuda_v3::packed_qkv_frontend_backward", "CPU")
def _packed_qkv_frontend_backward_cpu(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    qkv: Tensor,
    weight: Tensor,
    _preact: Tensor,
    _q: Tensor,
    _k: Tensor,
    _inv_q: Tensor,
    _inv_k: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """CPU/reference implementation for the packed frontend backward op.

    :param Tensor grad_q: Query gradients.
    :param Tensor grad_k: Key gradients.
    :param Tensor grad_v: Value gradients.
    :param Tensor qkv: Saved packed q/k/v projections.
    :param Tensor weight: Saved depthwise conv weights.
    :param Tensor _preact: Ignored saved preactivation placeholder.
    :param Tensor _q: Ignored saved q placeholder.
    :param Tensor _k: Ignored saved k placeholder.
    :param Tensor _inv_q: Ignored saved inverse-q placeholder.
    :param Tensor _inv_k: Ignored saved inverse-k placeholder.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor]: Gradients for ``qkv`` and ``weight``.
    """
    return packed_qkv_frontend_backward_reference(
        grad_q,
        grad_k,
        grad_v,
        qkv,
        weight,
        n_heads=int(n_heads),
        head_k_dim=int(head_k_dim),
        head_v_dim=int(head_v_dim),
        eps=float(eps),
    )


@torch.library.impl("hgdn_cuda_v3::packed_qkv_frontend_backward", "CUDA")
def _packed_qkv_frontend_backward_cuda(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    qkv: Tensor,
    weight: Tensor,
    preact: Tensor,
    q: Tensor,
    k: Tensor,
    inv_q: Tensor,
    inv_k: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """CUDA implementation for the packed frontend backward op.

    :param Tensor grad_q: Query gradients.
    :param Tensor grad_k: Key gradients.
    :param Tensor grad_v: Value gradients.
    :param Tensor qkv: Saved packed q/k/v projections.
    :param Tensor weight: Saved depthwise conv weights.
    :param Tensor preact: Saved packed preactivation tensor.
    :param Tensor q: Saved normalized q output.
    :param Tensor k: Saved normalized k output.
    :param Tensor inv_q: Saved inverse q norms.
    :param Tensor inv_k: Saved inverse k norms.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: Unused epsilon placeholder for API symmetry.
    :return tuple[Tensor, Tensor]: Gradients for ``qkv`` and ``weight``.
    """
    ext = _load_extension()
    if ext is None:
        _warn_once(
            "hgdn_cuda_frontend_v3_bwd_fallback",
            "GDN compile-visible packed frontend backward requested but the extension is unavailable; "
            "using the PyTorch reference path instead.",
        )
        return packed_qkv_frontend_backward_reference(
            grad_q,
            grad_k,
            grad_v,
            qkv,
            weight,
            n_heads=int(n_heads),
            head_k_dim=int(head_k_dim),
            head_v_dim=int(head_v_dim),
            eps=float(eps),
        )
    return ext.packed_qkv_frontend_backward(
        grad_q.contiguous(),
        grad_k.contiguous(),
        grad_v.contiguous(),
        qkv.contiguous(),
        weight.contiguous(),
        preact.contiguous(),
        q.contiguous(),
        k.contiguous(),
        inv_q.contiguous(),
        inv_k.contiguous(),
    )


@torch.library.impl("hgdn_cuda_v4::packed_qkv_split_l2norm", "CPU")
def _packed_qkv_split_l2norm_lib_cpu(
    packed: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CPU/reference implementation for the compile-visible split-norm op.

    :param Tensor packed: Packed post-conv activations in ``(batch, seq, channels)`` layout.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Normalized q/k, reshaped v, and inverse norms.
    """
    batch, seq, _channels = packed.shape
    q, k, v = packed_qkv_split_l2norm_reference(
        packed,
        n_heads=int(n_heads),
        head_k_dim=int(head_k_dim),
        head_v_dim=int(head_v_dim),
        eps=float(eps),
    )
    inv_dtype = packed.new_empty((), dtype=torch.float32).dtype
    q_pre, k_pre, _v_pre = packed.split(
        (
            int(n_heads) * int(head_k_dim),
            int(n_heads) * int(head_k_dim),
            int(n_heads) * int(head_v_dim),
        ),
        dim=-1,
    )
    q_pre = q_pre.view(batch, seq, int(n_heads), int(head_k_dim)).to(dtype=inv_dtype)
    k_pre = k_pre.view(batch, seq, int(n_heads), int(head_k_dim)).to(dtype=inv_dtype)
    inv_q = torch.rsqrt(q_pre.square().sum(dim=-1) + float(eps)).float()
    inv_k = torch.rsqrt(k_pre.square().sum(dim=-1) + float(eps)).float()
    return q, k, v, inv_q.contiguous(), inv_k.contiguous()


@torch.library.impl("hgdn_cuda_v4::packed_qkv_split_l2norm", "CUDA")
def _packed_qkv_split_l2norm_lib_cuda(
    packed: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CUDA implementation for the compile-visible split-norm op.

    :param Tensor packed: Packed post-conv activations in ``(batch, seq, channels)`` layout.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Normalized q/k, reshaped v, and inverse norms.
    """
    ext = _load_extension()
    if ext is None:
        _warn_once(
            "hgdn_cuda_split_norm_lib_fallback",
            "GDN compile-visible split-norm path requested but the extension is unavailable; "
            "using the PyTorch reference path instead.",
        )
        return _packed_qkv_split_l2norm_lib_cpu(
            packed,
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            float(eps),
        )
    return ext.packed_qkv_split_l2norm_forward(
        packed.contiguous(),
        int(n_heads),
        int(head_k_dim),
        int(head_v_dim),
        float(eps),
    )


@torch.library.impl("hgdn_cuda_v4::packed_qkv_split_l2norm_backward", "CPU")
def _packed_qkv_split_l2norm_lib_backward_cpu(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    packed: Tensor,
    _q: Tensor,
    _k: Tensor,
    _inv_q: Tensor,
    _inv_k: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> Tensor:
    """CPU/reference implementation for the compile-visible split-norm backward op.

    :param Tensor grad_q: Query gradients.
    :param Tensor grad_k: Key gradients.
    :param Tensor grad_v: Value gradients.
    :param Tensor packed: Saved packed post-conv activations.
    :param Tensor _q: Ignored saved q placeholder.
    :param Tensor _k: Ignored saved k placeholder.
    :param Tensor _inv_q: Ignored saved inverse-q placeholder.
    :param Tensor _inv_k: Ignored saved inverse-k placeholder.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return Tensor: Gradient for ``packed``.
    """
    return packed_qkv_split_l2norm_backward_reference(
        grad_q,
        grad_k,
        grad_v,
        packed,
        n_heads=int(n_heads),
        head_k_dim=int(head_k_dim),
        head_v_dim=int(head_v_dim),
        eps=float(eps),
    )


@torch.library.impl("hgdn_cuda_v4::packed_qkv_split_l2norm_backward", "CUDA")
def _packed_qkv_split_l2norm_lib_backward_cuda(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    packed: Tensor,
    q: Tensor,
    k: Tensor,
    inv_q: Tensor,
    inv_k: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> Tensor:
    """CUDA implementation for the compile-visible split-norm backward op.

    :param Tensor grad_q: Query gradients.
    :param Tensor grad_k: Key gradients.
    :param Tensor grad_v: Value gradients.
    :param Tensor packed: Saved packed post-conv activations.
    :param Tensor q: Saved normalized q output.
    :param Tensor k: Saved normalized k output.
    :param Tensor inv_q: Saved inverse q norms.
    :param Tensor inv_k: Saved inverse k norms.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: Unused epsilon placeholder for API symmetry.
    :return Tensor: Gradient for ``packed``.
    """
    ext = _load_extension()
    if ext is None:
        _warn_once(
            "hgdn_cuda_split_norm_lib_bwd_fallback",
            "GDN compile-visible split-norm backward requested but the extension is unavailable; "
            "using the PyTorch reference path instead.",
        )
        return packed_qkv_split_l2norm_backward_reference(
            grad_q,
            grad_k,
            grad_v,
            packed,
            n_heads=int(n_heads),
            head_k_dim=int(head_k_dim),
            head_v_dim=int(head_v_dim),
            eps=float(eps),
        )
    return ext.packed_qkv_split_l2norm_backward(
        grad_q.contiguous(),
        grad_k.contiguous(),
        grad_v.contiguous(),
        q.contiguous(),
        k.contiguous(),
        inv_q.contiguous(),
        inv_k.contiguous(),
    )


@torch.library.register_fake("hgdn_cuda_v2::preact_silu_split_l2norm_nct")
def _preact_silu_split_l2norm_nct_fake(
    preact_nct: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Meta kernel for the compile-visible NCT frontend op.

    :param Tensor preact_nct: Packed pre-activation tensor in ``(batch, channels, seq)`` layout.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Meta tensors matching the real op outputs.
    """
    if preact_nct.ndim != 3:
        raise ValueError(f"Expected preact_nct.ndim == 3, got {preact_nct.ndim}")
    batch, channels, seq = preact_nct.shape
    q_dim = int(n_heads) * int(head_k_dim)
    k_dim = int(n_heads) * int(head_k_dim)
    v_dim = int(n_heads) * int(head_v_dim)
    expected_channels = q_dim + k_dim + v_dim
    if channels != expected_channels:
        raise ValueError(
            f"Channel mismatch: got {channels}, expected {expected_channels}"
        )
    q = preact_nct.new_empty((batch, seq, int(n_heads), int(head_k_dim)))
    k = preact_nct.new_empty((batch, seq, int(n_heads), int(head_k_dim)))
    v = preact_nct.new_empty((batch, seq, int(n_heads), int(head_v_dim)))
    inv_q = preact_nct.new_empty((batch, seq, int(n_heads)), dtype=torch.float32)
    inv_k = preact_nct.new_empty((batch, seq, int(n_heads)), dtype=torch.float32)
    return q, k, v, inv_q, inv_k


@torch.library.register_fake("hgdn_cuda_v2::preact_silu_split_l2norm_nct_backward")
def _preact_silu_split_l2norm_nct_backward_fake(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    preact_nct: Tensor,
    q: Tensor,
    k: Tensor,
    inv_q: Tensor,
    inv_k: Tensor,
) -> Tensor:
    """Meta kernel for the NCT frontend backward op.

    :param Tensor grad_q: Query gradients.
    :param Tensor grad_k: Key gradients.
    :param Tensor grad_v: Value gradients.
    :param Tensor preact_nct: Saved packed pre-activation tensor.
    :param Tensor q: Saved normalized q output.
    :param Tensor k: Saved normalized k output.
    :param Tensor inv_q: Saved inverse q norms.
    :param Tensor inv_k: Saved inverse k norms.
    :return Tensor: Meta tensor matching the ``preact_nct`` gradient shape.
    """
    return preact_nct.new_empty(preact_nct.shape)


@torch.library.register_fake("hgdn_cuda_v3::packed_qkv_frontend")
def _packed_qkv_frontend_fake(
    qkv: Tensor,
    weight: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    _eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Meta kernel for the compile-visible packed frontend op.

    :param Tensor qkv: Packed q/k/v projections.
    :param Tensor weight: Packed depthwise conv weights.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float _eps: Ignored epsilon placeholder.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: Meta tensors matching the packed frontend outputs.
    """
    if qkv.ndim != 3:
        raise ValueError(f"Expected qkv.ndim == 3, got {qkv.ndim}")
    batch, seq, _channels = qkv.shape
    q = qkv.new_empty((batch, seq, int(n_heads), int(head_k_dim)))
    k = qkv.new_empty((batch, seq, int(n_heads), int(head_k_dim)))
    v = qkv.new_empty((batch, seq, int(n_heads), int(head_v_dim)))
    preact = qkv.new_empty(qkv.shape)
    inv_q = qkv.new_empty((batch, seq, int(n_heads)), dtype=torch.float32)
    inv_k = qkv.new_empty((batch, seq, int(n_heads)), dtype=torch.float32)
    return q, k, v, preact, inv_q, inv_k


@torch.library.register_fake("hgdn_cuda_v3::packed_qkv_frontend_backward")
def _packed_qkv_frontend_backward_fake(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    qkv: Tensor,
    weight: Tensor,
    _preact: Tensor,
    _q: Tensor,
    _k: Tensor,
    _inv_q: Tensor,
    _inv_k: Tensor,
    _n_heads: int,
    _head_k_dim: int,
    _head_v_dim: int,
    _eps: float,
) -> tuple[Tensor, Tensor]:
    """Meta kernel for the packed frontend backward op.

    :param Tensor grad_q: Query gradients.
    :param Tensor grad_k: Key gradients.
    :param Tensor grad_v: Value gradients.
    :param Tensor qkv: Saved packed q/k/v projections.
    :param Tensor weight: Saved depthwise conv weights.
    :param Tensor _preact: Ignored saved preactivation placeholder.
    :param Tensor _q: Ignored saved q placeholder.
    :param Tensor _k: Ignored saved k placeholder.
    :param Tensor _inv_q: Ignored saved inverse-q placeholder.
    :param Tensor _inv_k: Ignored saved inverse-k placeholder.
    :param int _n_heads: Ignored head-count placeholder.
    :param int _head_k_dim: Ignored q/k width placeholder.
    :param int _head_v_dim: Ignored value width placeholder.
    :param float _eps: Ignored epsilon placeholder.
    :return tuple[Tensor, Tensor]: Meta tensors matching the input and weight gradients.
    """
    del grad_q, grad_k, grad_v
    return qkv.new_empty(qkv.shape), weight.new_empty(weight.shape)


@torch.library.register_fake("hgdn_cuda_v4::packed_qkv_split_l2norm")
def _packed_qkv_split_l2norm_lib_fake(
    packed: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    _eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Meta kernel for the compile-visible packed split-norm op.

    :param Tensor packed: Packed post-conv activations.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float _eps: Ignored epsilon placeholder.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Meta tensors matching the real outputs.
    """
    if packed.ndim != 3:
        raise ValueError(f"Expected packed.ndim == 3, got {packed.ndim}")
    batch, seq, _channels = packed.shape
    q = packed.new_empty((batch, seq, int(n_heads), int(head_k_dim)))
    k = packed.new_empty((batch, seq, int(n_heads), int(head_k_dim)))
    v = packed.new_empty((batch, seq, int(n_heads), int(head_v_dim)))
    inv_q = packed.new_empty((batch, seq, int(n_heads)), dtype=torch.float32)
    inv_k = packed.new_empty((batch, seq, int(n_heads)), dtype=torch.float32)
    return q, k, v, inv_q, inv_k


@torch.library.register_fake("hgdn_cuda_v4::packed_qkv_split_l2norm_backward")
def _packed_qkv_split_l2norm_lib_backward_fake(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    packed: Tensor,
    _q: Tensor,
    _k: Tensor,
    _inv_q: Tensor,
    _inv_k: Tensor,
    _n_heads: int,
    _head_k_dim: int,
    _head_v_dim: int,
    _eps: float,
) -> Tensor:
    """Meta kernel for the compile-visible packed split-norm backward op.

    :param Tensor grad_q: Query gradients.
    :param Tensor grad_k: Key gradients.
    :param Tensor grad_v: Value gradients.
    :param Tensor packed: Saved packed activations.
    :param Tensor _q: Ignored saved q placeholder.
    :param Tensor _k: Ignored saved k placeholder.
    :param Tensor _inv_q: Ignored saved inverse-q placeholder.
    :param Tensor _inv_k: Ignored saved inverse-k placeholder.
    :param int _n_heads: Ignored head-count placeholder.
    :param int _head_k_dim: Ignored q/k width placeholder.
    :param int _head_v_dim: Ignored value width placeholder.
    :param float _eps: Ignored epsilon placeholder.
    :return Tensor: Meta tensor matching the packed-input gradient.
    """
    del grad_q, grad_k, grad_v
    return packed.new_empty(packed.shape)


@torch.library.register_fake("hgdn_fla_v1::chunk_gated_delta_rule")
def _fla_chunk_gated_delta_rule_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tensor:
    """Meta kernel for the compile-visible FLA recurrence op.

    :param Tensor q: Query tensor.
    :param Tensor k: Key tensor.
    :param Tensor v: Value tensor.
    :param Tensor g: Log-space gate tensor.
    :param Tensor beta: Beta tensor.
    :return Tensor: Meta tensor matching the recurrence output.
    """
    del q, k, g, beta
    return v.new_empty(v.shape)


def _setup_preact_silu_split_l2norm_nct_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[Tensor, int, int, int, float],
    output: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
) -> None:
    """Save tensors needed by the registered NCT frontend backward formula.

    :param torch.autograd.function.FunctionCtx ctx: Autograd context.
    :param tuple[Tensor, int, int, int, float] inputs: Forward inputs.
    :param tuple[Tensor, Tensor, Tensor, Tensor, Tensor] output: Forward outputs.
    :return None: Saves tensors onto ``ctx``.
    """
    preact_nct, _n_heads, _head_k_dim, _head_v_dim, _eps = inputs
    q, k, _v, inv_q, inv_k = output
    ctx.save_for_backward(preact_nct, q, k, inv_q, inv_k)


def _setup_packed_qkv_frontend_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[Tensor, Tensor, int, int, int, float],
    output: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
) -> None:
    """Save tensors needed by the packed frontend backward formula.

    :param torch.autograd.function.FunctionCtx ctx: Autograd context.
    :param tuple[Tensor, Tensor, int, int, int, float] inputs: Forward inputs.
    :param tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] output: Forward outputs.
    :return None: Saves tensors and metadata onto ``ctx``.
    """
    qkv, weight, n_heads, head_k_dim, head_v_dim, eps = inputs
    q, k, _v, preact, inv_q, inv_k = output
    ctx.save_for_backward(qkv, weight, preact, q, k, inv_q, inv_k)
    ctx.n_heads = int(n_heads)
    ctx.head_k_dim = int(head_k_dim)
    ctx.head_v_dim = int(head_v_dim)
    ctx.eps = float(eps)


def _setup_packed_qkv_split_l2norm_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[Tensor, int, int, int, float],
    output: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
) -> None:
    """Save tensors needed by the compile-visible split-norm backward formula.

    :param torch.autograd.function.FunctionCtx ctx: Autograd context.
    :param tuple[Tensor, int, int, int, float] inputs: Forward inputs.
    :param tuple[Tensor, Tensor, Tensor, Tensor, Tensor] output: Forward outputs.
    :return None: Saves tensors and metadata onto ``ctx``.
    """
    packed, n_heads, head_k_dim, head_v_dim, eps = inputs
    q, k, _v, inv_q, inv_k = output
    ctx.save_for_backward(packed, q, k, inv_q, inv_k)
    ctx.n_heads = int(n_heads)
    ctx.head_k_dim = int(head_k_dim)
    ctx.head_v_dim = int(head_v_dim)
    ctx.eps = float(eps)


def _setup_fla_chunk_gated_delta_rule_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    output: Tensor,
) -> None:
    """Save tensors needed by the compile-visible FLA recurrence backward formula.

    :param torch.autograd.function.FunctionCtx ctx: Autograd context.
    :param tuple[Tensor, Tensor, Tensor, Tensor, Tensor] inputs: Forward inputs.
    :param Tensor output: Forward output tensor.
    :return None: Saves tensors onto ``ctx``.
    """
    del output
    ctx.save_for_backward(*inputs)


def _preact_silu_split_l2norm_nct_backward_formula(
    ctx: torch.autograd.function.FunctionCtx,
    grad_q: Tensor | None,
    grad_k: Tensor | None,
    grad_v: Tensor | None,
    _grad_inv_q: Tensor | None,
    _grad_inv_k: Tensor | None,
) -> tuple[Tensor | None, None, None, None, None]:
    """Backward formula for the registered NCT frontend op.

    :param torch.autograd.function.FunctionCtx ctx: Autograd context.
    :param Tensor | None grad_q: Query-output gradient.
    :param Tensor | None grad_k: Key-output gradient.
    :param Tensor | None grad_v: Value-output gradient.
    :param Tensor | None _grad_inv_q: Ignored inverse-q gradient placeholder.
    :param Tensor | None _grad_inv_k: Ignored inverse-k gradient placeholder.
    :return tuple[Tensor | None, None, None, None, None]: Gradient for ``preact_nct`` plus ``None`` for non-tensor args.
    """
    preact_nct, q, k, inv_q, inv_k = ctx.saved_tensors
    grad_q = torch.zeros_like(q) if grad_q is None else grad_q
    grad_k = torch.zeros_like(k) if grad_k is None else grad_k
    if grad_v is None:
        batch, seq, heads = inv_q.shape
        head_v_dim = preact_nct.shape[1] // heads - 2 * q.shape[-1]
        grad_v = preact_nct.new_zeros((batch, seq, heads, head_v_dim))
    grad_preact = torch.ops.hgdn_cuda_v2.preact_silu_split_l2norm_nct_backward(
        grad_q,
        grad_k,
        grad_v,
        preact_nct,
        q,
        k,
        inv_q,
        inv_k,
    )
    return grad_preact, None, None, None, None


def _packed_qkv_frontend_backward_formula(
    ctx: torch.autograd.function.FunctionCtx,
    grad_q: Tensor | None,
    grad_k: Tensor | None,
    grad_v: Tensor | None,
    _grad_preact: Tensor | None,
    _grad_inv_q: Tensor | None,
    _grad_inv_k: Tensor | None,
) -> tuple[Tensor | None, Tensor | None, None, None, None, None]:
    """Backward formula for the compile-visible packed frontend op.

    :param torch.autograd.function.FunctionCtx ctx: Autograd context.
    :param Tensor | None grad_q: Query-output gradient.
    :param Tensor | None grad_k: Key-output gradient.
    :param Tensor | None grad_v: Value-output gradient.
    :param Tensor | None _grad_preact: Ignored preactivation-gradient placeholder.
    :param Tensor | None _grad_inv_q: Ignored inverse-q-gradient placeholder.
    :param Tensor | None _grad_inv_k: Ignored inverse-k-gradient placeholder.
    :return tuple[Tensor | None, Tensor | None, None, None, None, None]: Gradients for ``qkv`` and ``weight`` plus ``None`` for non-tensor args.
    """
    qkv, weight, preact, q, k, inv_q, inv_k = ctx.saved_tensors
    grad_q = torch.zeros_like(q) if grad_q is None else grad_q
    grad_k = torch.zeros_like(k) if grad_k is None else grad_k
    if grad_v is None:
        batch, seq, heads = inv_q.shape
        grad_v = qkv.new_zeros((batch, seq, heads, ctx.head_v_dim))
    grad_qkv, grad_weight = torch.ops.hgdn_cuda_v3.packed_qkv_frontend_backward(
        grad_q,
        grad_k,
        grad_v,
        qkv,
        weight,
        preact,
        q,
        k,
        inv_q,
        inv_k,
        int(ctx.n_heads),
        int(ctx.head_k_dim),
        int(ctx.head_v_dim),
        float(ctx.eps),
    )
    return grad_qkv, grad_weight, None, None, None, None


def _packed_qkv_split_l2norm_backward_formula(
    ctx: torch.autograd.function.FunctionCtx,
    grad_q: Tensor | None,
    grad_k: Tensor | None,
    grad_v: Tensor | None,
    _grad_inv_q: Tensor | None,
    _grad_inv_k: Tensor | None,
) -> tuple[Tensor | None, None, None, None, None]:
    """Backward formula for the compile-visible packed split-norm op.

    :param torch.autograd.function.FunctionCtx ctx: Autograd context.
    :param Tensor | None grad_q: Query-output gradient.
    :param Tensor | None grad_k: Key-output gradient.
    :param Tensor | None grad_v: Value-output gradient.
    :param Tensor | None _grad_inv_q: Ignored inverse-q-gradient placeholder.
    :param Tensor | None _grad_inv_k: Ignored inverse-k-gradient placeholder.
    :return tuple[Tensor | None, None, None, None, None]: Gradient for ``packed`` plus ``None`` for non-tensor args.
    """
    packed, q, k, inv_q, inv_k = ctx.saved_tensors
    grad_q = torch.zeros_like(q) if grad_q is None else grad_q
    grad_k = torch.zeros_like(k) if grad_k is None else grad_k
    if grad_v is None:
        batch, seq, heads = inv_q.shape
        grad_v = packed.new_zeros((batch, seq, heads, ctx.head_v_dim))
    grad_packed = torch.ops.hgdn_cuda_v4.packed_qkv_split_l2norm_backward(
        grad_q,
        grad_k,
        grad_v,
        packed,
        q,
        k,
        inv_q,
        inv_k,
        int(ctx.n_heads),
        int(ctx.head_k_dim),
        int(ctx.head_v_dim),
        float(ctx.eps),
    )
    return grad_packed, None, None, None, None


def _fla_chunk_gated_delta_rule_backward_formula(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: Tensor | None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """Backward formula for the compile-visible FLA recurrence op.

    :param torch.autograd.function.FunctionCtx ctx: Autograd context.
    :param Tensor | None grad_output: Gradient for the recurrence output.
    :return tuple[Tensor | None, ...]: Gradients for ``q, k, v, g, beta``.
    """
    q, k, v, g, beta = ctx.saved_tensors
    if grad_output is None:
        grad_output = torch.zeros_like(v)
    with torch.enable_grad():
        q_req = q.detach().requires_grad_(True)
        k_req = k.detach().requires_grad_(True)
        v_req = v.detach().requires_grad_(True)
        g_req = g.detach().requires_grad_(True)
        beta_req = beta.detach().requires_grad_(True)
        out = _run_fla_chunk_gated_delta_rule(q_req, k_req, v_req, g_req, beta_req)
    grads = torch.autograd.grad(
        out,
        (q_req, k_req, v_req, g_req, beta_req),
        grad_outputs=grad_output,
        allow_unused=False,
    )
    return grads


torch.library.register_autograd(
    "hgdn_cuda_v2::preact_silu_split_l2norm_nct",
    _preact_silu_split_l2norm_nct_backward_formula,
    setup_context=_setup_preact_silu_split_l2norm_nct_context,
)

torch.library.register_autograd(
    "hgdn_cuda_v3::packed_qkv_frontend",
    _packed_qkv_frontend_backward_formula,
    setup_context=_setup_packed_qkv_frontend_context,
)

torch.library.register_autograd(
    "hgdn_cuda_v4::packed_qkv_split_l2norm",
    _packed_qkv_split_l2norm_backward_formula,
    setup_context=_setup_packed_qkv_split_l2norm_context,
)

torch.library.register_autograd(
    "hgdn_fla_v1::chunk_gated_delta_rule",
    _fla_chunk_gated_delta_rule_backward_formula,
    setup_context=_setup_fla_chunk_gated_delta_rule_context,
)


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


class _PackedQKVConvFunction(torch.autograd.Function):
    """CUDA autograd wrapper for exact-length packed causal depthwise conv."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        qkv: Tensor,
        weight: Tensor,
    ) -> Tensor:
        """Run the packed causal depthwise conv forward pass.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor qkv: Packed q/k/v activations.
        :param Tensor weight: Packed depthwise conv weights.
        :raises RuntimeError: If the HGDN CUDA extension is unavailable.
        :return Tensor: Packed post-conv activations.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN CUDA extension is not available")
        qkv_c = qkv.contiguous()
        weight_c = weight.contiguous()
        packed_out, preact = ext.packed_qkv_conv_forward(qkv_c, weight_c)
        ctx.save_for_backward(qkv_c, weight_c, preact)
        return packed_out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_packed_out: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Run the packed causal depthwise conv backward pass.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor grad_packed_out: Packed output gradients.
        :raises RuntimeError: If the HGDN CUDA extension is unavailable.
        :return tuple[Tensor | None, Tensor | None]: Input and weight gradients.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN CUDA extension is not available")
        qkv, weight, preact = ctx.saved_tensors
        grad_input, grad_weight = ext.packed_qkv_conv_backward(
            grad_packed_out.contiguous(),
            qkv,
            weight,
            preact,
        )
        return grad_input, grad_weight


def _silu_backward(preact: Tensor, grad_output: Tensor) -> Tensor:
    """Differentiate SiLU with respect to its pre-activation.

    :param Tensor preact: Saved pre-activation tensor.
    :param Tensor grad_output: Upstream gradient after SiLU.
    :return Tensor: Gradient with respect to ``preact``.
    """
    sigma = torch.sigmoid(preact)
    return grad_output * sigma * (1.0 + preact * (1.0 - sigma))


class _PackedQKVConvAtenBackwardFunction(torch.autograd.Function):
    """CUDA forward wrapper that keeps packed-conv backward on ATen/cuDNN."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        qkv: Tensor,
        weight: Tensor,
    ) -> Tensor:
        """Run exact-length packed causal depthwise conv forward through CUDA.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor qkv: Packed q/k/v activations.
        :param Tensor weight: Packed depthwise conv weights shaped ``(channels, kernel)``.
        :raises RuntimeError: If the HGDN CUDA extension is unavailable.
        :return Tensor: Packed post-conv activations.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN CUDA extension is not available")
        qkv_c = qkv.contiguous()
        weight_c = weight.contiguous()
        packed_out, preact = ext.packed_qkv_conv_forward(qkv_c, weight_c)
        ctx.save_for_backward(qkv_c, weight_c, preact)
        return packed_out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_packed_out: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Backprop through the CUDA forward path with ATen depthwise conv grads.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor grad_packed_out: Packed output gradients.
        :return tuple[Tensor | None, Tensor | None]: Input and weight gradients.
        """
        qkv, weight, preact = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        groups = weight.shape[0]
        kernel = weight.shape[-1]
        weight_3d = weight.unsqueeze(1)

        with torch.autograd.profiler.record_function(
            "gdn.qkv_conv_cuda_aten_bwd_output_transpose"
        ):
            grad_out = grad_packed_out.contiguous()
        with torch.autograd.profiler.record_function("gdn.qkv_conv_cuda_aten_bwd_silu"):
            grad_preact = _silu_backward(preact, grad_out)
        with torch.autograd.profiler.record_function(
            "gdn.qkv_conv_cuda_aten_bwd_left_pad"
        ):
            qkv_t = qkv.transpose(1, 2)
            qkv_pad = F.pad(qkv_t, (kernel - 1, 0))
            grad_preact_t = grad_preact.transpose(1, 2).contiguous()
        if ctx.needs_input_grad[0]:
            with torch.autograd.profiler.record_function(
                "gdn.qkv_conv_cuda_aten_bwd_input_grad"
            ):
                grad_pad = nn_grad.conv1d_input(
                    qkv_pad.shape,
                    weight_3d,
                    grad_preact_t,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=groups,
                )
            with torch.autograd.profiler.record_function(
                "gdn.qkv_conv_cuda_aten_bwd_input_trim"
            ):
                grad_input = grad_pad[..., kernel - 1 :].transpose(1, 2).contiguous()
        if ctx.needs_input_grad[1]:
            with torch.autograd.profiler.record_function(
                "gdn.qkv_conv_cuda_aten_bwd_weight_grad"
            ):
                grad_weight_3d = nn_grad.conv1d_weight(
                    qkv_pad,
                    weight_3d.shape,
                    grad_preact_t,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=groups,
                )
            grad_weight = grad_weight_3d.squeeze(1).contiguous()
        return grad_input, grad_weight


class _PackedQKVConvAtenWeightBackwardFunction(torch.autograd.Function):
    """CUDA forward/input-grad wrapper with ATen/cuDNN weight grad."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        qkv: Tensor,
        weight: Tensor,
    ) -> Tensor:
        """Run exact-length packed causal depthwise conv forward through CUDA.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor qkv: Packed q/k/v activations.
        :param Tensor weight: Packed depthwise conv weights shaped ``(channels, kernel)``.
        :raises RuntimeError: If the HGDN CUDA extension is unavailable.
        :return Tensor: Packed post-conv activations.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN CUDA extension is not available")
        qkv_c = qkv.contiguous()
        weight_c = weight.contiguous()
        packed_out, preact = ext.packed_qkv_conv_forward(qkv_c, weight_c)
        ctx.save_for_backward(qkv_c, weight_c, preact)
        return packed_out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_packed_out: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Backprop with CUDA input-grad and ATen/cuDNN weight grad.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor grad_packed_out: Packed output gradients.
        :raises RuntimeError: If the HGDN CUDA extension is unavailable.
        :return tuple[Tensor | None, Tensor | None]: Input and weight gradients.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN CUDA extension is not available")
        qkv, weight, preact = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        groups = weight.shape[0]
        kernel = weight.shape[-1]
        weight_3d = weight.unsqueeze(1)

        with torch.autograd.profiler.record_function(
            "gdn.qkv_conv_cuda_aten_weight_bwd_cuda_input"
        ):
            grad_preact, grad_input = ext.packed_qkv_conv_input_backward(
                grad_packed_out.contiguous(),
                weight,
                preact,
            )
        if ctx.needs_input_grad[1]:
            with torch.autograd.profiler.record_function(
                "gdn.qkv_conv_cuda_aten_weight_bwd_left_pad"
            ):
                qkv_t = qkv.transpose(1, 2)
                qkv_pad = F.pad(qkv_t, (kernel - 1, 0))
                grad_preact_t = grad_preact.transpose(1, 2).contiguous()
            with torch.autograd.profiler.record_function(
                "gdn.qkv_conv_cuda_aten_weight_bwd_weight_grad"
            ):
                grad_weight_3d = nn_grad.conv1d_weight(
                    qkv_pad,
                    weight_3d.shape,
                    grad_preact_t,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=groups,
                )
            grad_weight = grad_weight_3d.squeeze(1).contiguous()
        return grad_input, grad_weight


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


class _PackedQKVSplitL2NormFunction(torch.autograd.Function):
    """CUDA autograd wrapper for packed split plus q/k L2 normalization."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        packed: Tensor,
        n_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        eps: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run packed split plus q/k L2 normalization on post-conv activations.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor packed: Packed post-conv activations.
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
        packed_c = packed.contiguous()
        q, k, v, inv_q, inv_k = ext.packed_qkv_split_l2norm_forward(
            packed_c,
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            float(eps),
        )
        ctx.save_for_backward(q, k, inv_q, inv_k)
        return q, k, v

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_q: Tensor,
        grad_k: Tensor,
        grad_v: Tensor,
    ) -> tuple[Tensor | None, None, None, None, None]:
        """Run the packed split plus q/k L2 normalization backward pass.

        :param torch.autograd.function.FunctionCtx ctx: Autograd context.
        :param Tensor grad_q: Query gradients.
        :param Tensor grad_k: Key gradients.
        :param Tensor grad_v: Value gradients.
        :raises RuntimeError: If the HGDN CUDA extension is unavailable.
        :return tuple[Tensor | None, None, None, None, None]: Gradient for the packed input only.
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("HGDN CUDA extension is not available")
        q, k, inv_q, inv_k = ctx.saved_tensors
        grad_packed = ext.packed_qkv_split_l2norm_backward(
            grad_q.contiguous(),
            grad_k.contiguous(),
            grad_v.contiguous(),
            q,
            k,
            inv_q,
            inv_k,
        )
        return grad_packed, None, None, None, None


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
def fused_packed_qkv_conv(
    qkv: Tensor,
    weight: Tensor,
    *,
    enabled: bool = True,
) -> Tensor:
    """Run packed causal depthwise conv through CUDA or the reference path.

    :param Tensor qkv: Packed q/k/v projections shaped ``(batch, seq, channels)``.
    :param Tensor weight: Depthwise conv weights shaped ``(channels, kernel)``.
    :param bool enabled: Whether the caller wants the CUDA path when available.
    :return Tensor: Packed post-conv activations.
    """
    use_cuda_ext = enabled and qkv.is_cuda and weight.is_cuda and extension_loaded()
    if use_cuda_ext:
        return _PackedQKVConvFunction.apply(qkv, weight)
    if enabled and qkv.is_cuda and weight.is_cuda:
        _warn_once(
            "hgdn_cuda_conv_fallback",
            "GDN CUDA packed-conv path requested but the extension is unavailable; "
            "using the PyTorch reference path instead.",
        )
    return packed_qkv_conv_reference(qkv, weight)


@_dynamo_disable
def fused_packed_qkv_conv_aten_backward(
    qkv: Tensor,
    weight: Tensor,
    *,
    enabled: bool = True,
) -> Tensor:
    """Run CUDA packed-conv forward while keeping backward on ATen/cuDNN.

    :param Tensor qkv: Packed q/k/v projections shaped ``(batch, seq, channels)``.
    :param Tensor weight: Depthwise conv weights shaped ``(channels, kernel)``.
    :param bool enabled: Whether the caller wants the CUDA path when available.
    :return Tensor: Packed post-conv activations.
    """
    use_cuda_ext = enabled and qkv.is_cuda and weight.is_cuda and extension_loaded()
    if use_cuda_ext:
        return _PackedQKVConvAtenBackwardFunction.apply(qkv, weight)
    if enabled and qkv.is_cuda and weight.is_cuda:
        _warn_once(
            "hgdn_cuda_conv_aten_backward_fallback",
            "GDN CUDA packed-conv ATen-backward path requested but the extension is "
            "unavailable; using the PyTorch reference path instead.",
        )
    return packed_qkv_conv_reference(qkv, weight)


@_dynamo_disable
def fused_packed_qkv_conv_aten_weight_backward(
    qkv: Tensor,
    weight: Tensor,
    *,
    enabled: bool = True,
) -> Tensor:
    """Run CUDA packed-conv forward/input-grad while keeping weight grad on ATen.

    :param Tensor qkv: Packed q/k/v projections shaped ``(batch, seq, channels)``.
    :param Tensor weight: Depthwise conv weights shaped ``(channels, kernel)``.
    :param bool enabled: Whether the caller wants the CUDA path when available.
    :return Tensor: Packed post-conv activations.
    """
    use_cuda_ext = enabled and qkv.is_cuda and weight.is_cuda and extension_loaded()
    if use_cuda_ext:
        return _PackedQKVConvAtenWeightBackwardFunction.apply(qkv, weight)
    if enabled and qkv.is_cuda and weight.is_cuda:
        _warn_once(
            "hgdn_cuda_conv_aten_weight_backward_fallback",
            "GDN CUDA packed-conv ATen-weight-backward path requested but the "
            "extension is unavailable; using the PyTorch reference path instead.",
        )
    return packed_qkv_conv_reference(qkv, weight)


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


@_dynamo_disable
def fused_packed_qkv_split_l2norm(
    packed: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
    enabled: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run packed split plus q/k L2 normalization through CUDA or reference ops.

    :param Tensor packed: Packed post-conv q/k/v activations shaped ``(batch, seq, channels)``.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :param bool enabled: Whether the caller wants the CUDA path when available.
    :return tuple[Tensor, Tensor, Tensor]: Normalized q/k and reshaped v.
    """
    use_cuda_ext = enabled and packed.is_cuda and extension_loaded()
    if use_cuda_ext:
        return _PackedQKVSplitL2NormFunction.apply(
            packed,
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            float(eps),
        )
    if enabled and packed.is_cuda:
        _warn_once(
            "hgdn_cuda_split_norm_fallback",
            "GDN CUDA split+l2norm path requested but the extension is unavailable; "
            "using the PyTorch reference path instead.",
        )
    return packed_qkv_split_l2norm_reference(
        packed,
        n_heads=n_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        eps=eps,
    )


def packed_qkv_split_l2norm_compile_visible(
    packed: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run packed split plus q/k norm through a compile-visible `torch.library` op.

    :param Tensor packed: Packed post-conv activations shaped ``(batch, seq, channels)``.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor, Tensor]: Normalized q/k and reshaped v.
    """
    q, k, v, _inv_q, _inv_k = torch.ops.hgdn_cuda_v4.packed_qkv_split_l2norm(
        packed,
        int(n_heads),
        int(head_k_dim),
        int(head_v_dim),
        float(eps),
    )
    return q, k, v


def fla_chunk_gated_delta_rule_compile_visible(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tensor:
    """Run the FLA gated-delta recurrence through a compile-visible library op.

    :param Tensor q: Query tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor k: Key tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor v: Value tensor shaped ``(batch, seq, heads, head_v)``.
    :param Tensor g: Log-space gate tensor shaped ``(batch, seq, heads)``.
    :param Tensor beta: Beta tensor shaped ``(batch, seq, heads)``.
    :return Tensor: Recurrence output shaped like ``v``.
    """
    return torch.ops.hgdn_fla_v1.chunk_gated_delta_rule(q, k, v, g, beta)


def frontend_preact_silu_split_l2norm_nct(
    preact_nct: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the compile-visible NCT preact frontend op.

    This op consumes the conv-native packed pre-activation tensor in
    ``(batch, channels, seq)`` layout, applies SiLU, splits q/k/v, and returns
    recurrence-ready q/k/v with q/k L2-normalized.

    :param Tensor preact_nct: Packed pre-activation tensor shaped ``(batch, channels, seq)``.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor, Tensor]: Normalized q/k and reshaped v.
    """
    q, k, v, _inv_q, _inv_k = torch.ops.hgdn_cuda_v2.preact_silu_split_l2norm_nct(
        preact_nct,
        int(n_heads),
        int(head_k_dim),
        int(head_v_dim),
        float(eps),
    )
    return q, k, v


def packed_qkv_frontend_compile_visible(
    qkv: Tensor,
    weight: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the packed fused frontend through a compile-visible `torch.library` op.

    :param Tensor qkv: Packed q/k/v projections shaped ``(batch, seq, channels)``.
    :param Tensor weight: Packed depthwise conv weights shaped ``(channels, kernel)``.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: L2-normalization epsilon.
    :return tuple[Tensor, Tensor, Tensor]: Normalized q/k and reshaped v.
    """
    q, k, v, _preact, _inv_q, _inv_k = torch.ops.hgdn_cuda_v3.packed_qkv_frontend(
        qkv,
        weight,
        int(n_heads),
        int(head_k_dim),
        int(head_v_dim),
        float(eps),
    )
    return q, k, v
