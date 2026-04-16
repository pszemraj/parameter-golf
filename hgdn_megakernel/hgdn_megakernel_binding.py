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
# Keep the fake/meta contract aligned with the current CUDA checkpoint cadence.
_REC_CHUNK_T = 8


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
            "HGDN megakernel JIT build failed; the extension remains unavailable. "
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


def _require_loaded_extension() -> Any:
    """Return the loaded megakernel extension or raise a clear error.

    :raises RuntimeError: If the extension is unavailable.
    :return Any: Imported extension module.
    """
    ext = _load_extension()
    if ext is None:
        raise RuntimeError(
            "HGDN megakernel extension is not available. Build "
            "`hgdn_megakernel_ext` or disable `GDN_USE_CUDA_MEGAKERNEL`."
        )
    return ext


def _require_cuda_tensor(
    name: str,
    tensor: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> None:
    """Validate one CUDA tensor precondition for the megakernel path.

    :param str name: Tensor label for error messages.
    :param Tensor tensor: Tensor to validate.
    :param torch.dtype dtype: Required dtype.
    :param torch.device | None device: Required device, defaults to `None`.
    :raises RuntimeError: If the tensor violates the CUDA megakernel contract.
    """
    if not tensor.is_cuda:
        raise RuntimeError(f"HGDN megakernel requires `{name}` to be a CUDA tensor")
    if tensor.dtype != dtype:
        raise RuntimeError(
            f"HGDN megakernel requires `{name}` dtype {dtype}, got {tensor.dtype}"
        )
    if not tensor.is_contiguous():
        raise RuntimeError(
            f"HGDN megakernel requires `{name}` to be contiguous in memory"
        )
    if device is not None and tensor.device != device:
        raise RuntimeError(
            "HGDN megakernel requires all tensors on the same CUDA device, "
            f"but `{name}` is on {tensor.device} and expected {device}"
        )


def _validate_forward_inputs(
    x: Tensor,
    w_qkv: Tensor,
    w_a: Tensor,
    w_b: Tensor,
    w_g: Tensor,
    w_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
) -> None:
    """Validate the forward tensor contract without inserting hidden kernels.

    :param Tensor x: Input activations.
    :param Tensor w_qkv: Packed qkv projection weight.
    :param Tensor w_a: Decay projection weight.
    :param Tensor w_b: Beta projection weight.
    :param Tensor w_g: Output-gate projection weight.
    :param Tensor w_out: Dense output projection weight.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Learnable decay magnitudes.
    :param Tensor dt_bias: Learnable decay biases.
    """
    device = x.device
    _require_cuda_tensor("x", x, dtype=torch.bfloat16)
    _require_cuda_tensor("w_qkv", w_qkv, dtype=torch.bfloat16, device=device)
    _require_cuda_tensor("w_a", w_a, dtype=torch.bfloat16, device=device)
    _require_cuda_tensor("w_b", w_b, dtype=torch.bfloat16, device=device)
    _require_cuda_tensor("w_g", w_g, dtype=torch.bfloat16, device=device)
    _require_cuda_tensor("w_out", w_out, dtype=torch.bfloat16, device=device)
    _require_cuda_tensor("conv_w", conv_w, dtype=torch.bfloat16, device=device)
    _require_cuda_tensor("A_log", A_log, dtype=torch.float32, device=device)
    _require_cuda_tensor("dt_bias", dt_bias, dtype=torch.float32, device=device)


def _meta_empty(
    example: Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Allocate a contiguous fake/meta tensor from explicit shape metadata.

    :param Tensor example: Tensor whose device/type family should be mirrored.
    :param tuple[int, ...] shape: Target output shape.
    :param torch.dtype | None dtype: Optional dtype override, defaults to `None`.
    :return Tensor: Empty tensor with the requested metadata.
    """
    return torch.empty(
        shape,
        device=example.device,
        dtype=example.dtype if dtype is None else dtype,
    )


_HGDN_MEGAKERNEL_V1_LIB = torch.library.Library("hgdn_megakernel_v1", "DEF")
_HGDN_MEGAKERNEL_V1_LIB.define(
    "run("
    "Tensor x, Tensor w_qkv, Tensor w_a, Tensor w_b, Tensor w_g, Tensor w_out, "
    "Tensor conv_w, Tensor A_log, Tensor dt_bias, int n_heads, int head_k_dim, "
    "int head_v_dim, int conv_size, bool allow_neg_eigval"
    ") -> ("
    "Tensor y, Tensor qkv, Tensor pre, Tensor q_norm, Tensor k_norm, Tensor v_post, "
    "Tensor inv_q, Tensor inv_k, Tensor g_pre, Tensor beta_pre, Tensor g_log, "
    "Tensor beta, Tensor g_out, Tensor o_raw, Tensor o_norm, Tensor z, Tensor state_ckpt)"
)
_HGDN_MEGAKERNEL_V1_LIB.define(
    "run_backward("
    "Tensor grad_y, Tensor x, Tensor w_qkv, Tensor w_a, Tensor w_b, Tensor w_g, "
    "Tensor w_out, Tensor conv_w, Tensor A_log, Tensor dt_bias, Tensor qkv, Tensor pre, "
    "Tensor q_norm, Tensor k_norm, Tensor v_post, Tensor inv_q, Tensor inv_k, "
    "Tensor g_pre, Tensor beta_pre, Tensor g_log, Tensor beta, Tensor g_out, "
    "Tensor o_raw, Tensor o_norm, Tensor z, Tensor state_ckpt, int n_heads, "
    "int head_k_dim, int head_v_dim, int conv_size, bool allow_neg_eigval"
    ") -> ("
    "Tensor dx, Tensor dw_qkv, Tensor dw_a, Tensor dw_b, Tensor dw_g, Tensor dw_out, "
    "Tensor dconv_w, Tensor dA_log, Tensor ddt_bias)"
)


def _run_megakernel_forward(
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
) -> tuple[Tensor, ...]:
    """Run the owned megakernel forward implementation.

    :return tuple[Tensor, ...]: Forward output plus saved activations.
    """
    ext = _require_loaded_extension()
    _validate_forward_inputs(
        x,
        w_qkv,
        w_a,
        w_b,
        w_g,
        w_out,
        conv_w,
        A_log,
        dt_bias,
    )
    return tuple(
        ext.forward(
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
    )


def _run_megakernel_backward(
    grad_y: Tensor,
    x: Tensor,
    w_qkv: Tensor,
    w_a: Tensor,
    w_b: Tensor,
    w_g: Tensor,
    w_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    qkv: Tensor,
    pre: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    v_post: Tensor,
    inv_q: Tensor,
    inv_k: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_log: Tensor,
    beta: Tensor,
    g_out: Tensor,
    o_raw: Tensor,
    o_norm: Tensor,
    z: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """Run the owned megakernel backward implementation.

    :return tuple[Tensor, ...]: Gradients for the differentiable inputs.
    """
    ext = _require_loaded_extension()
    _require_cuda_tensor("grad_y", grad_y, dtype=torch.bfloat16, device=x.device)
    return tuple(
        ext.backward(
            grad_y,
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
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            int(conv_size),
            bool(allow_neg_eigval),
        )
    )


@torch.library.impl("hgdn_megakernel_v1::run", "CPU")
def _hgdn_megakernel_run_cpu(
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
) -> tuple[Tensor, ...]:
    """CPU stub for the compile-visible megakernel op."""
    del (
        x,
        w_qkv,
        w_a,
        w_b,
        w_g,
        w_out,
        conv_w,
        A_log,
        dt_bias,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        allow_neg_eigval,
    )
    raise RuntimeError("HGDN megakernel only supports CUDA execution")


@torch.library.impl("hgdn_megakernel_v1::run", "CUDA")
def _hgdn_megakernel_run_cuda(
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
) -> tuple[Tensor, ...]:
    """CUDA implementation for the compile-visible megakernel op."""
    return _run_megakernel_forward(
        x,
        w_qkv,
        w_a,
        w_b,
        w_g,
        w_out,
        conv_w,
        A_log,
        dt_bias,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        allow_neg_eigval,
    )


@torch.library.impl("hgdn_megakernel_v1::run_backward", "CPU")
def _hgdn_megakernel_run_backward_cpu(
    grad_y: Tensor,
    x: Tensor,
    w_qkv: Tensor,
    w_a: Tensor,
    w_b: Tensor,
    w_g: Tensor,
    w_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    qkv: Tensor,
    pre: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    v_post: Tensor,
    inv_q: Tensor,
    inv_k: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_log: Tensor,
    beta: Tensor,
    g_out: Tensor,
    o_raw: Tensor,
    o_norm: Tensor,
    z: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """CPU stub for the compile-visible megakernel backward op."""
    del (
        grad_y,
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
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        allow_neg_eigval,
    )
    raise RuntimeError("HGDN megakernel only supports CUDA execution")


@torch.library.impl("hgdn_megakernel_v1::run_backward", "CUDA")
def _hgdn_megakernel_run_backward_cuda(
    grad_y: Tensor,
    x: Tensor,
    w_qkv: Tensor,
    w_a: Tensor,
    w_b: Tensor,
    w_g: Tensor,
    w_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    qkv: Tensor,
    pre: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    v_post: Tensor,
    inv_q: Tensor,
    inv_k: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_log: Tensor,
    beta: Tensor,
    g_out: Tensor,
    o_raw: Tensor,
    o_norm: Tensor,
    z: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """CUDA implementation for the compile-visible megakernel backward op."""
    return _run_megakernel_backward(
        grad_y,
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
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        allow_neg_eigval,
    )


@torch.library.register_fake("hgdn_megakernel_v1::run")
def _hgdn_megakernel_run_fake(
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
) -> tuple[Tensor, ...]:
    """Meta kernel for the compile-visible megakernel forward op."""
    del conv_w, w_a, w_b, w_g, w_out, A_log, dt_bias, conv_size, allow_neg_eigval
    batch, seq, d_model = x.shape
    heads = int(n_heads)
    dk = int(head_k_dim)
    dv = int(head_v_dim)
    channels = heads * (2 * dk + dv)
    n_chunks = (seq + _REC_CHUNK_T - 1) // _REC_CHUNK_T
    return (
        _meta_empty(x, x.shape),
        _meta_empty(x, (batch, seq, channels)),
        _meta_empty(x, (batch, seq, channels)),
        _meta_empty(x, (batch, seq, heads, dk)),
        _meta_empty(x, (batch, seq, heads, dk)),
        _meta_empty(x, (batch, seq, heads, dv)),
        _meta_empty(x, (batch, seq, heads), dtype=torch.float32),
        _meta_empty(x, (batch, seq, heads), dtype=torch.float32),
        _meta_empty(x, (batch, seq, heads)),
        _meta_empty(x, (batch, seq, heads)),
        _meta_empty(x, (batch, seq, heads)),
        _meta_empty(x, (batch, seq, heads)),
        _meta_empty(x, (batch, seq, d_model)),
        _meta_empty(x, (batch, seq, heads, dv)),
        _meta_empty(x, (batch, seq, heads, dv)),
        _meta_empty(x, (batch, seq, d_model)),
        _meta_empty(x, (batch, n_chunks, heads, dk, dv), dtype=torch.float32),
    )


@torch.library.register_fake("hgdn_megakernel_v1::run_backward")
def _hgdn_megakernel_run_backward_fake(
    grad_y: Tensor,
    x: Tensor,
    w_qkv: Tensor,
    w_a: Tensor,
    w_b: Tensor,
    w_g: Tensor,
    w_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    qkv: Tensor,
    pre: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    v_post: Tensor,
    inv_q: Tensor,
    inv_k: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_log: Tensor,
    beta: Tensor,
    g_out: Tensor,
    o_raw: Tensor,
    o_norm: Tensor,
    z: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """Meta kernel for the compile-visible megakernel backward op."""
    del (
        grad_y,
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
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        allow_neg_eigval,
    )
    return (
        _meta_empty(x, tuple(x.shape)),
        _meta_empty(w_qkv, tuple(w_qkv.shape)),
        _meta_empty(w_a, tuple(w_a.shape)),
        _meta_empty(w_b, tuple(w_b.shape)),
        _meta_empty(w_g, tuple(w_g.shape)),
        _meta_empty(w_out, tuple(w_out.shape)),
        _meta_empty(conv_w, tuple(conv_w.shape)),
        _meta_empty(A_log, tuple(A_log.shape)),
        _meta_empty(dt_bias, tuple(dt_bias.shape)),
    )


def _setup_hgdn_megakernel_context(
    ctx: Any,
    inputs: tuple[Tensor, ...],
    output: tuple[Tensor, ...],
) -> None:
    """Save tensors needed by the registered megakernel backward formula."""
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
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        allow_neg_eigval,
    ) = inputs
    (
        _y,
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
    ) = output
    ctx.set_materialize_grads(False)
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
    )
    ctx.n_heads = int(n_heads)
    ctx.head_k_dim = int(head_k_dim)
    ctx.head_v_dim = int(head_v_dim)
    ctx.conv_size = int(conv_size)
    ctx.allow_neg_eigval = bool(allow_neg_eigval)


def _hgdn_megakernel_backward_formula(
    ctx: Any,
    grad_y: Tensor | None,
    _grad_qkv: Tensor | None,
    _grad_pre: Tensor | None,
    _grad_q_norm: Tensor | None,
    _grad_k_norm: Tensor | None,
    _grad_v_post: Tensor | None,
    _grad_inv_q: Tensor | None,
    _grad_inv_k: Tensor | None,
    _grad_g_pre: Tensor | None,
    _grad_beta_pre: Tensor | None,
    _grad_g_log: Tensor | None,
    _grad_beta: Tensor | None,
    _grad_g_out: Tensor | None,
    _grad_o_raw: Tensor | None,
    _grad_o_norm: Tensor | None,
    _grad_z: Tensor | None,
    _grad_state_ckpt: Tensor | None,
) -> tuple[Tensor | None, ...]:
    """Backward formula for the compile-visible megakernel op."""
    del (
        _grad_qkv,
        _grad_pre,
        _grad_q_norm,
        _grad_k_norm,
        _grad_v_post,
        _grad_inv_q,
        _grad_inv_k,
        _grad_g_pre,
        _grad_beta_pre,
        _grad_g_log,
        _grad_beta,
        _grad_g_out,
        _grad_o_raw,
        _grad_o_norm,
        _grad_z,
        _grad_state_ckpt,
    )
    if grad_y is None:
        return (None,) * 14
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
    grads = torch.ops.hgdn_megakernel_v1.run_backward(
        grad_y,
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
        int(ctx.n_heads),
        int(ctx.head_k_dim),
        int(ctx.head_v_dim),
        int(ctx.conv_size),
        bool(ctx.allow_neg_eigval),
    )
    return (*grads, None, None, None, None, None)


torch.library.register_autograd(
    "hgdn_megakernel_v1::run",
    _hgdn_megakernel_backward_formula,
    setup_context=_setup_hgdn_megakernel_context,
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
    """Run the HGDN megakernel through the compile-visible custom-op path.

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
    y, *_saved = torch.ops.hgdn_megakernel_v1.run(
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
    return y


def run_from_gated_delta_net(module: torch.nn.Module, x: Tensor) -> Tensor:
    """Run the HGDN megakernel from a configured `GatedDeltaNet` module.

    :param torch.nn.Module module: Configured packed HGDN module.
    :param Tensor x: Input activations.
    :raises ValueError: If the module config is incompatible with the megakernel.
    :return Tensor: HGDN block output.
    """
    if not bool(getattr(module, "use_packed_qkv_proj", False)):
        raise ValueError("HGDN megakernel requires use_packed_qkv_proj=True")
    if not bool(getattr(module, "use_packed_qkv_conv", False)):
        raise ValueError("HGDN megakernel requires use_packed_qkv_conv=True")
    if not bool(getattr(module, "gates_fp32", True)):
        raise ValueError("HGDN megakernel requires gates_fp32=True")
    if not bool(getattr(module, "output_norm_fp32", True)):
        raise ValueError("HGDN megakernel requires output_norm_fp32=True")
    if module.w_a.weight.dtype != torch.bfloat16:
        raise ValueError(
            "HGDN megakernel requires bf16 `w_a`; set GDN_CONTROL_PROJ_FP32=0"
        )
    if module.w_b.weight.dtype != torch.bfloat16:
        raise ValueError(
            "HGDN megakernel requires bf16 `w_b`; set GDN_CONTROL_PROJ_FP32=0"
        )
    if module.w_g.weight.dtype != torch.bfloat16:
        raise ValueError(
            "HGDN megakernel requires bf16 `w_g`; set GDN_CONTROL_PROJ_FP32=0"
        )
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
