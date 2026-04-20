"""Python wrapper for the optional HGDN CUDA extension."""

from __future__ import annotations

import importlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_EXTENSION_NAME = "hgdn_megakernel_ext"
_WARNED_KEYS: set[str] = set()
_REC_CHUNK_T_DEFAULT = 8
_CACHED_EXTENSION: Any | None = None
_EXTENSION_LOAD_ATTEMPTED = False
_REC_CHUNK_T_MAX_CACHE: int | None = None
_MEGAKERNEL_REC_CHUNK_T_ENV = "GDN_MEGAKERNEL_REC_CHUNK_T"
_COREKERNEL_REC_CHUNK_T_ENV = "GDN_COREKERNEL_REC_CHUNK_T"


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


def _load_extension() -> Any | None:
    """Load the optional HGDN megakernel extension when available.

    Import order:

    1. Reuse an in-place build from `setup_hgdn_megakernel.py build_ext --inplace`.
    2. Optionally JIT-build from source when `GDN_MEGAKERNEL_ALLOW_JIT_BUILD=1`.

    :return Any | None: Imported extension module, or `None`.
    """
    global _CACHED_EXTENSION, _EXTENSION_LOAD_ATTEMPTED
    if _EXTENSION_LOAD_ATTEMPTED:
        return _CACHED_EXTENSION
    _EXTENSION_LOAD_ATTEMPTED = True
    try:
        _CACHED_EXTENSION = importlib.import_module(_EXTENSION_NAME)
        return _CACHED_EXTENSION
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
        _CACHED_EXTENSION = load(
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
        return _CACHED_EXTENSION
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
    ext = _load_extension()
    build_config: dict[str, object] | None = None
    module_file = getattr(ext, "__file__", None) if ext is not None else None
    if ext is not None and hasattr(ext, "build_config_json"):
        build_config = json.loads(str(ext.build_config_json()))
    return {
        "loaded": ext is not None,
        "allow_jit_build": bool(
            int(os.environ.get("GDN_MEGAKERNEL_ALLOW_JIT_BUILD", "0"))
        ),
        "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "module_file": module_file,
        "build_config": build_config,
        "rec_chunk_t_max": rec_chunk_t_max()
        if ext is not None
        else _REC_CHUNK_T_DEFAULT,
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


def rec_chunk_t_max() -> int:
    """Return the compiled megakernel checkpoint-chunk upper bound.

    :return int: Maximum runtime `rec_chunk_t` supported by the loaded extension.
    """
    global _REC_CHUNK_T_MAX_CACHE
    if _REC_CHUNK_T_MAX_CACHE is not None:
        return _REC_CHUNK_T_MAX_CACHE
    ext = _require_loaded_extension()
    _REC_CHUNK_T_MAX_CACHE = int(ext.rec_chunk_t_max())
    return _REC_CHUNK_T_MAX_CACHE


def resolve_runtime_rec_chunk_t(
    module: torch.nn.Module | None = None,
    *,
    prefer_corekernel: bool = False,
) -> int:
    """Return the runtime checkpoint cadence for an owned HGDN CUDA path.

    Resolution order:

    Megakernel path:

    1. `module.megakernel_rec_chunk_t` when present.
    2. `GDN_MEGAKERNEL_REC_CHUNK_T`.
    3. Current default cadence.

    Core-kernel path:

    1. `module.corekernel_rec_chunk_t` when present.
    2. `module.megakernel_rec_chunk_t` when present.
    3. `GDN_COREKERNEL_REC_CHUNK_T`.
    4. `GDN_MEGAKERNEL_REC_CHUNK_T`.
    5. Current default cadence.

    :param torch.nn.Module | None module: Optional HGDN module, defaults to `None`.
    :param bool prefer_corekernel: Whether to prefer the core-kernel alias and
        module attribute, defaults to False.
    :raises ValueError: If the requested cadence is not positive.
    :return int: Runtime checkpoint cadence.
    """
    if (
        module is not None
        and prefer_corekernel
        and hasattr(module, "corekernel_rec_chunk_t")
    ):
        value = int(getattr(module, "corekernel_rec_chunk_t"))
    elif module is not None and hasattr(module, "megakernel_rec_chunk_t"):
        value = int(getattr(module, "megakernel_rec_chunk_t"))
    elif prefer_corekernel and _COREKERNEL_REC_CHUNK_T_ENV in os.environ:
        value = int(os.environ[_COREKERNEL_REC_CHUNK_T_ENV])
    elif _MEGAKERNEL_REC_CHUNK_T_ENV in os.environ:
        value = int(os.environ[_MEGAKERNEL_REC_CHUNK_T_ENV])
    else:
        value = _REC_CHUNK_T_DEFAULT
    if value <= 0:
        env_name = (
            _COREKERNEL_REC_CHUNK_T_ENV
            if prefer_corekernel
            else _MEGAKERNEL_REC_CHUNK_T_ENV
        )
        fallback = (
            f" (fallback {_MEGAKERNEL_REC_CHUNK_T_ENV})" if prefer_corekernel else ""
        )
        raise ValueError(
            f"HGDN {'core kernel' if prefer_corekernel else 'megakernel'} requires "
            f"{env_name} > 0{fallback}, got {value}."
        )
    return value


def _resolve_rec_chunk_t(module: torch.nn.Module | None = None) -> int:
    """Return the runtime checkpoint cadence for the megakernel path.

    :param torch.nn.Module | None module: Optional HGDN module, defaults to `None`.
    :raises ValueError: If the requested cadence is not positive.
    :return int: Runtime checkpoint cadence.
    """
    return resolve_runtime_rec_chunk_t(module, prefer_corekernel=False)


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


def _validate_core_forward_inputs(
    qkv: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
) -> None:
    """Validate the HGDN core-kernel tensor contract without hidden kernels.

    :param Tensor qkv: Packed qkv activations.
    :param Tensor g_pre: Decay preactivations.
    :param Tensor beta_pre: Beta preactivations.
    :param Tensor g_out: Output-gate preactivations.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Learnable decay magnitudes.
    :param Tensor dt_bias: Learnable decay biases.
    """
    device = qkv.device
    _require_cuda_tensor("qkv", qkv, dtype=torch.bfloat16)
    _require_cuda_tensor("g_pre", g_pre, dtype=torch.bfloat16, device=device)
    _require_cuda_tensor("beta_pre", beta_pre, dtype=torch.bfloat16, device=device)
    _require_cuda_tensor("g_out", g_out, dtype=torch.bfloat16, device=device)
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
    "int head_v_dim, int conv_size, int rec_chunk_t, bool allow_neg_eigval"
    ") -> ("
    "Tensor y, Tensor qkv, Tensor g_pre, Tensor beta_pre, Tensor g_log, "
    "Tensor beta, Tensor g_out, Tensor o_raw, Tensor state_ckpt)"
)
_HGDN_MEGAKERNEL_V1_LIB.define(
    "run_backward("
    "Tensor grad_y, Tensor x, Tensor w_qkv, Tensor w_a, Tensor w_b, Tensor w_g, "
    "Tensor w_out, Tensor conv_w, Tensor A_log, Tensor dt_bias, Tensor qkv, "
    "Tensor g_pre, Tensor beta_pre, Tensor g_log, Tensor beta, Tensor g_out, "
    "Tensor o_raw, Tensor state_ckpt, int n_heads, "
    "int head_k_dim, int head_v_dim, int conv_size, int rec_chunk_t, bool allow_neg_eigval"
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
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """Run the owned megakernel forward implementation.

    :param Tensor x: Input activations.
    :param Tensor w_qkv: Packed qkv projection weight.
    :param Tensor w_a: Decay projection weight.
    :param Tensor w_b: Beta projection weight.
    :param Tensor w_g: Output-gate projection weight.
    :param Tensor w_out: Dense output projection weight.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Decay magnitude parameter.
    :param Tensor dt_bias: Decay bias parameter.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Forward output plus saved activations.
    """
    ext = _require_loaded_extension()
    max_chunk_t = rec_chunk_t_max()
    if rec_chunk_t > max_chunk_t:
        raise RuntimeError(
            "HGDN megakernel rec_chunk_t exceeds the compiled maximum: "
            f"got {rec_chunk_t}, max {max_chunk_t}."
        )
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
            int(rec_chunk_t),
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
    g_pre: Tensor,
    beta_pre: Tensor,
    g_log: Tensor,
    beta: Tensor,
    g_out: Tensor,
    o_raw: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """Run the owned megakernel backward implementation.

    :param Tensor grad_y: Output gradient.
    :param Tensor x: Input activations.
    :param Tensor w_qkv: Packed qkv projection weight.
    :param Tensor w_a: Decay projection weight.
    :param Tensor w_b: Beta projection weight.
    :param Tensor w_g: Output-gate projection weight.
    :param Tensor w_out: Dense output projection weight.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Decay magnitude parameter.
    :param Tensor dt_bias: Decay bias parameter.
    :param Tensor qkv: Saved packed qkv activations.
    :param Tensor g_pre: Saved decay preactivation.
    :param Tensor beta_pre: Saved beta preactivation.
    :param Tensor g_log: Saved log-decay value.
    :param Tensor beta: Saved beta value.
    :param Tensor g_out: Saved output-gate preactivation.
    :param Tensor o_raw: Saved recurrent readout before output RMSNorm.
    :param Tensor state_ckpt: Saved chunk-start recurrent checkpoints.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Gradients for the differentiable inputs.
    """
    ext = _require_loaded_extension()
    max_chunk_t = rec_chunk_t_max()
    if rec_chunk_t > max_chunk_t:
        raise RuntimeError(
            "HGDN megakernel rec_chunk_t exceeds the compiled maximum: "
            f"got {rec_chunk_t}, max {max_chunk_t}."
        )
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
            g_pre,
            beta_pre,
            g_log,
            beta,
            g_out,
            o_raw,
            state_ckpt,
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            int(conv_size),
            int(rec_chunk_t),
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
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """CPU stub for the compile-visible megakernel op.

    :param Tensor x: Input activations.
    :param Tensor w_qkv: Packed qkv projection weight.
    :param Tensor w_a: Decay projection weight.
    :param Tensor w_b: Beta projection weight.
    :param Tensor w_g: Output-gate projection weight.
    :param Tensor w_out: Dense output projection weight.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Decay magnitude parameter.
    :param Tensor dt_bias: Decay bias parameter.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :raises RuntimeError: Always, because the megakernel is CUDA-only.
    :return tuple[Tensor, ...]: Never returns.
    """
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
        rec_chunk_t,
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
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """CUDA implementation for the compile-visible megakernel op.

    :param Tensor x: Input activations.
    :param Tensor w_qkv: Packed qkv projection weight.
    :param Tensor w_a: Decay projection weight.
    :param Tensor w_b: Beta projection weight.
    :param Tensor w_g: Output-gate projection weight.
    :param Tensor w_out: Dense output projection weight.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Decay magnitude parameter.
    :param Tensor dt_bias: Decay bias parameter.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Forward output plus saved activations.
    """
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
        rec_chunk_t,
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
    g_pre: Tensor,
    beta_pre: Tensor,
    g_log: Tensor,
    beta: Tensor,
    g_out: Tensor,
    o_raw: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """CPU stub for the compile-visible megakernel backward op.

    :param Tensor grad_y: Output gradient.
    :param Tensor x: Input activations.
    :param Tensor w_qkv: Packed qkv projection weight.
    :param Tensor w_a: Decay projection weight.
    :param Tensor w_b: Beta projection weight.
    :param Tensor w_g: Output-gate projection weight.
    :param Tensor w_out: Dense output projection weight.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Decay magnitude parameter.
    :param Tensor dt_bias: Decay bias parameter.
    :param Tensor qkv: Saved packed qkv activations.
    :param Tensor g_pre: Saved decay preactivation.
    :param Tensor beta_pre: Saved beta preactivation.
    :param Tensor g_log: Saved log-decay value.
    :param Tensor beta: Saved beta value.
    :param Tensor g_out: Saved output-gate preactivation.
    :param Tensor o_raw: Saved recurrent readout before output RMSNorm.
    :param Tensor state_ckpt: Saved chunk-start recurrent checkpoints.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :raises RuntimeError: Always, because the megakernel is CUDA-only.
    :return tuple[Tensor, ...]: Never returns.
    """
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
        g_pre,
        beta_pre,
        g_log,
        beta,
        g_out,
        o_raw,
        state_ckpt,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        rec_chunk_t,
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
    g_pre: Tensor,
    beta_pre: Tensor,
    g_log: Tensor,
    beta: Tensor,
    g_out: Tensor,
    o_raw: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """CUDA implementation for the compile-visible megakernel backward op.

    :param Tensor grad_y: Output gradient.
    :param Tensor x: Input activations.
    :param Tensor w_qkv: Packed qkv projection weight.
    :param Tensor w_a: Decay projection weight.
    :param Tensor w_b: Beta projection weight.
    :param Tensor w_g: Output-gate projection weight.
    :param Tensor w_out: Dense output projection weight.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Decay magnitude parameter.
    :param Tensor dt_bias: Decay bias parameter.
    :param Tensor qkv: Saved packed qkv activations.
    :param Tensor g_pre: Saved decay preactivation.
    :param Tensor beta_pre: Saved beta preactivation.
    :param Tensor g_log: Saved log-decay value.
    :param Tensor beta: Saved beta value.
    :param Tensor g_out: Saved output-gate preactivation.
    :param Tensor o_raw: Saved recurrent readout before output RMSNorm.
    :param Tensor state_ckpt: Saved chunk-start recurrent checkpoints.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Gradients for the differentiable inputs.
    """
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
        g_pre,
        beta_pre,
        g_log,
        beta,
        g_out,
        o_raw,
        state_ckpt,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        rec_chunk_t,
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
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """Meta kernel for the compile-visible megakernel forward op.

    :param Tensor x: Input activations.
    :param Tensor w_qkv: Packed qkv projection weight.
    :param Tensor w_a: Decay projection weight.
    :param Tensor w_b: Beta projection weight.
    :param Tensor w_g: Output-gate projection weight.
    :param Tensor w_out: Dense output projection weight.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Decay magnitude parameter.
    :param Tensor dt_bias: Decay bias parameter.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Meta outputs matching the CUDA forward schema.
    """
    del conv_w, w_a, w_b, w_g, w_out, A_log, dt_bias, conv_size, allow_neg_eigval
    batch, seq, d_model = x.shape
    heads = int(n_heads)
    dk = int(head_k_dim)
    dv = int(head_v_dim)
    channels = heads * (2 * dk + dv)
    n_chunks = (seq + int(rec_chunk_t) - 1) // int(rec_chunk_t)
    return (
        _meta_empty(x, x.shape),
        _meta_empty(x, (batch, seq, channels)),
        _meta_empty(x, (batch, seq, heads)),
        _meta_empty(x, (batch, seq, heads)),
        _meta_empty(x, (batch, seq, heads)),
        _meta_empty(x, (batch, seq, heads)),
        _meta_empty(x, (batch, seq, d_model)),
        _meta_empty(x, (batch, seq, heads, dv)),
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
    g_pre: Tensor,
    beta_pre: Tensor,
    g_log: Tensor,
    beta: Tensor,
    g_out: Tensor,
    o_raw: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """Meta kernel for the compile-visible megakernel backward op.

    :param Tensor grad_y: Output gradient.
    :param Tensor x: Input activations.
    :param Tensor w_qkv: Packed qkv projection weight.
    :param Tensor w_a: Decay projection weight.
    :param Tensor w_b: Beta projection weight.
    :param Tensor w_g: Output-gate projection weight.
    :param Tensor w_out: Dense output projection weight.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Decay magnitude parameter.
    :param Tensor dt_bias: Decay bias parameter.
    :param Tensor qkv: Saved packed qkv activations.
    :param Tensor g_pre: Saved decay preactivation.
    :param Tensor beta_pre: Saved beta preactivation.
    :param Tensor g_log: Saved log-decay value.
    :param Tensor beta: Saved beta value.
    :param Tensor g_out: Saved output-gate preactivation.
    :param Tensor o_raw: Saved recurrent readout before output RMSNorm.
    :param Tensor state_ckpt: Saved chunk-start recurrent checkpoints.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Meta gradients matching the CUDA backward schema.
    """
    del (
        grad_y,
        qkv,
        g_pre,
        beta_pre,
        g_log,
        beta,
        g_out,
        o_raw,
        state_ckpt,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        rec_chunk_t,
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
    """Save tensors needed by the registered megakernel backward formula.

    :param Any ctx: Custom-op autograd context.
    :param tuple[Tensor, ...] inputs: Forward inputs passed to the custom op.
    :param tuple[Tensor, ...] output: Forward outputs returned by the custom op.
    :return None: Updates `ctx` in place.
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
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        rec_chunk_t,
        allow_neg_eigval,
    ) = inputs
    (
        _y,
        qkv,
        g_pre,
        beta_pre,
        g_log,
        beta,
        g_out,
        o_raw,
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
        g_pre,
        beta_pre,
        g_log,
        beta,
        g_out,
        o_raw,
        state_ckpt,
    )
    ctx.n_heads = int(n_heads)
    ctx.head_k_dim = int(head_k_dim)
    ctx.head_v_dim = int(head_v_dim)
    ctx.conv_size = int(conv_size)
    ctx.rec_chunk_t = int(rec_chunk_t)
    ctx.allow_neg_eigval = bool(allow_neg_eigval)


def _hgdn_megakernel_backward_formula(
    ctx: Any,
    grad_y: Tensor | None,
    _grad_qkv: Tensor | None,
    _grad_g_pre: Tensor | None,
    _grad_beta_pre: Tensor | None,
    _grad_g_log: Tensor | None,
    _grad_beta: Tensor | None,
    _grad_g_out: Tensor | None,
    _grad_o_raw: Tensor | None,
    _grad_state_ckpt: Tensor | None,
) -> tuple[Tensor | None, ...]:
    """Backward formula for the compile-visible megakernel op.

    :param Any ctx: Custom-op autograd context.
    :param Tensor | None grad_y: Output gradient for `y`.
    :param Tensor | None _grad_qkv: Unused gradient placeholder for saved qkv.
    :param Tensor | None _grad_g_pre: Unused gradient placeholder for saved g_pre.
    :param Tensor | None _grad_beta_pre: Unused gradient placeholder for saved beta_pre.
    :param Tensor | None _grad_g_log: Unused gradient placeholder for saved g_log.
    :param Tensor | None _grad_beta: Unused gradient placeholder for saved beta.
    :param Tensor | None _grad_g_out: Unused gradient placeholder for saved g_out.
    :param Tensor | None _grad_o_raw: Unused gradient placeholder for saved o_raw.
    :param Tensor | None _grad_state_ckpt: Unused gradient placeholder for saved checkpoints.
    :return tuple[Tensor | None, ...]: Input gradients plus `None` for scalar args.
    """
    del (
        _grad_qkv,
        _grad_g_pre,
        _grad_beta_pre,
        _grad_g_log,
        _grad_beta,
        _grad_g_out,
        _grad_o_raw,
        _grad_state_ckpt,
    )
    if grad_y is None:
        return (None,) * 15
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
        g_pre,
        beta_pre,
        g_log,
        beta,
        g_out,
        o_raw,
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
        g_pre,
        beta_pre,
        g_log,
        beta,
        g_out,
        o_raw,
        state_ckpt,
        int(ctx.n_heads),
        int(ctx.head_k_dim),
        int(ctx.head_v_dim),
        int(ctx.conv_size),
        int(ctx.rec_chunk_t),
        bool(ctx.allow_neg_eigval),
    )
    return (*grads, None, None, None, None, None, None)


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
    rec_chunk_t: int | None = None,
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
    :param int | None rec_chunk_t: Runtime checkpoint cadence, defaults to `None`.
    :param bool allow_neg_eigval: Whether to scale beta by `2.0`, defaults to True.
    :return Tensor: HGDN block output shaped `(batch, seq, d_model)`.
    """
    rec_chunk_t = (
        _resolve_rec_chunk_t(None) if rec_chunk_t is None else int(rec_chunk_t)
    )
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
        int(rec_chunk_t),
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
    rec_chunk_t = _resolve_rec_chunk_t(module)
    max_chunk_t = rec_chunk_t_max()
    if rec_chunk_t > max_chunk_t:
        raise ValueError(
            "HGDN megakernel rec_chunk_t exceeds the compiled maximum: "
            f"got {rec_chunk_t}, max {max_chunk_t}. "
            "Rebuild the extension with a larger HGDN_REC_CHUNK_T if needed."
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
        rec_chunk_t=rec_chunk_t,
        allow_neg_eigval=bool(module.allow_neg_eigval),
    )


_HGDN_COREKERNEL_V1_LIB = torch.library.Library("hgdn_corekernel_v1", "DEF")
_HGDN_COREKERNEL_V1_LIB.define(
    "run("
    "Tensor qkv, Tensor g_pre, Tensor beta_pre, Tensor g_out, Tensor conv_w, "
    "Tensor A_log, Tensor dt_bias, int n_heads, int head_k_dim, int head_v_dim, "
    "int conv_size, int rec_chunk_t, bool allow_neg_eigval"
    ") -> ("
    "Tensor z, Tensor g_log, Tensor beta, Tensor o_raw, Tensor state_ckpt)"
)
_HGDN_COREKERNEL_V1_LIB.define(
    "run_backward("
    "Tensor grad_z, Tensor qkv, Tensor g_pre, Tensor beta_pre, Tensor g_out, "
    "Tensor conv_w, Tensor A_log, Tensor dt_bias, Tensor g_log, Tensor beta, "
    "Tensor o_raw, Tensor state_ckpt, int n_heads, int head_k_dim, int head_v_dim, "
    "int conv_size, int rec_chunk_t, bool allow_neg_eigval"
    ") -> ("
    "Tensor dqkv, Tensor dg_pre, Tensor dbeta_pre, Tensor dg_out, Tensor dconv_w, "
    "Tensor dA_log, Tensor ddt_bias)"
)


def _run_corekernel_forward(
    qkv: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """Run the owned HGDN core-kernel forward implementation.

    :param Tensor qkv: Packed qkv activations.
    :param Tensor g_pre: Decay preactivations.
    :param Tensor beta_pre: Beta preactivations.
    :param Tensor g_out: Output-gate preactivations.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Decay magnitude parameter.
    :param Tensor dt_bias: Decay bias parameter.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Core output plus saved activations.
    """
    ext = _require_loaded_extension()
    max_chunk_t = rec_chunk_t_max()
    if rec_chunk_t > max_chunk_t:
        raise RuntimeError(
            "HGDN core kernel rec_chunk_t exceeds the compiled maximum: "
            f"got {rec_chunk_t}, max {max_chunk_t}."
        )
    _validate_core_forward_inputs(qkv, g_pre, beta_pre, g_out, conv_w, A_log, dt_bias)
    return tuple(
        ext.core_forward(
            qkv,
            g_pre,
            beta_pre,
            g_out,
            conv_w,
            A_log,
            dt_bias,
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            int(conv_size),
            int(rec_chunk_t),
            bool(allow_neg_eigval),
        )
    )


def _run_corekernel_backward(
    grad_z: Tensor,
    qkv: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    g_log: Tensor,
    beta: Tensor,
    o_raw: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """Run the owned HGDN core-kernel backward implementation.

    :param Tensor grad_z: Gradient for the core output.
    :param Tensor qkv: Packed qkv activations.
    :param Tensor g_pre: Saved decay preactivations.
    :param Tensor beta_pre: Saved beta preactivations.
    :param Tensor g_out: Saved output-gate preactivations.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Decay magnitude parameter.
    :param Tensor dt_bias: Decay bias parameter.
    :param Tensor g_log: Saved log-decay value.
    :param Tensor beta: Saved beta value.
    :param Tensor o_raw: Saved recurrent readout before output RMSNorm.
    :param Tensor state_ckpt: Saved chunk-start recurrent checkpoints.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Gradients for the differentiable inputs.
    """
    ext = _require_loaded_extension()
    max_chunk_t = rec_chunk_t_max()
    if rec_chunk_t > max_chunk_t:
        raise RuntimeError(
            "HGDN core kernel rec_chunk_t exceeds the compiled maximum: "
            f"got {rec_chunk_t}, max {max_chunk_t}."
        )
    _require_cuda_tensor("grad_z", grad_z, dtype=torch.bfloat16, device=qkv.device)
    return tuple(
        ext.core_backward(
            grad_z,
            qkv,
            g_pre,
            beta_pre,
            g_out,
            conv_w,
            A_log,
            dt_bias,
            g_log,
            beta,
            o_raw,
            state_ckpt,
            int(n_heads),
            int(head_k_dim),
            int(head_v_dim),
            int(conv_size),
            int(rec_chunk_t),
            bool(allow_neg_eigval),
        )
    )


@torch.library.impl("hgdn_corekernel_v1::run", "CPU")
def _hgdn_corekernel_run_cpu(
    qkv: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """CPU stub for the compile-visible HGDN core-kernel op.

    :param Tensor qkv: Packed qkv activations.
    :param Tensor g_pre: Decay preactivations.
    :param Tensor beta_pre: Beta preactivations.
    :param Tensor g_out: Output-gate preactivations.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Learnable decay magnitudes.
    :param Tensor dt_bias: Learnable decay biases.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Packed depthwise conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :raises RuntimeError: Always, because the core kernel is CUDA-only.
    :return tuple[Tensor, ...]: Never returns.
    """
    del (
        qkv,
        g_pre,
        beta_pre,
        g_out,
        conv_w,
        A_log,
        dt_bias,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        rec_chunk_t,
        allow_neg_eigval,
    )
    raise RuntimeError("HGDN core kernel only supports CUDA execution")


@torch.library.impl("hgdn_corekernel_v1::run", "CUDA")
def _hgdn_corekernel_run_cuda(
    qkv: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """CUDA implementation for the compile-visible HGDN core-kernel op.

    :param Tensor qkv: Packed qkv activations.
    :param Tensor g_pre: Decay preactivations.
    :param Tensor beta_pre: Beta preactivations.
    :param Tensor g_out: Output-gate preactivations.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Learnable decay magnitudes.
    :param Tensor dt_bias: Learnable decay biases.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Packed depthwise conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Forward output plus saved tensors for the
        registered autograd formula.
    """
    return _run_corekernel_forward(
        qkv,
        g_pre,
        beta_pre,
        g_out,
        conv_w,
        A_log,
        dt_bias,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        rec_chunk_t,
        allow_neg_eigval,
    )


@torch.library.impl("hgdn_corekernel_v1::run_backward", "CPU")
def _hgdn_corekernel_run_backward_cpu(
    grad_z: Tensor,
    qkv: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    g_log: Tensor,
    beta: Tensor,
    o_raw: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """CPU stub for the compile-visible HGDN core-kernel backward op.

    :param Tensor grad_z: Upstream gradient for the core output.
    :param Tensor qkv: Packed qkv activations.
    :param Tensor g_pre: Decay preactivations.
    :param Tensor beta_pre: Beta preactivations.
    :param Tensor g_out: Output-gate preactivations.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Learnable decay magnitudes.
    :param Tensor dt_bias: Learnable decay biases.
    :param Tensor g_log: Log-decay activations saved from forward.
    :param Tensor beta: Beta activations saved from forward.
    :param Tensor o_raw: Raw recurrence output saved from forward.
    :param Tensor state_ckpt: Recurrence checkpoints saved from forward.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Packed depthwise conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :raises RuntimeError: Always, because the core kernel is CUDA-only.
    :return tuple[Tensor, ...]: Never returns.
    """
    del (
        grad_z,
        qkv,
        g_pre,
        beta_pre,
        g_out,
        conv_w,
        A_log,
        dt_bias,
        g_log,
        beta,
        o_raw,
        state_ckpt,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        rec_chunk_t,
        allow_neg_eigval,
    )
    raise RuntimeError("HGDN core kernel only supports CUDA execution")


@torch.library.impl("hgdn_corekernel_v1::run_backward", "CUDA")
def _hgdn_corekernel_run_backward_cuda(
    grad_z: Tensor,
    qkv: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    g_log: Tensor,
    beta: Tensor,
    o_raw: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """CUDA implementation for the compile-visible HGDN core-kernel backward op.

    :param Tensor grad_z: Upstream gradient for the core output.
    :param Tensor qkv: Packed qkv activations.
    :param Tensor g_pre: Decay preactivations.
    :param Tensor beta_pre: Beta preactivations.
    :param Tensor g_out: Output-gate preactivations.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Learnable decay magnitudes.
    :param Tensor dt_bias: Learnable decay biases.
    :param Tensor g_log: Log-decay activations saved from forward.
    :param Tensor beta: Beta activations saved from forward.
    :param Tensor o_raw: Raw recurrence output saved from forward.
    :param Tensor state_ckpt: Recurrence checkpoints saved from forward.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Packed depthwise conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Gradients for the tensors owned by the core
        kernel contract.
    """
    return _run_corekernel_backward(
        grad_z,
        qkv,
        g_pre,
        beta_pre,
        g_out,
        conv_w,
        A_log,
        dt_bias,
        g_log,
        beta,
        o_raw,
        state_ckpt,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        rec_chunk_t,
        allow_neg_eigval,
    )


@torch.library.register_fake("hgdn_corekernel_v1::run")
def _hgdn_corekernel_run_fake(
    qkv: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """Meta kernel for the compile-visible HGDN core-kernel forward op.

    :param Tensor qkv: Packed qkv activations.
    :param Tensor g_pre: Decay preactivations.
    :param Tensor beta_pre: Beta preactivations.
    :param Tensor g_out: Output-gate preactivations.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Learnable decay magnitudes.
    :param Tensor dt_bias: Learnable decay biases.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Packed depthwise conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Fake forward outputs matching the real CUDA
        contract.
    """
    del conv_w, A_log, dt_bias, conv_size, allow_neg_eigval, g_pre, beta_pre, g_out
    batch, seq, _channels = qkv.shape
    heads = int(n_heads)
    dv = int(head_v_dim)
    n_chunks = (seq + int(rec_chunk_t) - 1) // int(rec_chunk_t)
    return (
        _meta_empty(qkv, (batch, seq, heads, dv)),
        _meta_empty(qkv, (batch, seq, heads)),
        _meta_empty(qkv, (batch, seq, heads)),
        _meta_empty(qkv, (batch, seq, heads, dv)),
        _meta_empty(
            qkv,
            (batch, n_chunks, heads, int(head_k_dim), dv),
            dtype=torch.float32,
        ),
    )


@torch.library.register_fake("hgdn_corekernel_v1::run_backward")
def _hgdn_corekernel_run_backward_fake(
    grad_z: Tensor,
    qkv: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    g_log: Tensor,
    beta: Tensor,
    o_raw: Tensor,
    state_ckpt: Tensor,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int,
    allow_neg_eigval: bool,
) -> tuple[Tensor, ...]:
    """Meta kernel for the compile-visible HGDN core-kernel backward op.

    :param Tensor grad_z: Upstream gradient for the core output.
    :param Tensor qkv: Packed qkv activations.
    :param Tensor g_pre: Decay preactivations.
    :param Tensor beta_pre: Beta preactivations.
    :param Tensor g_out: Output-gate preactivations.
    :param Tensor conv_w: Packed depthwise conv weights.
    :param Tensor A_log: Learnable decay magnitudes.
    :param Tensor dt_bias: Learnable decay biases.
    :param Tensor g_log: Log-decay activations saved from forward.
    :param Tensor beta: Beta activations saved from forward.
    :param Tensor o_raw: Raw recurrence output saved from forward.
    :param Tensor state_ckpt: Recurrence checkpoints saved from forward.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Packed depthwise conv width.
    :param int rec_chunk_t: Runtime checkpoint cadence.
    :param bool allow_neg_eigval: Whether beta is scaled by `2.0`.
    :return tuple[Tensor, ...]: Fake gradients matching the real CUDA contract.
    """
    del (
        grad_z,
        g_log,
        beta,
        o_raw,
        state_ckpt,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        rec_chunk_t,
        allow_neg_eigval,
    )
    return (
        _meta_empty(qkv, tuple(qkv.shape)),
        _meta_empty(g_pre, tuple(g_pre.shape)),
        _meta_empty(beta_pre, tuple(beta_pre.shape)),
        _meta_empty(g_out, tuple(g_out.shape)),
        _meta_empty(conv_w, tuple(conv_w.shape)),
        _meta_empty(A_log, tuple(A_log.shape)),
        _meta_empty(dt_bias, tuple(dt_bias.shape)),
    )


def _setup_hgdn_corekernel_context(
    ctx: Any,
    inputs: tuple[Tensor, ...],
    output: tuple[Tensor, ...],
) -> None:
    """Save tensors needed by the registered core-kernel backward formula.

    :param Any ctx: Autograd context to populate.
    :param tuple[Tensor, ...] inputs: Original forward inputs.
    :param tuple[Tensor, ...] output: Forward outputs plus saved tensors.
    """
    (
        qkv,
        g_pre,
        beta_pre,
        g_out,
        conv_w,
        A_log,
        dt_bias,
        n_heads,
        head_k_dim,
        head_v_dim,
        conv_size,
        rec_chunk_t,
        allow_neg_eigval,
    ) = inputs
    (_z, g_log, beta, o_raw, state_ckpt) = output
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(
        qkv,
        g_pre,
        beta_pre,
        g_out,
        conv_w,
        A_log,
        dt_bias,
        g_log,
        beta,
        o_raw,
        state_ckpt,
    )
    ctx.n_heads = int(n_heads)
    ctx.head_k_dim = int(head_k_dim)
    ctx.head_v_dim = int(head_v_dim)
    ctx.conv_size = int(conv_size)
    ctx.rec_chunk_t = int(rec_chunk_t)
    ctx.allow_neg_eigval = bool(allow_neg_eigval)


def _hgdn_corekernel_backward_formula(
    ctx: Any,
    grad_z: Tensor | None,
    _grad_g_log: Tensor | None,
    _grad_beta: Tensor | None,
    _grad_o_raw: Tensor | None,
    _grad_state_ckpt: Tensor | None,
) -> tuple[Tensor | None, ...]:
    """Backward formula for the compile-visible HGDN core-kernel op.

    :param Any ctx: Autograd context populated during forward.
    :param Tensor | None grad_z: Upstream gradient for the core output.
    :param Tensor | None _grad_g_log: Unused gradient placeholder.
    :param Tensor | None _grad_beta: Unused gradient placeholder.
    :param Tensor | None _grad_o_raw: Unused gradient placeholder.
    :param Tensor | None _grad_state_ckpt: Unused gradient placeholder.
    :return tuple[Tensor | None, ...]: Gradients for the registered forward
        inputs plus trailing `None` entries for non-tensor metadata.
    """
    del _grad_g_log, _grad_beta, _grad_o_raw, _grad_state_ckpt
    if grad_z is None:
        return (None,) * 13
    (
        qkv,
        g_pre,
        beta_pre,
        g_out,
        conv_w,
        A_log,
        dt_bias,
        g_log,
        beta,
        o_raw,
        state_ckpt,
    ) = ctx.saved_tensors
    grads = torch.ops.hgdn_corekernel_v1.run_backward(
        grad_z,
        qkv,
        g_pre,
        beta_pre,
        g_out,
        conv_w,
        A_log,
        dt_bias,
        g_log,
        beta,
        o_raw,
        state_ckpt,
        int(ctx.n_heads),
        int(ctx.head_k_dim),
        int(ctx.head_v_dim),
        int(ctx.conv_size),
        int(ctx.rec_chunk_t),
        bool(ctx.allow_neg_eigval),
    )
    return (*grads, None, None, None, None, None, None)


torch.library.register_autograd(
    "hgdn_corekernel_v1::run",
    _hgdn_corekernel_backward_formula,
    setup_context=_setup_hgdn_corekernel_context,
)


def hgdn_corekernel(
    qkv: Tensor,
    g_pre: Tensor,
    beta_pre: Tensor,
    g_out: Tensor,
    conv_w: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_size: int,
    rec_chunk_t: int | None = None,
    allow_neg_eigval: bool = True,
) -> Tensor:
    """Run the HGDN core kernel through the compile-visible custom-op path.

    :param Tensor qkv: Packed qkv activations shaped `(batch, seq, channels)`.
    :param Tensor g_pre: Decay preactivations shaped `(batch, seq, heads)`.
    :param Tensor beta_pre: Beta preactivations shaped `(batch, seq, heads)`.
    :param Tensor g_out: Output-gate preactivations shaped `(batch, seq, heads, d_v)`.
    :param Tensor conv_w: Packed depthwise conv weights shaped `(channels, kernel)`.
    :param Tensor A_log: Learnable decay magnitudes shaped `(heads,)`.
    :param Tensor dt_bias: Learnable decay biases shaped `(heads,)`.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head key width.
    :param int head_v_dim: Per-head value width.
    :param int conv_size: Causal conv width.
    :param int | None rec_chunk_t: Runtime checkpoint cadence, defaults to `None`.
    :param bool allow_neg_eigval: Whether to scale beta by `2.0`, defaults to True.
    :return Tensor: HGDN core output shaped `(batch, seq, heads, d_v)`.
    """
    rec_chunk_t = (
        resolve_runtime_rec_chunk_t(None, prefer_corekernel=True)
        if rec_chunk_t is None
        else int(rec_chunk_t)
    )
    z, *_saved = torch.ops.hgdn_corekernel_v1.run(
        qkv,
        g_pre,
        beta_pre,
        g_out,
        conv_w,
        A_log,
        dt_bias,
        int(n_heads),
        int(head_k_dim),
        int(head_v_dim),
        int(conv_size),
        int(rec_chunk_t),
        bool(allow_neg_eigval),
    )
    return z
