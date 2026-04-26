"""
GDN Hybrid Model for Parameter Golf — v3
=========================================
P0 fixes from code review:
  - Single n_heads for Q/K/V (matches FLA chunk_gated_delta_rule API)
  - Split projections: w_q/w_k/w_v (Muon) vs scalar/control routing
  - Correct naive recurrence (no broken grouped-key averaging)
  - kernels.py removed from submission
  - attention-only baseline preset added
"""

from __future__ import annotations

import json
import math
import os
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import grad as nn_grad

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func

    _HAS_FLASH_ATTN_3 = True
except ImportError:
    flash_attn_3_func = None
    _HAS_FLASH_ATTN_3 = False

from hgdn_fla import (
    HAS_FLA_GATED_DELTA_RULE,
    fla_chunk_gated_delta_rule_compile_visible,
    fla_chunk_gated_delta_rule_direct,
    fla_chunk_gated_delta_rule_direct_fused_gate_norm,
)

_HAS_FLA = HAS_FLA_GATED_DELTA_RULE

# Parameters routed to Adam (not Muon). Muon is for 2D feature maps only.
_BASE_SCALAR_PARAM_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weight",
    "w_a.",
    "w_b.",
    "A_log",
    "dt_bias",  # GDN decay params → Adam
    "o_norm_weight",
)
_W_G_SCALAR_PARAM_PATTERN = "w_g."
_VALID_GDN_W_G_OPTIMIZERS = {"scalar", "matrix"}
SCALAR_PARAM_PATTERNS = _BASE_SCALAR_PARAM_PATTERNS + (_W_G_SCALAR_PARAM_PATTERN,)
GDN_DECAY_PARAM_PATTERNS = ("A_log", "dt_bias")
GDN_FLA_RECURRENCE_MODES = {"compile_visible", "direct", "direct_fused"}

NormStyle = Literal["pre", "post", "keel"]
_PROFILE_RANGES = bool(int(os.environ.get("PROFILE_RANGES", "0")))
_LOG_GDN_LAYOUTS = bool(int(os.environ.get("GDN_LOG_LAYOUTS", "0")))
_GDN_LAYOUTS_REMAINING = int(os.environ.get("GDN_LOG_LAYOUTS_LIMIT", "1"))
_AUDIT_GDN_BOUNDARIES = bool(int(os.environ.get("GDN_AUDIT_BOUNDARIES", "0")))
_GDN_AUDIT_BOUNDARIES_PATH = os.environ.get("GDN_AUDIT_BOUNDARIES_PATH", "")
_GDN_AUDIT_BOUNDARIES_REMAINING = int(os.environ.get("GDN_AUDIT_BOUNDARIES_LIMIT", "1"))
_GDN_AUDIT_CALL_INDEX = 0
_HAS_SPLIT_WITH_SIZES_COPY = hasattr(torch.ops.aten, "split_with_sizes_copy")
_USE_FLASH_ATTN_3 = bool(int(os.environ.get("ATTN_USE_FLASH_ATTN3", "1")))


def attention_backend_name() -> str:
    """Return the active attention backend label for standard attention blocks.

    :return str: `"fa3"` when FlashAttention 3 is enabled and importable, else `"sdpa_flash"`.
    """
    if _USE_FLASH_ATTN_3 and _HAS_FLASH_ATTN_3:
        return "fa3"
    return "sdpa_flash"


def validate_gdn_w_g_optimizer(gdn_w_g_optimizer: str) -> str:
    """Normalize and validate the `w_g` optimizer-routing mode.

    :param str gdn_w_g_optimizer: Requested `w_g` routing mode.
    :raises ValueError: If the mode is unsupported.
    :return str: Validated routing mode.
    """
    normalized = gdn_w_g_optimizer.strip().lower()
    if normalized not in _VALID_GDN_W_G_OPTIMIZERS:
        raise ValueError(
            f"gdn_w_g_optimizer must be 'scalar' or 'matrix', got {gdn_w_g_optimizer!r}"
        )
    return normalized


def scalar_param_patterns(*, gdn_w_g_optimizer: str = "scalar") -> tuple[str, ...]:
    """Return the active scalar-optimizer name patterns.

    :param str gdn_w_g_optimizer: Whether `w_g` should ride Adam or Muon,
        defaults to `"scalar"`.
    :return tuple[str, ...]: Active scalar-pattern tuple.
    """
    normalized = validate_gdn_w_g_optimizer(gdn_w_g_optimizer)
    if normalized == "scalar":
        return _BASE_SCALAR_PARAM_PATTERNS + (_W_G_SCALAR_PARAM_PATTERN,)
    return _BASE_SCALAR_PARAM_PATTERNS


def uses_scalar_optimizer(
    param_name: str,
    *,
    param_ndim: int,
    gdn_w_g_optimizer: str = "scalar",
) -> bool:
    """Return whether one parameter should ride the scalar optimizer path.

    :param str param_name: Fully-qualified parameter name.
    :param int param_ndim: Parameter rank.
    :param str gdn_w_g_optimizer: Whether `w_g` should ride Adam or Muon,
        defaults to `"scalar"`.
    :return bool: `True` when the parameter belongs on the scalar path.
    """
    return param_ndim < 2 or any(
        pat in param_name
        for pat in scalar_param_patterns(gdn_w_g_optimizer=gdn_w_g_optimizer)
    )


def is_gdn_decay_param(param_name: str) -> bool:
    """Return whether one parameter controls the GDN decay timescale.

    :param str param_name: Fully-qualified parameter name.
    :return bool: `True` for `A_log` and `dt_bias` parameters.
    """
    return any(pattern in param_name for pattern in GDN_DECAY_PARAM_PATTERNS)


def init_gdn_decay_params(
    n_heads: int,
    *,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init_floor: float = 1e-4,
) -> tuple[nn.Parameter, nn.Parameter]:
    """Initialize GDN decay parameters with upstream FLA-style timescales.

    :param int n_heads: Number of GDN value heads.
    :param float dt_min: Minimum initial dt, defaults to `0.001`.
    :param float dt_max: Maximum initial dt, defaults to `0.1`.
    :param float dt_init_floor: Lower clamp for sampled dt, defaults to `1e-4`.
    :return tuple[nn.Parameter, nn.Parameter]: `A_log` and `dt_bias`.
    """
    A = torch.empty(n_heads, dtype=torch.float32).uniform_(1e-3, 16.0)
    A_log = nn.Parameter(torch.log(A))
    A_log._no_weight_decay = True

    dt = torch.exp(
        torch.rand(n_heads, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = torch.clamp(dt, min=dt_init_floor)
    dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
    dt_bias._no_weight_decay = True
    return A_log, dt_bias


def normalize_block_pattern(
    block_pattern: str | list[str] | None,
    *,
    num_layers: int,
    gdn_ratio: int,
) -> list[str]:
    """Resolve the model block schedule from an explicit pattern or ratio.

    :param str | list[str] | None block_pattern: Optional explicit block pattern.
    :param int num_layers: Total block count.
    :param int gdn_ratio: Legacy periodic GDN-to-attention ratio.
    :raises ValueError: If the pattern length or tokens are invalid.
    :return list[str]: Normalized block-type list.
    """
    if isinstance(block_pattern, str):
        normalized = [part.strip().lower() for part in block_pattern.split(",")]
        pattern = [part for part in normalized if part]
    elif block_pattern is None:
        pattern = []
    else:
        pattern = [
            str(part).strip().lower() for part in block_pattern if str(part).strip()
        ]
    if pattern:
        if len(pattern) != num_layers:
            raise ValueError(
                f"BLOCK_PATTERN length {len(pattern)} != num_layers {num_layers}"
            )
        invalid = sorted({part for part in pattern if part not in {"attn", "gdn"}})
        if invalid:
            raise ValueError(
                f"BLOCK_PATTERN entries must be only attn/gdn, got {invalid}"
            )
        return pattern

    period = gdn_ratio + 1
    return ["attn" if (i + 1) % period == 0 else "gdn" for i in range(num_layers)]


def profile_range(name: str) -> AbstractContextManager[None]:
    """Return a profiling context manager when range capture is enabled.

    :param str name: Label to emit into the profiler trace.
    :return: `record_function(name)` when profiling ranges are enabled, else a no-op context.
    """
    if _PROFILE_RANGES:
        return torch.autograd.profiler.record_function(name)
    return nullcontext()


def _tensor_layout_summary(name: str, tensor: Tensor) -> str:
    """Summarize tensor dtype and layout for one-shot debug logging.

    :param str name: User-facing tensor label.
    :param Tensor tensor: Tensor to summarize.
    :return str: Compact dtype/layout summary string.
    """
    return (
        f"{name}:dtype={tensor.dtype} shape={tuple(tensor.shape)} "
        f"stride={tuple(tensor.stride())} contiguous={int(tensor.is_contiguous())}"
    )


def log_gdn_layouts_once(**tensors: Tensor) -> None:
    """Emit one-shot HGDN layout diagnostics on rank 0 when enabled.

    :param Tensor tensors: Named tensors to summarize.
    """
    global _GDN_LAYOUTS_REMAINING

    if not _LOG_GDN_LAYOUTS or _GDN_LAYOUTS_REMAINING <= 0:
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return
    joined = " | ".join(
        _tensor_layout_summary(name, tensor) for name, tensor in tensors.items()
    )
    print(f"gdn_layout:{joined}")
    _GDN_LAYOUTS_REMAINING -= 1


def _tensor_layout_record(name: str, tensor: Tensor) -> dict[str, object]:
    """Build a structured tensor layout record for HGDN audit logs.

    :param str name: User-facing tensor label.
    :param Tensor tensor: Tensor to summarize.
    :return dict[str, object]: Serializable tensor record.
    """
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
    }


def begin_gdn_boundary_audit() -> int | None:
    """Reserve one GDN forward call for structured boundary logging.

    :return int | None: Audit call index, or `None` when audit is disabled.
    """
    global _GDN_AUDIT_BOUNDARIES_REMAINING, _GDN_AUDIT_CALL_INDEX

    if not _AUDIT_GDN_BOUNDARIES or _GDN_AUDIT_BOUNDARIES_REMAINING <= 0:
        return None
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    call_index = _GDN_AUDIT_CALL_INDEX
    _GDN_AUDIT_CALL_INDEX += 1
    _GDN_AUDIT_BOUNDARIES_REMAINING -= 1
    return call_index


def audit_gdn_boundary(
    boundary: str,
    call_index: int | None,
    **tensors: Tensor,
) -> None:
    """Emit a structured HGDN boundary audit record when enabled.

    :param str boundary: Logical HGDN boundary label.
    :param int | None call_index: Reserved audit-call identifier.
    :param Tensor tensors: Named tensors to summarize.
    """
    if call_index is None:
        return
    payload = {
        "call_index": call_index,
        "boundary": boundary,
        "tensors": [
            _tensor_layout_record(name, tensor) for name, tensor in tensors.items()
        ],
    }
    encoded = json.dumps(payload, sort_keys=True)
    if _GDN_AUDIT_BOUNDARIES_PATH:
        path = Path(_GDN_AUDIT_BOUNDARIES_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(encoded + "\n")
        return
    print(f"gdn_boundary:{encoded}")


def validate_norm_style(norm_style: str) -> NormStyle:
    """Normalize and validate a residual norm-placement string.

    :param str norm_style: Requested normalization style.
    :raises ValueError: If the style is unsupported.
    :return NormStyle: Validated normalization style.
    """
    normalized = norm_style.lower()
    if normalized not in {"pre", "post", "keel"}:
        raise ValueError(
            f"norm_style must be 'pre', 'post', or 'keel', got {norm_style!r}"
        )
    return normalized


def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Apply RMS normalization over the last dimension.

    :param Tensor x: Input activations.
    :param float eps: Numerical stability epsilon, defaults to 1e-6.
    :return Tensor: Normalized activations.
    """
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def l2_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Apply L2 normalization over the last dimension.

    :param Tensor x: Input activations.
    :param float eps: Numerical stability epsilon, defaults to 1e-6.
    :return Tensor: Unit-normalized activations.
    """
    return F.normalize(x, p=2, dim=-1, eps=eps)


def match_reference_tensor(tensor: Tensor, ref: Tensor) -> Tensor:
    """Match a tensor to a reference tensor's device and dtype only when needed.

    :param Tensor tensor: Source tensor to align.
    :param Tensor ref: Reference tensor providing target device and dtype.
    :return Tensor: `tensor` on the same device and dtype as `ref`.
    """

    if tensor.device == ref.device and tensor.dtype == ref.dtype:
        return tensor
    return tensor.to(device=ref.device, dtype=ref.dtype)


class CastedLinear(nn.Linear):
    """Linear projection that casts stored weights to the input dtype when needed."""

    def forward(self, x: Tensor) -> Tensor:
        """Project activations using weights cast to the input dtype.

        :param Tensor x: Input activations.
        :return Tensor: Projected activations.
        """
        bias = self.bias
        if self.weight.dtype == x.dtype and self.weight.device == x.device:
            if bias is None or (bias.dtype == x.dtype and bias.device == x.device):
                return F.linear(x, self.weight, bias)
        return F.linear(
            x,
            self.weight.to(x.dtype),
            bias.to(x.dtype) if bias is not None else None,
        )


class PackedCastedLinear(nn.Module):
    """Single packed projection that can be split into q/k/v feature maps."""

    def __init__(
        self,
        in_features: int,
        dims: tuple[int, int, int],
        *,
        bias: bool = False,
    ):
        """Initialize the packed q/k/v projection.

        :param int in_features: Input feature width.
        :param tuple[int, int, int] dims: Output widths for q, k, and v.
        :param bool bias: Whether to learn a bias term, defaults to False.
        """
        super().__init__()
        self.dims = dims
        self.linear = CastedLinear(in_features, sum(dims), bias=bias)

    @property
    def weight(self) -> nn.Parameter:
        """Expose the packed weight for optimizer routing and test parity.

        :return nn.Parameter: Packed q/k/v projection weight.
        """
        return self.linear.weight

    @property
    def bias(self) -> nn.Parameter | None:
        """Expose the packed bias parameter when present.

        :return nn.Parameter | None: Optional packed bias.
        """
        return self.linear.bias

    def forward_packed(self, x: Tensor) -> Tensor:
        """Project activations into one packed q/k/v buffer.

        :param Tensor x: Input activations.
        :return Tensor: Packed projection shaped `(batch, seq, q_dim + k_dim + v_dim)`.
        """
        return self.linear(x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project and split activations into q/k/v tensors.

        :param Tensor x: Input activations.
        :return tuple[Tensor, Tensor, Tensor]: Split q/k/v projections.
        """
        return self.forward_packed(x).split(self.dims, dim=-1)


class RMSNorm(nn.Module):
    """Parameter-free RMSNorm wrapper used throughout the hybrid stack."""

    def __init__(self, eps: float = 1e-6):
        """Initialize the RMSNorm wrapper.

        :param float eps: Numerical stability epsilon, defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize activations with RMSNorm.

        :param Tensor x: Input activations.
        :return Tensor: Normalized activations.
        """
        return rms_norm(x, self.eps)


class CausalConv1d(nn.Module):
    """Depthwise causal convolution used to preprocess sequence features."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 4,
        enabled: bool = True,
        output_contiguous: bool = False,
    ):
        """Initialize the causal depthwise convolution.

        :param int dim: Channel count.
        :param int kernel_size: Convolution kernel width, defaults to 4.
        :param bool enabled: Whether to apply the convolution, defaults to True.
        :param bool output_contiguous: Whether to materialize a contiguous `(batch, seq, dim)` result, defaults to False.
        """
        super().__init__()
        self.enabled = enabled
        self.output_contiguous = output_contiguous
        self.conv = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size - 1, groups=dim, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the causal convolution over the sequence axis.

        :param Tensor x: Input activations shaped `(batch, seq, dim)`.
        :return Tensor: Convolved activations with causal trimming applied, or the input unchanged when disabled.
        """
        if not self.enabled:
            return x
        with profile_range("gdn.conv_input_transpose"):
            x = x.transpose(1, 2)
        with profile_range("gdn.conv_depthwise"):
            x = self.conv(x)
        with profile_range("gdn.conv_trim"):
            x = x[..., : x.size(-1) - (self.conv.kernel_size[0] - 1)]
        with profile_range("gdn.conv_silu"):
            x = F.silu(x)
        with profile_range("gdn.conv_output_transpose"):
            x = x.transpose(1, 2)
        if self.output_contiguous:
            with profile_range("gdn.conv_output_contiguous"):
                x = x.contiguous()
        return x


class PackedCausalConv1d(nn.Module):
    """Packed depthwise causal conv over concatenated q/k/v feature maps."""

    def __init__(
        self,
        dims: tuple[int, int, int],
        kernel_size: int = 4,
        output_contiguous: bool = False,
        use_custom_backward: bool = False,
        use_single_packed_contiguous: bool = False,
        use_split_with_sizes_copy: bool = False,
    ):
        """Initialize the packed causal depthwise convolution.

        :param tuple[int, int, int] dims: Channel counts for q, k, and v paths.
        :param int kernel_size: Convolution kernel width, defaults to 4.
        :param bool output_contiguous: Whether to materialize contiguous `(batch, seq, dim)` outputs, defaults to False.
        :param bool use_custom_backward: Whether to route the packed depthwise conv through an exact-length custom autograd path, defaults to False.
        :param bool use_single_packed_contiguous: Whether to materialize one contiguous packed `(batch, seq, q_dim + k_dim + v_dim)` tensor before splitting, defaults to False.
        :param bool use_split_with_sizes_copy: Whether to materialize q/k/v with `aten.split_with_sizes_copy`, defaults to False.
        """
        super().__init__()
        self.dims = dims
        self.output_contiguous = output_contiguous
        self.use_custom_backward = use_custom_backward
        self.use_single_packed_contiguous = use_single_packed_contiguous
        self.use_split_with_sizes_copy = use_split_with_sizes_copy
        total_dim = sum(dims)
        self.conv = nn.Conv1d(
            total_dim,
            total_dim,
            kernel_size,
            padding=kernel_size - 1,
            groups=total_dim,
            bias=False,
        )

    @staticmethod
    def _silu_backward(preact: Tensor, grad_output: Tensor) -> Tensor:
        """Differentiate SiLU with respect to its pre-activation.

        :param Tensor preact: Saved pre-activation tensor.
        :param Tensor grad_output: Upstream gradient after SiLU.
        :return Tensor: Gradient with respect to ``preact``.
        """
        sigma = torch.sigmoid(preact)
        return grad_output * sigma * (1.0 + preact * (1.0 - sigma))

    class _PackedDepthwiseCustomBackward(torch.autograd.Function):
        """Exact-length packed causal depthwise conv with custom backward."""

        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            packed: Tensor,
            weight: Tensor,
        ) -> Tensor:
            """Run one packed causal depthwise conv without padded tail outputs.

            :param torch.autograd.function.FunctionCtx ctx: Autograd context.
            :param Tensor packed: Packed q/k/v tensor shaped ``(batch, seq, channels)``.
            :param Tensor weight: Depthwise conv weights shaped ``(channels, 1, kernel)``.
            :return Tensor: Packed convolved tensor shaped ``(batch, seq, channels)``.
            """
            kernel = weight.shape[-1]
            groups = weight.shape[0]
            with profile_range("gdn.qkv_conv_input_transpose"):
                packed_t = packed.transpose(1, 2)
            with profile_range("gdn.qkv_conv_left_pad"):
                packed_pad = F.pad(packed_t, (kernel - 1, 0))
            with profile_range("gdn.qkv_conv_depthwise"):
                preact = F.conv1d(packed_pad, weight, groups=groups)
            with profile_range("gdn.qkv_conv_silu"):
                packed_out = F.silu(preact)
            with profile_range("gdn.qkv_conv_output_transpose"):
                packed_out = packed_out.transpose(1, 2)
            ctx.save_for_backward(packed_pad, weight, preact)
            return packed_out

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_packed_out: Tensor,
        ) -> tuple[Tensor | None, Tensor | None]:
            """Backprop through the exact-length packed causal depthwise conv.

            :param torch.autograd.function.FunctionCtx ctx: Autograd context.
            :param Tensor grad_packed_out: Upstream packed gradient.
            :return tuple[Tensor | None, Tensor | None]: Gradients for packed input and conv weights.
            """
            packed_pad, weight, preact = ctx.saved_tensors
            grad_packed = None
            grad_weight = None
            with profile_range("gdn.qkv_conv_bwd_output_transpose"):
                grad_out_t = grad_packed_out.transpose(1, 2).contiguous()
            with profile_range("gdn.qkv_conv_bwd_silu"):
                grad_preact = PackedCausalConv1d._silu_backward(preact, grad_out_t)
            groups = weight.shape[0]
            kernel = weight.shape[-1]
            if ctx.needs_input_grad[0]:
                with profile_range("gdn.qkv_conv_bwd_input_grad"):
                    grad_pad = nn_grad.conv1d_input(
                        packed_pad.shape,
                        weight,
                        grad_preact,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=groups,
                    )
                with profile_range("gdn.qkv_conv_bwd_input_trim"):
                    grad_packed = (
                        grad_pad[..., kernel - 1 :].transpose(1, 2).contiguous()
                    )
            if ctx.needs_input_grad[1]:
                with profile_range("gdn.qkv_conv_bwd_weight_grad"):
                    grad_weight = nn_grad.conv1d_weight(
                        packed_pad,
                        weight.shape,
                        grad_preact,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=groups,
                    )
            return grad_packed, grad_weight

    class _PackedDepthwisePreactNCTCustomBackward(torch.autograd.Function):
        """Exact-length packed causal depthwise conv that returns pre-activation NCT."""

        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            packed: Tensor,
            weight: Tensor,
        ) -> Tensor:
            """Run one packed causal depthwise conv and return exact-length pre-activations.

            :param torch.autograd.function.FunctionCtx ctx: Autograd context.
            :param Tensor packed: Packed q/k/v tensor shaped ``(batch, seq, channels)``.
            :param Tensor weight: Depthwise conv weights shaped ``(channels, 1, kernel)``.
            :return Tensor: Packed pre-activation tensor shaped ``(batch, channels, seq)``.
            """
            kernel = weight.shape[-1]
            groups = weight.shape[0]
            with profile_range("gdn.qkv_conv_input_transpose"):
                packed_t = packed.transpose(1, 2)
            with profile_range("gdn.qkv_conv_left_pad"):
                packed_pad = F.pad(packed_t, (kernel - 1, 0))
            with profile_range("gdn.qkv_conv_depthwise"):
                preact_nct = F.conv1d(packed_pad, weight, groups=groups)
            ctx.save_for_backward(packed_pad, weight)
            return preact_nct

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_preact_nct: Tensor,
        ) -> tuple[Tensor | None, Tensor | None]:
            """Backprop through the exact-length packed pre-activation conv path.

            :param torch.autograd.function.FunctionCtx ctx: Autograd context.
            :param Tensor grad_preact_nct: Gradient for the packed pre-activation tensor.
            :return tuple[Tensor | None, Tensor | None]: Gradients for packed input and conv weights.
            """
            packed_pad, weight = ctx.saved_tensors
            grad_packed = None
            grad_weight = None
            groups = weight.shape[0]
            kernel = weight.shape[-1]
            grad_preact_nct = grad_preact_nct.contiguous()
            if ctx.needs_input_grad[0]:
                with profile_range("gdn.qkv_conv_bwd_input_grad"):
                    grad_pad = nn_grad.conv1d_input(
                        packed_pad.shape,
                        weight,
                        grad_preact_nct,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=groups,
                    )
                with profile_range("gdn.qkv_conv_bwd_input_trim"):
                    grad_packed = (
                        grad_pad[..., kernel - 1 :].transpose(1, 2).contiguous()
                    )
            if ctx.needs_input_grad[1]:
                with profile_range("gdn.qkv_conv_bwd_weight_grad"):
                    grad_weight = nn_grad.conv1d_weight(
                        packed_pad,
                        weight.shape,
                        grad_preact_nct,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=groups,
                    )
            return grad_packed, grad_weight

    def _forward_packed_tensor(self, x: Tensor) -> Tensor:
        """Convolve one packed q/k/v tensor without splitting the result.

        :param Tensor x: Packed activations shaped `(batch, seq, q_dim + k_dim + v_dim)`.
        :return Tensor: Convolved packed activations with the same packed trailing dimension.
        """
        if self.use_custom_backward:
            return self._PackedDepthwiseCustomBackward.apply(x, self.conv.weight)
        with profile_range("gdn.qkv_conv_input_transpose"):
            x = x.transpose(1, 2)
        with profile_range("gdn.qkv_conv_depthwise"):
            x = self.conv(x)
        with profile_range("gdn.qkv_conv_trim"):
            x = x[..., : x.size(-1) - (self.conv.kernel_size[0] - 1)]
        with profile_range("gdn.qkv_conv_silu"):
            x = F.silu(x)
        with profile_range("gdn.qkv_conv_output_transpose"):
            x = x.transpose(1, 2)
        return x

    def forward_packed_preact_nct(self, x: Tensor) -> Tensor:
        """Apply packed causal depthwise convolution and keep the exact-length NCT pre-activation.

        :param Tensor x: Packed activations shaped `(batch, seq, q_dim + k_dim + v_dim)`.
        :return Tensor: Packed pre-activation tensor shaped `(batch, q_dim + k_dim + v_dim, seq)`.
        """
        if self.use_custom_backward:
            return self._PackedDepthwisePreactNCTCustomBackward.apply(
                x, self.conv.weight
            )
        with profile_range("gdn.qkv_conv_input_transpose"):
            x = x.transpose(1, 2)
        with profile_range("gdn.qkv_conv_depthwise"):
            x = self.conv(x)
        with profile_range("gdn.qkv_conv_trim"):
            x = x[..., : x.size(-1) - (self.conv.kernel_size[0] - 1)]
        return x

    def _forward_packed_impl(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Convolve one packed q/k/v tensor and split the result.

        :param Tensor x: Packed activations shaped `(batch, seq, q_dim + k_dim + v_dim)`.
        :return tuple[Tensor, Tensor, Tensor]: Convolved q/k/v activations.
        """
        x = self._forward_packed_tensor(x)
        if self.output_contiguous and self.use_split_with_sizes_copy:
            with profile_range("gdn.qkv_conv_split_copy"):
                q, k, v = torch.ops.aten.split_with_sizes_copy.default(
                    x,
                    list(self.dims),
                    -1,
                )
            return q, k, v
        if self.output_contiguous and self.use_single_packed_contiguous:
            with profile_range("gdn.qkv_conv_output_contiguous_packed"):
                x = x.contiguous()
        with profile_range("gdn.qkv_conv_split"):
            q_dim, k_dim, v_dim = self.dims
            q, k, v = x.split((q_dim, k_dim, v_dim), dim=-1)
        if self.output_contiguous and not self.use_single_packed_contiguous:
            with profile_range("gdn.qkv_conv_output_contiguous"):
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
        return q, k, v

    def forward_packed(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply packed causal depthwise convolution to a packed q/k/v tensor.

        :param Tensor x: Packed activations shaped `(batch, seq, q_dim + k_dim + v_dim)`.
        :return tuple[Tensor, Tensor, Tensor]: Convolved q/k/v activations.
        """
        return self._forward_packed_impl(x)

    def forward_packed_tensor(self, x: Tensor) -> Tensor:
        """Apply packed causal depthwise convolution and keep the packed result.

        :param Tensor x: Packed activations shaped `(batch, seq, q_dim + k_dim + v_dim)`.
        :return Tensor: Packed convolved activations without q/k/v splitting.
        """
        return self._forward_packed_tensor(x)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply one packed causal depthwise convolution and split the result.

        :param Tensor q: Query activations shaped `(batch, seq, q_dim)`.
        :param Tensor k: Key activations shaped `(batch, seq, k_dim)`.
        :param Tensor v: Value activations shaped `(batch, seq, v_dim)`.
        :return tuple[Tensor, Tensor, Tensor]: Convolved q/k/v activations.
        """
        return self._forward_packed_impl(torch.cat((q, k, v), dim=-1))


# ── Naive GDN recurrence ─────────────────────────────────────────────


def gdn_recurrent_naive(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    initial_state: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Run the reference gated delta recurrence on CPU or fallback paths.

    :param Tensor q: Normalized queries shaped `(batch, seq, heads, d_k)`.
    :param Tensor k: Normalized keys shaped `(batch, seq, heads, d_k)`.
    :param Tensor v: Values shaped `(batch, seq, heads, d_v)`.
    :param Tensor alpha: Decay multipliers shaped `(batch, seq, heads)`.
    :param Tensor beta: Write gates shaped `(batch, seq, heads)`.
    :param Optional[Tensor] initial_state: Optional initial state shaped `(batch, heads, d_k, d_v)`, defaults to None.
    :return tuple[Tensor, Tensor]: Sequence outputs and final recurrence state.
    """
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    q, k, v = q.float(), k.float(), v.float()
    S = (
        initial_state.float()
        if initial_state is not None
        else torch.zeros(B, H, Dk, Dv, device=q.device)
    )
    outputs = []
    for t in range(T):
        q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
        a_t = alpha[:, t, :, None, None]
        b_t = beta[:, t, :, None, None]
        kS = torch.einsum("bhd,bhdv->bhv", k_t, S)
        kkS = torch.einsum("bhd,bhv->bhdv", k_t, kS)
        write = torch.einsum("bhd,bhv->bhdv", k_t, v_t)
        S = a_t * (S - b_t * kkS) + b_t * write
        outputs.append(torch.einsum("bhdv,bhd->bhv", S, q_t))
    return torch.stack(outputs, dim=1), S


# ── Gated DeltaNet layer ─────────────────────────────────────────────
#
# Matches FLA API: q, k, v all have n_heads heads.
# head_v_dim = head_k_dim * expand_v controls state capacity.
# Projections split: w_q/w_k/w_v are proper feature maps (→ Muon);
# w_a/w_b are tiny scalar controls, and w_g follows GDN_W_G_OPTIMIZER.


class GatedDeltaNet(nn.Module):
    """Hybrid sequence mixer that wraps the FLA Gated DeltaNet kernel."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        head_k_dim: int = 48,
        expand_v: float = 2.0,
        allow_neg_eigval: bool = True,
        conv_size: int = 4,
        use_fla: bool = True,
        use_q_conv: bool = True,
        use_k_conv: bool = True,
        use_v_conv: bool = True,
        use_packed_qkv_conv: bool = False,
        use_packed_qkv_proj: bool = False,
        conv_output_contiguous: bool = False,
        q_conv_output_contiguous: bool | None = None,
        k_conv_output_contiguous: bool | None = None,
        v_conv_output_contiguous: bool | None = None,
        gates_fp32: bool = True,
        output_norm_fp32: bool = True,
        use_packed_qkv_conv_custom_backward: bool = False,
        use_packed_qkv_single_contig: bool = False,
        use_packed_qkv_split_copy: bool = False,
        fla_recurrence_mode: str = "compile_visible",
    ):
        """Initialize a Gated DeltaNet block.

        :param int d_model: Model width.
        :param int n_heads: Number of recurrence heads, defaults to 4.
        :param int head_k_dim: Per-head key width, defaults to 48.
        :param float expand_v: Value expansion relative to `head_k_dim`, defaults to 2.0.
        :param bool allow_neg_eigval: Whether to allow negative eigenvalues in the write gate, defaults to True.
        :param int conv_size: Depthwise causal convolution width, defaults to 4.
        :param bool use_fla: Whether to use the FLA kernel when available, defaults to True.
        :param bool use_q_conv: Whether to apply the q-path convolution, defaults to True.
        :param bool use_k_conv: Whether to apply the k-path convolution, defaults to True.
        :param bool use_v_conv: Whether to apply the v-path convolution, defaults to True.
        :param bool use_packed_qkv_conv: Whether to replace separate q/k/v depthwise convs with one packed depthwise conv, defaults to False.
        :param bool use_packed_qkv_proj: Whether to replace separate q/k/v projections with one packed projection, defaults to False.
        :param bool conv_output_contiguous: Whether to materialize contiguous `(batch, seq, dim)` conv outputs before recurrence prep, defaults to False.
        :param bool | None q_conv_output_contiguous: Optional q-path override for contiguous conv outputs, defaults to None.
        :param bool | None k_conv_output_contiguous: Optional k-path override for contiguous conv outputs, defaults to None.
        :param bool | None v_conv_output_contiguous: Optional v-path override for contiguous conv outputs, defaults to None.
        :param bool gates_fp32: Whether to keep the decay-gate softplus path in fp32, defaults to True.
        :param bool output_norm_fp32: Whether to keep the post-recurrence RMSNorm in fp32 before casting back to the activation dtype, defaults to True.
        :param bool use_packed_qkv_conv_custom_backward: Whether to route the packed depthwise qkv conv through an exact-length custom autograd path, defaults to False.
        :param bool use_packed_qkv_single_contig: Whether to materialize one contiguous packed q/k/v tensor before splitting the packed conv output, defaults to False.
        :param bool use_packed_qkv_split_copy: Whether to materialize q/k/v with `aten.split_with_sizes_copy`, defaults to False.
        :param str fla_recurrence_mode: Public FLA recurrence path:
            `compile_visible` keeps the torch.library wrapper, `direct`
            bypasses it with the custom HGDN recurrence semantics, and
            `direct_fused` uses upstream-style in-kernel q/k norm and decay-gate
            activation, defaults to `"compile_visible"`.
        """
        super().__init__()
        if fla_recurrence_mode not in GDN_FLA_RECURRENCE_MODES:
            raise ValueError(
                f"fla_recurrence_mode must be one of {sorted(GDN_FLA_RECURRENCE_MODES)}, "
                f"got {fla_recurrence_mode!r}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = int(head_k_dim * expand_v)
        assert self.head_v_dim % 2 == 0, f"head_v_dim={self.head_v_dim} must be even"
        self.allow_neg_eigval = allow_neg_eigval
        self.use_fla = use_fla and _HAS_FLA
        self.gates_fp32 = gates_fp32
        self.output_norm_fp32 = output_norm_fp32
        self.use_packed_qkv_conv = use_packed_qkv_conv
        self.use_packed_qkv_proj = use_packed_qkv_proj
        self.use_packed_qkv_conv_custom_backward = use_packed_qkv_conv_custom_backward
        self.use_packed_qkv_single_contig = use_packed_qkv_single_contig
        self.use_packed_qkv_split_copy = use_packed_qkv_split_copy
        self.fla_recurrence_mode = fla_recurrence_mode
        q_conv_output_contiguous = (
            conv_output_contiguous
            if q_conv_output_contiguous is None
            else q_conv_output_contiguous
        )
        k_conv_output_contiguous = (
            conv_output_contiguous
            if k_conv_output_contiguous is None
            else k_conv_output_contiguous
        )
        v_conv_output_contiguous = (
            conv_output_contiguous
            if v_conv_output_contiguous is None
            else v_conv_output_contiguous
        )

        total_qk = n_heads * head_k_dim
        total_v = n_heads * self.head_v_dim

        if self.use_packed_qkv_conv:
            if not (use_q_conv and use_k_conv and use_v_conv):
                raise ValueError(
                    "use_packed_qkv_conv requires q/k/v convs to all be enabled"
                )
            if (
                len(
                    {
                        q_conv_output_contiguous,
                        k_conv_output_contiguous,
                        v_conv_output_contiguous,
                    }
                )
                != 1
            ):
                raise ValueError(
                    "use_packed_qkv_conv requires aligned q/k/v output_contiguous settings"
                )
        if self.use_packed_qkv_proj and not self.use_packed_qkv_conv:
            raise ValueError("use_packed_qkv_proj requires use_packed_qkv_conv")
        if self.use_packed_qkv_conv_custom_backward and not self.use_packed_qkv_conv:
            raise ValueError(
                "use_packed_qkv_conv_custom_backward requires use_packed_qkv_conv"
            )
        if self.use_packed_qkv_single_contig and not self.use_packed_qkv_conv:
            raise ValueError(
                "use_packed_qkv_single_contig requires use_packed_qkv_conv"
            )
        if self.use_packed_qkv_single_contig and not q_conv_output_contiguous:
            raise ValueError(
                "use_packed_qkv_single_contig requires packed qkv output_contiguous"
            )
        if self.use_packed_qkv_split_copy and not _HAS_SPLIT_WITH_SIZES_COPY:
            raise ValueError(
                "use_packed_qkv_split_copy requires aten.split_with_sizes_copy"
            )
        if self.use_packed_qkv_split_copy and not self.use_packed_qkv_conv:
            raise ValueError("use_packed_qkv_split_copy requires use_packed_qkv_conv")
        if self.use_packed_qkv_split_copy and not q_conv_output_contiguous:
            raise ValueError(
                "use_packed_qkv_split_copy requires packed qkv output_contiguous"
            )
        if self.use_packed_qkv_single_contig and self.use_packed_qkv_split_copy:
            raise ValueError(
                "use_packed_qkv_single_contig is incompatible with use_packed_qkv_split_copy"
            )

        # Feature-map projections → Muon
        if self.use_packed_qkv_proj:
            self.w_qkv = PackedCastedLinear(
                d_model, (total_qk, total_qk, total_v), bias=False
            )
            self.w_q = None
            self.w_k = None
            self.w_v = None
        else:
            self.w_qkv = None
            self.w_q = CastedLinear(d_model, total_qk, bias=False)
            self.w_k = CastedLinear(d_model, total_qk, bias=False)
            self.w_v = CastedLinear(d_model, total_v, bias=False)

        # Control projections → Adam (tiny: d_model → n_heads each)
        self.w_a = CastedLinear(d_model, n_heads, bias=False)
        self.w_b = CastedLinear(d_model, n_heads, bias=False)

        # Output gate → Adam (sigmoid path, not a feature map)
        self.w_g = CastedLinear(d_model, total_v, bias=False)

        # Output projection → Muon
        self.w_out = CastedLinear(total_v, d_model, bias=False)
        self.w_out._zero_init = True

        # Learnable recurrence controls → Adam.
        self.A_log, self.dt_bias = init_gdn_decay_params(n_heads)
        self.o_norm_weight = nn.Parameter(
            torch.ones(self.head_v_dim, dtype=torch.float32)
        )

        # Short causal convolutions
        self.conv_size = int(conv_size)
        if self.use_packed_qkv_conv:
            self.qkv_conv = PackedCausalConv1d(
                dims=(total_qk, total_qk, total_v),
                kernel_size=conv_size,
                output_contiguous=q_conv_output_contiguous,
                use_custom_backward=use_packed_qkv_conv_custom_backward,
                use_single_packed_contiguous=use_packed_qkv_single_contig,
                use_split_with_sizes_copy=use_packed_qkv_split_copy,
            )
            self.q_conv = None
            self.k_conv = None
            self.v_conv = None
        else:
            self.qkv_conv = None
            self.q_conv = CausalConv1d(
                total_qk,
                conv_size,
                enabled=use_q_conv,
                output_contiguous=q_conv_output_contiguous,
            )
            self.k_conv = CausalConv1d(
                total_qk,
                conv_size,
                enabled=use_k_conv,
                output_contiguous=k_conv_output_contiguous,
            )
            self.v_conv = CausalConv1d(
                total_v,
                conv_size,
                enabled=use_v_conv,
                output_contiguous=v_conv_output_contiguous,
            )

    def _project_recurrence_inputs(
        self,
        x: Tensor,
        *,
        audit_call_index: int | None = None,
        fused_fla_semantics: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Project activations into recurrence inputs with aligned dtypes.

        :param Tensor x: Input activations shaped `(batch, seq, d_model)`.
        :param int | None audit_call_index: Optional structured audit call index.
        :param bool fused_fla_semantics: Whether to leave q/k and the decay gate
            raw for FLA's in-kernel normalization/gate path, defaults to False.
        :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: q/k, values,
            decay gate input, and write gates.
        """
        B, T, _ = x.shape
        H, Dk, Dv = self.n_heads, self.head_k_dim, self.head_v_dim

        with profile_range("gdn.project_qkv"):
            if self.w_qkv is not None:
                with profile_range("gdn.project_qkv_packed"):
                    qkv = self.w_qkv.forward_packed(x)
                q, k, v = qkv.split((H * Dk, H * Dk, H * Dv), dim=-1)
            else:
                q = self.w_q(x)
                k = self.w_k(x)
                v = self.w_v(x)
            audit_gdn_boundary(
                "project_qkv",
                audit_call_index,
                q=q,
                k=k,
                v=v,
            )

        with profile_range("gdn.conv_qkv"):
            if self.qkv_conv is not None:
                with profile_range("gdn.qkv_conv_packed"):
                    if self.w_qkv is not None:
                        q, k, v = self.qkv_conv.forward_packed(qkv)
                    else:
                        q, k, v = self.qkv_conv(q, k, v)
            else:
                with profile_range("gdn.q_conv"):
                    q = self.q_conv(q)
                with profile_range("gdn.k_conv"):
                    k = self.k_conv(k)
                with profile_range("gdn.v_conv"):
                    v = self.v_conv(v)
            audit_gdn_boundary(
                "conv_qkv",
                audit_call_index,
                q=q,
                k=k,
                v=v,
            )

            with profile_range("gdn.norm_qkv"):
                if fused_fla_semantics:
                    with profile_range("gdn.q_reshape_raw"):
                        q = match_reference_tensor(q.view(B, T, H, Dk), x)
                    with profile_range("gdn.k_reshape_raw"):
                        k = match_reference_tensor(k.view(B, T, H, Dk), x)
                else:
                    with profile_range("gdn.q_norm"):
                        q = match_reference_tensor(l2_norm(q.view(B, T, H, Dk)), x)
                    with profile_range("gdn.k_norm"):
                        k = match_reference_tensor(l2_norm(k.view(B, T, H, Dk)), x)
                with profile_range("gdn.v_reshape"):
                    v = v.view(B, T, H, Dv)
                    if self.use_packed_qkv_single_contig and not v.is_contiguous():
                        with profile_range("gdn.v_contiguous"):
                            v = v.contiguous()
                    v = match_reference_tensor(v, x)
            audit_gdn_boundary(
                "norm_qkv",
                audit_call_index,
                q=q,
                k=k,
                v=v,
            )

        with profile_range("gdn.gates"):
            if fused_fla_semantics:
                with profile_range("gdn.g_proj_raw"):
                    g = self.w_a(x)
            elif self.gates_fp32:
                with profile_range("gdn.g_proj"):
                    g_pre = self.w_a(x).float()
                with profile_range("gdn.g_pointwise"):
                    g = -self.A_log.exp() * F.softplus(
                        g_pre + self.dt_bias
                    )  # (B,T,H) log-space
            else:
                with profile_range("gdn.g_proj"):
                    g_pre = self.w_a(x)
                with profile_range("gdn.g_pointwise"):
                    g = -self.A_log.to(dtype=x.dtype).exp() * F.softplus(
                        g_pre + self.dt_bias.to(dtype=x.dtype)
                    )
            with profile_range("gdn.beta_proj"):
                beta_pre = self.w_b(x)
            with profile_range("gdn.beta_pointwise"):
                beta = torch.sigmoid(beta_pre)  # (B,T,H)
        if self.allow_neg_eigval:
            with profile_range("gdn.beta_scale"):
                beta = beta * 2.0
        with profile_range("gdn.gate_casts"):
            g = g.to(dtype=x.dtype)
            beta = beta.to(dtype=x.dtype)
        audit_gdn_boundary(
            "recurrence_inputs",
            audit_call_index,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
        )
        return q, k, v, g, beta

    def decay_init_stats(self) -> dict[str, float]:
        """Return summary statistics for the current decay timescale parameters.

        :return dict[str, float]: Means and extrema for `A`, `dt`, and `alpha`.
        """
        with torch.no_grad():
            initial_dt = F.softplus(self.dt_bias.float())
            initial_A = self.A_log.float().exp()
            initial_alpha = (-initial_A * initial_dt).exp()
            return {
                "A_mean": float(initial_A.mean().item()),
                "dt_mean": float(initial_dt.mean().item()),
                "alpha_mean": float(initial_alpha.mean().item()),
                "alpha_min": float(initial_alpha.min().item()),
                "alpha_max": float(initial_alpha.max().item()),
            }

    def forward(self, x: Tensor) -> Tensor:
        """Apply the Gated DeltaNet layer to a sequence batch.

        :param Tensor x: Input activations shaped `(batch, seq, d_model)`.
        :return Tensor: Updated activations shaped `(batch, seq, d_model)`.
        """
        B, T, _ = x.shape
        H, Dv = self.n_heads, self.head_v_dim
        with profile_range("gdn.forward"):
            audit_call_index = begin_gdn_boundary_audit()
            fused_fla_semantics = self.fla_recurrence_mode == "direct_fused"
            q, k, v, g, beta = self._project_recurrence_inputs(
                x,
                audit_call_index=audit_call_index,
                fused_fla_semantics=fused_fla_semantics,
            )
            log_gdn_layouts_once(q=q, k=k, v=v, g=g, beta=beta)

            with profile_range("gdn.recurrence"):
                if fused_fla_semantics and not (self.use_fla and x.is_cuda):
                    raise RuntimeError(
                        "fla_recurrence_mode='direct_fused' requires CUDA and "
                        "the public FLA recurrence stack."
                    )
                if self.use_fla and x.is_cuda:
                    if self.fla_recurrence_mode == "direct_fused":
                        o = fla_chunk_gated_delta_rule_direct_fused_gate_norm(
                            q,
                            k,
                            v,
                            g,
                            beta,
                            self.A_log,
                            self.dt_bias,
                        )
                    elif self.fla_recurrence_mode == "direct":
                        o = fla_chunk_gated_delta_rule_direct(q, k, v, g, beta)
                    else:
                        o = fla_chunk_gated_delta_rule_compile_visible(q, k, v, g, beta)
                else:
                    o, _ = gdn_recurrent_naive(q, k, v, g.exp(), beta)
            audit_gdn_boundary("recurrence_output", audit_call_index, o=o)

            with profile_range("gdn.output_gate"):
                with profile_range("gdn.output_gate_proj"):
                    g_out = self.w_g(x).view(B, T, H, Dv)
                with profile_range("gdn.output_norm"):
                    if self.output_norm_fp32:
                        o = match_reference_tensor(rms_norm(o.float()), x)
                    else:
                        o = match_reference_tensor(rms_norm(o), x)
                with profile_range("gdn.output_norm_weight"):
                    o = o * self.o_norm_weight.to(dtype=o.dtype, device=o.device).view(
                        1, 1, 1, -1
                    )
                audit_gdn_boundary(
                    "output_gate_inputs", audit_call_index, o=o, g_out=g_out
                )
                with profile_range("gdn.output_gate_mul"):
                    o = o * F.silu(g_out)
            audit_gdn_boundary("output_proj_input", audit_call_index, o=o)

            with profile_range("gdn.output_proj"):
                return self.w_out(o.reshape(B, T, -1))


# ── Standard attention ────────────────────────────────────────────────


class Rotary(nn.Module):
    """Rotary embedding cache for attention heads."""

    def __init__(self, dim: int, base: float = 10000.0):
        """Initialize the rotary cache.

        :param int dim: Per-head dimension.
        :param float base: Rotary frequency base, defaults to 10000.0.
        """
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)),
            persistent=False,
        )
        self._cache_key: tuple[int, str, int | None, torch.dtype] | None = None
        self._cos: Optional[Tensor] = None
        self._sin: Optional[Tensor] = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Return cached cosine and sine tables for a sequence length.

        :param int seq_len: Sequence length to cover.
        :param torch.device device: Target device.
        :param torch.dtype dtype: Target dtype for the cache.
        :return tuple[Tensor, Tensor]: Cosine and sine rotary tables.
        """
        cache_key = (seq_len, device.type, device.index, dtype)
        if self._cos is None or self._sin is None or self._cache_key != cache_key:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            cos = freqs.cos()[None, :, None, :]
            sin = freqs.sin()[None, :, None, :]
            if dtype != torch.float32:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            self._cos = cos
            self._sin = sin
            self._cache_key = cache_key
        return self._cos, self._sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to the last dimension of a head tensor.

    :param Tensor x: Input tensor with paired rotary dimensions.
    :param Tensor cos: Cached cosine table.
    :param Tensor sin: Cached sine table.
    :return Tensor: Rotated tensor.
    """
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """Grouped-query causal self-attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
    ):
        """Initialize the causal attention module.

        :param int dim: Model width.
        :param int num_heads: Attention query head count.
        :param int num_kv_heads: Key/value head count for GQA.
        :param float rope_base: Rotary frequency base, defaults to 10000.0.
        :param float qk_gain_init: Initial per-head query gain, defaults to 1.5.
        """
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal grouped-query attention.

        :param Tensor x: Input activations shaped `(batch, seq, dim)`.
        :return Tensor: Attention outputs shaped `(batch, seq, dim)`.
        """
        B, T, D = x.shape
        with profile_range("attn.forward"):
            with profile_range("attn.project_qkv"):
                q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim)
                k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
                v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim)

            with profile_range("attn.norm_rope"):
                q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
                cos, sin = self.rotary(T, x.device, q.dtype)
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                q_gain = match_reference_tensor(self.q_gain, q)[None, None, :, None]
                q = q * q_gain

            use_fa3 = (
                _USE_FLASH_ATTN_3
                and _HAS_FLASH_ATTN_3
                and x.is_cuda
                and torch.cuda.get_device_capability(x.device) >= (9, 0)
                and q.dtype in {torch.float16, torch.bfloat16}
            )
            if use_fa3:
                with profile_range("attn.fa3"):
                    assert flash_attn_3_func is not None
                    y = flash_attn_3_func(q, k, v, causal=True)
            else:
                with profile_range("attn.sdpa"):
                    y = F.scaled_dot_product_attention(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        is_causal=True,
                        enable_gqa=(self.num_kv_heads != self.num_heads),
                    ).transpose(1, 2)

            with profile_range("attn.output_proj"):
                return self.proj(y.contiguous().reshape(B, T, D))


# ── MLP ───────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """Squared LeakyReLU feed-forward block."""

    def __init__(self, dim: int, mult: float = 3.0, leaky_slope: float = 0.5):
        """Initialize the MLP block.

        :param int dim: Model width.
        :param float mult: Hidden width multiplier, defaults to 3.0.
        :param float leaky_slope: LeakyReLU slope, defaults to 0.5.
        """
        super().__init__()
        hidden = int(dim * mult)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.leaky_slope = leaky_slope

    def forward(self, x: Tensor) -> Tensor:
        """Apply the squared LeakyReLU MLP.

        :param Tensor x: Input activations.
        :return Tensor: MLP outputs.
        """
        with profile_range("mlp.forward"):
            h = F.leaky_relu(self.fc(x), negative_slope=self.leaky_slope)
            return self.proj(h * h)


# ── Blocks ────────────────────────────────────────────────────────────


class GDNBlock(nn.Module):
    """Residual transformer block with Gated DeltaNet mixing."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        mlp_mult: float = 3.0,
        head_k_dim: int = 48,
        expand_v: float = 2.0,
        allow_neg_eigval: bool = True,
        conv_size: int = 4,
        use_q_conv: bool = True,
        use_k_conv: bool = True,
        use_v_conv: bool = True,
        use_packed_qkv_conv: bool = False,
        use_packed_qkv_proj: bool = False,
        conv_output_contiguous: bool = False,
        q_conv_output_contiguous: bool | None = None,
        k_conv_output_contiguous: bool | None = None,
        v_conv_output_contiguous: bool | None = None,
        gates_fp32: bool = True,
        output_norm_fp32: bool = True,
        use_packed_qkv_conv_custom_backward: bool = False,
        use_packed_qkv_single_contig: bool = False,
        use_packed_qkv_split_copy: bool = False,
        fla_recurrence_mode: str = "compile_visible",
        leaky_slope: float = 0.5,
        norm_style: NormStyle = "pre",
        residual_alpha: float = 1.0,
        is_first_block: bool = False,
    ):
        """Initialize the GDN residual block.

        :param int dim: Model width.
        :param int n_heads: GDN head count, defaults to 4.
        :param float mlp_mult: MLP expansion factor, defaults to 3.0.
        :param int head_k_dim: GDN key width per head, defaults to 48.
        :param float expand_v: GDN value expansion factor, defaults to 2.0.
        :param bool allow_neg_eigval: Whether to allow negative eigenvalues, defaults to True.
        :param int conv_size: Causal convolution width, defaults to 4.
        :param bool use_q_conv: Whether to apply the q-path convolution, defaults to True.
        :param bool use_k_conv: Whether to apply the k-path convolution, defaults to True.
        :param bool use_v_conv: Whether to apply the v-path convolution, defaults to True.
        :param bool use_packed_qkv_conv: Whether to replace separate q/k/v depthwise convs with one packed depthwise conv, defaults to False.
        :param bool use_packed_qkv_proj: Whether to replace separate q/k/v feature-map projections with one packed projection, defaults to False.
        :param bool conv_output_contiguous: Whether to materialize contiguous `(batch, seq, dim)` conv outputs before recurrence prep, defaults to False.
        :param bool | None q_conv_output_contiguous: Optional q-path override for contiguous conv outputs, defaults to None.
        :param bool | None k_conv_output_contiguous: Optional k-path override for contiguous conv outputs, defaults to None.
        :param bool | None v_conv_output_contiguous: Optional v-path override for contiguous conv outputs, defaults to None.
        :param bool gates_fp32: Whether to keep the decay-gate softplus path in fp32, defaults to True.
        :param bool output_norm_fp32: Whether to keep the post-recurrence RMSNorm in fp32 before casting back to the activation dtype, defaults to True.
        :param bool use_packed_qkv_conv_custom_backward: Whether to route the packed depthwise qkv conv through an exact-length custom autograd path, defaults to False.
        :param bool use_packed_qkv_single_contig: Whether to materialize one contiguous packed q/k/v tensor before splitting the packed conv output, defaults to False.
        :param bool use_packed_qkv_split_copy: Whether to materialize q/k/v with `aten.split_with_sizes_copy`, defaults to False.
        :param str fla_recurrence_mode: Public FLA recurrence path used by the GDN layer, defaults to `"compile_visible"`.
        :param float leaky_slope: LeakyReLU slope, defaults to 0.5.
        :param NormStyle norm_style: Residual norm placement, defaults to "pre".
        :param float residual_alpha: Residual scaling factor used by KEEL-style blocks, defaults to 1.0.
        :param bool is_first_block: Whether this is the first block in the stack, defaults to False.
        """
        super().__init__()
        self.norm_style = validate_norm_style(norm_style)
        self.residual_alpha = residual_alpha
        self.is_first_block = is_first_block
        self.attn_in_norm, self.attn_out_norm = RMSNorm(), RMSNorm()
        self.mlp_in_norm, self.mlp_out_norm = RMSNorm(), RMSNorm()
        self.gdn = GatedDeltaNet(
            dim,
            n_heads,
            head_k_dim,
            expand_v,
            allow_neg_eigval,
            conv_size,
            use_q_conv=use_q_conv,
            use_k_conv=use_k_conv,
            use_v_conv=use_v_conv,
            use_packed_qkv_conv=use_packed_qkv_conv,
            use_packed_qkv_proj=use_packed_qkv_proj,
            conv_output_contiguous=conv_output_contiguous,
            q_conv_output_contiguous=q_conv_output_contiguous,
            k_conv_output_contiguous=k_conv_output_contiguous,
            v_conv_output_contiguous=v_conv_output_contiguous,
            gates_fp32=gates_fp32,
            output_norm_fp32=output_norm_fp32,
            use_packed_qkv_conv_custom_backward=use_packed_qkv_conv_custom_backward,
            use_packed_qkv_single_contig=use_packed_qkv_single_contig,
            use_packed_qkv_split_copy=use_packed_qkv_split_copy,
            fla_recurrence_mode=fla_recurrence_mode,
        )
        self.mlp = MLP(dim, mlp_mult, leaky_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack([torch.ones(dim), torch.zeros(dim)]).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        """Apply the residual GDN block.

        :param Tensor x: Current activations.
        :param Tensor x0: Block-stack input used by residual mixing.
        :return Tensor: Updated activations.
        """
        with profile_range("block.gdn"):
            mix = match_reference_tensor(self.resid_mix, x)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            attn_scale = match_reference_tensor(self.attn_scale, x)[None, None, :]
            mlp_scale = match_reference_tensor(self.mlp_scale, x)[None, None, :]
            if self.norm_style == "pre":
                x = x + attn_scale * self.gdn(self.attn_in_norm(x))
                x = x + mlp_scale * self.mlp(self.mlp_in_norm(x))
                return x
            if self.norm_style == "post":
                x = self.attn_out_norm(x + attn_scale * self.gdn(x))
                x = self.mlp_out_norm(x + mlp_scale * self.mlp(x))
                return x
            if self.is_first_block:
                x = x + attn_scale * self.gdn(self.attn_in_norm(x))
                x = x + mlp_scale * self.mlp(self.mlp_in_norm(x))
                return x
            x = self.attn_out_norm(
                self.residual_alpha * x + attn_scale * self.gdn(self.attn_in_norm(x))
            )
            x = self.mlp_out_norm(
                self.residual_alpha * x + mlp_scale * self.mlp(self.mlp_in_norm(x))
            )
            return x


class AttnBlock(nn.Module):
    """Residual transformer block with causal attention mixing."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float = 3.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        leaky_slope: float = 0.5,
        norm_style: NormStyle = "pre",
        residual_alpha: float = 1.0,
        is_first_block: bool = False,
    ):
        """Initialize the attention residual block.

        :param int dim: Model width.
        :param int num_heads: Query head count.
        :param int num_kv_heads: Key/value head count.
        :param float mlp_mult: MLP expansion factor, defaults to 3.0.
        :param float rope_base: Rotary frequency base, defaults to 10000.0.
        :param float qk_gain_init: Initial query gain, defaults to 1.5.
        :param float leaky_slope: LeakyReLU slope, defaults to 0.5.
        :param NormStyle norm_style: Residual norm placement, defaults to "pre".
        :param float residual_alpha: Residual scaling factor used by KEEL-style blocks, defaults to 1.0.
        :param bool is_first_block: Whether this is the first block in the stack, defaults to False.
        """
        super().__init__()
        self.norm_style = validate_norm_style(norm_style)
        self.residual_alpha = residual_alpha
        self.is_first_block = is_first_block
        self.attn_in_norm, self.attn_out_norm = RMSNorm(), RMSNorm()
        self.mlp_in_norm, self.mlp_out_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init
        )
        self.mlp = MLP(dim, mlp_mult, leaky_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack([torch.ones(dim), torch.zeros(dim)]).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        """Apply the residual attention block.

        :param Tensor x: Current activations.
        :param Tensor x0: Block-stack input used by residual mixing.
        :return Tensor: Updated activations.
        """
        with profile_range("block.attn"):
            mix = match_reference_tensor(self.resid_mix, x)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            attn_scale = match_reference_tensor(self.attn_scale, x)[None, None, :]
            mlp_scale = match_reference_tensor(self.mlp_scale, x)[None, None, :]
            if self.norm_style == "pre":
                x = x + attn_scale * self.attn(self.attn_in_norm(x))
                x = x + mlp_scale * self.mlp(self.mlp_in_norm(x))
                return x
            if self.norm_style == "post":
                x = self.attn_out_norm(x + attn_scale * self.attn(x))
                x = self.mlp_out_norm(x + mlp_scale * self.mlp(x))
                return x
            if self.is_first_block:
                x = x + attn_scale * self.attn(self.attn_in_norm(x))
                x = x + mlp_scale * self.mlp(self.mlp_in_norm(x))
                return x
            x = self.attn_out_norm(
                self.residual_alpha * x + attn_scale * self.attn(self.attn_in_norm(x))
            )
            x = self.mlp_out_norm(
                self.residual_alpha * x + mlp_scale * self.mlp(self.mlp_in_norm(x))
            )
            return x


# ── Hybrid GPT ────────────────────────────────────────────────────────


class HybridGPT(nn.Module):
    """Interleaved GDN-attention language model used by the hybrid trainer."""

    def __init__(
        self,
        vocab_size: int = 1024,
        num_layers: int = 16,
        d_model: int = 384,
        # Attention config
        attn_heads: int = 8,
        attn_kv_heads: int = 4,
        # GDN config (single n_heads, expand_v controls state)
        gdn_n_heads: int = 4,
        gdn_head_k_dim: int = 48,
        gdn_expand_v: float = 2.0,
        gdn_allow_neg_eigval: bool = True,
        gdn_conv_size: int = 4,
        gdn_use_q_conv: bool = True,
        gdn_use_k_conv: bool = True,
        gdn_use_v_conv: bool = True,
        gdn_use_packed_qkv_conv: bool = False,
        gdn_use_packed_qkv_proj: bool = False,
        gdn_conv_output_contiguous: bool = False,
        gdn_q_conv_output_contiguous: bool | None = None,
        gdn_k_conv_output_contiguous: bool | None = None,
        gdn_v_conv_output_contiguous: bool | None = None,
        gdn_gates_fp32: bool = True,
        gdn_output_norm_fp32: bool = True,
        gdn_use_packed_qkv_conv_custom_backward: bool = False,
        gdn_use_packed_qkv_single_contig: bool = False,
        gdn_use_packed_qkv_split_copy: bool = False,
        gdn_fla_recurrence_mode: str = "compile_visible",
        # Shared
        mlp_mult: float = 3.0,
        leaky_slope: float = 0.5,
        gdn_ratio: int = 3,
        block_pattern: str | list[str] | None = None,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        logit_softcap: float = 30.0,
        tie_embeddings: bool = True,
        tied_embed_init_std: float = 0.005,
        norm_style: NormStyle = "pre",
        residual_alpha: float | None = None,
    ):
        """Initialize the hybrid language model.

        :param int vocab_size: Token vocabulary size, defaults to 1024.
        :param int num_layers: Total block count, defaults to 16.
        :param int d_model: Model width, defaults to 384.
        :param int attn_heads: Attention query head count, defaults to 8.
        :param int attn_kv_heads: Attention key/value head count, defaults to 4.
        :param int gdn_n_heads: GDN head count, defaults to 4.
        :param int gdn_head_k_dim: GDN key width per head, defaults to 48.
        :param float gdn_expand_v: GDN value expansion factor, defaults to 2.0.
        :param bool gdn_allow_neg_eigval: Whether to allow negative eigenvalues, defaults to True.
        :param int gdn_conv_size: GDN convolution width, defaults to 4.
        :param bool gdn_use_q_conv: Whether to apply the q-path convolution, defaults to True.
        :param bool gdn_use_k_conv: Whether to apply the k-path convolution, defaults to True.
        :param bool gdn_use_v_conv: Whether to apply the v-path convolution, defaults to True.
        :param bool gdn_use_packed_qkv_conv: Whether to replace separate q/k/v depthwise convs with one packed depthwise conv, defaults to False.
        :param bool gdn_use_packed_qkv_proj: Whether to replace separate q/k/v feature-map projections with one packed projection, defaults to False.
        :param bool gdn_conv_output_contiguous: Whether to materialize contiguous `(batch, seq, dim)` conv outputs before recurrence prep, defaults to False.
        :param bool | None gdn_q_conv_output_contiguous: Optional q-path override for contiguous conv outputs, defaults to None.
        :param bool | None gdn_k_conv_output_contiguous: Optional k-path override for contiguous conv outputs, defaults to None.
        :param bool | None gdn_v_conv_output_contiguous: Optional v-path override for contiguous conv outputs, defaults to None.
        :param bool gdn_gates_fp32: Whether to keep the decay-gate softplus path in fp32, defaults to True.
        :param bool gdn_output_norm_fp32: Whether to keep the post-recurrence RMSNorm in fp32 before casting back to the activation dtype, defaults to True.
        :param bool gdn_use_packed_qkv_conv_custom_backward: Whether to route the packed depthwise qkv conv through an exact-length custom autograd path, defaults to False.
        :param bool gdn_use_packed_qkv_single_contig: Whether to materialize one contiguous packed q/k/v tensor before splitting the packed conv output, defaults to False.
        :param bool gdn_use_packed_qkv_split_copy: Whether to materialize q/k/v with `aten.split_with_sizes_copy`, defaults to False.
        :param str gdn_fla_recurrence_mode: Public FLA recurrence path used by GDN layers, defaults to `"compile_visible"`.
        :param float mlp_mult: MLP expansion factor, defaults to 3.0.
        :param float leaky_slope: LeakyReLU slope, defaults to 0.5.
        :param int gdn_ratio: Number of GDN layers per attention layer, defaults to 3.
        :param str | list[str] | None block_pattern: Optional explicit
            comma-separated or list-valued block schedule using `attn`/`gdn`
            tokens, defaults to None.
        :param float rope_base: Rotary frequency base, defaults to 10000.0.
        :param float qk_gain_init: Initial attention query gain, defaults to 1.5.
        :param float logit_softcap: Tanh logit softcap, defaults to 30.0.
        :param bool tie_embeddings: Whether to tie embedding and output weights, defaults to True.
        :param float tied_embed_init_std: Embedding init std when tied, defaults to 0.005.
        :param NormStyle norm_style: Residual norm placement inside each block, defaults to "pre".
        :param float | None residual_alpha: Optional residual scaling for KEEL-style blocks, defaults to None.
        """
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.d_model = d_model
        self.norm_style = validate_norm_style(norm_style)
        self.residual_alpha = (
            float(2 * num_layers)
            if residual_alpha is None and self.norm_style == "keel"
            else float(1.0 if residual_alpha is None else residual_alpha)
        )
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        pattern = normalize_block_pattern(
            block_pattern, num_layers=num_layers, gdn_ratio=gdn_ratio
        )
        blocks, self.block_types = [], []
        for i, block_type in enumerate(pattern):
            if block_type == "attn":
                blocks.append(
                    AttnBlock(
                        d_model,
                        attn_heads,
                        attn_kv_heads,
                        mlp_mult,
                        rope_base,
                        qk_gain_init,
                        leaky_slope,
                        self.norm_style,
                        self.residual_alpha,
                        i == 0,
                    )
                )
                self.block_types.append("attn")
            elif block_type == "gdn":
                blocks.append(
                    GDNBlock(
                        d_model,
                        gdn_n_heads,
                        mlp_mult,
                        gdn_head_k_dim,
                        gdn_expand_v,
                        gdn_allow_neg_eigval,
                        gdn_conv_size,
                        gdn_use_q_conv,
                        gdn_use_k_conv,
                        gdn_use_v_conv,
                        gdn_use_packed_qkv_conv,
                        gdn_use_packed_qkv_proj,
                        gdn_conv_output_contiguous,
                        gdn_q_conv_output_contiguous,
                        gdn_k_conv_output_contiguous,
                        gdn_v_conv_output_contiguous,
                        gdn_gates_fp32,
                        gdn_output_norm_fp32,
                        gdn_use_packed_qkv_conv_custom_backward,
                        gdn_use_packed_qkv_single_contig,
                        gdn_use_packed_qkv_split_copy,
                        gdn_fla_recurrence_mode,
                        leaky_slope,
                        self.norm_style,
                        self.residual_alpha,
                        i == 0,
                    )
                )
                self.block_types.append("gdn")
            else:  # pragma: no cover - normalize_block_pattern guards this
                raise ValueError(f"Unknown block type {block_type!r}")
        self.blocks = nn.ModuleList(blocks)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, d_model, dtype=torch.float32)
        )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings else CastedLinear(d_model, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights(tied_embed_init_std)

    def _init_weights(self, tied_std: float) -> None:
        """Initialize embeddings and zero-init marked projections.

        :param float tied_std: Standard deviation for tied embedding init.
        """
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Compute the autoregressive training loss.

        :param Tensor input_ids: Input token ids shaped `(batch, seq)`.
        :param Tensor target_ids: Next-token targets shaped `(batch, seq)`.
        :return Tensor: Mean cross-entropy loss.
        """
        x = rms_norm(self.tok_emb(input_ids))
        x0, skips = x, []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                skip_weight = match_reference_tensor(self.skip_weights[i], x)
                x = x + skip_weight[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = (
            F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        )
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")


# ── Presets ──────────────────────────────────────────────────────────
#
# These constructors mirror the historical sweep families. Earlier artifact-size
# comments here were based on a stale proxy estimate; use the trainer's final
# quantized artifact audit or docs/README.md for current measured bytes.


def make_hybrid_tight(vocab_size: int = 1024) -> HybridGPT:
    """Build the primary budget-filling hybrid preset.

    :param int vocab_size: Vocabulary size, defaults to 1024.
    :return HybridGPT: Hybrid preset with matched `Dk=Dv=48`.
    """
    return HybridGPT(
        vocab_size=vocab_size,
        num_layers=16,
        d_model=384,
        gdn_n_heads=8,
        gdn_head_k_dim=48,
        gdn_expand_v=1.0,
        gdn_ratio=3,
        mlp_mult=3.0,
    )


def make_hybrid_wide(vocab_size: int = 1024) -> HybridGPT:
    """Build the wider-state hybrid preset.

    :param int vocab_size: Vocabulary size, defaults to 1024.
    :return HybridGPT: Hybrid preset with wider per-head state.
    """
    return HybridGPT(
        vocab_size=vocab_size,
        num_layers=16,
        d_model=384,
        gdn_n_heads=4,
        gdn_head_k_dim=48,
        gdn_expand_v=2.0,
        gdn_ratio=3,
        mlp_mult=3.25,
    )


def make_baseline_fill(vocab_size: int = 1024) -> HybridGPT:
    """Build the width-matched pure-attention baseline.

    :param int vocab_size: Vocabulary size, defaults to 1024.
    :return HybridGPT: Pure-attention baseline preset.
    """
    return HybridGPT(
        vocab_size=vocab_size, num_layers=11, d_model=512, gdn_ratio=0, mlp_mult=2.75
    )


def make_attention_only_baseline(vocab_size: int = 1024) -> HybridGPT:
    """Build the depth-matched attention-only baseline preset.

    :param int vocab_size: Vocabulary size, defaults to 1024.
    :return HybridGPT: Attention-only baseline with matched depth.
    """
    return HybridGPT(
        vocab_size=vocab_size, num_layers=16, d_model=384, gdn_ratio=0, mlp_mult=3.75
    )


make_depth_control = make_attention_only_baseline
