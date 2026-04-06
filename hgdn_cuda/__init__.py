"""HGDN fused CUDA kernels and safe fallbacks."""

from .ops import (
    extension_loaded,
    extension_status,
    fused_packed_qkv_conv,
    fused_packed_qkv_frontend,
    fused_packed_qkv_split_l2norm,
    fused_rmsnorm_silu_gate,
)
from .reference import (
    packed_qkv_conv_reference,
    packed_qkv_frontend_reference,
    packed_qkv_split_l2norm_reference,
    rmsnorm_silu_gate_reference,
)

__all__ = [
    "extension_loaded",
    "extension_status",
    "fused_packed_qkv_conv",
    "fused_packed_qkv_frontend",
    "fused_packed_qkv_split_l2norm",
    "fused_rmsnorm_silu_gate",
    "packed_qkv_conv_reference",
    "packed_qkv_frontend_reference",
    "packed_qkv_split_l2norm_reference",
    "rmsnorm_silu_gate_reference",
]
