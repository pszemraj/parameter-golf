"""Reference HGDN fused-kernel paths.

These implementations preserve the intended semantics using stock PyTorch ops.
They are used both as CPU fallbacks and as correctness references for the CUDA
extension path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_LOW_PRECISION_DTYPES = {torch.float16, torch.bfloat16}


def _norm_accum_dtype(x: Tensor) -> torch.dtype:
    """Return the accumulation dtype used for normalization-like reductions.

    :param Tensor x: Input tensor.
    :return torch.dtype: Dtype to use for reduction accumulation.
    """
    return torch.float32 if x.dtype in _LOW_PRECISION_DTYPES else x.dtype


def packed_qkv_frontend_reference(
    qkv: Tensor,
    weight: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference packed HGDN front-end.

    Applies the packed causal depthwise conv + SiLU activation, then splits the
    packed tensor into q/k/v and performs q/k L2 normalization with fp32
    accumulation when the activation dtype is low precision.

    :param Tensor qkv: Packed q/k/v projections shaped ``(batch, seq, channels)``.
    :param Tensor weight: Depthwise conv weights shaped ``(channels, kernel)`` or
        ``(channels, 1, kernel)``.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: Numerical stability epsilon.
    :raises ValueError: If the packed tensor or conv weight shape is unsupported.
    :return tuple[Tensor, Tensor, Tensor]: Normalized q/k and reshaped v.
    """
    if qkv.ndim != 3:
        raise ValueError(f"Expected qkv.ndim == 3, got {qkv.ndim}")
    if weight.ndim == 3:
        weight = weight.view(weight.shape[0], weight.shape[-1])
    if weight.ndim != 2:
        raise ValueError(f"Expected weight.ndim in {{2, 3}}, got {weight.ndim}")

    batch, seq, channels = qkv.shape
    q_dim = n_heads * head_k_dim
    k_dim = n_heads * head_k_dim
    v_dim = n_heads * head_v_dim
    if channels != q_dim + k_dim + v_dim:
        raise ValueError(
            f"Packed frontend channel mismatch: got {channels}, expected {q_dim + k_dim + v_dim}"
        )

    packed = F.conv1d(
        qkv.transpose(1, 2),
        weight[:, None, :],
        padding=weight.shape[-1] - 1,
        groups=channels,
    )[..., :seq]
    packed = F.silu(packed).transpose(1, 2)
    q_pre, k_pre, v = packed.split((q_dim, k_dim, v_dim), dim=-1)

    accum_dtype = _norm_accum_dtype(q_pre)
    q = F.normalize(
        q_pre.view(batch, seq, n_heads, head_k_dim).to(dtype=accum_dtype),
        p=2,
        dim=-1,
        eps=eps,
    ).to(dtype=qkv.dtype)
    k = F.normalize(
        k_pre.view(batch, seq, n_heads, head_k_dim).to(dtype=accum_dtype),
        p=2,
        dim=-1,
        eps=eps,
    ).to(dtype=qkv.dtype)
    v = v.view(batch, seq, n_heads, head_v_dim).to(dtype=qkv.dtype)
    return q.contiguous(), k.contiguous(), v.contiguous()


def rmsnorm_silu_gate_reference(
    o: Tensor,
    gate: Tensor,
    *,
    eps: float = 1e-6,
    fp32_accum: bool = True,
) -> Tensor:
    """Reference post-recurrence fused output path.

    Computes ``rms_norm(o) * silu(gate)`` with optional fp32 accumulation in the
    RMSNorm reduction.

    :param Tensor o: Recurrence outputs shaped ``(..., dim)``.
    :param Tensor gate: Output-gate pre-activations with the same shape as ``o``.
    :param float eps: Numerical stability epsilon.
    :param bool fp32_accum: Whether to accumulate the RMS reduction in fp32.
    :raises ValueError: If ``o`` and ``gate`` shapes do not match.
    :return Tensor: Gated normalized outputs with the same dtype as ``o``.
    """
    if o.shape != gate.shape:
        raise ValueError(f"Shape mismatch: o={tuple(o.shape)} gate={tuple(gate.shape)}")
    if fp32_accum and o.dtype in _LOW_PRECISION_DTYPES:
        normed = F.rms_norm(o.float(), (o.size(-1),), eps=eps).to(dtype=o.dtype)
    else:
        normed = F.rms_norm(o, (o.size(-1),), eps=eps).to(dtype=o.dtype)
    return normed * F.silu(gate)
