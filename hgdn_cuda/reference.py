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


def _silu_backward_from_preact(preact: Tensor) -> Tensor:
    """Return the exact SiLU derivative evaluated at the pre-activation.

    :param Tensor preact: Input pre-activation tensor.
    :return Tensor: SiLU derivative with the same dtype as ``preact``.
    """
    preact_f = preact.float()
    sigma = torch.sigmoid(preact_f)
    return (sigma * (1.0 + preact_f * (1.0 - sigma))).to(dtype=preact.dtype)


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


def packed_qkv_conv_reference(
    qkv: Tensor,
    weight: Tensor,
) -> Tensor:
    """Reference exact-length packed causal depthwise conv plus SiLU.

    This matches the promoted packed HGDN front-end at the point immediately
    before q/k/v splitting and q/k normalization.

    :param Tensor qkv: Packed q/k/v projections shaped ``(batch, seq, channels)``.
    :param Tensor weight: Depthwise conv weights shaped ``(channels, kernel)`` or
        ``(channels, 1, kernel)``.
    :raises ValueError: If the packed tensor or conv weight shape is unsupported.
    :return Tensor: Packed post-conv activations shaped ``(batch, seq, channels)``.
    """
    if qkv.ndim != 3:
        raise ValueError(f"Expected qkv.ndim == 3, got {qkv.ndim}")
    if weight.ndim == 3:
        weight = weight.view(weight.shape[0], weight.shape[-1])
    if weight.ndim != 2:
        raise ValueError(f"Expected weight.ndim in {{2, 3}}, got {weight.ndim}")
    if qkv.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"Packed conv channel mismatch: qkv={qkv.shape[-1]} weight={weight.shape[0]}"
        )

    seq = qkv.shape[1]
    packed = F.conv1d(
        qkv.transpose(1, 2),
        weight[:, None, :],
        padding=weight.shape[-1] - 1,
        groups=qkv.shape[-1],
    )[..., :seq]
    return F.silu(packed).transpose(1, 2).contiguous()


def packed_qkv_split_l2norm_reference(
    packed: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference post-conv packed split plus q/k L2 normalization.

    This matches the active packed HGDN front-end contract after the packed
    depthwise conv and SiLU have already been applied.

    :param Tensor packed: Packed post-conv activations shaped ``(batch, seq, channels)``.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: Numerical stability epsilon.
    :raises ValueError: If the packed tensor shape is unsupported.
    :return tuple[Tensor, Tensor, Tensor]: Normalized q/k and reshaped v.
    """
    if packed.ndim != 3:
        raise ValueError(f"Expected packed.ndim == 3, got {packed.ndim}")

    batch, seq, channels = packed.shape
    q_dim = n_heads * head_k_dim
    k_dim = n_heads * head_k_dim
    v_dim = n_heads * head_v_dim
    if channels != q_dim + k_dim + v_dim:
        raise ValueError(
            f"Packed split/norm channel mismatch: got {channels}, expected {q_dim + k_dim + v_dim}"
        )

    q_pre, k_pre, v = packed.split((q_dim, k_dim, v_dim), dim=-1)
    accum_dtype = _norm_accum_dtype(q_pre)
    q = F.normalize(
        q_pre.view(batch, seq, n_heads, head_k_dim).to(dtype=accum_dtype),
        p=2,
        dim=-1,
        eps=eps,
    ).to(dtype=packed.dtype)
    k = F.normalize(
        k_pre.view(batch, seq, n_heads, head_k_dim).to(dtype=accum_dtype),
        p=2,
        dim=-1,
        eps=eps,
    ).to(dtype=packed.dtype)
    v = v.view(batch, seq, n_heads, head_v_dim).to(dtype=packed.dtype)
    return q.contiguous(), k.contiguous(), v.contiguous()


def packed_qkv_frontend_reference_with_ctx(
    qkv: Tensor,
    weight: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reference packed frontend with saved tensors for custom autograd formulas.

    :param Tensor qkv: Packed q/k/v projections shaped ``(batch, seq, channels)``.
    :param Tensor weight: Depthwise conv weights shaped ``(channels, kernel)``.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: Numerical stability epsilon.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: Normalized q/k, reshaped v, packed preactivation, and fp32 inverse norms.
    """
    if qkv.ndim != 3:
        raise ValueError(f"Expected qkv.ndim == 3, got {qkv.ndim}")
    if weight.ndim == 3:
        weight = weight.view(weight.shape[0], weight.shape[-1])
    if weight.ndim != 2:
        raise ValueError(f"Expected weight.ndim in {{2, 3}}, got {weight.ndim}")
    if qkv.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"Packed conv channel mismatch: qkv={qkv.shape[-1]} weight={weight.shape[0]}"
        )

    batch, seq, channels = qkv.shape
    q_dim = n_heads * head_k_dim
    k_dim = n_heads * head_k_dim
    v_dim = n_heads * head_v_dim
    expected_channels = q_dim + k_dim + v_dim
    if channels != expected_channels:
        raise ValueError(
            f"Packed frontend channel mismatch: got {channels}, expected {expected_channels}"
        )

    preact = (
        F.conv1d(
            qkv.transpose(1, 2),
            weight[:, None, :],
            padding=weight.shape[-1] - 1,
            groups=qkv.shape[-1],
        )[..., :seq]
        .transpose(1, 2)
        .contiguous()
    )
    activated = F.silu(preact)
    q_pre, k_pre, v = activated.split((q_dim, k_dim, v_dim), dim=-1)
    q_pre = q_pre.view(batch, seq, n_heads, head_k_dim)
    k_pre = k_pre.view(batch, seq, n_heads, head_k_dim)
    v = v.view(batch, seq, n_heads, head_v_dim)

    accum_dtype = _norm_accum_dtype(q_pre)
    q_pre_accum = q_pre.to(dtype=accum_dtype)
    k_pre_accum = k_pre.to(dtype=accum_dtype)
    inv_q = torch.rsqrt(q_pre_accum.square().sum(dim=-1) + eps).float()
    inv_k = torch.rsqrt(k_pre_accum.square().sum(dim=-1) + eps).float()
    q = (q_pre_accum * inv_q.unsqueeze(-1)).to(dtype=qkv.dtype)
    k = (k_pre_accum * inv_k.unsqueeze(-1)).to(dtype=qkv.dtype)
    return (
        q.contiguous(),
        k.contiguous(),
        v.to(dtype=qkv.dtype).contiguous(),
        preact,
        inv_q.contiguous(),
        inv_k.contiguous(),
    )


def packed_qkv_frontend_backward_reference(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    qkv: Tensor,
    weight: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Reference backward for the packed frontend op.

    :param Tensor grad_q: Query gradients shaped ``(batch, seq, heads, head_k_dim)``.
    :param Tensor grad_k: Key gradients shaped ``(batch, seq, heads, head_k_dim)``.
    :param Tensor grad_v: Value gradients shaped ``(batch, seq, heads, head_v_dim)``.
    :param Tensor qkv: Saved packed q/k/v projections.
    :param Tensor weight: Saved depthwise conv weights.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: Numerical stability epsilon.
    :return tuple[Tensor, Tensor]: Gradients for ``qkv`` and ``weight``.
    """
    with torch.enable_grad():
        qkv_ref = qkv.detach().requires_grad_(True)
        weight_ref = weight.detach().requires_grad_(True)
        q, k, v, _preact, _inv_q, _inv_k = packed_qkv_frontend_reference_with_ctx(
            qkv_ref,
            weight_ref,
            n_heads=n_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            eps=eps,
        )
        grad_input, grad_weight = torch.autograd.grad(
            outputs=(q, k, v),
            inputs=(qkv_ref, weight_ref),
            grad_outputs=(grad_q, grad_k, grad_v),
            allow_unused=False,
        )
    return grad_input.contiguous(), grad_weight.contiguous()


def preact_silu_split_l2norm_nct_reference(
    preact_nct: Tensor,
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reference NCT preact frontend op.

    This matches the proposed compile-visible frontend boundary:

    - input: pre-activation tensor in conv-native ``(batch, channels, seq)``
    - work: SiLU, q/k/v split, q/k L2 norm
    - output: q/k/v in recurrence layout plus cached inverse norms

    :param Tensor preact_nct: Packed pre-activations shaped ``(batch, channels, seq)``.
    :param int n_heads: Number of HGDN heads.
    :param int head_k_dim: Per-head q/k width.
    :param int head_v_dim: Per-head value width.
    :param float eps: Numerical stability epsilon.
    :raises ValueError: If the pre-activation tensor shape is unsupported.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Normalized q/k, reshaped v, and fp32 inverse norms.
    """
    if preact_nct.ndim != 3:
        raise ValueError(f"Expected preact_nct.ndim == 3, got {preact_nct.ndim}")

    batch, channels, seq = preact_nct.shape
    q_dim = n_heads * head_k_dim
    k_dim = n_heads * head_k_dim
    v_dim = n_heads * head_v_dim
    expected_channels = q_dim + k_dim + v_dim
    if channels != expected_channels:
        raise ValueError(
            f"NCT preact channel mismatch: got {channels}, expected {expected_channels}"
        )

    activated_btc = F.silu(preact_nct).transpose(1, 2).contiguous()
    q_pre, k_pre, v = activated_btc.split((q_dim, k_dim, v_dim), dim=-1)
    q_pre = q_pre.view(batch, seq, n_heads, head_k_dim)
    k_pre = k_pre.view(batch, seq, n_heads, head_k_dim)
    v = v.view(batch, seq, n_heads, head_v_dim)

    accum_dtype = _norm_accum_dtype(q_pre)
    q_pre_accum = q_pre.to(dtype=accum_dtype)
    k_pre_accum = k_pre.to(dtype=accum_dtype)
    inv_q = torch.rsqrt(q_pre_accum.square().sum(dim=-1) + eps).float()
    inv_k = torch.rsqrt(k_pre_accum.square().sum(dim=-1) + eps).float()
    q = (q_pre_accum * inv_q.unsqueeze(-1)).to(dtype=preact_nct.dtype)
    k = (k_pre_accum * inv_k.unsqueeze(-1)).to(dtype=preact_nct.dtype)
    return (
        q.contiguous(),
        k.contiguous(),
        v.to(dtype=preact_nct.dtype).contiguous(),
        inv_q.contiguous(),
        inv_k.contiguous(),
    )


def preact_silu_split_l2norm_nct_backward_reference(
    grad_q: Tensor,
    grad_k: Tensor,
    grad_v: Tensor,
    preact_nct: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    inv_q: Tensor,
    inv_k: Tensor,
) -> Tensor:
    """Reference backward for the NCT preact frontend op.

    :param Tensor grad_q: Query gradients shaped ``(batch, seq, heads, head_k_dim)``.
    :param Tensor grad_k: Key gradients shaped ``(batch, seq, heads, head_k_dim)``.
    :param Tensor grad_v: Value gradients shaped ``(batch, seq, heads, head_v_dim)``.
    :param Tensor preact_nct: Saved pre-activation tensor shaped ``(batch, channels, seq)``.
    :param Tensor q_norm: Normalized q output from the forward pass.
    :param Tensor k_norm: Normalized k output from the forward pass.
    :param Tensor inv_q: Saved inverse q norms shaped ``(batch, seq, heads)``.
    :param Tensor inv_k: Saved inverse k norms shaped ``(batch, seq, heads)``.
    :return Tensor: Gradient for ``preact_nct`` with the same shape as the input.
    """
    batch, channels, seq = preact_nct.shape
    _, _, n_heads, head_k_dim = grad_q.shape
    head_v_dim = grad_v.shape[-1]
    q_dim = n_heads * head_k_dim
    k_dim = n_heads * head_k_dim
    v_dim = n_heads * head_v_dim
    if channels != q_dim + k_dim + v_dim:
        raise ValueError(
            f"NCT preact channel mismatch: got {channels}, expected {q_dim + k_dim + v_dim}"
        )

    accum_dtype = _norm_accum_dtype(preact_nct)
    grad_q_accum = grad_q.to(dtype=accum_dtype)
    grad_k_accum = grad_k.to(dtype=accum_dtype)
    grad_v_accum = grad_v.to(dtype=accum_dtype)
    q_norm_accum = q_norm.to(dtype=accum_dtype)
    k_norm_accum = k_norm.to(dtype=accum_dtype)
    inv_q_accum = inv_q.to(dtype=accum_dtype).unsqueeze(-1)
    inv_k_accum = inv_k.to(dtype=accum_dtype).unsqueeze(-1)

    q_dot = (grad_q_accum * q_norm_accum).sum(dim=-1, keepdim=True)
    k_dot = (grad_k_accum * k_norm_accum).sum(dim=-1, keepdim=True)
    grad_q_pre = (grad_q_accum - q_norm_accum * q_dot) * inv_q_accum
    grad_k_pre = (grad_k_accum - k_norm_accum * k_dot) * inv_k_accum

    preact_btc = preact_nct.transpose(1, 2).contiguous()
    q_preact, k_preact, v_preact = preact_btc.split((q_dim, k_dim, v_dim), dim=-1)
    q_preact = q_preact.view(batch, seq, n_heads, head_k_dim)
    k_preact = k_preact.view(batch, seq, n_heads, head_k_dim)
    v_preact = v_preact.view(batch, seq, n_heads, head_v_dim)

    grad_q_preact = grad_q_pre * _silu_backward_from_preact(q_preact).to(
        dtype=accum_dtype
    )
    grad_k_preact = grad_k_pre * _silu_backward_from_preact(k_preact).to(
        dtype=accum_dtype
    )
    grad_v_preact = grad_v_accum * _silu_backward_from_preact(v_preact).to(
        dtype=accum_dtype
    )

    grad_preact_btc = torch.cat(
        (
            grad_q_preact.reshape(batch, seq, q_dim),
            grad_k_preact.reshape(batch, seq, k_dim),
            grad_v_preact.reshape(batch, seq, v_dim),
        ),
        dim=-1,
    )
    return grad_preact_btc.transpose(1, 2).contiguous().to(dtype=preact_nct.dtype)


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
