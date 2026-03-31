#!/usr/bin/env python3
"""Benchmark a fused ALlama attention-prep kernel candidate.

This isolates the attention boundary immediately after the fused qkv projection:

- split qkv into q, k, v
- reshape into SDPA layout
- apply q/k RMSNorm
- apply RoPE
- apply q_gain to q

The goal is to test whether a larger attention-side boundary can beat the
compiled PyTorch reference before any training-path integration work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_func
except Exception:  # pragma: no cover
    flash_attn_func = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from allama_shared import apply_rotary_emb, sdpa_enable_gqa_available  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the attention-prep benchmark.

    :return argparse.Namespace: Parsed benchmark configuration.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs_allama_validation/attention_prep"),
        help="Directory for benchmark JSON outputs.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=50,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--measured-iters",
        type=int,
        default=200,
        help="Measured iterations for timing.",
    )
    parser.add_argument(
        "--backend",
        choices=("sdpa", "fa2"),
        default="fa2",
        help="Attention backend paired with the prep boundary benchmark.",
    )
    return parser.parse_args()


def next_power_of_2(value: int) -> int:
    """Return the next power-of-two integer greater than or equal to ``value``."""
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def rotary_cache(
    seq_len: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct RoPE cos/sin tensors matching the ALlama attention path."""
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    return cos, sin


def attention_prep_reference(
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qk_norm: bool = True,
    bshd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference PyTorch implementation of the attention-prep boundary."""
    bsz, seqlen, qkv_dim = qkv.shape
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    if qkv_dim != q_dim + 2 * kv_dim:
        raise ValueError("qkv shape does not match the requested head layout")
    q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
    if bshd:
        q = q.view(bsz, seqlen, num_heads, head_dim)
        k = k.view(bsz, seqlen, num_kv_heads, head_dim)
        v = v.view(bsz, seqlen, num_kv_heads, head_dim)
    else:
        q = q.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2)
    if qk_norm:
        q = F.rms_norm(q, (head_dim,))
        k = F.rms_norm(k, (head_dim,))
    if bshd:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * q_gain.to(dtype=q.dtype)[None, None, :, None]
    else:
        q = apply_rotary_emb(q, cos[None, None, :, :], sin[None, None, :, :])
        k = apply_rotary_emb(k, cos[None, None, :, :], sin[None, None, :, :])
        q = q * q_gain.to(dtype=q.dtype)[None, :, None, None]
    return q, k, v


def attention_forward_reference(
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qk_norm: bool = True,
    backend: str = "sdpa",
) -> torch.Tensor:
    """Reference prep plus flash-attention forward."""
    use_fa2 = backend == "fa2"
    q, k, v = attention_prep_reference(
        qkv,
        q_gain,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm=qk_norm,
        bshd=use_fa2,
    )
    if use_fa2:
        if flash_attn_func is None:
            raise RuntimeError("backend='fa2' requires flash-attn to be installed")
        return flash_attn_func(q, k, v, causal=True)
    if num_heads != num_kv_heads and sdpa_enable_gqa_available():
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=True,
        )
    if num_heads != num_kv_heads:
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


@triton.jit
def attention_prep_kernel(
    qkv_ptr,
    q_gain_ptr,
    cos_ptr,
    sin_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    batch_size,
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    q_dim,
    kv_dim,
    qkv_stride_b,
    qkv_stride_t,
    q_stride_b,
    q_stride_h,
    q_stride_t,
    q_stride_d,
    kv_stride_b,
    kv_stride_h,
    kv_stride_t,
    kv_stride_d,
    BLOCK_D: tl.constexpr,
):
    """Fused Triton kernel for the ALlama attention-prep boundary."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    b = token_idx // seq_len
    t = token_idx % seq_len
    cols = tl.arange(0, BLOCK_D)
    mask = cols < head_dim

    qkv_base = qkv_ptr + b * qkv_stride_b + t * qkv_stride_t
    half = head_dim // 2
    half_cols = tl.arange(0, BLOCK_D // 2)
    half_mask = half_cols < half
    cos_ptrs = cos_ptr + t * half + half_cols
    sin_ptrs = sin_ptr + t * half + half_cols
    cos_vals = tl.load(cos_ptrs, mask=half_mask, other=0.0).to(tl.float32)
    sin_vals = tl.load(sin_ptrs, mask=half_mask, other=0.0).to(tl.float32)

    if head_idx < num_heads:
        q_head_base = qkv_base + head_idx * head_dim
        q1 = tl.load(q_head_base + half_cols, mask=half_mask, other=0.0).to(tl.float32)
        q2 = tl.load(q_head_base + half + half_cols, mask=half_mask, other=0.0).to(
            tl.float32
        )
        rms = tl.sqrt(
            (tl.sum(q1 * q1, axis=0) + tl.sum(q2 * q2, axis=0)) / head_dim + 1e-5
        )
        q1 = q1 / rms
        q2 = q2 / rms
        rot_first = q1 * cos_vals + q2 * sin_vals
        rot_second = q2 * cos_vals - q1 * sin_vals
        gain = tl.load(q_gain_ptr + head_idx).to(tl.float32)
        q_out_base = q_ptr + b * q_stride_b + head_idx * q_stride_h + t * q_stride_t
        tl.store(
            q_out_base + half_cols * q_stride_d,
            (rot_first * gain).to(tl.bfloat16),
            mask=half_mask,
        )
        tl.store(
            q_out_base + (half + half_cols) * q_stride_d,
            (rot_second * gain).to(tl.bfloat16),
            mask=half_mask,
        )
        return

    kv_head = head_idx - num_heads
    if kv_head >= num_kv_heads:
        return

    k_head_base = qkv_base + q_dim + kv_head * head_dim
    v_head_base = qkv_base + q_dim + kv_dim + kv_head * head_dim
    k1 = tl.load(k_head_base + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_head_base + half + half_cols, mask=half_mask, other=0.0).to(
        tl.float32
    )
    v_vals = tl.load(v_head_base + cols, mask=mask, other=0.0)
    rms = tl.sqrt((tl.sum(k1 * k1, axis=0) + tl.sum(k2 * k2, axis=0)) / head_dim + 1e-5)
    k1 = k1 / rms
    k2 = k2 / rms
    rot_first = k1 * cos_vals + k2 * sin_vals
    rot_second = k2 * cos_vals - k1 * sin_vals
    k_out_base = k_ptr + b * kv_stride_b + kv_head * kv_stride_h + t * kv_stride_t
    v_out_ptrs = (
        v_ptr
        + b * kv_stride_b
        + kv_head * kv_stride_h
        + t * kv_stride_t
        + cols * kv_stride_d
    )
    tl.store(
        k_out_base + half_cols * kv_stride_d,
        rot_first.to(tl.bfloat16),
        mask=half_mask,
    )
    tl.store(
        k_out_base + (half + half_cols) * kv_stride_d,
        rot_second.to(tl.bfloat16),
        mask=half_mask,
    )
    tl.store(v_out_ptrs, v_vals.to(tl.bfloat16), mask=mask)


@triton.jit
def attention_prep_backward_kernel(
    qkv_ptr,
    q_gain_ptr,
    cos_ptr,
    sin_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_v_ptr,
    grad_qkv_ptr,
    grad_q_gain_ptr,
    batch_size,
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    q_dim,
    kv_dim,
    qkv_stride_b,
    qkv_stride_t,
    grad_q_stride_b,
    grad_q_stride_h,
    grad_q_stride_t,
    grad_q_stride_d,
    grad_k_stride_b,
    grad_k_stride_h,
    grad_k_stride_t,
    grad_k_stride_d,
    grad_qkv_stride_b,
    grad_qkv_stride_t,
    BLOCK_D: tl.constexpr,
):
    """Fuse backward through q/k RMSNorm, RoPE, q_gain, and v copy."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    b = token_idx // seq_len
    t = token_idx % seq_len
    cols = tl.arange(0, BLOCK_D)
    mask = cols < head_dim
    half = head_dim // 2
    half_cols = tl.arange(0, BLOCK_D // 2)
    half_mask = half_cols < half

    qkv_base = qkv_ptr + b * qkv_stride_b + t * qkv_stride_t
    grad_qkv_base = grad_qkv_ptr + b * grad_qkv_stride_b + t * grad_qkv_stride_t
    cos_ptrs = cos_ptr + t * half + half_cols
    sin_ptrs = sin_ptr + t * half + half_cols
    cos_vals = tl.load(cos_ptrs, mask=half_mask, other=0.0).to(tl.float32)
    sin_vals = tl.load(sin_ptrs, mask=half_mask, other=0.0).to(tl.float32)

    if head_idx < num_heads:
        q_head_base = qkv_base + head_idx * head_dim
        q1 = tl.load(q_head_base + half_cols, mask=half_mask, other=0.0).to(tl.float32)
        q2 = tl.load(q_head_base + half + half_cols, mask=half_mask, other=0.0).to(
            tl.float32
        )
        rms = tl.sqrt(
            (tl.sum(q1 * q1, axis=0) + tl.sum(q2 * q2, axis=0)) / head_dim + 1e-5
        )
        q1_norm = q1 / rms
        q2_norm = q2 / rms
        rot_first = q1_norm * cos_vals + q2_norm * sin_vals
        rot_second = q2_norm * cos_vals - q1_norm * sin_vals
        gain = tl.load(q_gain_ptr + head_idx).to(tl.float32)

        grad_q_base = (
            grad_q_ptr
            + b * grad_q_stride_b
            + head_idx * grad_q_stride_h
            + t * grad_q_stride_t
        )
        grad_q1 = tl.load(
            grad_q_base + half_cols * grad_q_stride_d,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        grad_q2 = tl.load(
            grad_q_base + (half + half_cols) * grad_q_stride_d,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        grad_rot_first = grad_q1 * gain
        grad_rot_second = grad_q2 * gain
        grad_q1_norm = grad_rot_first * cos_vals - grad_rot_second * sin_vals
        grad_q2_norm = grad_rot_first * sin_vals + grad_rot_second * cos_vals
        dot = (
            tl.sum(grad_q1_norm * q1_norm, axis=0)
            + tl.sum(grad_q2_norm * q2_norm, axis=0)
        ) / head_dim
        grad_q1_orig = (grad_q1_norm - q1_norm * dot) / rms
        grad_q2_orig = (grad_q2_norm - q2_norm * dot) / rms
        grad_qkv_q_base = grad_qkv_base + head_idx * head_dim
        tl.store(
            grad_qkv_q_base + half_cols,
            grad_q1_orig.to(tl.bfloat16),
            mask=half_mask,
        )
        tl.store(
            grad_qkv_q_base + half + half_cols,
            grad_q2_orig.to(tl.bfloat16),
            mask=half_mask,
        )
        grad_gain = tl.sum(grad_q1 * rot_first + grad_q2 * rot_second, axis=0)
        tl.atomic_add(grad_q_gain_ptr + head_idx, grad_gain)
        return

    kv_head = head_idx - num_heads
    if kv_head >= num_kv_heads:
        return

    k_head_base = qkv_base + q_dim + kv_head * head_dim
    k1 = tl.load(k_head_base + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_head_base + half + half_cols, mask=half_mask, other=0.0).to(
        tl.float32
    )
    rms = tl.sqrt((tl.sum(k1 * k1, axis=0) + tl.sum(k2 * k2, axis=0)) / head_dim + 1e-5)
    k1_norm = k1 / rms
    k2_norm = k2 / rms

    grad_k_base = (
        grad_k_ptr
        + b * grad_k_stride_b
        + kv_head * grad_k_stride_h
        + t * grad_k_stride_t
    )
    grad_k1 = tl.load(
        grad_k_base + half_cols * grad_k_stride_d,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    grad_k2 = tl.load(
        grad_k_base + (half + half_cols) * grad_k_stride_d,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    grad_k1_norm = grad_k1 * cos_vals - grad_k2 * sin_vals
    grad_k2_norm = grad_k1 * sin_vals + grad_k2 * cos_vals
    dot = (
        tl.sum(grad_k1_norm * k1_norm, axis=0) + tl.sum(grad_k2_norm * k2_norm, axis=0)
    ) / head_dim
    grad_k1_orig = (grad_k1_norm - k1_norm * dot) / rms
    grad_k2_orig = (grad_k2_norm - k2_norm * dot) / rms
    grad_qkv_k_base = grad_qkv_base + q_dim + kv_head * head_dim
    tl.store(
        grad_qkv_k_base + half_cols,
        grad_k1_orig.to(tl.bfloat16),
        mask=half_mask,
    )
    tl.store(
        grad_qkv_k_base + half + half_cols,
        grad_k2_orig.to(tl.bfloat16),
        mask=half_mask,
    )

    grad_v_ptrs = (
        grad_v_ptr
        + b * grad_k_stride_b
        + kv_head * grad_k_stride_h
        + t * grad_k_stride_t
        + cols * grad_k_stride_d
    )
    grad_v_vals = tl.load(grad_v_ptrs, mask=mask, other=0.0)
    grad_qkv_v_ptrs = grad_qkv_base + q_dim + kv_dim + kv_head * head_dim + cols
    tl.store(grad_qkv_v_ptrs, grad_v_vals.to(tl.bfloat16), mask=mask)


def attention_prep_triton(
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    bshd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused Triton attention-prep kernel."""
    batch_size, seq_len, _ = qkv.shape
    if bshd:
        q = torch.empty(
            (batch_size, seq_len, num_heads, head_dim),
            device=qkv.device,
            dtype=qkv.dtype,
        )
        k = torch.empty(
            (batch_size, seq_len, num_kv_heads, head_dim),
            device=qkv.device,
            dtype=qkv.dtype,
        )
        q_stride_h = q.stride(2)
        q_stride_t = q.stride(1)
        kv_stride_h = k.stride(2)
        kv_stride_t = k.stride(1)
    else:
        q = torch.empty(
            (batch_size, num_heads, seq_len, head_dim),
            device=qkv.device,
            dtype=qkv.dtype,
        )
        k = torch.empty(
            (batch_size, num_kv_heads, seq_len, head_dim),
            device=qkv.device,
            dtype=qkv.dtype,
        )
        q_stride_h = q.stride(1)
        q_stride_t = q.stride(2)
        kv_stride_h = k.stride(1)
        kv_stride_t = k.stride(2)
    v = torch.empty_like(k)
    block_d = next_power_of_2(head_dim)
    grid = (batch_size * seq_len, num_heads + num_kv_heads)
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    attention_prep_kernel[grid](
        qkv,
        q_gain,
        cos,
        sin,
        q,
        k,
        v,
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        q_dim,
        kv_dim,
        qkv.stride(0),
        qkv.stride(1),
        q.stride(0),
        q_stride_h,
        q_stride_t,
        q.stride(3),
        k.stride(0),
        kv_stride_h,
        kv_stride_t,
        k.stride(3),
        BLOCK_D=block_d,
        num_warps=2 if head_dim <= 64 else 4,
    )
    return q, k, v


def attention_prep_backward_triton(
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused Triton backward kernel for the attention-prep boundary."""
    batch_size, seq_len, qkv_dim = qkv.shape
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    if qkv_dim != q_dim + 2 * kv_dim:
        raise ValueError("qkv shape does not match the requested head layout")
    grad_qkv = torch.empty_like(qkv)
    grad_q_gain = torch.zeros_like(q_gain, dtype=torch.float32)
    block_d = next_power_of_2(head_dim)
    grid = (batch_size * seq_len, num_heads + num_kv_heads)
    attention_prep_backward_kernel[grid](
        qkv,
        q_gain,
        cos,
        sin,
        grad_q,
        grad_k,
        grad_v,
        grad_qkv,
        grad_q_gain,
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        q_dim,
        kv_dim,
        qkv.stride(0),
        qkv.stride(1),
        grad_q.stride(0),
        grad_q.stride(1),
        grad_q.stride(2),
        grad_q.stride(3),
        grad_k.stride(0),
        grad_k.stride(1),
        grad_k.stride(2),
        grad_k.stride(3),
        grad_qkv.stride(0),
        grad_qkv.stride(1),
        BLOCK_D=block_d,
        num_warps=2 if head_dim <= 64 else 4,
    )
    return grad_qkv, grad_q_gain


def register_attention_prep_custom_op() -> Any:
    """Register a benchmark-local custom op for the attention-prep boundary."""

    @torch.library.custom_op(
        "allama_triton_bench::attention_prep_backward",
        mutates_args=(),
    )
    def attention_prep_backward_op(
        qkv: torch.Tensor,
        q_gain: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        grad_q: torch.Tensor,
        grad_k: torch.Tensor,
        grad_v: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return attention_prep_backward_triton(
            qkv,
            q_gain,
            cos,
            sin,
            grad_q,
            grad_k,
            grad_v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

    @attention_prep_backward_op.register_fake
    def _attention_prep_backward_op_fake(
        qkv: torch.Tensor,
        q_gain: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        grad_q: torch.Tensor,
        grad_k: torch.Tensor,
        grad_v: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cos, sin, grad_q, grad_k, grad_v, num_heads, num_kv_heads, head_dim
        return qkv.new_empty(qkv.shape), q_gain.new_empty(
            q_gain.shape, dtype=torch.float32
        )

    @torch.library.custom_op(
        "allama_triton_bench::attention_prep",
        mutates_args=(),
    )
    def attention_prep_op(
        qkv: torch.Tensor,
        q_gain: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bshd: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return attention_prep_triton(
            qkv,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bshd=bool(bshd),
        )

    @attention_prep_op.register_fake
    def _attention_prep_op_fake(
        qkv: torch.Tensor,
        q_gain: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bshd: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del q_gain, cos, sin
        batch_size, seq_len, _ = qkv.shape
        if bool(bshd):
            q = qkv.new_empty((batch_size, seq_len, num_heads, head_dim))
            k = qkv.new_empty((batch_size, seq_len, num_kv_heads, head_dim))
            v = qkv.new_empty((batch_size, seq_len, num_kv_heads, head_dim))
        else:
            q = qkv.new_empty((batch_size, num_heads, seq_len, head_dim))
            k = qkv.new_empty((batch_size, num_kv_heads, seq_len, head_dim))
            v = qkv.new_empty((batch_size, num_kv_heads, seq_len, head_dim))
        return q, k, v

    def setup_context(
        ctx: Any,
        inputs: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int
        ],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        del output
        qkv, q_gain, cos, sin, num_heads, num_kv_heads, head_dim, bshd = inputs
        ctx.save_for_backward(qkv, q_gain, cos, sin)
        ctx.num_heads = int(num_heads)
        ctx.num_kv_heads = int(num_kv_heads)
        ctx.head_dim = int(head_dim)
        ctx.bshd = int(bshd)

    def backward(
        ctx: Any,
        grad_q: torch.Tensor,
        grad_k: torch.Tensor,
        grad_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None, None]:
        qkv, q_gain, cos, sin = ctx.saved_tensors
        grad_qkv, grad_q_gain = attention_prep_backward_op(
            qkv,
            q_gain,
            cos,
            sin,
            grad_q,
            grad_k,
            grad_v,
            ctx.num_heads,
            ctx.num_kv_heads,
            ctx.head_dim,
        )
        return grad_qkv, grad_q_gain, None, None, None, None, None, None

    torch.library.register_autograd(
        "allama_triton_bench::attention_prep",
        backward,
        setup_context=setup_context,
    )
    return attention_prep_op


def attention_forward_triton(
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    backend: str = "sdpa",
) -> torch.Tensor:
    """Run the fused Triton prep kernel followed by flash-attention forward."""
    use_fa2 = backend == "fa2"
    q, k, v = attention_prep_triton(
        qkv,
        q_gain,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bshd=use_fa2,
    )
    if use_fa2:
        if flash_attn_func is None:
            raise RuntimeError("backend='fa2' requires flash-attn to be installed")
        return flash_attn_func(q, k, v, causal=True)
    if num_heads != num_kv_heads and sdpa_enable_gqa_available():
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=True,
        )
    if num_heads != num_kv_heads:
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def attention_forward_custom(
    op: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    backend: str = "sdpa",
) -> torch.Tensor:
    """Run the custom-op prep path followed by flash attention."""
    use_fa2 = backend == "fa2"
    q, k, v = op(
        qkv,
        q_gain,
        cos,
        sin,
        num_heads,
        num_kv_heads,
        head_dim,
        1 if use_fa2 else 0,
    )
    if use_fa2:
        if flash_attn_func is None:
            raise RuntimeError("backend='fa2' requires flash-attn to be installed")
        return flash_attn_func(q, k, v, causal=True)
    if num_heads != num_kv_heads and sdpa_enable_gqa_available():
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=True,
        )
    if num_heads != num_kv_heads:
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def measure_ms(fn: Callable[[], Any], warmup_iters: int, measured_iters: int) -> float:
    """Measure mean CUDA runtime for a callable that launches work."""
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measured_iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / measured_iters


def max_diff(ref: torch.Tensor, out: torch.Tensor) -> tuple[float, float]:
    """Return max absolute and relative error for two tensors."""
    diff = (ref.float() - out.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff.max() / ref.float().abs().max().clamp_min(1e-6)).item()
    return float(max_abs), float(max_rel)


def loss_reference_prep(
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    bshd: bool = False,
) -> torch.Tensor:
    """Return a scalar loss that exercises prep-only backward."""
    q, k, v = attention_prep_reference(
        qkv,
        q_gain,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bshd=bshd,
    )
    return (q * grad_q).sum() + (k * grad_k).sum() + (v * grad_v).sum()


def loss_custom_prep(
    op: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    bshd: bool = False,
) -> torch.Tensor:
    """Return a scalar loss that exercises custom prep backward."""
    q, k, v = op(
        qkv,
        q_gain,
        cos,
        sin,
        num_heads,
        num_kv_heads,
        head_dim,
        1 if bshd else 0,
    )
    return (q * grad_q).sum() + (k * grad_k).sum() + (v * grad_v).sum()


def loss_reference_flash(
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    backend: str = "sdpa",
) -> torch.Tensor:
    """Return a scalar loss that exercises flash-attention through prep."""
    return (
        attention_forward_reference(
            qkv,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            backend=backend,
        )
        * grad_out
    ).sum()


def loss_custom_flash(
    op: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    qkv: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    backend: str = "sdpa",
) -> torch.Tensor:
    """Return a scalar loss that exercises custom prep plus flash backward."""
    return (
        attention_forward_custom(
            op,
            qkv,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            backend=backend,
        )
        * grad_out
    ).sum()


def main() -> None:
    """Run the standalone attention-prep benchmark."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the attention-prep benchmark")

    device = torch.device("cuda")
    torch.manual_seed(1337)
    op = register_attention_prep_custom_op()
    batch = 4
    seq_len = 1024
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    use_fa2 = args.backend == "fa2"

    qkv = torch.randn(batch, seq_len, qkv_dim, device=device, dtype=torch.bfloat16)
    q_gain = torch.randn(num_heads, device=device, dtype=torch.float32)
    cos, sin = rotary_cache(seq_len, head_dim, device=device, dtype=torch.bfloat16)

    ref_q, ref_k, ref_v = attention_prep_reference(
        qkv,
        q_gain,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bshd=use_fa2,
    )
    tri_q, tri_k, tri_v = attention_prep_triton(
        qkv,
        q_gain,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bshd=use_fa2,
    )
    prep_abs = 0.0
    prep_rel = 0.0
    for ref_tensor, tri_tensor in ((ref_q, tri_q), (ref_k, tri_k), (ref_v, tri_v)):
        max_abs, max_rel = max_diff(ref_tensor, tri_tensor)
        prep_abs = max(prep_abs, max_abs)
        prep_rel = max(prep_rel, max_rel)

    ref_prep_ms = measure_ms(
        lambda: attention_prep_reference(
            qkv,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bshd=use_fa2,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_ref_prep = torch.compile(
        attention_prep_reference,
        dynamic=False,
        fullgraph=True,
    )
    compiled_ref_prep_ms = measure_ms(
        lambda: compiled_ref_prep(
            qkv,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bshd=use_fa2,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    triton_prep_ms = measure_ms(
        lambda: attention_prep_triton(
            qkv,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bshd=use_fa2,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )

    ref_attn = attention_forward_reference(
        qkv,
        q_gain,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        backend=args.backend,
    )
    tri_attn = attention_forward_triton(
        qkv,
        q_gain,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        backend=args.backend,
    )
    attn_abs, attn_rel = max_diff(ref_attn, tri_attn)
    ref_attn_ms = measure_ms(
        lambda: attention_forward_reference(
            qkv,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            backend=args.backend,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_ref_attn = torch.compile(
        attention_forward_reference,
        dynamic=False,
        fullgraph=True,
    )
    compiled_ref_attn_ms = measure_ms(
        lambda: compiled_ref_attn(
            qkv,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            backend=args.backend,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    triton_attn_ms = measure_ms(
        lambda: attention_forward_triton(
            qkv,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            backend=args.backend,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )

    grad_q = torch.randn_like(ref_q)
    grad_k = torch.randn_like(ref_k)
    grad_v = torch.randn_like(ref_v)
    qkv_ref = qkv.detach().clone().requires_grad_(True)
    q_gain_ref = q_gain.detach().clone().requires_grad_(True)
    qkv_custom = qkv.detach().clone().requires_grad_(True)
    q_gain_custom = q_gain.detach().clone().requires_grad_(True)

    ref_prep_grads = torch.autograd.grad(
        loss_reference_prep(
            qkv_ref,
            q_gain_ref,
            cos,
            sin,
            grad_q,
            grad_k,
            grad_v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bshd=use_fa2,
        ),
        (qkv_ref, q_gain_ref),
    )
    custom_prep_grads = torch.autograd.grad(
        loss_custom_prep(
            op,
            qkv_custom,
            q_gain_custom,
            cos,
            sin,
            grad_q,
            grad_k,
            grad_v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bshd=use_fa2,
        ),
        (qkv_custom, q_gain_custom),
    )
    prep_backward_abs = max(
        (ref_grad.float() - custom_grad.float()).abs().max().item()
        for ref_grad, custom_grad in zip(ref_prep_grads, custom_prep_grads, strict=True)
    )
    prep_backward_rel = max(
        (
            (ref_grad.float() - custom_grad.float()).abs().max()
            / ref_grad.float().abs().max().clamp_min(1e-6)
        ).item()
        for ref_grad, custom_grad in zip(ref_prep_grads, custom_prep_grads, strict=True)
    )

    ref_prep_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            loss_reference_prep(
                qkv_ref,
                q_gain_ref,
                cos,
                sin,
                grad_q,
                grad_k,
                grad_v,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                bshd=use_fa2,
            ),
            (qkv_ref, q_gain_ref),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_ref_prep_loss = torch.compile(
        loss_reference_prep, dynamic=False, fullgraph=True
    )
    compiled_ref_prep_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_ref_prep_loss(
                qkv_ref,
                q_gain_ref,
                cos,
                sin,
                grad_q,
                grad_k,
                grad_v,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                bshd=use_fa2,
            ),
            (qkv_ref, q_gain_ref),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom_prep_loss = torch.compile(
        lambda qkv_, q_gain_, cos_, sin_, grad_q_, grad_k_, grad_v_: loss_custom_prep(
            op,
            qkv_,
            q_gain_,
            cos_,
            sin_,
            grad_q_,
            grad_k_,
            grad_v_,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bshd=use_fa2,
        ),
        dynamic=False,
        fullgraph=True,
    )
    custom_prep_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            loss_custom_prep(
                op,
                qkv_custom,
                q_gain_custom,
                cos,
                sin,
                grad_q,
                grad_k,
                grad_v,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                bshd=use_fa2,
            ),
            (qkv_custom, q_gain_custom),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom_prep_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_custom_prep_loss(
                qkv_custom,
                q_gain_custom,
                cos,
                sin,
                grad_q,
                grad_k,
                grad_v,
            ),
            (qkv_custom, q_gain_custom),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )

    grad_out = torch.randn_like(ref_attn)
    qkv_ref_flash = qkv.detach().clone().requires_grad_(True)
    q_gain_ref_flash = q_gain.detach().clone().requires_grad_(True)
    qkv_custom_flash = qkv.detach().clone().requires_grad_(True)
    q_gain_custom_flash = q_gain.detach().clone().requires_grad_(True)
    ref_flash_grads = torch.autograd.grad(
        loss_reference_flash(
            qkv_ref_flash,
            q_gain_ref_flash,
            cos,
            sin,
            grad_out,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            backend=args.backend,
        ),
        (qkv_ref_flash, q_gain_ref_flash),
    )
    custom_flash_grads = torch.autograd.grad(
        loss_custom_flash(
            op,
            qkv_custom_flash,
            q_gain_custom_flash,
            cos,
            sin,
            grad_out,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            backend=args.backend,
        ),
        (qkv_custom_flash, q_gain_custom_flash),
    )
    flash_backward_abs = max(
        (ref_grad.float() - custom_grad.float()).abs().max().item()
        for ref_grad, custom_grad in zip(
            ref_flash_grads, custom_flash_grads, strict=True
        )
    )
    flash_backward_rel = max(
        (
            (ref_grad.float() - custom_grad.float()).abs().max()
            / ref_grad.float().abs().max().clamp_min(1e-6)
        ).item()
        for ref_grad, custom_grad in zip(
            ref_flash_grads, custom_flash_grads, strict=True
        )
    )
    ref_flash_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            loss_reference_flash(
                qkv_ref_flash,
                q_gain_ref_flash,
                cos,
                sin,
                grad_out,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                backend=args.backend,
            ),
            (qkv_ref_flash, q_gain_ref_flash),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_ref_flash = torch.compile(
        loss_reference_flash, dynamic=False, fullgraph=True
    )
    compiled_ref_flash_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_ref_flash(
                qkv_ref_flash,
                q_gain_ref_flash,
                cos,
                sin,
                grad_out,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                backend=args.backend,
            ),
            (qkv_ref_flash, q_gain_ref_flash),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom_flash = torch.compile(
        lambda qkv_, q_gain_, cos_, sin_, grad_out_: loss_custom_flash(
            op,
            qkv_,
            q_gain_,
            cos_,
            sin_,
            grad_out_,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            backend=args.backend,
        ),
        dynamic=False,
        fullgraph=True,
    )
    custom_flash_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            loss_custom_flash(
                op,
                qkv_custom_flash,
                q_gain_custom_flash,
                cos,
                sin,
                grad_out,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                backend=args.backend,
            ),
            (qkv_custom_flash, q_gain_custom_flash),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom_flash_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_custom_flash(
                qkv_custom_flash,
                q_gain_custom_flash,
                cos,
                sin,
                grad_out,
            ),
            (qkv_custom_flash, q_gain_custom_flash),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )

    summary = {
        "shape": [batch, seq_len, qkv_dim],
        "backend": args.backend,
        "heads": {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "warmup_iters": int(args.warmup_iters),
        "measured_iters": int(args.measured_iters),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "cases": [
            {
                "case": "attention_prep",
                "reference_ms": float(ref_prep_ms),
                "compiled_reference_ms": float(compiled_ref_prep_ms),
                "triton_ms": float(triton_prep_ms),
                "speedup": float(ref_prep_ms / triton_prep_ms),
                "speedup_vs_compiled": float(compiled_ref_prep_ms / triton_prep_ms),
                "max_abs": prep_abs,
                "max_rel": prep_rel,
            },
            {
                "case": "attention_prep_backward",
                "reference_ms": float(ref_prep_backward_ms),
                "compiled_reference_ms": float(compiled_ref_prep_backward_ms),
                "triton_ms": float(custom_prep_backward_ms),
                "compiled_triton_ms": float(compiled_custom_prep_backward_ms),
                "speedup": float(ref_prep_backward_ms / custom_prep_backward_ms),
                "speedup_vs_compiled": float(
                    compiled_ref_prep_backward_ms / compiled_custom_prep_backward_ms
                ),
                "max_abs": prep_backward_abs,
                "max_rel": prep_backward_rel,
            },
            {
                "case": "attention_prep_plus_flash_fwd",
                "reference_ms": float(ref_attn_ms),
                "compiled_reference_ms": float(compiled_ref_attn_ms),
                "triton_ms": float(triton_attn_ms),
                "speedup": float(ref_attn_ms / triton_attn_ms),
                "speedup_vs_compiled": float(compiled_ref_attn_ms / triton_attn_ms),
                "max_abs": attn_abs,
                "max_rel": attn_rel,
            },
            {
                "case": "attention_prep_plus_flash_bwd",
                "reference_ms": float(ref_flash_backward_ms),
                "compiled_reference_ms": float(compiled_ref_flash_backward_ms),
                "triton_ms": float(custom_flash_backward_ms),
                "compiled_triton_ms": float(compiled_custom_flash_backward_ms),
                "speedup": float(ref_flash_backward_ms / custom_flash_backward_ms),
                "speedup_vs_compiled": float(
                    compiled_ref_flash_backward_ms / compiled_custom_flash_backward_ms
                ),
                "max_abs": flash_backward_abs,
                "max_rel": flash_backward_rel,
            },
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
