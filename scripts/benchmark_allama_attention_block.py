#!/usr/bin/env python3
"""Benchmark a larger ALlama attention output boundary with a Triton kernel.

This targets the boundary after flash attention:

- flash-attention output in head-major layout ``[B, H, T, Dh]``
- Triton fused output projection + residual add/scale

The kernel avoids materializing the transposed ``[B, T, D]`` attention output
before the projection matmul. Backward now uses Triton kernels for the
post-flash boundary as well:

- fused branch recompute + reduction for ``grad_scale``
- direct ``grad_attn_y`` writeback in head-major layout
- direct ``grad_proj_weight_t`` without flattening the attention input
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "1")

import torch
import triton
import triton.language as tl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the attention-block benchmark.

    :return argparse.Namespace: Parsed benchmark configuration.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs_allama_validation/attention_block_v1"),
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
        "--layout",
        choices=("head_major", "bshd"),
        default="bshd",
        help="Attention-output layout to benchmark.",
    )
    return parser.parse_args()


def measure_ms(fn: Callable[[], Any], warmup_iters: int, measured_iters: int) -> float:
    """Measure mean CUDA runtime for a callable.

    :param Callable[[], Any] fn: Callable under test.
    :param int warmup_iters: Warmup iterations.
    :param int measured_iters: Measured iterations.
    :return float: Mean milliseconds per call.
    """
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measured_iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / measured_iters


def attn_layout_meta(attn_y: torch.Tensor, *, bshd: bool) -> tuple[int, int, int, int]:
    """Return ``(batch, num_heads, seq_len, head_dim)`` for the active layout."""
    if attn_y.dim() != 4:
        raise ValueError("attn_y must be rank-4")
    if bshd:
        batch, seq_len, num_heads, head_dim = attn_y.shape
    else:
        batch, num_heads, seq_len, head_dim = attn_y.shape
    return batch, num_heads, seq_len, head_dim


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def attn_outproj_residual_kernel(
    attn_y_ptr,
    proj_weight_t_ptr,
    residual_x_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    seq_len,
    head_dim,
    attn_stride_b,
    attn_stride_h,
    attn_stride_t,
    attn_stride_d,
    proj_stride_k,
    proj_stride_n,
    residual_stride_m,
    residual_stride_n,
    out_stride_m,
    out_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fuse head-major attention output projection and residual add."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    b_idx = offs_m // seq_len
    t_idx = offs_m % seq_len

    for k_base in range(0, K, BLOCK_K):
        k_idx = k_base + offs_k
        mask_k = k_idx < K
        h_idx = k_idx // head_dim
        d_idx = k_idx % head_dim

        attn_ptrs = (
            attn_y_ptr
            + b_idx[:, None] * attn_stride_b
            + h_idx[None, :] * attn_stride_h
            + t_idx[:, None] * attn_stride_t
            + d_idx[None, :] * attn_stride_d
        )
        attn_vals = tl.load(
            attn_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )
        weight_ptrs = (
            proj_weight_t_ptr
            + k_idx[:, None] * proj_stride_k
            + offs_n[None, :] * proj_stride_n
        )
        weight = tl.load(weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(attn_vals, weight)

    residual_ptrs = (
        residual_x_ptr
        + offs_m[:, None] * residual_stride_m
        + offs_n[None, :] * residual_stride_n
    )
    residual = tl.load(
        residual_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
    ).to(tl.float32)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    out = residual + acc * scale[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
    reset_to_zero=["grad_scale_ptr"],
)
@triton.jit
def attn_grad_scale_kernel(
    attn_y_ptr,
    proj_weight_t_ptr,
    grad_out_ptr,
    grad_scale_ptr,
    M,
    N,
    K,
    seq_len,
    head_dim,
    attn_stride_b,
    attn_stride_h,
    attn_stride_t,
    attn_stride_d,
    proj_stride_k,
    proj_stride_n,
    grad_out_stride_m,
    grad_out_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fuse branch recompute with the reduction for ``grad_scale``."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    b_idx = offs_m // seq_len
    t_idx = offs_m % seq_len

    for k_base in range(0, K, BLOCK_K):
        k_idx = k_base + offs_k
        mask_k = k_idx < K
        h_idx = k_idx // head_dim
        d_idx = k_idx % head_dim

        attn_ptrs = (
            attn_y_ptr
            + b_idx[:, None] * attn_stride_b
            + h_idx[None, :] * attn_stride_h
            + t_idx[:, None] * attn_stride_t
            + d_idx[None, :] * attn_stride_d
        )
        attn_vals = tl.load(
            attn_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )
        weight_ptrs = (
            proj_weight_t_ptr
            + k_idx[:, None] * proj_stride_k
            + offs_n[None, :] * proj_stride_n
        )
        weight = tl.load(weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(attn_vals, weight)

    grad_out_ptrs = (
        grad_out_ptr
        + offs_m[:, None] * grad_out_stride_m
        + offs_n[None, :] * grad_out_stride_n
    )
    grad_out = tl.load(
        grad_out_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
    ).to(tl.float32)
    partial = tl.sum(acc * grad_out, axis=0)
    tl.atomic_add(grad_scale_ptr + offs_n, partial, mask=mask_n)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def attn_grad_attn_y_kernel(
    grad_branch_ptr,
    proj_weight_t_ptr,
    grad_attn_y_ptr,
    M,
    N,
    K,
    seq_len,
    head_dim,
    grad_branch_stride_m,
    grad_branch_stride_n,
    proj_stride_k,
    proj_stride_n,
    grad_attn_stride_b,
    grad_attn_stride_h,
    grad_attn_stride_t,
    grad_attn_stride_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute ``grad_attn_y`` directly into head-major layout."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_k = offs_k < K
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n_base in range(0, N, BLOCK_N):
        n_idx = n_base + offs_n
        mask_n = n_idx < N
        grad_branch_ptrs = (
            grad_branch_ptr
            + offs_m[:, None] * grad_branch_stride_m
            + n_idx[None, :] * grad_branch_stride_n
        )
        grad_branch_vals = tl.load(
            grad_branch_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
        )
        weight_ptrs = (
            proj_weight_t_ptr
            + offs_k[:, None] * proj_stride_k
            + n_idx[None, :] * proj_stride_n
        )
        weight = tl.load(weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(grad_branch_vals, tl.trans(weight))

    b_idx = offs_m // seq_len
    t_idx = offs_m % seq_len
    h_idx = offs_k // head_dim
    d_idx = offs_k % head_dim
    out_ptrs = (
        grad_attn_y_ptr
        + b_idx[:, None] * grad_attn_stride_b
        + h_idx[None, :] * grad_attn_stride_h
        + t_idx[:, None] * grad_attn_stride_t
        + d_idx[None, :] * grad_attn_stride_d
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_k[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def attn_grad_proj_weight_t_kernel(
    attn_y_ptr,
    grad_branch_ptr,
    grad_proj_weight_t_ptr,
    M,
    N,
    K,
    seq_len,
    head_dim,
    attn_stride_b,
    attn_stride_h,
    attn_stride_t,
    attn_stride_d,
    grad_branch_stride_m,
    grad_branch_stride_n,
    grad_proj_stride_k,
    grad_proj_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute ``grad_proj_weight_t`` without flattening the head-major input."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    mask_k = offs_k < K
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

    h_idx = offs_k // head_dim
    d_idx = offs_k % head_dim

    for m_base in range(0, M, BLOCK_M):
        m_idx = m_base + offs_m
        mask_m = m_idx < M
        b_idx = m_idx // seq_len
        t_idx = m_idx % seq_len
        attn_ptrs = (
            attn_y_ptr
            + b_idx[None, :] * attn_stride_b
            + h_idx[:, None] * attn_stride_h
            + t_idx[None, :] * attn_stride_t
            + d_idx[:, None] * attn_stride_d
        )
        attn_vals = tl.load(
            attn_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0
        )
        grad_branch_ptrs = (
            grad_branch_ptr
            + m_idx[:, None] * grad_branch_stride_m
            + offs_n[None, :] * grad_branch_stride_n
        )
        grad_branch_vals = tl.load(
            grad_branch_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
        )
        acc += tl.dot(attn_vals, grad_branch_vals)

    out_ptrs = (
        grad_proj_weight_t_ptr
        + offs_k[:, None] * grad_proj_stride_k
        + offs_n[None, :] * grad_proj_stride_n
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_k[:, None] & mask_n[None, :])


def attn_outproj_residual_triton(
    residual_x: torch.Tensor,
    attn_y: torch.Tensor,
    proj_weight_t: torch.Tensor,
    scale: torch.Tensor,
    *,
    bshd: bool = False,
) -> torch.Tensor:
    """Run the fused Triton attention-outproj-residual kernel.

    :param torch.Tensor residual_x: Residual input flattened to ``[B*T, D]``.
    :param torch.Tensor attn_y: Flash-attention output in ``[B, H, T, Dh]`` layout.
    :param torch.Tensor proj_weight_t: Output projection weight transposed to ``[D, D]``.
    :param torch.Tensor scale: Per-channel residual scale vector ``[D]``.
    :return torch.Tensor: Residual-updated output tensor ``[B*T, D]``.
    """
    if residual_x.dtype != torch.bfloat16 or attn_y.dtype != torch.bfloat16:
        raise TypeError("this benchmark currently targets bfloat16 tensors")
    if residual_x.dim() != 2 or attn_y.dim() != 4 or proj_weight_t.dim() != 2:
        raise ValueError(
            "expected residual_x=[M,D], attn_y rank-4, proj_weight_t=[D,D]"
        )
    batch, num_heads, seq_len, head_dim = attn_layout_meta(attn_y, bshd=bshd)
    model_dim = num_heads * head_dim
    if residual_x.shape != (batch * seq_len, model_dim):
        raise ValueError("residual_x shape must match the flattened attention output")
    if proj_weight_t.shape != (model_dim, model_dim):
        raise ValueError("proj_weight_t shape must be [model_dim, model_dim]")
    out = torch.empty_like(residual_x)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(residual_x.size(0), meta["BLOCK_M"]),
            triton.cdiv(residual_x.size(1), meta["BLOCK_N"]),
        )

    attn_outproj_residual_kernel[grid](
        attn_y,
        proj_weight_t,
        residual_x,
        scale,
        out,
        residual_x.size(0),
        residual_x.size(1),
        model_dim,
        seq_len,
        head_dim,
        attn_y.stride(0),
        attn_y.stride(2) if bshd else attn_y.stride(1),
        attn_y.stride(1) if bshd else attn_y.stride(2),
        attn_y.stride(3),
        proj_weight_t.stride(0),
        proj_weight_t.stride(1),
        residual_x.stride(0),
        residual_x.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def attn_grad_scale_triton(
    attn_y: torch.Tensor,
    proj_weight_t: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    bshd: bool = False,
) -> torch.Tensor:
    """Run the fused branch-recompute reduction for ``grad_scale``."""
    batch, num_heads, seq_len, head_dim = attn_layout_meta(attn_y, bshd=bshd)
    model_dim = num_heads * head_dim
    grad_scale = torch.zeros(model_dim, device=grad_out.device, dtype=torch.float32)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(grad_out.size(0), meta["BLOCK_M"]),
            triton.cdiv(grad_out.size(1), meta["BLOCK_N"]),
        )

    attn_grad_scale_kernel[grid](
        attn_y,
        proj_weight_t,
        grad_out,
        grad_scale,
        grad_out.size(0),
        grad_out.size(1),
        model_dim,
        seq_len,
        head_dim,
        attn_y.stride(0),
        attn_y.stride(2) if bshd else attn_y.stride(1),
        attn_y.stride(1) if bshd else attn_y.stride(2),
        attn_y.stride(3),
        proj_weight_t.stride(0),
        proj_weight_t.stride(1),
        grad_out.stride(0),
        grad_out.stride(1),
    )
    return grad_scale


def attn_grad_attn_y_triton(
    grad_branch: torch.Tensor,
    proj_weight_t: torch.Tensor,
    seq_len: int,
    head_dim: int,
    *,
    bshd: bool = False,
) -> torch.Tensor:
    """Run the Triton backward kernel for ``grad_attn_y``."""
    model_dim = proj_weight_t.size(0)
    batch = grad_branch.size(0) // seq_len
    num_heads = model_dim // head_dim
    if bshd:
        grad_attn_y = torch.empty(
            batch,
            seq_len,
            num_heads,
            head_dim,
            device=grad_branch.device,
            dtype=grad_branch.dtype,
        )
    else:
        grad_attn_y = torch.empty(
            batch,
            num_heads,
            seq_len,
            head_dim,
            device=grad_branch.device,
            dtype=grad_branch.dtype,
        )

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(grad_branch.size(0), meta["BLOCK_M"]),
            triton.cdiv(model_dim, meta["BLOCK_K"]),
        )

    attn_grad_attn_y_kernel[grid](
        grad_branch,
        proj_weight_t,
        grad_attn_y,
        grad_branch.size(0),
        grad_branch.size(1),
        model_dim,
        seq_len,
        head_dim,
        grad_branch.stride(0),
        grad_branch.stride(1),
        proj_weight_t.stride(0),
        proj_weight_t.stride(1),
        grad_attn_y.stride(0),
        grad_attn_y.stride(2) if bshd else grad_attn_y.stride(1),
        grad_attn_y.stride(1) if bshd else grad_attn_y.stride(2),
        grad_attn_y.stride(3),
    )
    return grad_attn_y


def attn_grad_proj_weight_t_triton(
    attn_y: torch.Tensor,
    grad_branch: torch.Tensor,
    *,
    bshd: bool = False,
) -> torch.Tensor:
    """Run the Triton backward kernel for ``grad_proj_weight_t``."""
    batch, num_heads, seq_len, head_dim = attn_layout_meta(attn_y, bshd=bshd)
    model_dim = num_heads * head_dim
    grad_proj_weight_t = torch.empty(
        model_dim,
        grad_branch.size(1),
        device=grad_branch.device,
        dtype=grad_branch.dtype,
    )

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(model_dim, meta["BLOCK_K"]),
            triton.cdiv(grad_branch.size(1), meta["BLOCK_N"]),
        )

    attn_grad_proj_weight_t_kernel[grid](
        attn_y,
        grad_branch,
        grad_proj_weight_t,
        grad_branch.size(0),
        grad_branch.size(1),
        model_dim,
        seq_len,
        head_dim,
        attn_y.stride(0),
        attn_y.stride(2) if bshd else attn_y.stride(1),
        attn_y.stride(1) if bshd else attn_y.stride(2),
        attn_y.stride(3),
        grad_branch.stride(0),
        grad_branch.stride(1),
        grad_proj_weight_t.stride(0),
        grad_proj_weight_t.stride(1),
    )
    return grad_proj_weight_t


def attn_outproj_residual_reference(
    residual_x: torch.Tensor,
    attn_y: torch.Tensor,
    proj_weight_t: torch.Tensor,
    scale: torch.Tensor,
    *,
    bshd: bool = False,
) -> torch.Tensor:
    """Reference PyTorch implementation of the post-flash attention boundary."""
    if bshd:
        attn_flat = attn_y.contiguous().view(residual_x.size(0), -1)
    else:
        attn_flat = attn_y.permute(0, 2, 1, 3).contiguous().view(residual_x.size(0), -1)
    branch = attn_flat @ proj_weight_t
    return residual_x + branch * scale.to(dtype=residual_x.dtype)[None, :]


def register_attention_block_custom_op() -> Any:
    """Register a benchmark-local custom op for the attention output boundary."""

    @torch.library.custom_op(
        "allama_triton_bench::attn_grad_scale",
        mutates_args=(),
    )
    def attn_grad_scale_op(
        attn_y: torch.Tensor,
        proj_weight_t: torch.Tensor,
        grad_out: torch.Tensor,
        bshd: int,
    ) -> torch.Tensor:
        return attn_grad_scale_triton(attn_y, proj_weight_t, grad_out, bshd=bool(bshd))

    @attn_grad_scale_op.register_fake
    def _attn_grad_scale_op_fake(
        attn_y: torch.Tensor,
        proj_weight_t: torch.Tensor,
        grad_out: torch.Tensor,
        bshd: int,
    ) -> torch.Tensor:
        del attn_y, proj_weight_t, bshd
        return grad_out.new_empty((grad_out.size(1),), dtype=torch.float32)

    @torch.library.custom_op(
        "allama_triton_bench::attn_grad_attn_y",
        mutates_args=(),
    )
    def attn_grad_attn_y_op(
        grad_branch: torch.Tensor,
        proj_weight_t: torch.Tensor,
        seq_len: int,
        head_dim: int,
        bshd: int,
    ) -> torch.Tensor:
        return attn_grad_attn_y_triton(
            grad_branch,
            proj_weight_t,
            seq_len,
            head_dim,
            bshd=bool(bshd),
        )

    @attn_grad_attn_y_op.register_fake
    def _attn_grad_attn_y_op_fake(
        grad_branch: torch.Tensor,
        proj_weight_t: torch.Tensor,
        seq_len: int,
        head_dim: int,
        bshd: int,
    ) -> torch.Tensor:
        model_dim = proj_weight_t.size(0)
        batch = grad_branch.size(0) // seq_len
        num_heads = model_dim // head_dim
        if bool(bshd):
            return grad_branch.new_empty((batch, seq_len, num_heads, head_dim))
        return grad_branch.new_empty((batch, num_heads, seq_len, head_dim))

    @torch.library.custom_op(
        "allama_triton_bench::attn_grad_proj_weight_t",
        mutates_args=(),
    )
    def attn_grad_proj_weight_t_op(
        attn_y: torch.Tensor,
        grad_branch: torch.Tensor,
        bshd: int,
    ) -> torch.Tensor:
        return attn_grad_proj_weight_t_triton(attn_y, grad_branch, bshd=bool(bshd))

    @attn_grad_proj_weight_t_op.register_fake
    def _attn_grad_proj_weight_t_op_fake(
        attn_y: torch.Tensor,
        grad_branch: torch.Tensor,
        bshd: int,
    ) -> torch.Tensor:
        del bshd
        model_dim = attn_y.size(-2) * attn_y.size(-1)
        return grad_branch.new_empty((model_dim, grad_branch.size(1)))

    @torch.library.custom_op(
        "allama_triton_bench::attn_outproj_residual",
        mutates_args=(),
    )
    def attn_outproj_residual_op(
        residual_x: torch.Tensor,
        attn_y: torch.Tensor,
        proj_weight_t: torch.Tensor,
        scale: torch.Tensor,
        bshd: int,
    ) -> torch.Tensor:
        return attn_outproj_residual_triton(
            residual_x, attn_y, proj_weight_t, scale, bshd=bool(bshd)
        )

    @attn_outproj_residual_op.register_fake
    def _attn_outproj_residual_op_fake(
        residual_x: torch.Tensor,
        attn_y: torch.Tensor,
        proj_weight_t: torch.Tensor,
        scale: torch.Tensor,
        bshd: int,
    ) -> torch.Tensor:
        del attn_y, proj_weight_t, scale, bshd
        return residual_x.new_empty(residual_x.shape)

    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
        output: torch.Tensor,
    ) -> None:
        residual_x, attn_y, proj_weight_t, scale, bshd = inputs
        del residual_x, output
        ctx.save_for_backward(attn_y, proj_weight_t, scale)
        ctx.bshd = int(bshd)

    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        attn_y, proj_weight_t, scale = ctx.saved_tensors
        scale_cast = scale.to(dtype=grad_out.dtype)[None, :]
        grad_x = grad_out
        grad_scale = attn_grad_scale_op(attn_y, proj_weight_t, grad_out, ctx.bshd).to(
            dtype=scale.dtype
        )
        grad_branch = grad_out * scale_cast
        grad_proj_weight_t = attn_grad_proj_weight_t_op(attn_y, grad_branch, ctx.bshd)
        grad_attn_y = attn_grad_attn_y_op(
            grad_branch,
            proj_weight_t,
            attn_y.size(1) if ctx.bshd else attn_y.size(2),
            attn_y.size(3),
            ctx.bshd,
        )
        return grad_x, grad_attn_y, grad_proj_weight_t, grad_scale, None

    torch.library.register_autograd(
        "allama_triton_bench::attn_outproj_residual",
        backward,
        setup_context=setup_context,
    )
    return attn_outproj_residual_op


def loss_reference(
    residual_x: torch.Tensor,
    attn_y: torch.Tensor,
    proj_weight_t: torch.Tensor,
    scale: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    bshd: bool = False,
) -> torch.Tensor:
    """Return a scalar loss that exercises the reference attention boundary."""
    return (
        attn_outproj_residual_reference(
            residual_x, attn_y, proj_weight_t, scale, bshd=bshd
        )
        * grad_out
    ).sum()


def loss_custom(
    op: Callable[..., torch.Tensor],
    residual_x: torch.Tensor,
    attn_y: torch.Tensor,
    proj_weight_t: torch.Tensor,
    scale: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    bshd: bool = False,
) -> torch.Tensor:
    """Return a scalar loss that exercises the custom attention boundary."""
    return (
        op(residual_x, attn_y, proj_weight_t, scale, 1 if bshd else 0) * grad_out
    ).sum()


def summarize_case(
    *,
    case: str,
    reference_ms: float,
    compiled_reference_ms: float,
    custom_ms: float,
    compiled_custom_ms: float,
    max_abs: float,
    max_rel: float,
) -> dict[str, float | str]:
    """Create a compact benchmark summary for one attention case."""
    return {
        "case": case,
        "reference_ms": float(reference_ms),
        "compiled_reference_ms": float(compiled_reference_ms),
        "custom_ms": float(custom_ms),
        "compiled_custom_ms": float(compiled_custom_ms),
        "speedup_vs_reference": (
            float(reference_ms / custom_ms) if custom_ms > 0.0 else 0.0
        ),
        "speedup_vs_compiled_reference": (
            float(compiled_reference_ms / compiled_custom_ms)
            if compiled_custom_ms > 0.0
            else 0.0
        ),
        "max_abs": float(max_abs),
        "max_rel": float(max_rel),
    }


def main() -> None:
    """Build and benchmark the fused attention output boundary."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the attention-block benchmark")

    device = torch.device("cuda")
    torch.manual_seed(1337)
    op = register_attention_block_custom_op()
    bshd = args.layout == "bshd"

    batch = 4
    seq_len = 1024
    num_heads = 14
    head_dim = 64
    model_dim = num_heads * head_dim

    residual_x = torch.randn(
        batch * seq_len, model_dim, device=device, dtype=torch.bfloat16
    )
    if bshd:
        attn_y = torch.randn(
            batch, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16
        )
    else:
        attn_y = torch.randn(
            batch, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16
        )
    proj_weight_t = (
        torch.randn(model_dim, model_dim, device=device, dtype=torch.bfloat16)
        / model_dim**0.5
    )
    scale = torch.randn(model_dim, device=device, dtype=torch.float32)
    grad_out = torch.randn_like(residual_x)

    ref_out = attn_outproj_residual_reference(
        residual_x, attn_y, proj_weight_t, scale, bshd=bshd
    )
    custom_out = op(residual_x, attn_y, proj_weight_t, scale, 1 if bshd else 0)
    forward_max_abs = (ref_out.float() - custom_out.float()).abs().max().item()
    forward_max_rel = (
        (ref_out.float() - custom_out.float()).abs().max()
        / ref_out.float().abs().max().clamp_min(1e-6)
    ).item()

    ref_ms = measure_ms(
        lambda: attn_outproj_residual_reference(
            residual_x, attn_y, proj_weight_t, scale, bshd=bshd
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_ref = torch.compile(
        attn_outproj_residual_reference,
        dynamic=False,
        fullgraph=True,
    )
    compiled_ref_ms = measure_ms(
        lambda: compiled_ref(residual_x, attn_y, proj_weight_t, scale, bshd=bshd),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    custom_ms = measure_ms(
        lambda: op(residual_x, attn_y, proj_weight_t, scale, 1 if bshd else 0),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom = torch.compile(
        lambda x_, y_, w_t_, scale_: op(x_, y_, w_t_, scale_, 1 if bshd else 0),
        dynamic=False,
        fullgraph=True,
    )
    compiled_custom_ms = measure_ms(
        lambda: compiled_custom(residual_x, attn_y, proj_weight_t, scale),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    forward_summary = summarize_case(
        case="attention_block_forward",
        reference_ms=ref_ms,
        compiled_reference_ms=compiled_ref_ms,
        custom_ms=custom_ms,
        compiled_custom_ms=compiled_custom_ms,
        max_abs=forward_max_abs,
        max_rel=forward_max_rel,
    )

    residual_x_ref = residual_x.detach().clone().requires_grad_(True)
    attn_y_ref = attn_y.detach().clone().requires_grad_(True)
    proj_weight_t_ref = proj_weight_t.detach().clone().requires_grad_(True)
    scale_ref = scale.detach().clone().requires_grad_(True)

    residual_x_custom = residual_x.detach().clone().requires_grad_(True)
    attn_y_custom = attn_y.detach().clone().requires_grad_(True)
    proj_weight_t_custom = proj_weight_t.detach().clone().requires_grad_(True)
    scale_custom = scale.detach().clone().requires_grad_(True)

    ref_grads = torch.autograd.grad(
        loss_reference(
            residual_x_ref,
            attn_y_ref,
            proj_weight_t_ref,
            scale_ref,
            grad_out,
            bshd=bshd,
        ),
        (residual_x_ref, attn_y_ref, proj_weight_t_ref, scale_ref),
    )
    custom_grads = torch.autograd.grad(
        loss_custom(
            op,
            residual_x_custom,
            attn_y_custom,
            proj_weight_t_custom,
            scale_custom,
            grad_out,
            bshd=bshd,
        ),
        (residual_x_custom, attn_y_custom, proj_weight_t_custom, scale_custom),
    )
    backward_max_abs = max(
        (ref_grad.float() - custom_grad.float()).abs().max().item()
        for ref_grad, custom_grad in zip(ref_grads, custom_grads, strict=True)
    )
    backward_max_rel = max(
        (
            (ref_grad.float() - custom_grad.float()).abs().max()
            / ref_grad.float().abs().max().clamp_min(1e-6)
        ).item()
        for ref_grad, custom_grad in zip(ref_grads, custom_grads, strict=True)
    )

    ref_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            loss_reference(
                residual_x_ref,
                attn_y_ref,
                proj_weight_t_ref,
                scale_ref,
                grad_out,
                bshd=bshd,
            ),
            (residual_x_ref, attn_y_ref, proj_weight_t_ref, scale_ref),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_ref_loss = torch.compile(loss_reference, dynamic=False, fullgraph=True)
    compiled_ref_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_ref_loss(
                residual_x_ref,
                attn_y_ref,
                proj_weight_t_ref,
                scale_ref,
                grad_out,
                bshd=bshd,
            ),
            (residual_x_ref, attn_y_ref, proj_weight_t_ref, scale_ref),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom_loss = torch.compile(
        lambda x_, y_, w_t_, scale_, grad_out_: loss_custom(
            op, x_, y_, w_t_, scale_, grad_out_, bshd=bshd
        ),
        dynamic=False,
        fullgraph=True,
    )
    custom_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            loss_custom(
                op,
                residual_x_custom,
                attn_y_custom,
                proj_weight_t_custom,
                scale_custom,
                grad_out,
                bshd=bshd,
            ),
            (
                residual_x_custom,
                attn_y_custom,
                proj_weight_t_custom,
                scale_custom,
            ),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_custom_loss(
                residual_x_custom,
                attn_y_custom,
                proj_weight_t_custom,
                scale_custom,
                grad_out,
            ),
            (
                residual_x_custom,
                attn_y_custom,
                proj_weight_t_custom,
                scale_custom,
            ),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    backward_summary = summarize_case(
        case="attention_block_backward",
        reference_ms=ref_backward_ms,
        compiled_reference_ms=compiled_ref_backward_ms,
        custom_ms=custom_backward_ms,
        compiled_custom_ms=compiled_custom_backward_ms,
        max_abs=backward_max_abs,
        max_rel=backward_max_rel,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "shape": {
            "batch": batch,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "model_dim": model_dim,
            "layout": args.layout,
        },
        "triton_version": triton.__version__,
        "forward": forward_summary,
        "backward": backward_summary,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
