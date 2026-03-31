#!/usr/bin/env python3
"""Benchmark a larger ALlama MLP gate-up boundary with a Triton kernel.

This targets the other obvious MLP-side fusion boundary:

- vendor RMSNorm
- Triton fused gate projection + up projection + SwiGLU epilogue
- vendor down projection
- vendor residual add/scale

The forward kernel avoids materializing the larger ``[M, 2H]`` gate-up tensor.
Backward now uses a real fused Triton derivative kernel for the gate/up
recompute plus SwiGLU epilogue, while still leaving the larger weight-gradient
and input-gradient matmuls to vendor kernels.
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
import torch.nn.functional as F
import triton
import triton.language as tl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MLP gate-up benchmark.

    :return argparse.Namespace: Parsed benchmark configuration.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs_allama_validation/mlp_gateup_block_v1"),
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


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=8,
            num_stages=5,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def gateup_swiglu_kernel(
    x_ptr,
    gate_weight_t_ptr,
    up_weight_t_ptr,
    hidden_ptr,
    M,
    N,
    K,
    x_stride_m,
    x_stride_k,
    gate_stride_k,
    gate_stride_n,
    up_stride_k,
    up_stride_n,
    hidden_stride_m,
    hidden_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Fuse the gate/up projections and SwiGLU epilogue."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_base in range(0, K, BLOCK_K):
        k_idx = k_base + offs_k
        mask_k = k_idx < K
        x_ptrs = x_ptr + offs_m[:, None] * x_stride_m + k_idx[None, :] * x_stride_k
        x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        gate_ptrs = (
            gate_weight_t_ptr
            + k_idx[:, None] * gate_stride_k
            + offs_n[None, :] * gate_stride_n
        )
        up_ptrs = (
            up_weight_t_ptr
            + k_idx[:, None] * up_stride_k
            + offs_n[None, :] * up_stride_n
        )
        gate_weight = tl.load(
            gate_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
        )
        up_weight = tl.load(up_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc_gate += tl.dot(x_vals, gate_weight)
        acc_up += tl.dot(x_vals, up_weight)

    hidden = acc_gate * tl.sigmoid(acc_gate) * acc_up
    hidden_ptrs = (
        hidden_ptr
        + offs_m[:, None] * hidden_stride_m
        + offs_n[None, :] * hidden_stride_n
    )
    tl.store(
        hidden_ptrs,
        hidden.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=8,
            num_stages=5,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def gateup_swiglu_backward_parts_kernel(
    x_ptr,
    gate_weight_t_ptr,
    up_weight_t_ptr,
    grad_hidden_ptr,
    grad_gate_ptr,
    grad_up_ptr,
    M,
    N,
    K,
    x_stride_m,
    x_stride_k,
    gate_stride_k,
    gate_stride_n,
    up_stride_k,
    up_stride_n,
    grad_hidden_stride_m,
    grad_hidden_stride_n,
    grad_gate_stride_m,
    grad_gate_stride_n,
    grad_up_stride_m,
    grad_up_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Fuse gate/up recompute with the SwiGLU derivative epilogue."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_base in range(0, K, BLOCK_K):
        k_idx = k_base + offs_k
        mask_k = k_idx < K
        x_ptrs = x_ptr + offs_m[:, None] * x_stride_m + k_idx[None, :] * x_stride_k
        x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        gate_ptrs = (
            gate_weight_t_ptr
            + k_idx[:, None] * gate_stride_k
            + offs_n[None, :] * gate_stride_n
        )
        up_ptrs = (
            up_weight_t_ptr
            + k_idx[:, None] * up_stride_k
            + offs_n[None, :] * up_stride_n
        )
        gate_weight = tl.load(
            gate_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
        )
        up_weight = tl.load(up_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc_gate += tl.dot(x_vals, gate_weight)
        acc_up += tl.dot(x_vals, up_weight)

    grad_hidden_ptrs = (
        grad_hidden_ptr
        + offs_m[:, None] * grad_hidden_stride_m
        + offs_n[None, :] * grad_hidden_stride_n
    )
    grad_hidden = tl.load(
        grad_hidden_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
    ).to(tl.float32)
    sigmoid_gate = tl.sigmoid(acc_gate)
    silu_gate = acc_gate * sigmoid_gate
    silu_prime = sigmoid_gate * (1.0 + acc_gate * (1.0 - sigmoid_gate))
    grad_gate = grad_hidden * acc_up * silu_prime
    grad_up = grad_hidden * silu_gate

    grad_gate_ptrs = (
        grad_gate_ptr
        + offs_m[:, None] * grad_gate_stride_m
        + offs_n[None, :] * grad_gate_stride_n
    )
    grad_up_ptrs = (
        grad_up_ptr
        + offs_m[:, None] * grad_up_stride_m
        + offs_n[None, :] * grad_up_stride_n
    )
    tl.store(
        grad_gate_ptrs,
        grad_gate.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(
        grad_up_ptrs,
        grad_up.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def gateup_swiglu_triton(
    x_norm: torch.Tensor,
    gate_weight_t: torch.Tensor,
    up_weight_t: torch.Tensor,
) -> torch.Tensor:
    """Run the fused Triton gate/up projection and SwiGLU kernel.

    :param torch.Tensor x_norm: Normalized activations flattened to ``[M, D]``.
    :param torch.Tensor gate_weight_t: Gate projection weight transposed to ``[D, H]``.
    :param torch.Tensor up_weight_t: Up projection weight transposed to ``[D, H]``.
    :return torch.Tensor: Hidden activations ``[M, H]``.
    """
    if (
        x_norm.dtype != torch.bfloat16
        or gate_weight_t.dtype != torch.bfloat16
        or up_weight_t.dtype != torch.bfloat16
    ):
        raise TypeError("this benchmark currently targets bfloat16 tensors")
    if x_norm.dim() != 2 or gate_weight_t.dim() != 2 or up_weight_t.dim() != 2:
        raise ValueError(
            "expected x_norm=[M,D], gate_weight_t=[D,H], up_weight_t=[D,H]"
        )
    if gate_weight_t.shape != up_weight_t.shape:
        raise ValueError("gate_weight_t and up_weight_t must have matching shapes")
    if gate_weight_t.size(0) != x_norm.size(1):
        raise ValueError("weight input dimension must match x_norm last dimension")
    hidden = torch.empty(
        x_norm.size(0),
        gate_weight_t.size(1),
        device=x_norm.device,
        dtype=x_norm.dtype,
    )

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (
            triton.cdiv(x_norm.size(0), meta["BLOCK_M"])
            * triton.cdiv(gate_weight_t.size(1), meta["BLOCK_N"]),
        )

    gateup_swiglu_kernel[grid](
        x_norm,
        gate_weight_t,
        up_weight_t,
        hidden,
        x_norm.size(0),
        gate_weight_t.size(1),
        x_norm.size(1),
        x_norm.stride(0),
        x_norm.stride(1),
        gate_weight_t.stride(0),
        gate_weight_t.stride(1),
        up_weight_t.stride(0),
        up_weight_t.stride(1),
        hidden.stride(0),
        hidden.stride(1),
    )
    return hidden


def gateup_swiglu_backward_parts_triton(
    x_norm: torch.Tensor,
    gate_weight_t: torch.Tensor,
    up_weight_t: torch.Tensor,
    grad_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Triton fused gate/up derivative kernel.

    :param torch.Tensor x_norm: Normalized activations flattened to ``[M, D]``.
    :param torch.Tensor gate_weight_t: Gate projection weight transposed to ``[D, H]``.
    :param torch.Tensor up_weight_t: Up projection weight transposed to ``[D, H]``.
    :param torch.Tensor grad_hidden: Gradient of the hidden activation ``[M, H]``.
    :return tuple[torch.Tensor, torch.Tensor]: ``grad_gate`` and ``grad_up`` tensors.
    """
    if (
        x_norm.dtype != torch.bfloat16
        or gate_weight_t.dtype != torch.bfloat16
        or up_weight_t.dtype != torch.bfloat16
        or grad_hidden.dtype != torch.bfloat16
    ):
        raise TypeError("this benchmark currently targets bfloat16 tensors")
    grad_gate = torch.empty_like(grad_hidden)
    grad_up = torch.empty_like(grad_hidden)

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (
            triton.cdiv(x_norm.size(0), meta["BLOCK_M"])
            * triton.cdiv(gate_weight_t.size(1), meta["BLOCK_N"]),
        )

    gateup_swiglu_backward_parts_kernel[grid](
        x_norm,
        gate_weight_t,
        up_weight_t,
        grad_hidden,
        grad_gate,
        grad_up,
        x_norm.size(0),
        gate_weight_t.size(1),
        x_norm.size(1),
        x_norm.stride(0),
        x_norm.stride(1),
        gate_weight_t.stride(0),
        gate_weight_t.stride(1),
        up_weight_t.stride(0),
        up_weight_t.stride(1),
        grad_hidden.stride(0),
        grad_hidden.stride(1),
        grad_gate.stride(0),
        grad_gate.stride(1),
        grad_up.stride(0),
        grad_up.stride(1),
    )
    return grad_gate, grad_up


def gateup_swiglu_reference(
    x_norm: torch.Tensor,
    gate_weight_t: torch.Tensor,
    up_weight_t: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation of the fused gate/up boundary."""
    gate = x_norm @ gate_weight_t
    up = x_norm @ up_weight_t
    return F.silu(gate) * up


def register_mlp_gateup_custom_op() -> Any:
    """Register a benchmark-local custom op for the gate-up boundary."""

    @torch.library.custom_op(
        "allama_triton_bench::gateup_swiglu_backward_parts",
        mutates_args=(),
    )
    def gateup_swiglu_backward_parts_op(
        x_norm: torch.Tensor,
        gate_weight_t: torch.Tensor,
        up_weight_t: torch.Tensor,
        grad_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return gateup_swiglu_backward_parts_triton(
            x_norm, gate_weight_t, up_weight_t, grad_hidden
        )

    @gateup_swiglu_backward_parts_op.register_fake
    def _gateup_swiglu_backward_parts_op_fake(
        x_norm: torch.Tensor,
        gate_weight_t: torch.Tensor,
        up_weight_t: torch.Tensor,
        grad_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del x_norm, gate_weight_t, up_weight_t
        return grad_hidden.new_empty(grad_hidden.shape), grad_hidden.new_empty(
            grad_hidden.shape
        )

    @torch.library.custom_op(
        "allama_triton_bench::gateup_swiglu",
        mutates_args=(),
    )
    def gateup_swiglu_op(
        x_norm: torch.Tensor,
        gate_weight_t: torch.Tensor,
        up_weight_t: torch.Tensor,
    ) -> torch.Tensor:
        return gateup_swiglu_triton(x_norm, gate_weight_t, up_weight_t)

    @gateup_swiglu_op.register_fake
    def _gateup_swiglu_op_fake(
        x_norm: torch.Tensor,
        gate_weight_t: torch.Tensor,
        up_weight_t: torch.Tensor,
    ) -> torch.Tensor:
        del up_weight_t
        return x_norm.new_empty((x_norm.size(0), gate_weight_t.size(1)))

    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        del output
        x_norm, gate_weight_t, up_weight_t = inputs
        ctx.save_for_backward(x_norm, gate_weight_t, up_weight_t)

    def backward(
        ctx: Any, grad_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm, gate_weight_t, up_weight_t = ctx.saved_tensors
        grad_gate, grad_up = gateup_swiglu_backward_parts_op(
            x_norm, gate_weight_t, up_weight_t, grad_hidden
        )
        grad_x = grad_gate @ gate_weight_t.transpose(0, 1)
        grad_x = grad_x + grad_up @ up_weight_t.transpose(0, 1)
        grad_gate_weight_t = x_norm.transpose(0, 1) @ grad_gate
        grad_up_weight_t = x_norm.transpose(0, 1) @ grad_up
        return grad_x, grad_gate_weight_t, grad_up_weight_t

    torch.library.register_autograd(
        "allama_triton_bench::gateup_swiglu",
        backward,
        setup_context=setup_context,
    )
    return gateup_swiglu_op


def mlp_block_reference(
    x: torch.Tensor,
    gate_weight_t: torch.Tensor,
    up_weight_t: torch.Tensor,
    down_weight_t: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference prenorm MLP sublayer using separate gate/up matmuls."""
    x_norm = F.rms_norm(x, (x.size(-1),), norm_weight.to(dtype=x.dtype), eps=eps)
    hidden = gateup_swiglu_reference(
        x_norm.view(-1, x.size(-1)),
        gate_weight_t,
        up_weight_t,
    )
    branch = hidden @ down_weight_t
    out = x.view(-1, x.size(-1)) + branch * scale.to(dtype=x.dtype)[None, :]
    return out.view_as(x)


def mlp_block_custom(
    op: Callable[..., torch.Tensor],
    x: torch.Tensor,
    gate_weight_t: torch.Tensor,
    up_weight_t: torch.Tensor,
    down_weight_t: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the MLP sublayer using the Triton fused gate-up custom op."""
    x_norm = F.rms_norm(x, (x.size(-1),), norm_weight.to(dtype=x.dtype), eps=eps)
    hidden = op(x_norm.view(-1, x.size(-1)), gate_weight_t, up_weight_t)
    branch = hidden @ down_weight_t
    out = x.view(-1, x.size(-1)) + branch * scale.to(dtype=x.dtype)[None, :]
    return out.view_as(x)


def loss_reference(
    x: torch.Tensor,
    gate_weight_t: torch.Tensor,
    up_weight_t: torch.Tensor,
    down_weight_t: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return a scalar loss that exercises the reference MLP path."""
    return (
        mlp_block_reference(
            x,
            gate_weight_t,
            up_weight_t,
            down_weight_t,
            norm_weight,
            scale,
            eps,
        )
        * grad_out
    ).sum()


def loss_custom(
    op: Callable[..., torch.Tensor],
    x: torch.Tensor,
    gate_weight_t: torch.Tensor,
    up_weight_t: torch.Tensor,
    down_weight_t: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return a scalar loss that exercises the custom MLP path."""
    return (
        mlp_block_custom(
            op,
            x,
            gate_weight_t,
            up_weight_t,
            down_weight_t,
            norm_weight,
            scale,
            eps,
        )
        * grad_out
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
    """Create a compact benchmark summary for one gate-up MLP case."""
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
    """Build and benchmark the fused gate-up MLP boundary."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MLP gate-up benchmark")

    device = torch.device("cuda")
    torch.manual_seed(1337)
    op = register_mlp_gateup_custom_op()

    batch = 4
    seq_len = 1024
    dim = 896
    hidden = 1408
    eps = 1e-5

    x = torch.randn(batch, seq_len, dim, device=device, dtype=torch.bfloat16)
    gate_weight_t = (
        torch.randn(dim, hidden, device=device, dtype=torch.bfloat16) / dim**0.5
    )
    up_weight_t = (
        torch.randn(dim, hidden, device=device, dtype=torch.bfloat16) / dim**0.5
    )
    down_weight_t = (
        torch.randn(hidden, dim, device=device, dtype=torch.bfloat16) / hidden**0.5
    )
    norm_weight = torch.randn(dim, device=device, dtype=torch.float32)
    scale = torch.randn(dim, device=device, dtype=torch.float32)
    grad_out = torch.randn_like(x)

    ref_out = mlp_block_reference(
        x,
        gate_weight_t,
        up_weight_t,
        down_weight_t,
        norm_weight,
        scale,
        eps,
    )
    custom_out = mlp_block_custom(
        op,
        x,
        gate_weight_t,
        up_weight_t,
        down_weight_t,
        norm_weight,
        scale,
        eps,
    )
    forward_max_abs = (ref_out.float() - custom_out.float()).abs().max().item()
    forward_max_rel = (
        (ref_out.float() - custom_out.float()).abs().max()
        / ref_out.float().abs().max().clamp_min(1e-6)
    ).item()

    ref_ms = measure_ms(
        lambda: mlp_block_reference(
            x,
            gate_weight_t,
            up_weight_t,
            down_weight_t,
            norm_weight,
            scale,
            eps,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_ref = torch.compile(mlp_block_reference, dynamic=False, fullgraph=True)
    compiled_ref_ms = measure_ms(
        lambda: compiled_ref(
            x,
            gate_weight_t,
            up_weight_t,
            down_weight_t,
            norm_weight,
            scale,
            eps,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    custom_ms = measure_ms(
        lambda: mlp_block_custom(
            op,
            x,
            gate_weight_t,
            up_weight_t,
            down_weight_t,
            norm_weight,
            scale,
            eps,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom = torch.compile(
        lambda x_,
        gate_w_t_,
        up_w_t_,
        down_w_t_,
        norm_w_,
        scale_,
        eps_: mlp_block_custom(
            op,
            x_,
            gate_w_t_,
            up_w_t_,
            down_w_t_,
            norm_w_,
            scale_,
            eps_,
        ),
        dynamic=False,
        fullgraph=True,
    )
    compiled_custom_ms = measure_ms(
        lambda: compiled_custom(
            x,
            gate_weight_t,
            up_weight_t,
            down_weight_t,
            norm_weight,
            scale,
            eps,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    forward_summary = summarize_case(
        case="mlp_gateup_block_forward",
        reference_ms=ref_ms,
        compiled_reference_ms=compiled_ref_ms,
        custom_ms=custom_ms,
        compiled_custom_ms=compiled_custom_ms,
        max_abs=forward_max_abs,
        max_rel=forward_max_rel,
    )

    x_ref = x.detach().clone().requires_grad_(True)
    gate_weight_t_ref = gate_weight_t.detach().clone().requires_grad_(True)
    up_weight_t_ref = up_weight_t.detach().clone().requires_grad_(True)
    down_weight_t_ref = down_weight_t.detach().clone().requires_grad_(True)
    norm_weight_ref = norm_weight.detach().clone().requires_grad_(True)
    scale_ref = scale.detach().clone().requires_grad_(True)

    x_custom = x.detach().clone().requires_grad_(True)
    gate_weight_t_custom = gate_weight_t.detach().clone().requires_grad_(True)
    up_weight_t_custom = up_weight_t.detach().clone().requires_grad_(True)
    down_weight_t_custom = down_weight_t.detach().clone().requires_grad_(True)
    norm_weight_custom = norm_weight.detach().clone().requires_grad_(True)
    scale_custom = scale.detach().clone().requires_grad_(True)

    ref_grads = torch.autograd.grad(
        loss_reference(
            x_ref,
            gate_weight_t_ref,
            up_weight_t_ref,
            down_weight_t_ref,
            norm_weight_ref,
            scale_ref,
            grad_out,
            eps,
        ),
        (
            x_ref,
            gate_weight_t_ref,
            up_weight_t_ref,
            down_weight_t_ref,
            norm_weight_ref,
            scale_ref,
        ),
    )
    custom_grads = torch.autograd.grad(
        loss_custom(
            op,
            x_custom,
            gate_weight_t_custom,
            up_weight_t_custom,
            down_weight_t_custom,
            norm_weight_custom,
            scale_custom,
            grad_out,
            eps,
        ),
        (
            x_custom,
            gate_weight_t_custom,
            up_weight_t_custom,
            down_weight_t_custom,
            norm_weight_custom,
            scale_custom,
        ),
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
                x_ref,
                gate_weight_t_ref,
                up_weight_t_ref,
                down_weight_t_ref,
                norm_weight_ref,
                scale_ref,
                grad_out,
                eps,
            ),
            (
                x_ref,
                gate_weight_t_ref,
                up_weight_t_ref,
                down_weight_t_ref,
                norm_weight_ref,
                scale_ref,
            ),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_ref_loss = torch.compile(loss_reference, dynamic=False, fullgraph=True)
    compiled_ref_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_ref_loss(
                x_ref,
                gate_weight_t_ref,
                up_weight_t_ref,
                down_weight_t_ref,
                norm_weight_ref,
                scale_ref,
                grad_out,
                eps,
            ),
            (
                x_ref,
                gate_weight_t_ref,
                up_weight_t_ref,
                down_weight_t_ref,
                norm_weight_ref,
                scale_ref,
            ),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom_loss = torch.compile(
        lambda x_,
        gate_w_t_,
        up_w_t_,
        down_w_t_,
        norm_w_,
        scale_,
        grad_out_,
        eps_: loss_custom(
            op,
            x_,
            gate_w_t_,
            up_w_t_,
            down_w_t_,
            norm_w_,
            scale_,
            grad_out_,
            eps_,
        ),
        dynamic=False,
        fullgraph=True,
    )
    custom_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            loss_custom(
                op,
                x_custom,
                gate_weight_t_custom,
                up_weight_t_custom,
                down_weight_t_custom,
                norm_weight_custom,
                scale_custom,
                grad_out,
                eps,
            ),
            (
                x_custom,
                gate_weight_t_custom,
                up_weight_t_custom,
                down_weight_t_custom,
                norm_weight_custom,
                scale_custom,
            ),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_custom_loss(
                x_custom,
                gate_weight_t_custom,
                up_weight_t_custom,
                down_weight_t_custom,
                norm_weight_custom,
                scale_custom,
                grad_out,
                eps,
            ),
            (
                x_custom,
                gate_weight_t_custom,
                up_weight_t_custom,
                down_weight_t_custom,
                norm_weight_custom,
                scale_custom,
            ),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    backward_summary = summarize_case(
        case="mlp_gateup_block_backward",
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
            "model_dim": dim,
            "hidden_dim": hidden,
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
