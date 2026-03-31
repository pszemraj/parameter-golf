#!/usr/bin/env python3
"""Benchmark a larger ALlama MLP boundary with a Triton forward kernel.

This targets a materially larger boundary than the earlier glue kernels:

- vendor RMSNorm
- vendor gate_up GEMM
- Triton fused SwiGLU + down projection + residual add/scale

The forward kernel removes the materialized hidden activation between the
SwiGLU and down projection. Backward deliberately uses straightforward PyTorch
matmuls and elementwise ops so we can measure whether the larger forward
boundary still pays off under a real training-style contract.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MLP-block benchmark.

    :return argparse.Namespace: Parsed benchmark configuration.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs_allama_validation/mlp_block_v1"),
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
def swiglu_down_residual_kernel(
    gate_up_ptr,
    down_weight_t_ptr,
    residual_x_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    gate_up_stride_m,
    gate_up_stride_k,
    down_stride_k,
    down_stride_n,
    residual_stride_m,
    residual_stride_n,
    out_stride_m,
    out_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Fuse SwiGLU, down projection, and the residual add epilogue."""
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_base in range(0, K, BLOCK_K):
        k_idx = k_base + offs_k
        mask_k = k_idx < K

        gate_ptrs = (
            gate_up_ptr
            + offs_m[:, None] * gate_up_stride_m
            + k_idx[None, :] * gate_up_stride_k
        )
        up_ptrs = (
            gate_up_ptr
            + offs_m[:, None] * gate_up_stride_m
            + (K + k_idx)[None, :] * gate_up_stride_k
        )
        gate = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        up = tl.load(up_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        gate_f = gate.to(tl.float32)
        hidden = gate_f * tl.sigmoid(gate_f) * up.to(tl.float32)
        hidden_bf16 = hidden.to(tl.bfloat16)

        weight_ptrs = (
            down_weight_t_ptr
            + k_idx[:, None] * down_stride_k
            + offs_n[None, :] * down_stride_n
        )
        weight = tl.load(weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(hidden_bf16, weight)

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


def swiglu_down_residual_triton(
    residual_x: torch.Tensor,
    gate_up: torch.Tensor,
    down_weight_t: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Run the fused Triton SwiGLU-down-residual kernel.

    :param torch.Tensor residual_x: Residual input flattened to ``[M, N]``.
    :param torch.Tensor gate_up: Gate/up activation flattened to ``[M, 2K]``.
    :param torch.Tensor down_weight_t: Down-projection weight transposed to ``[K, N]``.
    :param torch.Tensor scale: Per-channel residual scale vector ``[N]``.
    :return torch.Tensor: Residual-updated output tensor ``[M, N]``.
    """
    if residual_x.dtype != torch.bfloat16 or gate_up.dtype != torch.bfloat16:
        raise TypeError("this benchmark currently targets bfloat16 tensors")
    if residual_x.dim() != 2 or gate_up.dim() != 2 or down_weight_t.dim() != 2:
        raise ValueError("expected 2D residual, gate_up, and down_weight_t tensors")
    if gate_up.size(0) != residual_x.size(0):
        raise ValueError("gate_up and residual_x must have matching row counts")
    if gate_up.size(1) % 2 != 0:
        raise ValueError("gate_up last dimension must be divisible by 2")
    hidden = gate_up.size(1) // 2
    if down_weight_t.shape != (hidden, residual_x.size(1)):
        raise ValueError("down_weight_t shape must be [hidden, model_dim]")
    out = torch.empty_like(residual_x)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(residual_x.size(0), meta["BLOCK_M"])
            * triton.cdiv(residual_x.size(1), meta["BLOCK_N"]),
        )

    swiglu_down_residual_kernel[grid](
        gate_up,
        down_weight_t,
        residual_x,
        scale,
        out,
        residual_x.size(0),
        residual_x.size(1),
        hidden,
        gate_up.stride(0),
        gate_up.stride(1),
        down_weight_t.stride(0),
        down_weight_t.stride(1),
        residual_x.stride(0),
        residual_x.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def swiglu_residual_reference(
    residual_x: torch.Tensor,
    gate_up: torch.Tensor,
    down_weight_t: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation of the fused MLP epilogue boundary."""
    hidden = gate_up.size(-1) // 2
    gate, up = gate_up.split(hidden, dim=-1)
    branch = (F.silu(gate) * up) @ down_weight_t
    return residual_x + branch * scale.to(dtype=residual_x.dtype)[None, :]


def mlp_block_reference(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference prenorm MLP sublayer on the anchor shape."""
    x_norm = F.rms_norm(x, (x.size(-1),), norm_weight.to(dtype=x.dtype), eps=eps)
    gate_up = F.linear(x_norm, gate_up_weight)
    return swiglu_residual_reference(
        x.view(-1, x.size(-1)),
        gate_up.view(-1, gate_up.size(-1)),
        down_weight_t,
        scale,
    ).view_as(x)


def register_mlp_block_custom_op() -> Any:
    """Register a benchmark-local custom op for the fused MLP boundary."""

    @torch.library.custom_op(
        "allama_triton_bench::swiglu_down_residual",
        mutates_args=(),
    )
    def swiglu_down_residual_op(
        residual_x: torch.Tensor,
        gate_up: torch.Tensor,
        down_weight_t: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return swiglu_down_residual_triton(residual_x, gate_up, down_weight_t, scale)

    @swiglu_down_residual_op.register_fake
    def _swiglu_down_residual_op_fake(
        residual_x: torch.Tensor,
        gate_up: torch.Tensor,
        down_weight_t: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        del gate_up, down_weight_t, scale
        return residual_x.new_empty(residual_x.shape)

    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        residual_x, gate_up, down_weight_t, scale = inputs
        del residual_x, output
        ctx.save_for_backward(gate_up, down_weight_t, scale)

    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_up, down_weight_t, scale = ctx.saved_tensors
        hidden = gate_up.size(-1) // 2
        gate, up = gate_up.split(hidden, dim=-1)
        silu_gate = F.silu(gate)
        branch = (silu_gate * up) @ down_weight_t
        scale_cast = scale.to(dtype=grad_out.dtype)[None, :]
        grad_x = grad_out
        grad_scale = (grad_out * branch).sum(dim=0).to(dtype=scale.dtype)
        grad_branch = grad_out * scale_cast
        grad_down_weight_t = (silu_gate * up).transpose(0, 1) @ grad_branch
        grad_hidden = grad_branch @ down_weight_t.transpose(0, 1)
        sigmoid_gate = torch.sigmoid(gate.float()).to(dtype=gate.dtype)
        silu_prime = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        grad_gate = grad_hidden * up * silu_prime
        grad_up = grad_hidden * silu_gate
        grad_gate_up = torch.cat((grad_gate, grad_up), dim=-1)
        return grad_x, grad_gate_up, grad_down_weight_t, grad_scale

    torch.library.register_autograd(
        "allama_triton_bench::swiglu_down_residual",
        backward,
        setup_context=setup_context,
    )
    return swiglu_down_residual_op


def mlp_block_custom(
    op: Callable[..., torch.Tensor],
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the MLP sublayer using the Triton fused down-path custom op."""
    x_norm = F.rms_norm(x, (x.size(-1),), norm_weight.to(dtype=x.dtype), eps=eps)
    gate_up = F.linear(x_norm, gate_up_weight)
    out = op(
        x.view(-1, x.size(-1)),
        gate_up.view(-1, gate_up.size(-1)),
        down_weight_t,
        scale,
    )
    return out.view_as(x)


def loss_reference(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return a scalar loss that exercises the full MLP reference path."""
    return (
        mlp_block_reference(x, gate_up_weight, down_weight_t, norm_weight, scale, eps)
        * grad_out
    ).sum()


def loss_custom(
    op: Callable[..., torch.Tensor],
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return a scalar loss that exercises the custom MLP path."""
    return (
        mlp_block_custom(op, x, gate_up_weight, down_weight_t, norm_weight, scale, eps)
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
    """Create a compact benchmark summary for one MLP case."""
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
    """Build and benchmark the fused MLP boundary against compiled PyTorch."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MLP-block benchmark")

    device = torch.device("cuda")
    torch.manual_seed(1337)
    op = register_mlp_block_custom_op()

    batch = 4
    seq_len = 1024
    dim = 896
    hidden = 1408
    eps = 1e-5

    x = torch.randn(batch, seq_len, dim, device=device, dtype=torch.bfloat16)
    gate_up_weight = (
        torch.randn(hidden * 2, dim, device=device, dtype=torch.bfloat16) / dim**0.5
    )
    down_weight_t = (
        torch.randn(hidden, dim, device=device, dtype=torch.bfloat16) / hidden**0.5
    )
    norm_weight = torch.randn(dim, device=device, dtype=torch.float32)
    scale = torch.randn(dim, device=device, dtype=torch.float32)
    grad_out = torch.randn_like(x)

    ref_out = mlp_block_reference(
        x, gate_up_weight, down_weight_t, norm_weight, scale, eps
    )
    custom_out = mlp_block_custom(
        op, x, gate_up_weight, down_weight_t, norm_weight, scale, eps
    )
    forward_max_abs = (ref_out.float() - custom_out.float()).abs().max().item()
    forward_max_rel = (
        (ref_out.float() - custom_out.float()).abs().max()
        / ref_out.float().abs().max().clamp_min(1e-6)
    ).item()

    ref_ms = measure_ms(
        lambda: mlp_block_reference(
            x, gate_up_weight, down_weight_t, norm_weight, scale, eps
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_ref = torch.compile(mlp_block_reference, dynamic=False, fullgraph=True)
    compiled_ref_ms = measure_ms(
        lambda: compiled_ref(x, gate_up_weight, down_weight_t, norm_weight, scale, eps),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    custom_ms = measure_ms(
        lambda: mlp_block_custom(
            op, x, gate_up_weight, down_weight_t, norm_weight, scale, eps
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom = torch.compile(
        lambda x_, gu_w_, down_w_t_, norm_w_, scale_, eps_: mlp_block_custom(
            op, x_, gu_w_, down_w_t_, norm_w_, scale_, eps_
        ),
        dynamic=False,
        fullgraph=True,
    )
    compiled_custom_ms = measure_ms(
        lambda: compiled_custom(
            x, gate_up_weight, down_weight_t, norm_weight, scale, eps
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    forward_summary = summarize_case(
        case="mlp_block_forward",
        reference_ms=ref_ms,
        compiled_reference_ms=compiled_ref_ms,
        custom_ms=custom_ms,
        compiled_custom_ms=compiled_custom_ms,
        max_abs=forward_max_abs,
        max_rel=forward_max_rel,
    )

    x_ref = x.detach().clone().requires_grad_(True)
    gate_up_weight_ref = gate_up_weight.detach().clone().requires_grad_(True)
    down_weight_t_ref = down_weight_t.detach().clone().requires_grad_(True)
    norm_weight_ref = norm_weight.detach().clone().requires_grad_(True)
    scale_ref = scale.detach().clone().requires_grad_(True)

    x_custom = x.detach().clone().requires_grad_(True)
    gate_up_weight_custom = gate_up_weight.detach().clone().requires_grad_(True)
    down_weight_t_custom = down_weight_t.detach().clone().requires_grad_(True)
    norm_weight_custom = norm_weight.detach().clone().requires_grad_(True)
    scale_custom = scale.detach().clone().requires_grad_(True)

    ref_grads = torch.autograd.grad(
        loss_reference(
            x_ref,
            gate_up_weight_ref,
            down_weight_t_ref,
            norm_weight_ref,
            scale_ref,
            grad_out,
            eps,
        ),
        (
            x_ref,
            gate_up_weight_ref,
            down_weight_t_ref,
            norm_weight_ref,
            scale_ref,
        ),
    )
    custom_grads = torch.autograd.grad(
        loss_custom(
            op,
            x_custom,
            gate_up_weight_custom,
            down_weight_t_custom,
            norm_weight_custom,
            scale_custom,
            grad_out,
            eps,
        ),
        (
            x_custom,
            gate_up_weight_custom,
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
                gate_up_weight_ref,
                down_weight_t_ref,
                norm_weight_ref,
                scale_ref,
                grad_out,
                eps,
            ),
            (
                x_ref,
                gate_up_weight_ref,
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
                gate_up_weight_ref,
                down_weight_t_ref,
                norm_weight_ref,
                scale_ref,
                grad_out,
                eps,
            ),
            (
                x_ref,
                gate_up_weight_ref,
                down_weight_t_ref,
                norm_weight_ref,
                scale_ref,
            ),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_custom_loss = torch.compile(
        lambda x_, gu_w_, down_w_t_, norm_w_, scale_, grad_out_, eps_: loss_custom(
            op,
            x_,
            gu_w_,
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
                gate_up_weight_custom,
                down_weight_t_custom,
                norm_weight_custom,
                scale_custom,
                grad_out,
                eps,
            ),
            (
                x_custom,
                gate_up_weight_custom,
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
                gate_up_weight_custom,
                down_weight_t_custom,
                norm_weight_custom,
                scale_custom,
                grad_out,
                eps,
            ),
            (
                x_custom,
                gate_up_weight_custom,
                down_weight_t_custom,
                norm_weight_custom,
                scale_custom,
            ),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    backward_summary = summarize_case(
        case="mlp_block_backward",
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
