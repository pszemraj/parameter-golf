#!/usr/bin/env python3
"""Benchmark a larger FA2 attention-core boundary.

This targets the branch up to, but not including, the output projection:

- qkv projection from normalized model states
- q/k RMSNorm + RoPE + q-gain
- FlashAttention-2

The goal is to test whether the qkv/prologue/attention boundary is promising on
its own, without bundling the post-attention output projection epilogue.
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

try:
    from flash_attn import flash_attn_func
    import flash_attn.flash_attn_interface as flash_attn_interface
except Exception:  # pragma: no cover
    flash_attn_func = None
    flash_attn_interface = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from allama_shared import (  # noqa: E402
    apply_rotary_emb,
    register_allama_attention_prep_cpp_custom_op,
    register_allama_attention_prep_custom_op,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the FA2 attention-core benchmark."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs_allama_validation/attention_core_fa2"),
        help="Directory for benchmark JSON outputs.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=20,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--measured-iters",
        type=int,
        default=50,
        help="Measured iterations for timing.",
    )
    parser.add_argument(
        "--prep-kernel",
        choices=("triton", "cpp"),
        default="triton",
        help="Prep kernel implementation to benchmark inside the FA2 core path.",
    )
    return parser.parse_args()


def reset_compile_state() -> None:
    """Reset compiler caches between measured cases."""
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        torch.compiler.reset()
    else:
        torch._dynamo.reset()


def measure_ms(fn: Callable[[], Any], warmup_iters: int, measured_iters: int) -> float:
    """Measure mean CUDA runtime for a callable."""
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measured_iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / measured_iters


def rotary_cache(
    seq_len: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct 2D RoPE tables matching the FA2-native path."""
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    return cos, sin


def fa2_forward_capture(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run FA2 forward and return the saved state needed for direct backward."""
    if flash_attn_interface is None:
        raise RuntimeError("flash-attn interface is required for direct FA2 kernels")
    out, softmax_lse, _s_dmask, rng_state = (
        flash_attn_interface._wrapped_flash_attn_forward(
            q,
            k,
            v,
            0.0,
            q.size(-1) ** (-0.5),
            causal=True,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    )
    return out, softmax_lse, rng_state


def fa2_backward_direct(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    rng_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the direct FA2 backward entrypoint without autograd re-entry."""
    if flash_attn_interface is None:
        raise RuntimeError("flash-attn interface is required for direct FA2 kernels")
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    flash_attn_interface._wrapped_flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        0.0,
        q.size(-1) ** (-0.5),
        True,
        -1,
        -1,
        0.0,
        None,
        False,
        rng_state=rng_state,
    )
    return dq, dk, dv


def attention_core_reference(
    attn_in: torch.Tensor,
    qkv_weight_t: torch.Tensor,
    q_gain: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reference PyTorch implementation of the FA2 attention core."""
    if flash_attn_func is None:
        raise RuntimeError("flash-attn is required for this benchmark")
    bsz, seqlen, model_dim = attn_in.shape
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv = attn_in.view(-1, model_dim) @ qkv_weight_t
    qkv = qkv.view(bsz, seqlen, q_dim + 2 * kv_dim)
    q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
    q = q.view(bsz, seqlen, num_heads, head_dim)
    k = k.view(bsz, seqlen, num_kv_heads, head_dim)
    v = v.view(bsz, seqlen, num_kv_heads, head_dim)
    q = F.rms_norm(q, (head_dim,))
    k = F.rms_norm(k, (head_dim,))
    q = apply_rotary_emb(q, cos[None, :, None, :], sin[None, :, None, :])
    k = apply_rotary_emb(k, cos[None, :, None, :], sin[None, :, None, :])
    q = q * q_gain.to(dtype=q.dtype)[None, None, :, None]
    return flash_attn_func(q, k, v, causal=True)


def register_attention_core_custom_op(
    prep_op: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    prep_backward_op: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., torch.Tensor]:
    """Register a benchmark-local FA2 attention-core op."""

    @torch.library.triton_op(
        "allama_triton_bench::attention_core_forward_fa2",
        mutates_args=(),
    )
    def attention_core_forward_op(
        attn_in: torch.Tensor,
        qkv_weight_t: torch.Tensor,
        q_gain: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        bsz, seqlen, model_dim = attn_in.shape
        qkv = attn_in.view(-1, model_dim) @ qkv_weight_t
        qkv = qkv.view(bsz, seqlen, qkv_weight_t.size(1))
        q, k, v = prep_op(
            qkv,
            q_gain,
            cos,
            sin,
            num_heads,
            num_kv_heads,
            head_dim,
        )
        out, softmax_lse, rng_state = fa2_forward_capture(q, k, v)
        return qkv, q, k, v, out, softmax_lse, rng_state

    @attention_core_forward_op.register_fake
    def _attention_core_forward_op_fake(
        attn_in: torch.Tensor,
        qkv_weight_t: torch.Tensor,
        q_gain: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        bsz, seqlen, _ = attn_in.shape
        qkv = attn_in.new_empty((bsz, seqlen, qkv_weight_t.size(1)))
        q = attn_in.new_empty((bsz, seqlen, num_heads, head_dim))
        k = attn_in.new_empty((bsz, seqlen, num_kv_heads, head_dim))
        v = attn_in.new_empty((bsz, seqlen, num_kv_heads, head_dim))
        out = attn_in.new_empty((bsz, seqlen, num_heads, head_dim))
        softmax_lse = torch.empty(
            (bsz, num_heads, seqlen), device=attn_in.device, dtype=torch.float32
        )
        rng_state = torch.empty((2,), device=attn_in.device, dtype=torch.int64)
        del q_gain, cos, sin
        return qkv, q, k, v, out, softmax_lse, rng_state

    @torch.library.triton_op(
        "allama_triton_bench::attention_core_fa2",
        mutates_args=(),
    )
    def attention_core_op(
        attn_in: torch.Tensor,
        qkv_weight_t: torch.Tensor,
        q_gain: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        return attention_core_forward_op(
            attn_in,
            qkv_weight_t,
            q_gain,
            cos,
            sin,
            num_heads,
            num_kv_heads,
            head_dim,
        )[4]

    @attention_core_op.register_fake
    def _attention_core_op_fake(
        attn_in: torch.Tensor,
        qkv_weight_t: torch.Tensor,
        q_gain: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        del qkv_weight_t, q_gain, cos, sin
        return attn_in.new_empty(
            (attn_in.size(0), attn_in.size(1), num_heads, head_dim)
        )

    def setup_context(
        ctx: Any,
        inputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            int,
            int,
        ],
        output: torch.Tensor,
    ) -> None:
        del output
        attn_in, qkv_weight_t, q_gain, cos, sin, num_heads, num_kv_heads, head_dim = (
            inputs
        )
        ctx.save_for_backward(attn_in, qkv_weight_t, q_gain, cos, sin)
        ctx.num_heads = int(num_heads)
        ctx.num_kv_heads = int(num_kv_heads)
        ctx.head_dim = int(head_dim)

    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        attn_in, qkv_weight_t, q_gain, cos, sin = ctx.saved_tensors
        qkv, q, k, v, out, softmax_lse, rng_state = attention_core_forward_op(
            attn_in,
            qkv_weight_t,
            q_gain,
            cos,
            sin,
            ctx.num_heads,
            ctx.num_kv_heads,
            ctx.head_dim,
        )
        grad_q, grad_k, grad_v = fa2_backward_direct(
            grad_out,
            q,
            k,
            v,
            out,
            softmax_lse,
            rng_state,
        )
        grad_qkv, grad_q_gain = prep_backward_op(
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
        grad_attn_in = grad_qkv.view(-1, grad_qkv.size(-1)) @ qkv_weight_t.transpose(
            0, 1
        )
        grad_attn_in = grad_attn_in.view_as(attn_in)
        grad_qkv_weight_t = attn_in.view(-1, attn_in.size(-1)).transpose(
            0, 1
        ) @ grad_qkv.view(-1, grad_qkv.size(-1))
        return (
            grad_attn_in,
            grad_qkv_weight_t,
            grad_q_gain,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        "allama_triton_bench::attention_core_fa2",
        backward,
        setup_context=setup_context,
    )
    return attention_core_op


def summarize_case(
    *,
    case: str,
    reference_ms: float,
    compiled_reference_ms: float,
    custom_ms: float,
    compiled_custom_ms: float,
    max_abs: float,
    max_rel: float,
    error: str = "",
) -> dict[str, float | str]:
    """Create a compact benchmark summary for one core case."""
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
        "error": error,
    }


def main() -> None:
    """Build and benchmark the larger FA2 attention-core boundary."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FA2 attention-core benchmark")
    if flash_attn_func is None or flash_attn_interface is None:
        raise RuntimeError("flash-attn is required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(1337)
    if args.prep_kernel == "triton":
        prep_op = register_allama_attention_prep_custom_op()
        prep_backward_op = torch.ops.allama_triton.attention_prep_bshd_backward
    else:
        prep_op = register_allama_attention_prep_cpp_custom_op()
        prep_backward_op = torch.ops.allama_cpp.attention_prep_bshd_backward
    if prep_op is None:
        raise RuntimeError(
            f"expected attention prep op for {args.prep_kernel!r} to be available"
        )
    core_op = register_attention_core_custom_op(prep_op, prep_backward_op)

    batch = 4
    seq_len = 1024
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    model_dim = num_heads * head_dim
    qkv_dim = model_dim + 2 * (num_kv_heads * head_dim)

    attn_in = torch.randn(
        batch, seq_len, model_dim, device=device, dtype=torch.bfloat16
    )
    qkv_weight_t = (
        torch.randn(model_dim, qkv_dim, device=device, dtype=torch.bfloat16)
        / model_dim**0.5
    )
    q_gain = torch.randn(num_heads, device=device, dtype=torch.float32)
    cos, sin = rotary_cache(seq_len, head_dim, device=device, dtype=torch.bfloat16)
    grad_out = torch.randn(
        batch, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    ref_out = attention_core_reference(
        attn_in,
        qkv_weight_t,
        q_gain,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    custom_out = core_op(
        attn_in,
        qkv_weight_t,
        q_gain,
        cos,
        sin,
        num_heads,
        num_kv_heads,
        head_dim,
    )
    forward_max_abs = (ref_out.float() - custom_out.float()).abs().max().item()
    forward_max_rel = (
        (ref_out.float() - custom_out.float()).abs().max()
        / ref_out.float().abs().max().clamp_min(1e-6)
    ).item()

    reset_compile_state()
    ref_ms = measure_ms(
        lambda: attention_core_reference(
            attn_in,
            qkv_weight_t,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    reset_compile_state()
    compiled_ref = torch.compile(
        attention_core_reference, dynamic=False, fullgraph=True
    )
    compiled_ref_ms = measure_ms(
        lambda: compiled_ref(
            attn_in,
            qkv_weight_t,
            q_gain,
            cos,
            sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    reset_compile_state()
    custom_ms = measure_ms(
        lambda: core_op(
            attn_in,
            qkv_weight_t,
            q_gain,
            cos,
            sin,
            num_heads,
            num_kv_heads,
            head_dim,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    reset_compile_state()
    compiled_custom = torch.compile(core_op, dynamic=False, fullgraph=True)
    compiled_custom_ms = measure_ms(
        lambda: compiled_custom(
            attn_in,
            qkv_weight_t,
            q_gain,
            cos,
            sin,
            num_heads,
            num_kv_heads,
            head_dim,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )

    forward_summary = summarize_case(
        case="attention_core_forward",
        reference_ms=ref_ms,
        compiled_reference_ms=compiled_ref_ms,
        custom_ms=custom_ms,
        compiled_custom_ms=compiled_custom_ms,
        max_abs=forward_max_abs,
        max_rel=forward_max_rel,
    )

    def loss_reference(
        attn_in_: torch.Tensor,
        qkv_weight_t_: torch.Tensor,
        q_gain_: torch.Tensor,
    ) -> torch.Tensor:
        return (
            attention_core_reference(
                attn_in_,
                qkv_weight_t_,
                q_gain_,
                cos,
                sin,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            * grad_out
        ).sum()

    def loss_custom(
        attn_in_: torch.Tensor,
        qkv_weight_t_: torch.Tensor,
        q_gain_: torch.Tensor,
    ) -> torch.Tensor:
        return (
            core_op(
                attn_in_,
                qkv_weight_t_,
                q_gain_,
                cos,
                sin,
                num_heads,
                num_kv_heads,
                head_dim,
            )
            * grad_out
        ).sum()

    attn_in_ref = attn_in.detach().clone().requires_grad_(True)
    qkv_weight_t_ref = qkv_weight_t.detach().clone().requires_grad_(True)
    q_gain_ref = q_gain.detach().clone().requires_grad_(True)
    ref_grads = torch.autograd.grad(
        loss_reference(
            attn_in_ref,
            qkv_weight_t_ref,
            q_gain_ref,
        ),
        (attn_in_ref, qkv_weight_t_ref, q_gain_ref),
    )

    attn_in_custom = attn_in.detach().clone().requires_grad_(True)
    qkv_weight_t_custom = qkv_weight_t.detach().clone().requires_grad_(True)
    q_gain_custom = q_gain.detach().clone().requires_grad_(True)
    custom_grads = torch.autograd.grad(
        loss_custom(
            attn_in_custom,
            qkv_weight_t_custom,
            q_gain_custom,
        ),
        (attn_in_custom, qkv_weight_t_custom, q_gain_custom),
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

    reset_compile_state()
    ref_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            loss_reference(attn_in_ref, qkv_weight_t_ref, q_gain_ref),
            (attn_in_ref, qkv_weight_t_ref, q_gain_ref),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    reset_compile_state()
    compiled_ref_loss = torch.compile(loss_reference, dynamic=False, fullgraph=True)
    compiled_ref_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_ref_loss(attn_in_ref, qkv_weight_t_ref, q_gain_ref),
            (attn_in_ref, qkv_weight_t_ref, q_gain_ref),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    reset_compile_state()
    custom_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            loss_custom(attn_in_custom, qkv_weight_t_custom, q_gain_custom),
            (attn_in_custom, qkv_weight_t_custom, q_gain_custom),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    reset_compile_state()
    compiled_custom_loss = torch.compile(loss_custom, dynamic=False, fullgraph=True)
    compiled_custom_backward_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_custom_loss(
                attn_in_custom,
                qkv_weight_t_custom,
                q_gain_custom,
            ),
            (attn_in_custom, qkv_weight_t_custom, q_gain_custom),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )

    backward_summary = summarize_case(
        case="attention_core_backward",
        reference_ms=ref_backward_ms,
        compiled_reference_ms=compiled_ref_backward_ms,
        custom_ms=custom_backward_ms,
        compiled_custom_ms=compiled_custom_backward_ms,
        max_abs=backward_max_abs,
        max_rel=backward_max_rel,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "prep_kernel": args.prep_kernel,
        "shape": {
            "batch": batch,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "model_dim": model_dim,
            "qkv_dim": qkv_dim,
        },
        "torch_version": torch.__version__,
        "forward": forward_summary,
        "backward": backward_summary,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
