#!/usr/bin/env python3
"""Benchmark dense causal GQA attention backends on ALlama anchor shapes.

This script compares the current raw attention options that matter for ALlama:

- SDPA flash
- SDPA cuDNN
- FlexAttention Triton backends
- FlexAttention FLASH backend when CuTe/FA4 is installed

The benchmark uses the real ALlama attention shape:
`B=4, H=14, H_kv=2, S=1024, D=64`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the flex-attention backend benchmark."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs_allama_validation/flex_attention"),
        help="Directory for JSON outputs.",
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
        help="Measured iterations per case.",
    )
    return parser.parse_args()


def measure_ms(fn, warmup_iters: int, measured_iters: int) -> float:
    """Measure mean CUDA runtime for a callable in milliseconds."""
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measured_iters):
        fn()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / measured_iters


def causal_mask(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
):
    """Return a standard causal mask mod for FlexAttention block-mask creation."""
    del b, h
    return q_idx >= kv_idx


def configure_sdpa_backend(name: str) -> None:
    """Set the active SDPA backend knobs for a raw attention benchmark."""
    torch.backends.cuda.enable_flash_sdp(name == "flash")
    torch.backends.cuda.enable_math_sdp(name == "math")
    torch.backends.cuda.enable_mem_efficient_sdp(name == "efficient")
    torch.backends.cuda.enable_cudnn_sdp(name == "cudnn")


def make_qkv(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create representative ALlama GQA tensors."""
    q = torch.randn(
        (4, 14, 1024, 64),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        (4, 2, 1024, 64),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn(
        (4, 2, 1024, 64),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    return q, k, v


def clone_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clone benchmark tensors with fresh grad history."""
    return (
        q.detach().clone().requires_grad_(True),
        k.detach().clone().requires_grad_(True),
        v.detach().clone().requires_grad_(True),
    )


def run_forward_backward(
    case_fn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> None:
    """Run one forward/backward iteration and clear gradients."""
    out = case_fn(q, k, v)
    loss = out.square().mean()
    loss.backward()
    q.grad = None
    k.grad = None
    v.grad = None


def benchmark_case(
    name: str,
    compile_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_iters: int,
    measured_iters: int,
) -> dict[str, Any]:
    """Compile and benchmark one attention case, returning a serializable summary."""
    try:
        compiled = compile_fn()
        q_case, k_case, v_case = clone_qkv(q, k, v)
        run_forward_backward(compiled, q_case, k_case, v_case)

        def run_bwd():
            run_forward_backward(compiled, q_case, k_case, v_case)

        q_fwd, k_fwd, v_fwd = clone_qkv(q, k, v)

        def run_fwd():
            compiled(q_fwd, k_fwd, v_fwd)

        return {
            "name": name,
            "forward_ms": float(measure_ms(run_fwd, warmup_iters, measured_iters)),
            "forward_backward_ms": float(
                measure_ms(run_bwd, warmup_iters, measured_iters)
            ),
        }
    except Exception as exc:
        return {
            "name": name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main() -> None:
    """Run the dense causal GQA backend benchmark and write a summary JSON."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    q, k, v = make_qkv(device)
    block_mask = torch.compile(create_block_mask, dynamic=False)(
        causal_mask,
        q.size(0),
        None,
        q.size(2),
        k.size(2),
        device=device,
    )

    results: list[dict[str, Any]] = []

    configure_sdpa_backend("flash")
    results.append(
        benchmark_case(
            "sdpa_flash",
            lambda: torch.compile(
                lambda q_, k_, v_: F.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    is_causal=True,
                    enable_gqa=True,
                ),
                dynamic=False,
                fullgraph=True,
            ),
            q,
            k,
            v,
            args.warmup_iters,
            args.measured_iters,
        )
    )

    configure_sdpa_backend("cudnn")
    results.append(
        benchmark_case(
            "sdpa_cudnn",
            lambda: torch.compile(
                lambda q_, k_, v_: F.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    is_causal=True,
                    enable_gqa=True,
                ),
                dynamic=False,
                fullgraph=True,
            ),
            q,
            k,
            v,
            args.warmup_iters,
            args.measured_iters,
        )
    )

    flex_cases = [
        ("flex_triton_auto", None),
        (
            "flex_triton_tuned",
            {"ROWS_GUARANTEED_SAFE": True, "BLOCKS_ARE_CONTIGUOUS": True},
        ),
        (
            "flex_triton_tuned_tma",
            {
                "ROWS_GUARANTEED_SAFE": True,
                "BLOCKS_ARE_CONTIGUOUS": True,
                "USE_TMA": True,
            },
        ),
        ("flex_flash_backend", {"BACKEND": "FLASH"}),
    ]
    for name, kernel_options in flex_cases:
        results.append(
            benchmark_case(
                name,
                lambda kernel_options=kernel_options: torch.compile(
                    lambda q_, k_, v_: flex_attention(
                        q_,
                        k_,
                        v_,
                        block_mask=block_mask,
                        enable_gqa=True,
                        kernel_options=kernel_options,
                    ),
                    dynamic=False,
                    fullgraph=True,
                    mode="max-autotune-no-cudagraphs",
                ),
                q,
                k,
                v,
                args.warmup_iters,
                args.measured_iters,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "shape": {
            "batch": int(q.size(0)),
            "num_heads": int(q.size(1)),
            "num_kv_heads": int(k.size(1)),
            "seq_len": int(q.size(2)),
            "head_dim": int(q.size(3)),
        },
        "block_mask_sparsity": float(block_mask.sparsity()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "results": results,
    }
    out_path = args.out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
