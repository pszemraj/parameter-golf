#!/usr/bin/env python3
"""Time HGDN recurrence wrapper, direct public FLA, and native FLA layer paths."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median

import torch
import torch.nn.functional as F

from _repo_bootstrap import ensure_repo_root_on_sys_path

REPO_ROOT = ensure_repo_root_on_sys_path()

from hgdn_cuda import (  # noqa: E402
    fla_chunk_gated_delta_rule_compile_visible,
    fla_chunk_gated_delta_rule_direct,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-k-dim", type=int, default=48)
    parser.add_argument("--expand-v", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--native-head-dim", type=int, default=None)
    parser.add_argument("--native-expand-v", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--skip-native-layer",
        action="store_true",
        help="Only benchmark recurrence-boundary wrapper/direct paths.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    """Map a CLI dtype name to a torch dtype.

    :param str name: CLI dtype name.
    :return torch.dtype: Torch dtype.
    """
    return torch.bfloat16 if name == "bf16" else torch.float16


def sync() -> None:
    """Synchronize the current CUDA device."""
    torch.cuda.synchronize()


def bench_cuda(fn: Callable[[], None], *, warmup: int, iters: int) -> dict[str, float]:
    """Time one CUDA callable with event timing.

    :param Callable[[], None] fn: Callable that runs forward and backward.
    :param int warmup: Untimed warmup iterations.
    :param int iters: Timed iterations.
    :return dict[str, float]: Timing summary in milliseconds.
    """
    for _ in range(warmup):
        fn()
    sync()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        sync()
        samples.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": mean(samples),
        "median_ms": median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "iters": float(iters),
    }


def make_recurrence_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    """Create recurrence-boundary tensors shared by wrapper/direct paths.

    :param argparse.Namespace args: Parsed arguments.
    :return tuple[torch.Tensor, ...]: `q, k, v, g, beta` CUDA leaf tensors.
    """
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    b, t, h, dk = args.batch_size, args.seq_len, args.heads, args.head_k_dim
    dv = int(round(dk * args.expand_v))
    q = F.normalize(torch.randn(b, t, h, dk, device=device, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(b, t, h, dk, device=device, dtype=dtype), dim=-1)
    v = torch.randn(b, t, h, dv, device=device, dtype=dtype)
    g = -torch.rand(b, t, h, device=device, dtype=dtype).clamp_min(1e-4)
    beta = torch.rand(b, t, h, device=device, dtype=dtype)
    return tuple(x.detach().requires_grad_(True) for x in (q, k, v, g, beta))


def zero_leaf_grads(tensors: tuple[torch.Tensor, ...]) -> None:
    """Clear gradients on leaf tensors.

    :param tuple[torch.Tensor, ...] tensors: Leaf tensors.
    """
    for tensor in tensors:
        tensor.grad = None


def make_recurrence_step(
    op: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    tensors: tuple[torch.Tensor, ...],
) -> Callable[[], None]:
    """Build one recurrence forward/backward benchmark step.

    :param Callable op: Recurrence op returning an output tensor.
    :param tuple[torch.Tensor, ...] tensors: Shared recurrence inputs.
    :return Callable[[], None]: Benchmark closure.
    """

    def step() -> None:
        zero_leaf_grads(tensors)
        q, k, v, g, beta = tensors
        out = op(q, k, v, g, beta)
        out.float().square().mean().backward()

    return step


def make_native_layer_step(args: argparse.Namespace) -> Callable[[], None]:
    """Build one native `fla.layers.GatedDeltaNet` forward/backward step.

    :param argparse.Namespace args: Parsed arguments.
    :return Callable[[], None]: Benchmark closure.
    """
    from fla.layers import GatedDeltaNet

    dtype = dtype_from_name(args.dtype)
    head_dim = args.native_head_dim or args.head_k_dim
    expand_v = (
        args.native_expand_v if args.native_expand_v is not None else args.expand_v
    )
    layer = (
        GatedDeltaNet(
            hidden_size=args.model_dim,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=args.heads,
            mode="chunk",
            use_gate=True,
            use_short_conv=True,
            allow_neg_eigval=True,
            conv_size=4,
            conv_bias=False,
        )
        .cuda()
        .to(dtype=dtype)
        .train()
    )
    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.model_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    def step() -> None:
        x.grad = None
        layer.zero_grad(set_to_none=True)
        out, _attn, _cache = layer(x)
        out.float().square().mean().backward()

    return step


def recurrence_parity(tensors: tuple[torch.Tensor, ...]) -> dict[str, float]:
    """Compare wrapper and direct recurrence outputs once.

    :param tuple[torch.Tensor, ...] tensors: Shared recurrence inputs.
    :return dict[str, float]: Error summary.
    """
    with torch.no_grad():
        q, k, v, g, beta = tensors
        wrapper = fla_chunk_gated_delta_rule_compile_visible(q, k, v, g, beta).float()
        direct = fla_chunk_gated_delta_rule_direct(q, k, v, g, beta).float()
        delta = wrapper - direct
        return {
            "max_abs": float(delta.abs().max().item()),
            "rmse": float(delta.square().mean().sqrt().item()),
            "norm_rel": float(
                delta.norm().item() / direct.norm().clamp_min(1e-6).item()
            ),
        }


def main() -> None:
    """Run the benchmark and print a compact report."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLA recurrence timing")
    torch.manual_seed(1337)
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    tensors = make_recurrence_inputs(args)
    payload: dict[str, object] = {
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": int(round(args.head_k_dim * args.expand_v)),
            "model_dim": args.model_dim,
            "dtype": args.dtype,
        },
        "parity": recurrence_parity(tensors),
        "results": {},
    }
    results = payload["results"]
    assert isinstance(results, dict)
    results["wrapper_recurrence"] = bench_cuda(
        make_recurrence_step(fla_chunk_gated_delta_rule_compile_visible, tensors),
        warmup=args.warmup,
        iters=args.iters,
    )
    results["direct_recurrence"] = bench_cuda(
        make_recurrence_step(fla_chunk_gated_delta_rule_direct, tensors),
        warmup=args.warmup,
        iters=args.iters,
    )
    if not args.skip_native_layer:
        results["native_fla_layer"] = bench_cuda(
            make_native_layer_step(args),
            warmup=args.warmup,
            iters=args.iters,
        )

    print(
        "device:"
        f"{payload['device']} cc:{payload['compute_capability']} "
        f"shape:B{args.batch_size} T{args.seq_len} H{args.heads} "
        f"Dk{args.head_k_dim} Dv{int(round(args.head_k_dim * args.expand_v))} "
        f"dtype:{args.dtype}"
    )
    parity = payload["parity"]
    assert isinstance(parity, dict)
    print(
        "parity:"
        f"wrapper_vs_direct max_abs={parity['max_abs']:.6g} "
        f"rmse={parity['rmse']:.6g} norm_rel={parity['norm_rel']:.6g}"
    )
    for name, row in results.items():
        print(
            f"{name}: mean_ms={row['mean_ms']:.3f} median_ms={row['median_ms']:.3f} "
            f"min_ms={row['min_ms']:.3f} max_ms={row['max_ms']:.3f}"
        )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
