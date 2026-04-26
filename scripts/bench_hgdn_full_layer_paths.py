#!/usr/bin/env python3
"""Time full custom HGDN GDN layers against native FLA GatedDeltaNet.

This is the fairer calibration companion to `bench_fla_recurrence_paths.py`:
it times the complete custom `model.GatedDeltaNet` layer under each public FLA
recurrence mode, plus a native `fla.layers.GatedDeltaNet` layer with matching
shape. No MLP or residual shell is included.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median

import torch
from torch import nn

from _repo_bootstrap import ensure_repo_root_on_sys_path

REPO_ROOT = ensure_repo_root_on_sys_path()

from hgdn_runtime_utils import restore_low_dim_params_to_fp32  # noqa: E402
from model import GDN_FLA_RECURRENCE_MODES, GatedDeltaNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-k-dim", type=int, default=48)
    parser.add_argument("--expand-v", type=float, default=2.0)
    parser.add_argument("--conv-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--custom-mode",
        action="append",
        choices=sorted(GDN_FLA_RECURRENCE_MODES),
        help="Custom recurrence mode to benchmark. Defaults to all modes.",
    )
    parser.add_argument(
        "--skip-native-layer",
        action="store_true",
        help="Only benchmark custom HGDN full-layer modes.",
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


def nonzero_custom_output_projection(layer: GatedDeltaNet) -> None:
    """Avoid zero-output backward degeneracy in an isolated layer benchmark.

    :param GatedDeltaNet layer: Custom GDN layer.
    """
    with torch.no_grad():
        layer.w_out.weight.normal_(mean=0.0, std=0.02)


def build_custom_layer(args: argparse.Namespace, mode: str) -> nn.Module:
    """Build one custom HGDN GDN layer.

    :param argparse.Namespace args: Parsed arguments.
    :param str mode: FLA recurrence mode.
    :return nn.Module: CUDA module ready for timing.
    """
    dtype = dtype_from_name(args.dtype)
    layer = GatedDeltaNet(
        args.model_dim,
        n_heads=args.heads,
        head_k_dim=args.head_k_dim,
        expand_v=args.expand_v,
        allow_neg_eigval=True,
        conv_size=args.conv_size,
        use_fla=True,
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
        conv_output_contiguous=True,
        gates_fp32=True,
        output_norm_fp32=True,
        fla_recurrence_mode=mode,
    )
    layer = layer.cuda().to(dtype=dtype).train()
    restore_low_dim_params_to_fp32(
        layer,
        gdn_control_proj_fp32=True,
        gdn_w_g_optimizer="matrix",
    )
    nonzero_custom_output_projection(layer)
    return layer


def build_native_layer(args: argparse.Namespace) -> nn.Module:
    """Build one native FLA GatedDeltaNet layer.

    :param argparse.Namespace args: Parsed arguments.
    :return nn.Module: CUDA module ready for timing.
    """
    from fla.layers import GatedDeltaNet as NativeGatedDeltaNet

    dtype = dtype_from_name(args.dtype)
    layer = NativeGatedDeltaNet(
        hidden_size=args.model_dim,
        expand_v=args.expand_v,
        head_dim=args.head_k_dim,
        num_heads=args.heads,
        mode="chunk",
        use_gate=True,
        use_short_conv=True,
        allow_neg_eigval=True,
        conv_size=args.conv_size,
        conv_bias=False,
    )
    return layer.cuda().to(dtype=dtype).train()


def make_input(args: argparse.Namespace) -> torch.Tensor:
    """Create one CUDA input tensor.

    :param argparse.Namespace args: Parsed arguments.
    :return torch.Tensor: Input tensor requiring gradients.
    """
    return torch.randn(
        args.batch_size,
        args.seq_len,
        args.model_dim,
        device="cuda",
        dtype=dtype_from_name(args.dtype),
        requires_grad=True,
    )


def make_layer_step(layer: nn.Module, x: torch.Tensor) -> Callable[[], None]:
    """Build one forward/backward timing closure.

    :param nn.Module layer: Layer to time.
    :param torch.Tensor x: Shared input tensor.
    :return Callable[[], None]: Timing closure.
    """

    def step() -> None:
        x.grad = None
        layer.zero_grad(set_to_none=True)
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        out.float().square().mean().backward()

    return step


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


def main() -> None:
    """Run full-layer timing."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-layer GDN timing")
    torch.manual_seed(1337)
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    modes = args.custom_mode or ["direct", "direct_fused", "compile_visible"]
    payload: dict[str, object] = {
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "model_dim": args.model_dim,
            "heads": args.heads,
            "head_k_dim": args.head_k_dim,
            "head_v_dim": int(round(args.head_k_dim * args.expand_v)),
            "conv_size": args.conv_size,
            "dtype": args.dtype,
        },
        "results": {},
    }
    results = payload["results"]
    assert isinstance(results, dict)
    for mode in modes:
        layer = build_custom_layer(args, mode)
        x = make_input(args)
        results[f"custom_hgdn_{mode}"] = bench_cuda(
            make_layer_step(layer, x),
            warmup=args.warmup,
            iters=args.iters,
        )
        del layer, x
        torch.cuda.empty_cache()
    if not args.skip_native_layer:
        native_layer = build_native_layer(args)
        native_x = make_input(args)
        results["native_fla_gated_deltanet"] = bench_cuda(
            make_layer_step(native_layer, native_x),
            warmup=args.warmup,
            iters=args.iters,
        )

    print(
        "device:"
        f"{payload['device']} cc:{payload['compute_capability']} "
        f"shape:B{args.batch_size} T{args.seq_len} H{args.heads} "
        f"Dk{args.head_k_dim} Dv{int(round(args.head_k_dim * args.expand_v))} "
        f"Dmodel{args.model_dim} dtype:{args.dtype}"
    )
    for name, row in results.items():
        print(
            f"{name}: mean_ms={row['mean_ms']:.3f} median_ms={row['median_ms']:.3f} "
            f"min_ms={row['min_ms']:.3f} max_ms={row['max_ms']:.3f}"
        )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
