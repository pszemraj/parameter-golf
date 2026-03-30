#!/usr/bin/env python3
"""Microbench the hottest ALlama kernels under representative anchor shapes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class HotKernelSummary:
    """Serializable hot-kernel microbenchmark summary.

    :param str case: Benchmark case name.
    :param int warmup_iters: Warmup iterations before timing.
    :param int measured_iters: Timed or profiled iterations.
    :param float mean_iter_ms: Mean iteration time in milliseconds.
    :param int peak_cuda_mem_bytes: Peak CUDA memory during measured iterations.
    """

    case: str
    warmup_iters: int
    measured_iters: int
    mean_iter_ms: float
    peak_cuda_mem_bytes: int


ALLAMA_FLASH_Q = (4, 14, 1024, 64)
ALLAMA_FLASH_KV = (4, 2, 1024, 64)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the hot-kernel microbench.

    :return argparse.Namespace: Parsed command-line configuration.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--case",
        choices=(
            "allama_mlp_up",
            "allama_mlp_down",
            "allama_attn_proj",
            "allama_qkv",
            "allama_flash_bwd",
        ),
        required=True,
        help="Representative kernel case to benchmark.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Warmup iterations before timing or Nsight capture.",
    )
    parser.add_argument(
        "--measured-iters",
        type=int,
        default=20,
        help="Timed or profiled iterations after warmup.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible synthetic tensors.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs_allama_validation/hot_kernels"),
        help="Directory for JSON summaries.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Bracket the measured region with cudaProfilerStart/Stop for Nsight.",
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to disk, creating parents first.

    :param Path path: File path to write.
    :param str text: UTF-8 text payload.
    :return None: Writes the file to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def cuda_profiler_toggle(enabled: bool, start: bool) -> None:
    """Start or stop the CUDA profiler API when requested.

    :param bool enabled: Whether profiler-range control is active.
    :param bool start: Whether to start or stop the CUDA profiler.
    :return None: Issues CUDA profiler API calls when requested.
    """
    if not enabled:
        return
    cudart = torch.cuda.cudart()
    if start:
        cudart.cudaProfilerStart()
    else:
        cudart.cudaProfilerStop()


def set_fast_math() -> None:
    """Enable the same CUDA fast-math knobs used by the trainers.

    :return None: Updates global CUDA backend settings.
    """
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"


def make_linear_case(
    in_features: int,
    out_features: int,
    batch_rows: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Create bf16 linear operands with gradients enabled.

    :param int in_features: Input feature width.
    :param int out_features: Output feature width.
    :param int batch_rows: Flattened leading rows.
    :param torch.device device: CUDA device for tensors.
    :return tuple[Tensor, Tensor]: Input activations and linear weight.
    """
    x = torch.randn(
        batch_rows,
        in_features,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    w = torch.randn(
        out_features,
        in_features,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    return x, w


def step_linear(x: Tensor, w: Tensor) -> Tensor:
    """Run one forward/backward linear training iteration.

    :param Tensor x: Input activation matrix.
    :param Tensor w: Weight matrix shaped `(out, in)`.
    :return Tensor: Scalar loss for the iteration.
    """
    y = F.linear(x, w)
    loss = y.square().mean()
    loss.backward()
    return loss


def make_flash_case(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Create representative ALlama flash-attention tensors.

    :param torch.device device: CUDA device for tensors.
    :return tuple[Tensor, Tensor, Tensor]: Query, key, and value tensors.
    """
    q = torch.randn(
        ALLAMA_FLASH_Q,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        ALLAMA_FLASH_KV,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn(
        ALLAMA_FLASH_KV,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    return q, k, v


def step_flash(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Run one forward/backward flash-attention training iteration.

    :param Tensor q: Query tensor.
    :param Tensor k: Key tensor.
    :param Tensor v: Value tensor.
    :return Tensor: Scalar loss for the iteration.
    """
    y = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=True,
        enable_gqa=True,
    )
    loss = y.square().mean()
    loss.backward()
    return loss


def zero_grads(*tensors: Tensor) -> None:
    """Clear gradients on the provided tensors.

    :param Tensor tensors: Tensors whose gradients should be cleared.
    :return None: Clears gradients in place.
    """
    for tensor in tensors:
        tensor.grad = None


def run_case(args: argparse.Namespace, device: torch.device) -> HotKernelSummary:
    """Run the selected hot-kernel microbenchmark.

    :param argparse.Namespace args: Parsed CLI arguments.
    :param torch.device device: CUDA device for tensors.
    :return HotKernelSummary: Serializable microbenchmark summary.
    """
    if args.case == "allama_mlp_up":
        tensors = make_linear_case(896, 2816, 4096, device)
        step_fn = step_linear
    elif args.case == "allama_mlp_down":
        tensors = make_linear_case(2816, 896, 4096, device)
        step_fn = step_linear
    elif args.case == "allama_attn_proj":
        tensors = make_linear_case(896, 896, 4096, device)
        step_fn = step_linear
    elif args.case == "allama_qkv":
        tensors = make_linear_case(896, 1152, 4096, device)
        step_fn = step_linear
    elif args.case == "allama_flash_bwd":
        tensors = make_flash_case(device)
        step_fn = step_flash
    else:
        raise ValueError(f"unknown benchmark case {args.case!r}")

    for _ in range(args.warmup_iters):
        zero_grads(*tensors)
        step_fn(*tensors)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.nvtx.range_push(args.case)
    cuda_profiler_toggle(args.cuda_profiler_range, start=True)
    try:
        t0 = time.perf_counter()
        for _ in range(args.measured_iters):
            zero_grads(*tensors)
            step_fn(*tensors)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    finally:
        cuda_profiler_toggle(args.cuda_profiler_range, start=False)
        torch.cuda.nvtx.range_pop()
    return HotKernelSummary(
        case=str(args.case),
        warmup_iters=int(args.warmup_iters),
        measured_iters=int(args.measured_iters),
        mean_iter_ms=float(1000.0 * elapsed / args.measured_iters),
        peak_cuda_mem_bytes=int(torch.cuda.max_memory_allocated()),
    )


def main() -> None:
    """Run the requested hot-kernel microbenchmark and write a JSON summary.

    :return None: Writes run artifacts under the requested output directory.
    """
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the hot-kernel microbench")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    set_fast_math()
    summary = run_case(args, device=device)
    payload = {
        "summary": asdict(summary),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
    }
    write_text(
        args.out_dir / f"{args.case}_summary.json",
        json.dumps(payload, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
