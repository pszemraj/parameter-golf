#!/usr/bin/env python3
"""Build and benchmark candidate ALlama C++/CUDA custom operators.

This stays outside the trainer path on purpose. The goal is to prove a custom
operator can beat the compiled PyTorch baseline on representative ALlama glue
work before we consider integrating it into `allama_shared.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.cpp_extension import load

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the standalone C++/CUDA operator benchmark.

    :return argparse.Namespace: Parsed benchmark configuration.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs_allama_validation/cpp_ops"),
        help="Directory for benchmark JSON outputs.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("./runs_allama_validation/torch_extensions"),
        help="Build cache directory for the C++ extension.",
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


def load_extension(build_dir: Path):
    """Build and import the candidate ALlama C++/CUDA extension.

    :param Path build_dir: Build cache directory.
    :return Any: Imported extension module.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="allama_cpp_ops",
        sources=[
            str(REPO_ROOT / "csrc" / "allama_ops.cpp"),
            str(REPO_ROOT / "csrc" / "allama_residual_scale_rms_norm.cu"),
        ],
        build_directory=str(build_dir),
        extra_cuda_cflags=["--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def residual_scale_rms_norm_reference(
    x: torch.Tensor,
    branch: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference PyTorch implementation of the fused residual+RMSNorm op."""
    scale_cast = scale.to(dtype=x.dtype)[None, None, :]
    mixed = x + scale_cast * branch
    return torch.nn.functional.rms_norm(
        mixed,
        (mixed.size(-1),),
        weight.to(dtype=x.dtype),
        eps=eps,
    )


def residual_scale_rms_norm_pair_reference(
    x: torch.Tensor,
    branch: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference PyTorch implementation that also returns the mixed residual."""
    scale_cast = scale.to(dtype=x.dtype)[None, None, :]
    mixed = x + scale_cast * branch
    normed = torch.nn.functional.rms_norm(
        mixed,
        (mixed.size(-1),),
        weight.to(dtype=x.dtype),
        eps=eps,
    )
    return mixed, normed


def measure_ms(fn: Callable[[], Any], warmup_iters: int, measured_iters: int) -> float:
    """Measure mean CUDA runtime for a callable that launches work.

    :param Callable[[], torch.Tensor] fn: Callable under test.
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


def summarize_case(
    *,
    case: str,
    reference_ms: float,
    compiled_reference_ms: float,
    extension_ms: float,
    max_abs: float,
    max_rel: float,
) -> dict[str, float | str]:
    """Create a compact benchmark summary for one operator case."""
    return {
        "case": case,
        "reference_ms": float(reference_ms),
        "compiled_reference_ms": float(compiled_reference_ms),
        "extension_ms": float(extension_ms),
        "speedup": float(reference_ms / extension_ms) if extension_ms > 0.0 else 0.0,
        "speedup_vs_compiled": (
            float(compiled_reference_ms / extension_ms) if extension_ms > 0.0 else 0.0
        ),
        "max_abs": float(max_abs),
        "max_rel": float(max_rel),
    }


def main() -> None:
    """Build the candidate extension and benchmark it against the reference."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the C++/CUDA operator benchmark")

    device = torch.device("cuda")
    torch.manual_seed(1337)

    ext = load_extension(args.build_dir)
    batch = 4
    seq_len = 1024
    dim = 896
    eps = 1e-5

    x = torch.randn(batch, seq_len, dim, device=device, dtype=torch.bfloat16)
    branch = torch.randn_like(x)
    scale = torch.randn(dim, device=device, dtype=torch.float32)
    weight = torch.randn(dim, device=device, dtype=torch.float32)

    ref_out = residual_scale_rms_norm_reference(x, branch, scale, weight, eps)
    ext_out = ext.residual_scale_rms_norm(x, branch, scale, weight, eps)
    single_max_abs = (ref_out.float() - ext_out.float()).abs().max().item()
    single_max_rel = (
        (ref_out.float() - ext_out.float()).abs().max()
        / ref_out.float().abs().max().clamp_min(1e-6)
    ).item()

    ref_ms = measure_ms(
        lambda: residual_scale_rms_norm_reference(x, branch, scale, weight, eps),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_ref = torch.compile(
        residual_scale_rms_norm_reference,
        dynamic=False,
        fullgraph=True,
    )
    compiled_ref_ms = measure_ms(
        lambda: compiled_ref(x, branch, scale, weight, eps),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    ext_ms = measure_ms(
        lambda: ext.residual_scale_rms_norm(x, branch, scale, weight, eps),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    single_summary = summarize_case(
        case="residual_scale_rms_norm",
        reference_ms=ref_ms,
        compiled_reference_ms=compiled_ref_ms,
        extension_ms=ext_ms,
        max_abs=single_max_abs,
        max_rel=single_max_rel,
    )

    ref_mixed, ref_normed = residual_scale_rms_norm_pair_reference(
        x, branch, scale, weight, eps
    )
    ext_mixed, ext_normed = ext.residual_scale_rms_norm_pair(
        x, branch, scale, weight, eps
    )
    pair_max_abs = max(
        (ref_mixed.float() - ext_mixed.float()).abs().max().item(),
        (ref_normed.float() - ext_normed.float()).abs().max().item(),
    )
    pair_max_rel = max(
        (
            (ref_mixed.float() - ext_mixed.float()).abs().max()
            / ref_mixed.float().abs().max().clamp_min(1e-6)
        ).item(),
        (
            (ref_normed.float() - ext_normed.float()).abs().max()
            / ref_normed.float().abs().max().clamp_min(1e-6)
        ).item(),
    )

    pair_ref_ms = measure_ms(
        lambda: residual_scale_rms_norm_pair_reference(x, branch, scale, weight, eps),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_pair_ref = torch.compile(
        residual_scale_rms_norm_pair_reference,
        dynamic=False,
        fullgraph=True,
    )
    compiled_pair_ref_ms = measure_ms(
        lambda: compiled_pair_ref(x, branch, scale, weight, eps),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    pair_ext_ms = measure_ms(
        lambda: ext.residual_scale_rms_norm_pair(x, branch, scale, weight, eps),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    pair_summary = summarize_case(
        case="residual_scale_rms_norm_pair",
        reference_ms=pair_ref_ms,
        compiled_reference_ms=compiled_pair_ref_ms,
        extension_ms=pair_ext_ms,
        max_abs=pair_max_abs,
        max_rel=pair_max_rel,
    )

    summary = {
        "shape": [batch, seq_len, dim],
        "warmup_iters": int(args.warmup_iters),
        "measured_iters": int(args.measured_iters),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "cases": [single_summary, pair_summary],
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for case_summary in summary["cases"]:
        (out_dir / f"{case_summary['case']}_summary.json").write_text(
            json.dumps(
                {
                    **{k: v for k, v in summary.items() if k != "cases"},
                    **case_summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
