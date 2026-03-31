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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from allama_cpp_extension import load_allama_cpp_extension  # noqa: E402


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
    return load_allama_cpp_extension(str(build_dir))


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


def register_pair_autograd_custom_op(ext: Any) -> Any:
    """Register a benchmark-local custom op with C++ forward and backward."""

    @torch.library.custom_op(
        "allama_cpp_bench::residual_scale_rms_norm_pair_autograd",
        mutates_args=(),
    )
    def pair_autograd_op(
        x: torch.Tensor,
        branch: torch.Tensor,
        scale: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ext.residual_scale_rms_norm_pair(x, branch, scale, weight, eps)

    @pair_autograd_op.register_fake
    def _pair_autograd_op_fake(
        x: torch.Tensor,
        branch: torch.Tensor,
        scale: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del branch, scale, weight, eps
        return x.new_empty(x.shape), x.new_empty(x.shape)

    @torch.library.custom_op(
        "allama_cpp_bench::residual_scale_rms_norm_pair_backward_op",
        mutates_args=(),
    )
    def pair_backward_op(
        mixed: torch.Tensor,
        branch: torch.Tensor,
        scale: torch.Tensor,
        weight: torch.Tensor,
        grad_mixed_out: torch.Tensor,
        grad_normed_out: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return ext.residual_scale_rms_norm_pair_backward(
            mixed,
            branch,
            scale,
            weight,
            grad_mixed_out,
            grad_normed_out,
            eps,
        )

    @pair_backward_op.register_fake
    def _pair_backward_op_fake(
        mixed: torch.Tensor,
        branch: torch.Tensor,
        scale: torch.Tensor,
        weight: torch.Tensor,
        grad_mixed_out: torch.Tensor,
        grad_normed_out: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del grad_mixed_out, grad_normed_out, eps
        return (
            mixed.new_empty(mixed.shape),
            branch.new_empty(branch.shape),
            scale.new_empty(scale.shape),
            weight.new_empty(weight.shape),
        )

    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        _, branch, scale, weight, eps = inputs
        mixed, _ = output
        ctx.save_for_backward(mixed, branch, scale, weight)
        ctx.eps = float(eps)

    def backward(
        ctx: Any,
        grad_mixed_out: torch.Tensor | None,
        grad_normed_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        mixed, branch, scale, weight = ctx.saved_tensors
        if grad_mixed_out is None:
            grad_mixed_out = torch.zeros_like(mixed)
        if grad_normed_out is None:
            grad_normed_out = torch.zeros_like(mixed)
        grad_x, grad_branch, grad_scale, grad_weight = pair_backward_op(
            mixed,
            branch,
            scale,
            weight,
            grad_mixed_out,
            grad_normed_out,
            ctx.eps,
        )
        return grad_x, grad_branch, grad_scale, grad_weight, None

    torch.library.register_autograd(
        "allama_cpp_bench::residual_scale_rms_norm_pair_autograd",
        backward,
        setup_context=setup_context,
    )
    return pair_autograd_op


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


def backward_step_reference(
    x: torch.Tensor,
    branch: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    grad_mixed: torch.Tensor,
    grad_normed: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one reference forward+backward step for the pair boundary."""
    mixed, normed = residual_scale_rms_norm_pair_reference(
        x, branch, scale, weight, eps
    )
    loss = (mixed * grad_mixed).sum() + (normed * grad_normed).sum()
    return torch.autograd.grad(loss, (x, branch, scale, weight))


def pair_loss_reference(
    x: torch.Tensor,
    branch: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    grad_mixed: torch.Tensor,
    grad_normed: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return a scalar loss that exercises both pair outputs."""
    mixed, normed = residual_scale_rms_norm_pair_reference(
        x, branch, scale, weight, eps
    )
    return (mixed * grad_mixed).sum() + (normed * grad_normed).sum()


def backward_step_custom(
    op: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    x: torch.Tensor,
    branch: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    grad_mixed: torch.Tensor,
    grad_normed: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one custom forward+backward step for the pair boundary."""
    mixed, normed = op(x, branch, scale, weight, eps)
    loss = (mixed * grad_mixed).sum() + (normed * grad_normed).sum()
    return torch.autograd.grad(loss, (x, branch, scale, weight))


def pair_loss_custom(
    op: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    x: torch.Tensor,
    branch: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    grad_mixed: torch.Tensor,
    grad_normed: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return a scalar loss for the custom pair op."""
    mixed, normed = op(x, branch, scale, weight, eps)
    return (mixed * grad_mixed).sum() + (normed * grad_normed).sum()


def main() -> None:
    """Build the candidate extension and benchmark it against the reference."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the C++/CUDA operator benchmark")

    device = torch.device("cuda")
    torch.manual_seed(1337)

    ext = load_extension(args.build_dir)
    pair_autograd_op = register_pair_autograd_custom_op(ext)
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

    x_ref = x.detach().clone().requires_grad_(True)
    branch_ref = branch.detach().clone().requires_grad_(True)
    scale_ref = scale.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    x_custom = x.detach().clone().requires_grad_(True)
    branch_custom = branch.detach().clone().requires_grad_(True)
    scale_custom = scale.detach().clone().requires_grad_(True)
    weight_custom = weight.detach().clone().requires_grad_(True)
    grad_mixed = torch.randn_like(x)
    grad_normed = torch.randn_like(x)

    ref_grads = backward_step_reference(
        x_ref,
        branch_ref,
        scale_ref,
        weight_ref,
        grad_mixed,
        grad_normed,
        eps,
    )
    custom_grads = backward_step_custom(
        pair_autograd_op,
        x_custom,
        branch_custom,
        scale_custom,
        weight_custom,
        grad_mixed,
        grad_normed,
        eps,
    )
    pair_backward_max_abs = max(
        (ref_grad.float() - custom_grad.float()).abs().max().item()
        for ref_grad, custom_grad in zip(ref_grads, custom_grads, strict=True)
    )
    pair_backward_max_rel = max(
        (
            (ref_grad.float() - custom_grad.float()).abs().max()
            / ref_grad.float().abs().max().clamp_min(1e-6)
        ).item()
        for ref_grad, custom_grad in zip(ref_grads, custom_grads, strict=True)
    )

    pair_backward_ref_ms = measure_ms(
        lambda: backward_step_reference(
            x_ref,
            branch_ref,
            scale_ref,
            weight_ref,
            grad_mixed,
            grad_normed,
            eps,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_pair_loss_ref = torch.compile(
        pair_loss_reference,
        dynamic=False,
        fullgraph=True,
    )
    compiled_pair_backward_ref_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_pair_loss_ref(
                x_ref,
                branch_ref,
                scale_ref,
                weight_ref,
                grad_mixed,
                grad_normed,
                eps,
            ),
            (x_ref, branch_ref, scale_ref, weight_ref),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    pair_backward_custom_ms = measure_ms(
        lambda: backward_step_custom(
            pair_autograd_op,
            x_custom,
            branch_custom,
            scale_custom,
            weight_custom,
            grad_mixed,
            grad_normed,
            eps,
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    compiled_pair_loss_custom = torch.compile(
        lambda x_, branch_, scale_, weight_, grad_mixed_, grad_normed_, eps_: (
            pair_loss_custom(
                pair_autograd_op,
                x_,
                branch_,
                scale_,
                weight_,
                grad_mixed_,
                grad_normed_,
                eps_,
            )
        ),
        dynamic=False,
        fullgraph=True,
    )
    compiled_pair_backward_custom_ms = measure_ms(
        lambda: torch.autograd.grad(
            compiled_pair_loss_custom(
                x_custom,
                branch_custom,
                scale_custom,
                weight_custom,
                grad_mixed,
                grad_normed,
                eps,
            ),
            (x_custom, branch_custom, scale_custom, weight_custom),
        ),
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    pair_backward_summary = {
        "case": "residual_scale_rms_norm_pair_backward",
        "reference_ms": float(pair_backward_ref_ms),
        "compiled_reference_ms": float(compiled_pair_backward_ref_ms),
        "custom_ms": float(pair_backward_custom_ms),
        "compiled_custom_ms": float(compiled_pair_backward_custom_ms),
        "speedup": (
            float(pair_backward_ref_ms / pair_backward_custom_ms)
            if pair_backward_custom_ms > 0.0
            else 0.0
        ),
        "speedup_vs_compiled_reference": (
            float(compiled_pair_backward_ref_ms / compiled_pair_backward_custom_ms)
            if compiled_pair_backward_custom_ms > 0.0
            else 0.0
        ),
        "max_abs": float(pair_backward_max_abs),
        "max_rel": float(pair_backward_max_rel),
    }

    summary = {
        "shape": [batch, seq_len, dim],
        "warmup_iters": int(args.warmup_iters),
        "measured_iters": int(args.measured_iters),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "cases": [single_summary, pair_summary, pair_backward_summary],
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
