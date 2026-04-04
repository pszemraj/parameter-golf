"""Tiny single-process CUDA preflight for HGDN kernel-path changes.

This script avoids `torchrun` and DDP so it can catch FLA and compile regressions
locally before handing longer commands to the user.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import CastedLinear, GatedDeltaNet, HybridGPT
from train_gpt_hybrid import prepare_hybrid_compile, restore_low_dim_params_to_fp32


@dataclass(frozen=True)
class PreflightResult:
    """Structured output for one preflight case.

    :param str name: Case label.
    :param float loss: Scalar loss or surrogate loss.
    :param float step_ms: End-to-end wall time in milliseconds.
    :param float peak_mem_mib: Peak allocated CUDA memory in MiB.
    """

    name: str
    loss: float
    step_ms: float
    peak_mem_mib: float


def prepare_module(module: torch.nn.Module) -> torch.nn.Module:
    """Apply the trainer's mixed-precision parameter policy.

    :param torch.nn.Module module: Module to prepare in place.
    :return torch.nn.Module: Prepared module.
    """
    module = module.cuda().bfloat16()
    for submodule in module.modules():
        if isinstance(submodule, CastedLinear):
            submodule.float()
    restore_low_dim_params_to_fp32(module)
    return module


def run_gdn_eager() -> PreflightResult:
    """Exercise the bare FLA-backed GDN layer on CUDA.

    :return PreflightResult: Timings and loss for the eager GDN smoke.
    """
    layer = prepare_module(
        GatedDeltaNet(
            d_model=384,
            n_heads=8,
            head_k_dim=48,
            expand_v=1.0,
            allow_neg_eigval=True,
            conv_size=4,
            use_fla=True,
        )
    )
    x = torch.randn(2, 256, 384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = layer(x)
    loss = out.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    return PreflightResult(
        name="gdn_eager",
        loss=float(loss.item()),
        step_ms=(time.perf_counter() - t0) * 1000.0,
        peak_mem_mib=torch.cuda.max_memory_allocated() / 1024**2,
    )


def run_hybrid_case(*, compiled: bool) -> PreflightResult:
    """Run a tiny HybridGPT forward/backward with at least one GDN block.

    :param bool compiled: Whether to compile the model with the trainer's default strategy.
    :return PreflightResult: Timings and loss for the hybrid smoke.
    """
    base_model = prepare_module(
        HybridGPT(
            vocab_size=1024,
            num_layers=4,
            d_model=128,
            attn_heads=4,
            attn_kv_heads=2,
            gdn_n_heads=4,
            gdn_head_k_dim=16,
            gdn_expand_v=1.0,
            gdn_allow_neg_eigval=True,
            gdn_conv_size=4,
            gdn_ratio=1,
            mlp_mult=2.0,
            norm_style="pre",
        )
    )
    model, _ = prepare_hybrid_compile(
        base_model, enabled=compiled, strategy="model"
    )
    ids = torch.randint(0, 1024, (2, 256), device="cuda", dtype=torch.int64)
    tgt = torch.randint(0, 1024, (2, 256), device="cuda", dtype=torch.int64)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    loss = model(ids, tgt)
    loss.backward()
    torch.cuda.synchronize()
    return PreflightResult(
        name="hybrid_compiled" if compiled else "hybrid_eager",
        loss=float(loss.item()),
        step_ms=(time.perf_counter() - t0) * 1000.0,
        peak_mem_mib=torch.cuda.max_memory_allocated() / 1024**2,
    )


def main() -> None:
    """Run the direct CUDA preflight suite and print compact summaries."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HGDN preflight")
    os.environ.setdefault("USE_WANDB", "0")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results = [
        run_gdn_eager(),
        run_hybrid_case(compiled=False),
        run_hybrid_case(compiled=True),
    ]
    for result in results:
        print(
            f"preflight:{result.name} "
            f"loss={result.loss:.6f} step_ms={result.step_ms:.2f} "
            f"peak_mem_mib={result.peak_mem_mib:.1f}"
        )
    print("HGDN CUDA preflight passed")


if __name__ == "__main__":
    main()
