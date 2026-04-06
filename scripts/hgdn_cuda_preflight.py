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

from hgdn_cuda import extension_status  # noqa: E402
from local_env import env_flag  # noqa: E402
from model import GatedDeltaNet, HybridGPT  # noqa: E402
from train_gpt_hybrid import (  # noqa: E402
    prepare_hybrid_compile,
    restore_low_dim_params_to_fp32,
)


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
    restore_low_dim_params_to_fp32(
        module, gdn_control_proj_fp32=env_flag("GDN_CONTROL_PROJ_FP32", True)
    )
    maybe_freeze_gdn_conv_weights(module)
    return module


def maybe_freeze_gdn_conv_weights(module: torch.nn.Module) -> None:
    """Optionally freeze wrapped GDN depthwise-conv weights for attribution screens.

    :param torch.nn.Module module: Module tree to mutate in place.
    """
    if not env_flag("GDN_FREEZE_CONV_WEIGHTS"):
        return
    for submodule in module.modules():
        conv = getattr(submodule, "conv", None)
        if isinstance(conv, torch.nn.Conv1d) and submodule.__class__.__name__ in {
            "CausalConv1d",
            "PackedCausalConv1d",
        }:
            conv.weight.requires_grad_(False)


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
            use_packed_qkv_conv=env_flag("GDN_USE_PACKED_QKV_CONV"),
            use_packed_qkv_proj=env_flag("GDN_USE_PACKED_QKV_PROJ"),
            use_q_conv=env_flag("GDN_USE_Q_CONV", True),
            use_k_conv=env_flag("GDN_USE_K_CONV", True),
            use_v_conv=env_flag("GDN_USE_V_CONV", True),
            conv_output_contiguous=env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
            q_conv_output_contiguous=env_flag(
                "GDN_Q_CONV_OUTPUT_CONTIGUOUS",
                env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
            ),
            k_conv_output_contiguous=env_flag(
                "GDN_K_CONV_OUTPUT_CONTIGUOUS",
                env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
            ),
            v_conv_output_contiguous=env_flag(
                "GDN_V_CONV_OUTPUT_CONTIGUOUS",
                env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
            ),
            gates_fp32=env_flag("GDN_GATES_FP32", True),
            output_norm_fp32=env_flag("GDN_OUTPUT_NORM_FP32", True),
            use_cuda_fused_frontend=env_flag("GDN_USE_CUDA_FUSED_FRONTEND"),
            use_cuda_fused_output=env_flag("GDN_USE_CUDA_FUSED_OUTPUT"),
            use_cuda_split_norm=env_flag("GDN_USE_CUDA_SPLIT_NORM"),
            use_packed_qkv_conv_custom_backward=env_flag(
                "GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD"
            ),
            use_packed_qkv_single_contig=env_flag("GDN_PACKED_QKV_SINGLE_CONTIG"),
            use_packed_qkv_split_copy=env_flag("GDN_PACKED_QKV_SPLIT_COPY"),
        )
    )
    x = torch.randn(
        2, 256, 384, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
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
            gdn_use_packed_qkv_conv=env_flag("GDN_USE_PACKED_QKV_CONV"),
            gdn_use_packed_qkv_proj=env_flag("GDN_USE_PACKED_QKV_PROJ"),
            gdn_use_q_conv=env_flag("GDN_USE_Q_CONV", True),
            gdn_use_k_conv=env_flag("GDN_USE_K_CONV", True),
            gdn_use_v_conv=env_flag("GDN_USE_V_CONV", True),
            gdn_conv_output_contiguous=env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
            gdn_q_conv_output_contiguous=env_flag(
                "GDN_Q_CONV_OUTPUT_CONTIGUOUS",
                env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
            ),
            gdn_k_conv_output_contiguous=env_flag(
                "GDN_K_CONV_OUTPUT_CONTIGUOUS",
                env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
            ),
            gdn_v_conv_output_contiguous=env_flag(
                "GDN_V_CONV_OUTPUT_CONTIGUOUS",
                env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
            ),
            gdn_gates_fp32=env_flag("GDN_GATES_FP32", True),
            gdn_output_norm_fp32=env_flag("GDN_OUTPUT_NORM_FP32", True),
            gdn_use_cuda_fused_frontend=env_flag("GDN_USE_CUDA_FUSED_FRONTEND"),
            gdn_use_cuda_fused_output=env_flag("GDN_USE_CUDA_FUSED_OUTPUT"),
            gdn_use_cuda_split_norm=env_flag("GDN_USE_CUDA_SPLIT_NORM"),
            gdn_use_packed_qkv_conv_custom_backward=env_flag(
                "GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD"
            ),
            gdn_use_packed_qkv_single_contig=env_flag("GDN_PACKED_QKV_SINGLE_CONTIG"),
            gdn_use_packed_qkv_split_copy=env_flag("GDN_PACKED_QKV_SPLIT_COPY"),
            gdn_ratio=1,
            mlp_mult=2.0,
            norm_style="pre",
        )
    )
    model, _ = prepare_hybrid_compile(base_model, enabled=compiled, strategy="model")
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
    seed = int(os.environ.get("SEED", "1337"))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = env_flag("CUDNN_BENCHMARK")
    print(f"hgdn_cuda_extension:{extension_status()}")
    print(
        "preflight_contract:"
        f"seed={seed} "
        f"pythonhashseed={os.environ.get('PYTHONHASHSEED', '<unset>')} "
        f"cudnn_benchmark={int(env_flag('CUDNN_BENCHMARK'))} "
        f"inductor_force_disable_caches={os.environ.get('TORCHINDUCTOR_FORCE_DISABLE_CACHES', '<unset>')}"
    )

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
