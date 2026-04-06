"""Profile local HGDN hot paths without DDP or dataloader noise.

Use this helper for local 4070 iteration on HGDN-specific bottlenecks before
rechecking promising changes on H100. It isolates three useful views:

- bare GDN forward/backward
- HybridGPT forward/backward
- optimizer step only after one backward pass
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("PROFILE_RANGES", "1")
os.environ.setdefault("USE_WANDB", "0")
os.environ.setdefault("WANDB_MODE", "offline")

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GatedDeltaNet = None
HybridGPT = None
Muon = None
restore_low_dim_params_to_fp32 = None
SCALAR_PARAM_PATTERNS = ()
build_profile_report = None
build_profile_rows = None
write_profile_report = None

from local_env import env_flag  # noqa: E402


def load_repo_symbols() -> None:
    """Import repo modules after env-backed profiling knobs are finalized."""
    global GatedDeltaNet, HybridGPT, Muon, restore_low_dim_params_to_fp32
    global SCALAR_PARAM_PATTERNS

    if GatedDeltaNet is not None:
        return
    from model import GatedDeltaNet as gdn_cls  # noqa: WPS433
    from model import HybridGPT as hybrid_cls  # noqa: WPS433
    from model import SCALAR_PARAM_PATTERNS as scalar_patterns  # noqa: WPS433
    from profiler_report import (  # noqa: WPS433
        build_profile_report,
        build_profile_rows,
        write_profile_report,
    )
    from train_gpt_hybrid import Muon as muon_cls  # noqa: WPS433
    from train_gpt_hybrid import (  # noqa: WPS433
        restore_low_dim_params_to_fp32 as restore_fp32,
    )

    globals()["GatedDeltaNet"] = gdn_cls
    globals()["HybridGPT"] = hybrid_cls
    globals()["SCALAR_PARAM_PATTERNS"] = scalar_patterns
    globals()["Muon"] = muon_cls
    globals()["restore_low_dim_params_to_fp32"] = restore_fp32
    globals()["build_profile_report"] = build_profile_report
    globals()["build_profile_rows"] = build_profile_rows
    globals()["write_profile_report"] = write_profile_report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the local HGDN profiler.

    :return argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("gdn", "hybrid-fwd-bwd", "hybrid-opt", "all"),
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "profiles" / "local_hotpath",
    )
    parser.add_argument(
        "--audit-boundaries-path",
        type=Path,
        default=None,
        help="Optional JSONL file for HGDN boundary dtype/layout audit.",
    )
    parser.add_argument(
        "--audit-boundaries-limit",
        type=int,
        default=1,
        help="Number of GDN forward calls to audit when --audit-boundaries-path is set.",
    )
    return parser.parse_args()


def configure_cuda(seed: int) -> None:
    """Set deterministic seeds and CUDA math defaults.

    :param int seed: Random seed.
    :raises RuntimeError: If CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local HGDN hotpath profiling")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = env_flag("CUDNN_BENCHMARK")


def configure_boundary_audit(args: argparse.Namespace) -> None:
    """Enable optional HGDN boundary audit logging before repo imports.

    :param argparse.Namespace args: Parsed command-line arguments.
    """
    if args.audit_boundaries_path is None:
        return
    os.environ["GDN_AUDIT_BOUNDARIES"] = "1"
    os.environ["GDN_AUDIT_BOUNDARIES_PATH"] = str(args.audit_boundaries_path)
    os.environ["GDN_AUDIT_BOUNDARIES_LIMIT"] = str(args.audit_boundaries_limit)


def prepare_model(module: torch.nn.Module) -> torch.nn.Module:
    """Apply the trainer's mixed-precision parameter policy.

    :param torch.nn.Module module: Module to prepare.
    :return torch.nn.Module: CUDA bf16 module with low-dimensional params restored to fp32.
    """
    module = module.cuda().bfloat16()
    load_repo_symbols()
    restore_low_dim_params_to_fp32(module)
    maybe_freeze_gdn_conv_weights(module)
    return module


def maybe_freeze_gdn_conv_weights(module: torch.nn.Module) -> None:
    """Optionally freeze wrapped GDN depthwise-conv weights for attribution screens.

    This is a profiling-only knob used to isolate how much of the packed front-end
    loss comes from depthwise-conv weight gradients versus everything else.

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


def build_hybrid_model() -> HybridGPT:
    """Construct the current HGDN operating point for local profiling.

    :return HybridGPT: Prepared hybrid model.
    """
    return prepare_model(
        HybridGPT(
            vocab_size=1024,
            num_layers=16,
            d_model=384,
            attn_heads=8,
            attn_kv_heads=2,
            gdn_n_heads=8,
            gdn_head_k_dim=48,
            gdn_expand_v=1.0,
            gdn_allow_neg_eigval=True,
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
            gdn_use_packed_qkv_conv_custom_backward=env_flag(
                "GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD"
            ),
            gdn_use_packed_qkv_single_contig=env_flag("GDN_PACKED_QKV_SINGLE_CONTIG"),
            gdn_ratio=1,
            mlp_mult=3.25,
            norm_style="pre",
        )
    )


def build_optimizers(model: HybridGPT) -> list[torch.optim.Optimizer]:
    """Recreate the trainer's Muon/Adam split for hotpath profiling.

    :param HybridGPT model: Hybrid model to optimize.
    :return list[torch.optim.Optimizer]: Optimizers in trainer order.
    """
    block_named_params = list(model.blocks.named_parameters())
    matrix_params = [
        p
        for n, p in block_named_params
        if p.ndim == 2 and not any(pat in n for pat in SCALAR_PARAM_PATTERNS)
    ]
    scalar_params = [
        p
        for n, p in block_named_params
        if p.ndim < 2 or any(pat in n for pat in SCALAR_PARAM_PATTERNS)
    ]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)

    optimizers: list[torch.optim.Optimizer] = [
        torch.optim.Adam(
            [model.tok_emb.weight],
            lr=0.02,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            fused=True,
        ),
        Muon(
            matrix_params,
            lr=0.05,
            momentum=0.95,
            backend_steps=5,
            weight_decay=0.0,
        ),
        torch.optim.Adam(
            scalar_params,
            lr=0.01,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            fused=True,
        ),
    ]
    if model.lm_head is not None:
        optimizers.insert(
            1,
            torch.optim.Adam(
                [model.lm_head.weight],
                lr=0.01,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.0,
                fused=True,
            ),
        )
    return optimizers


def make_tokens(batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random token ids and targets for local profiling.

    :param int batch_size: Batch size.
    :param int seq_len: Sequence length.
    :return tuple[torch.Tensor, torch.Tensor]: Token ids and targets on CUDA.
    """
    ids = torch.randint(
        0, 1024, (batch_size, seq_len), device="cuda", dtype=torch.int64
    )
    tgt = torch.randint(
        0, 1024, (batch_size, seq_len), device="cuda", dtype=torch.int64
    )
    return ids, tgt


def run_profile(
    name: str,
    body: Callable[[], None],
    *,
    row_limit: int,
    output_dir: Path,
) -> None:
    """Profile one callable and write a table plus chrome trace.

    :param str name: Profile name.
    :param callable body: Body to execute under the profiler.
    :param int row_limit: Table row limit.
    :param Path output_dir: Directory for outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{name}.trace.json"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        body()
    prof.export_chrome_trace(str(trace_path))
    rows = build_profile_rows(prof.key_averages())
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=row_limit,
    )
    report = build_profile_report(
        rows,
        sort_by="self_cuda_time_total",
        metadata={
            "profile_name": name,
            "trace_path": str(trace_path),
        },
    )
    write_profile_report(
        output_dir,
        report=report,
        stem=name,
    )
    print(f"\n=== {name} ===")
    print(table)
    print(
        f"peak_mem_mib={torch.cuda.max_memory_allocated() / 1024**2:.1f} "
        f"trace={trace_path}"
    )


def profile_gdn(
    batch_size: int, seq_len: int, row_limit: int, output_dir: Path
) -> None:
    """Profile the bare GDN forward/backward path.

    :param int batch_size: Batch size.
    :param int seq_len: Sequence length.
    :param int row_limit: Table row limit.
    :param Path output_dir: Output directory.
    """
    layer = prepare_model(
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
            use_packed_qkv_conv_custom_backward=env_flag(
                "GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD"
            ),
            use_packed_qkv_single_contig=env_flag("GDN_PACKED_QKV_SINGLE_CONTIG"),
        )
    )
    x = torch.randn(
        batch_size,
        seq_len,
        384,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def body() -> None:
        """Run one bare-GDN forward/backward step."""
        out = layer(x)
        out.float().square().mean().backward()

    run_profile("gdn_fwd_bwd", body, row_limit=row_limit, output_dir=output_dir)


def profile_hybrid_fwd_bwd(
    batch_size: int, seq_len: int, row_limit: int, output_dir: Path
) -> None:
    """Profile HybridGPT forward/backward without optimizer step.

    :param int batch_size: Batch size.
    :param int seq_len: Sequence length.
    :param int row_limit: Table row limit.
    :param Path output_dir: Output directory.
    """
    model = build_hybrid_model()
    ids, tgt = make_tokens(batch_size, seq_len)

    def body() -> None:
        """Run one HybridGPT forward/backward step without optimizer updates."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(ids, tgt)
        loss.backward()

    run_profile("hybrid_fwd_bwd", body, row_limit=row_limit, output_dir=output_dir)


def profile_hybrid_opt(
    batch_size: int, seq_len: int, row_limit: int, output_dir: Path
) -> None:
    """Profile the optimizer step after one backward pass.

    :param int batch_size: Batch size.
    :param int seq_len: Sequence length.
    :param int row_limit: Table row limit.
    :param Path output_dir: Output directory.
    """
    model = build_hybrid_model()
    optimizers = build_optimizers(model)
    ids, tgt = make_tokens(batch_size, seq_len)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(ids, tgt)
    loss.backward()

    def body() -> None:
        """Run one optimizer update across the trainer-style optimizer split."""
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

    run_profile("hybrid_optimizer", body, row_limit=row_limit, output_dir=output_dir)


def resolve_modes(mode: str) -> Iterable[str]:
    """Expand a mode selector into concrete profile names.

    :param str mode: Requested mode.
    :return Iterable[str]: Concrete modes to run in order.
    """
    if mode == "all":
        return ("gdn", "hybrid-fwd-bwd", "hybrid-opt")
    return (mode,)


def main() -> None:
    """Run the requested local HGDN hotpath profiles."""
    args = parse_args()
    configure_boundary_audit(args)
    load_repo_symbols()
    configure_cuda(args.seed)
    for mode in resolve_modes(args.mode):
        if mode == "gdn":
            profile_gdn(args.batch_size, args.seq_len, args.row_limit, args.output_dir)
        elif mode == "hybrid-fwd-bwd":
            profile_hybrid_fwd_bwd(
                args.batch_size, args.seq_len, args.row_limit, args.output_dir
            )
        elif mode == "hybrid-opt":
            profile_hybrid_opt(
                args.batch_size, args.seq_len, args.row_limit, args.output_dir
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
