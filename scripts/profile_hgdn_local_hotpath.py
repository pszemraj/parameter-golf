"""Profile local HGDN hot paths without DDP or dataloader noise.

Use this helper for local 4070 iteration on HGDN-specific bottlenecks before
rechecking promising changes on H100. It isolates three useful views:

- bare GDN forward/backward
- HybridGPT forward/backward
- optimizer step only after one backward pass
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("PROFILE_RANGES", "1")
os.environ.setdefault("USE_WANDB", "0")
os.environ.setdefault("WANDB_MODE", "offline")

import torch

from _repo_bootstrap import ensure_repo_root_on_sys_path

REPO_ROOT = ensure_repo_root_on_sys_path()

from hgdn_runtime_utils import (  # noqa: E402
    maybe_freeze_gdn_conv_weights,
    prepare_cuda_module,
    restore_low_dim_params_to_fp32 as restore_fp32,
)
from local_env import env_flag  # noqa: E402

GatedDeltaNet = None
HybridGPT = None
Muon = None
uses_scalar_optimizer = None
validate_gdn_w_g_optimizer = None
build_profile_report = None
build_profile_rows = None
write_profile_report = None


def load_repo_symbols() -> None:
    """Import repo modules after env-backed profiling knobs are finalized."""
    global GatedDeltaNet, HybridGPT, Muon
    global uses_scalar_optimizer, validate_gdn_w_g_optimizer

    if GatedDeltaNet is not None:
        return
    from model import GatedDeltaNet as gdn_cls  # noqa: WPS433
    from model import HybridGPT as hybrid_cls  # noqa: WPS433
    from model import uses_scalar_optimizer as uses_scalar_optimizer_fn  # noqa: WPS433
    from model import validate_gdn_w_g_optimizer as validate_w_g_optimizer_fn  # noqa: WPS433
    from profiler_report import (  # noqa: WPS433
        build_profile_report,
        build_profile_rows,
        write_profile_report,
    )
    from train_gpt_hybrid import Muon as muon_cls  # noqa: WPS433

    globals()["GatedDeltaNet"] = gdn_cls
    globals()["HybridGPT"] = hybrid_cls
    globals()["uses_scalar_optimizer"] = uses_scalar_optimizer_fn
    globals()["validate_gdn_w_g_optimizer"] = validate_w_g_optimizer_fn
    globals()["Muon"] = muon_cls
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
    parser.add_argument("--seq-len", type=int, default=1024)
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
    load_repo_symbols()
    gdn_control_proj_fp32 = env_flag("GDN_CONTROL_PROJ_FP32", True)
    gdn_w_g_optimizer = validate_gdn_w_g_optimizer(
        os.environ.get("GDN_W_G_OPTIMIZER", "scalar")
    )
    prepare_kwargs: dict[str, object] = {
        "restore_low_dim_params_to_fp32": restore_fp32,
        "freeze_conv_weights": env_flag("GDN_FREEZE_CONV_WEIGHTS"),
        "gdn_control_proj_fp32": gdn_control_proj_fp32,
    }
    if "gdn_w_g_optimizer" in inspect.signature(prepare_cuda_module).parameters:
        prepare_kwargs["gdn_w_g_optimizer"] = gdn_w_g_optimizer
        return prepare_cuda_module(module, **prepare_kwargs)

    module = module.cuda().bfloat16()
    restore_fp32(
        module,
        gdn_control_proj_fp32=gdn_control_proj_fp32,
        gdn_w_g_optimizer=gdn_w_g_optimizer,
    )
    maybe_freeze_gdn_conv_weights(
        module, enabled=bool(prepare_kwargs["freeze_conv_weights"])
    )
    return module


def env_int(name: str, default: int) -> int:
    """Read one integer environment variable with a typed default.

    :param str name: Environment variable name.
    :param int default: Default value when unset.
    :return int: Parsed integer value.
    """
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    """Read one float environment variable with a typed default.

    :param str name: Environment variable name.
    :param float default: Default value when unset.
    :return float: Parsed float value.
    """
    return float(os.environ.get(name, str(default)))


def resolve_block_pattern(*, num_layers: int) -> str | None:
    """Resolve the exact-contract block pattern for local profiling.

    :param int num_layers: Requested depth.
    :return str | None: Explicit block pattern when configured, else `None`.
    """
    block_pattern = os.environ.get("BLOCK_PATTERN", "").strip()
    if block_pattern:
        return block_pattern
    if num_layers == 8:
        return "attn,attn,gdn,attn,gdn,attn,attn,attn"
    if num_layers == 9:
        return "attn,attn,gdn,attn,attn,gdn,attn,attn,attn"
    return None


def build_gdn_kwargs() -> dict[str, object]:
    """Build the shared HGDN frontend/runtime kwargs from the environment.

    :return dict[str, object]: Keyword arguments shared by local GDN model builders.
    """
    return {
        "expand_v": env_float("GDN_EXPAND_V", 1.0),
        "allow_neg_eigval": env_flag("GDN_ALLOW_NEG_EIGVAL", True),
        "conv_size": env_int("GDN_CONV_SIZE", 4),
        "use_fla": True,
        "use_packed_qkv_conv": env_flag("GDN_USE_PACKED_QKV_CONV"),
        "use_packed_qkv_proj": env_flag("GDN_USE_PACKED_QKV_PROJ"),
        "use_q_conv": env_flag("GDN_USE_Q_CONV", True),
        "use_k_conv": env_flag("GDN_USE_K_CONV", True),
        "use_v_conv": env_flag("GDN_USE_V_CONV", True),
        "conv_output_contiguous": env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
        "q_conv_output_contiguous": env_flag(
            "GDN_Q_CONV_OUTPUT_CONTIGUOUS",
            env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
        ),
        "k_conv_output_contiguous": env_flag(
            "GDN_K_CONV_OUTPUT_CONTIGUOUS",
            env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
        ),
        "v_conv_output_contiguous": env_flag(
            "GDN_V_CONV_OUTPUT_CONTIGUOUS",
            env_flag("GDN_CONV_OUTPUT_CONTIGUOUS"),
        ),
        "gates_fp32": env_flag("GDN_GATES_FP32", True),
        "output_norm_fp32": env_flag("GDN_OUTPUT_NORM_FP32", True),
        "use_cuda_frontend_nct": env_flag("GDN_USE_CUDA_FRONTEND_NCT"),
        "use_cuda_packed_conv": env_flag("GDN_USE_CUDA_PACKED_CONV"),
        "use_cuda_packed_conv_aten_backward": env_flag(
            "GDN_USE_CUDA_PACKED_CONV_ATEN_BACKWARD"
        ),
        "use_cuda_packed_conv_aten_weight_backward": env_flag(
            "GDN_USE_CUDA_PACKED_CONV_ATEN_WEIGHT_BACKWARD"
        ),
        "use_cuda_split_norm": env_flag("GDN_USE_CUDA_SPLIT_NORM"),
        "use_packed_qkv_conv_custom_backward": env_flag(
            "GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD"
        ),
        "use_packed_qkv_single_contig": env_flag("GDN_PACKED_QKV_SINGLE_CONTIG"),
        "use_packed_qkv_split_copy": env_flag("GDN_PACKED_QKV_SPLIT_COPY"),
    }


def build_hybrid_model() -> HybridGPT:
    """Construct the current HGDN operating point for local profiling.

    :return HybridGPT: Prepared hybrid model.
    """
    load_repo_symbols()
    num_layers = env_int("NUM_LAYERS", 8)
    block_pattern = resolve_block_pattern(num_layers=num_layers)
    return prepare_model(
        HybridGPT(
            vocab_size=env_int("VOCAB_SIZE", 1024),
            num_layers=num_layers,
            d_model=env_int("MODEL_DIM", 512),
            attn_heads=env_int("NUM_HEADS", 8),
            attn_kv_heads=env_int("NUM_KV_HEADS", 4),
            gdn_n_heads=env_int("GDN_N_HEADS", env_int("NUM_HEADS", 8)),
            gdn_head_k_dim=env_int("GDN_HEAD_K_DIM", 48),
            gdn_expand_v=env_float("GDN_EXPAND_V", 1.0),
            gdn_allow_neg_eigval=env_flag("GDN_ALLOW_NEG_EIGVAL", True),
            gdn_conv_size=env_int("GDN_CONV_SIZE", 4),
            gdn_use_q_conv=env_flag("GDN_USE_Q_CONV", True),
            gdn_use_k_conv=env_flag("GDN_USE_K_CONV", True),
            gdn_use_v_conv=env_flag("GDN_USE_V_CONV", True),
            gdn_use_packed_qkv_conv=env_flag("GDN_USE_PACKED_QKV_CONV"),
            gdn_use_packed_qkv_proj=env_flag("GDN_USE_PACKED_QKV_PROJ"),
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
            gdn_use_cuda_frontend_nct=env_flag("GDN_USE_CUDA_FRONTEND_NCT"),
            gdn_use_cuda_packed_conv=env_flag("GDN_USE_CUDA_PACKED_CONV"),
            gdn_use_cuda_packed_conv_aten_backward=env_flag(
                "GDN_USE_CUDA_PACKED_CONV_ATEN_BACKWARD"
            ),
            gdn_use_cuda_packed_conv_aten_weight_backward=env_flag(
                "GDN_USE_CUDA_PACKED_CONV_ATEN_WEIGHT_BACKWARD"
            ),
            gdn_use_cuda_split_norm=env_flag("GDN_USE_CUDA_SPLIT_NORM"),
            gdn_use_packed_qkv_conv_custom_backward=env_flag(
                "GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD"
            ),
            gdn_use_packed_qkv_single_contig=env_flag("GDN_PACKED_QKV_SINGLE_CONTIG"),
            gdn_use_packed_qkv_split_copy=env_flag("GDN_PACKED_QKV_SPLIT_COPY"),
            gdn_ratio=env_int("GDN_RATIO", 1),
            block_pattern=block_pattern,
            mlp_mult=env_float("MLP_MULT", 2.0),
            norm_style=os.environ.get("NORM_STYLE", "pre"),
        )
    )


def build_optimizers(model: HybridGPT) -> list[torch.optim.Optimizer]:
    """Recreate the trainer's Muon/Adam split for hotpath profiling.

    :param HybridGPT model: Hybrid model to optimize.
    :return list[torch.optim.Optimizer]: Optimizers in trainer order.
    """
    load_repo_symbols()
    w_g_mode = validate_gdn_w_g_optimizer(os.environ.get("GDN_W_G_OPTIMIZER", "scalar"))
    block_named_params = list(model.blocks.named_parameters())
    matrix_params = [
        p
        for n, p in block_named_params
        if p.ndim == 2
        and not uses_scalar_optimizer(n, param_ndim=p.ndim, gdn_w_g_optimizer=w_g_mode)
    ]
    scalar_params = [
        p
        for n, p in block_named_params
        if uses_scalar_optimizer(n, param_ndim=p.ndim, gdn_w_g_optimizer=w_g_mode)
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
            distributed_mode=os.environ.get(
                "MUON_DISTRIBUTED_MODE", "packed_allreduce"
            ),
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
    load_repo_symbols()
    d_model = env_int("MODEL_DIM", 512)
    n_heads = env_int("GDN_N_HEADS", env_int("NUM_HEADS", 8))
    layer = prepare_model(
        GatedDeltaNet(
            d_model=d_model,
            n_heads=n_heads,
            head_k_dim=env_int("GDN_HEAD_K_DIM", 48),
            **build_gdn_kwargs(),
        )
    )
    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
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
