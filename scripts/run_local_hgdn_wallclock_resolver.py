#!/usr/bin/env python3
"""Run the local HGDN true-wallclock resolver through argparse."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _repo_bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path()

from hgdn_helper_cli import parse_bool_flag, write_json  # noqa: E402
from hgdn_local_runner import (  # noqa: E402
    CommandSpec,
    REPO_ROOT,
    RECURRENCE_MODES,
    bool_flag_value,
    check_cuda_jobs,
    common_thread_env,
    copy_existing,
    create_7z_archive,
    csv_items,
    diagnostic_env,
    enable_line_buffering,
    ensure_py7zr_available,
    filtered_config_env,
    format_seconds,
    log_completion_state,
    resolve_grad_accum_steps,
    resolve_val_batch_size,
    run_command,
    write_command_log,
)


@dataclass(frozen=True)
class RunSpec:
    """One resolver run."""

    key: str
    label: str
    trainer: str
    run_id: str
    config: Path | None = None
    recurrence_mode: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--use-wandb", type=parse_bool_flag, default=False)
    parser.add_argument(
        "--wandb-mode", default="offline", choices=["online", "offline"]
    )
    parser.add_argument("--wandb-project", default="pg-hgdn-ablations")
    parser.add_argument("--wandb-watch", default="none")
    parser.add_argument("--wandb-watch-log-freq", type=int, default=25)
    parser.add_argument("--run-prefix-base", default="localhgdn_wallclock1")
    parser.add_argument("--bundle-stage-dir", type=Path, default=None)
    parser.add_argument("--archive-output", type=Path, default=None)
    parser.add_argument("--command-log", type=Path, default=None)
    parser.add_argument("--size-screen-output", type=Path, default=None)
    parser.add_argument(
        "--size-screen-config",
        type=Path,
        default=Path("configs/hgdn/naive_contract_search.toml"),
    )
    parser.add_argument("--run-plan", default="primary")
    parser.add_argument("--skip-completed-runs", type=parse_bool_flag, default=True)
    parser.add_argument("--ignore-incomplete-logs", type=parse_bool_flag, default=False)
    parser.add_argument("--allow-existing-logs", type=parse_bool_flag, default=False)
    parser.add_argument("--check-cuda-idle", type=parse_bool_flag, default=True)
    parser.add_argument("--allow-active-cuda-jobs", type=parse_bool_flag, default=False)
    parser.add_argument("--torchinductor-max-autotune", type=int, default=0)
    parser.add_argument("--torchinductor-max-autotune-gemm", type=int, default=0)
    parser.add_argument("--torch-logs", default="")
    parser.add_argument("--torch-trace", default="")
    parser.add_argument("--artifact-limit-bytes", type=int, default=16_000_000)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--train-batch-tokens", type=int, default=65_536)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--val-loss-every", type=int, default=100)
    parser.add_argument("--train-log-every", type=int, default=25)
    parser.add_argument("--min-val-seqs", type=int, default=512)
    parser.add_argument("--val-max-seqs", type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-batch-seqs", type=int, default=None)
    parser.add_argument("--max-wallclock-seconds", type=float, default=600.0)
    parser.add_argument("--compile", type=parse_bool_flag, default=True)
    parser.add_argument("--compile-strategy", default="hybrid")
    parser.add_argument("--distributed-mode", default="parallel_muon")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data/datasets/fineweb10B_sp1024",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model",
    )
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument(
        "--primary-hgdn-config",
        type=Path,
        default=Path("configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml"),
    )
    parser.add_argument(
        "--primary-control-config",
        type=Path,
        default=Path("configs/hgdn/naive_contract_l8_d512_r0_m2.toml"),
    )
    parser.add_argument(
        "--secondary-hgdn-config",
        type=Path,
        default=Path("configs/hgdn/naive_contract_l8_d512_olmoish_6g2a_v2_m1p25.toml"),
    )
    parser.add_argument(
        "--secondary-control-config",
        type=Path,
        default=Path("configs/hgdn/naive_contract_l8_d512_r0_m1p25.toml"),
    )
    parser.add_argument("--gdn-fla-recurrence-mode", default="")
    parser.add_argument("--primary-gdn-fla-recurrence-mode", default="")
    parser.add_argument("--primary-control-gdn-fla-recurrence-mode", default="")
    parser.add_argument("--secondary-gdn-fla-recurrence-mode", default="")
    parser.add_argument("--secondary-control-gdn-fla-recurrence-mode", default="")
    parser.add_argument("--primary-control-margin", type=float, default=0.003)
    parser.add_argument("--secondary-primary-margin", type=float, default=0.005)
    parser.add_argument("--omp-num-threads", type=int, default=1)
    parser.add_argument("--mkl-num-threads", type=int, default=1)
    parser.add_argument("--openblas-num-threads", type=int, default=1)
    parser.add_argument("--numexpr-num-threads", type=int, default=1)
    parser.add_argument("--nccl-ib-disable", type=int, default=1)
    parser.add_argument("--gpt-naive-run-id", default="")
    parser.add_argument("--primary-hgdn-run-id", default="")
    parser.add_argument("--primary-control-run-id", default="")
    parser.add_argument("--secondary-hgdn-run-id", default="")
    parser.add_argument("--secondary-control-run-id", default="")
    return parser.parse_args()


def recurrence_modes(args: argparse.Namespace) -> dict[str, str]:
    """Resolve per-run recurrence modes.

    :param argparse.Namespace args: Parsed arguments.
    :return dict[str, str]: Run key to recurrence mode.
    """
    base = args.gdn_fla_recurrence_mode
    primary = args.primary_gdn_fla_recurrence_mode or base or "direct_fused"
    primary_control = args.primary_control_gdn_fla_recurrence_mode or primary
    secondary = args.secondary_gdn_fla_recurrence_mode or base or "direct_fused"
    secondary_control = args.secondary_control_gdn_fla_recurrence_mode or secondary
    modes = {
        "primary_hgdn": primary,
        "primary_control": primary_control,
        "secondary_hgdn": secondary,
        "secondary_control": secondary_control,
    }
    bad = {key: mode for key, mode in modes.items() if mode not in RECURRENCE_MODES}
    if bad:
        raise SystemExit(f"Unsupported GDN_FLA_RECURRENCE_MODE values: {bad}")
    return modes


def resolve_run_keys(run_plan: str) -> list[str]:
    """Resolve a run-plan string into ordered run keys.

    :param str run_plan: Run-plan value.
    :return list[str]: Ordered selected keys.
    """
    all_keys = [
        "exact",
        "primary_hgdn",
        "primary_control",
        "secondary_hgdn",
        "secondary_control",
    ]
    aliases = {
        "exact": ["exact"],
        "gpt_naive": ["exact"],
        "gpt_naive_baseline": ["exact"],
        "primary": ["exact", "primary_hgdn", "primary_control"],
        "secondary": ["secondary_hgdn", "secondary_control"],
        "full": all_keys,
        "primary_hgdn": ["primary_hgdn"],
        "primary_control": ["primary_control"],
        "secondary_hgdn": ["secondary_hgdn"],
        "secondary_control": ["secondary_control"],
    }
    out: list[str] = []
    for token in csv_items(run_plan):
        if token not in aliases:
            raise SystemExit(
                f"Unsupported --run-plan token {token!r}; expected primary, secondary, "
                "full, exact, or comma-listed run keys."
            )
        for key in aliases[token]:
            if key not in out:
                out.append(key)
    if not out:
        raise SystemExit(f"--run-plan resolved to zero runs: {run_plan}")
    return out


def run_specs(args: argparse.Namespace, modes: dict[str, str]) -> dict[str, RunSpec]:
    """Build all known resolver run specs.

    :param argparse.Namespace args: Parsed arguments.
    :param dict[str, str] modes: Resolved recurrence modes.
    :return dict[str, RunSpec]: Run specs by key.
    """
    wallclock_tag = format_seconds(args.max_wallclock_seconds)
    prefix = args.run_prefix_base
    exact_run_id = (
        args.gpt_naive_run_id
        or f"{prefix}_gpt_naive_baseline_seq{args.train_seq_len}_wall{wallclock_tag}"
    )
    primary_hgdn_run_id = (
        args.primary_hgdn_run_id
        or f"{prefix}_primary_hgdn_{args.primary_hgdn_config.stem}_seq{args.train_seq_len}_wall{wallclock_tag}"
    )
    primary_control_run_id = (
        args.primary_control_run_id
        or f"{prefix}_primary_control_{args.primary_control_config.stem}_seq{args.train_seq_len}_wall{wallclock_tag}"
    )
    secondary_hgdn_run_id = (
        args.secondary_hgdn_run_id
        or f"{prefix}_secondary_hgdn_{args.secondary_hgdn_config.stem}_seq{args.train_seq_len}_wall{wallclock_tag}"
    )
    secondary_control_run_id = (
        args.secondary_control_run_id
        or f"{prefix}_secondary_control_{args.secondary_control_config.stem}_seq{args.train_seq_len}_wall{wallclock_tag}"
    )
    return {
        "exact": RunSpec(
            key="exact",
            label="exact repo naive baseline local wallclock resolver",
            trainer="train_gpt.py",
            run_id=exact_run_id,
        ),
        "primary_hgdn": RunSpec(
            key="primary_hgdn",
            label="primary HGDN local wallclock resolver",
            trainer="train_gpt_hybrid.py",
            run_id=primary_hgdn_run_id,
            config=args.primary_hgdn_config,
            recurrence_mode=modes["primary_hgdn"],
        ),
        "primary_control": RunSpec(
            key="primary_control",
            label="primary attention-only baseline diagnostic control local wallclock resolver",
            trainer="train_gpt_hybrid.py",
            run_id=primary_control_run_id,
            config=args.primary_control_config,
            recurrence_mode=modes["primary_control"],
        ),
        "secondary_hgdn": RunSpec(
            key="secondary_hgdn",
            label="secondary HGDN local wallclock resolver",
            trainer="train_gpt_hybrid.py",
            run_id=secondary_hgdn_run_id,
            config=args.secondary_hgdn_config,
            recurrence_mode=modes["secondary_hgdn"],
        ),
        "secondary_control": RunSpec(
            key="secondary_control",
            label="secondary attention-only baseline diagnostic control local wallclock resolver",
            trainer="train_gpt_hybrid.py",
            run_id=secondary_control_run_id,
            config=args.secondary_control_config,
            recurrence_mode=modes["secondary_control"],
        ),
    }


def training_env(
    args: argparse.Namespace, grad_accum_steps: int, val_batch_size: int
) -> dict[str, str]:
    """Build the common training environment.

    :param argparse.Namespace args: Parsed arguments.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    :param int val_batch_size: Resolved validation batch tokens.
    :return dict[str, str]: Environment overlay.
    """
    env = common_thread_env(
        omp_num_threads=args.omp_num_threads,
        mkl_num_threads=args.mkl_num_threads,
        openblas_num_threads=args.openblas_num_threads,
        numexpr_num_threads=args.numexpr_num_threads,
        nccl_ib_disable=args.nccl_ib_disable,
    )
    env.update(diagnostic_env(args.torch_logs, args.torch_trace))
    env.update(
        {
            "TORCHINDUCTOR_MAX_AUTOTUNE": str(args.torchinductor_max_autotune),
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": str(
                args.torchinductor_max_autotune_gemm
            ),
            "DATA_PATH": str(args.data_path),
            "TOKENIZER_PATH": str(args.tokenizer_path),
            "VOCAB_SIZE": str(args.vocab_size),
            "GRAD_ACCUM_STEPS": str(grad_accum_steps),
            "ITERATIONS": str(args.iterations),
            "MAX_WALLCLOCK_SECONDS": f"{args.max_wallclock_seconds:g}",
            "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
            "TRAIN_SEQ_LEN": str(args.train_seq_len),
            "VAL_LOSS_EVERY": str(args.val_loss_every),
            "TRAIN_LOG_EVERY": str(args.train_log_every),
            "MIN_VAL_SEQS": str(args.min_val_seqs),
            "VAL_MAX_SEQS": str(args.val_max_seqs),
            "VAL_BATCH_SIZE": str(val_batch_size),
            "ARTIFACT_LIMIT_BYTES": str(args.artifact_limit_bytes),
        }
    )
    return env


def command_for_spec(
    args: argparse.Namespace,
    spec: RunSpec,
    base_env: dict[str, str],
) -> CommandSpec:
    """Build the executable command for one run.

    :param argparse.Namespace args: Parsed arguments.
    :param RunSpec spec: Run spec.
    :param dict[str, str] base_env: Common training environment.
    :return CommandSpec: Executable command.
    """
    env = dict(base_env)
    env["RUN_ID"] = spec.run_id
    if spec.key == "exact":
        env.update(
            {
                "NUM_LAYERS": "9",
                "MODEL_DIM": "512",
                "NUM_HEADS": "8",
                "NUM_KV_HEADS": "4",
                "MLP_MULT": "2",
            }
        )
        command = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={args.ngpu}",
            "train_gpt.py",
        ]
    else:
        assert spec.config is not None
        assert spec.recurrence_mode is not None
        env.update(
            {
                "NGPU": str(args.ngpu),
                "USE_WANDB": bool_flag_value(args.use_wandb),
                "WANDB_MODE": args.wandb_mode,
                "WANDB_PROJECT": args.wandb_project,
                "WANDB_WATCH": args.wandb_watch,
                "WANDB_WATCH_LOG_FREQ": str(args.wandb_watch_log_freq),
                "COMPILE": bool_flag_value(args.compile),
                "COMPILE_STRATEGY": args.compile_strategy,
                "DISTRIBUTED_MODE": args.distributed_mode,
                "WEIGHT_DECAY": f"{args.weight_decay:g}",
                "PERF_SKIP_FINAL_EVAL": "0",
            }
        )
        env.update(filtered_config_env(spec.config, spec.recurrence_mode))
        command = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={args.ngpu}",
            "train_gpt_hybrid.py",
        ]
    return CommandSpec(label=spec.label, run_id=spec.run_id, env=env, command=command)


def print_plan(
    args: argparse.Namespace,
    selected_specs: list[RunSpec],
    grad_accum_steps: int,
    val_batch_size: int,
) -> None:
    """Print the launch plan.

    :param argparse.Namespace args: Parsed arguments.
    :param list[RunSpec] selected_specs: Selected runs.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    :param int val_batch_size: Resolved validation batch size.
    """
    print()
    print(">>> Local HGDN true-wallclock resolver")
    print(f"run_prefix_base={args.run_prefix_base}")
    print(f"run_plan={args.run_plan}")
    print(f"selected_run_keys={','.join(spec.key for spec in selected_specs)}")
    print(f"iterations={args.iterations}  # safety cap")
    print(f"max_wallclock_seconds={args.max_wallclock_seconds:g}")
    print(
        f"estimated_max_train_seconds={len(selected_specs) * args.max_wallclock_seconds:g} plus validation/serialization overhead"
    )
    print(f"train_batch_tokens={args.train_batch_tokens}")
    print(f"train_seq_len={args.train_seq_len}")
    print(f"grad_accum_steps={grad_accum_steps}")
    print(f"val_batch_size={val_batch_size}")
    print(f"compile={int(args.compile)} compile_strategy={args.compile_strategy}")
    print(f"skip_completed_runs={int(args.skip_completed_runs)}")
    print(f"ignore_incomplete_logs={int(args.ignore_incomplete_logs)}")
    for spec in selected_specs:
        suffix = f" config={spec.config}" if spec.config else ""
        mode = f" recurrence={spec.recurrence_mode}" if spec.recurrence_mode else ""
        print(f"  - {spec.key}: {spec.run_id}{suffix}{mode}")


def check_existing_logs(
    args: argparse.Namespace, selected_specs: list[RunSpec]
) -> None:
    """Validate existing logs before launch.

    :param argparse.Namespace args: Parsed arguments.
    :param list[RunSpec] selected_specs: Selected runs.
    :raises SystemExit: If a conflicting log is present.
    """
    if args.allow_existing_logs:
        return
    conflicts: list[str] = []
    for spec in selected_specs:
        log_path = REPO_ROOT / "logs" / f"{spec.run_id}.txt"
        state = log_completion_state(log_path)
        if state == "missing":
            continue
        if state == "complete" and args.skip_completed_runs:
            continue
        if state == "incomplete" and args.ignore_incomplete_logs:
            continue
        conflicts.append(f"{state}: {log_path}")
    if conflicts:
        joined = "\n".join(conflicts)
        raise SystemExit(
            "Refusing to append to existing run logs:\n"
            f"{joined}\n"
            "Use a fresh --run-prefix-base, --skip-completed-runs for completed logs, "
            "or --allow-existing-logs only for intentional append."
        )


def run_size_screen(args: argparse.Namespace) -> None:
    """Run the artifact size screen.

    :param argparse.Namespace args: Parsed arguments.
    """
    print()
    print(">>> artifact-size screen")
    subprocess.run(
        [
            args.python_bin,
            "scripts/screen_hgdn_arch_sizes.py",
            "--config",
            str(args.size_screen_config),
            "--gdn-fla-recurrence-mode",
            args.primary_gdn_fla_recurrence_mode
            or args.gdn_fla_recurrence_mode
            or "direct",
            "--output-dir",
            str(args.size_screen_output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def should_skip_run(args: argparse.Namespace, spec: RunSpec) -> bool:
    """Return whether an existing run should be skipped.

    :param argparse.Namespace args: Parsed arguments.
    :param RunSpec spec: Run spec.
    :return bool: Whether to skip launch.
    """
    log_path = REPO_ROOT / "logs" / f"{spec.run_id}.txt"
    state = log_completion_state(log_path)
    if state == "complete" and args.skip_completed_runs:
        print()
        print(f">>> skip completed {spec.key}: {spec.run_id}")
        return True
    if state == "incomplete" and args.ignore_incomplete_logs:
        print()
        print(f">>> skip ignored incomplete {spec.key}: {spec.run_id}")
        return True
    return False


def write_manifest(
    args: argparse.Namespace,
    selected_specs: list[RunSpec],
    *,
    grad_accum_steps: int,
    val_batch_size: int,
    exit_status: int,
    completed_run_ids: list[str],
    incomplete_run_ids: list[str],
    missing_run_ids: list[str],
) -> None:
    """Write the resolver bundle manifest.

    :param argparse.Namespace args: Parsed arguments.
    :param list[RunSpec] selected_specs: Selected runs.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    :param int val_batch_size: Resolved validation batch size.
    :param int exit_status: Batch exit status.
    :param list[str] completed_run_ids: Completed run IDs.
    :param list[str] incomplete_run_ids: Incomplete run IDs.
    :param list[str] missing_run_ids: Missing run IDs.
    """
    manifest = {
        "run_prefix_base": args.run_prefix_base,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "archive_output": str(args.archive_output),
        "command_log": str(args.command_log),
        "exit_status": exit_status,
        "matched_logs": not incomplete_run_ids and not missing_run_ids,
        "run_plan": args.run_plan,
        "selected_run_keys": [spec.key for spec in selected_specs],
        "completed_log_count": len(completed_run_ids),
        "completed_run_ids": completed_run_ids,
        "incomplete_run_ids": incomplete_run_ids,
        "missing_run_ids": missing_run_ids,
        "size_screen": {
            "config": str(args.size_screen_config),
            "output_dir": str(args.size_screen_output),
        },
        "contract": {
            "ngpu": args.ngpu,
            "grad_accum_steps": grad_accum_steps,
            "iterations": args.iterations,
            "train_batch_tokens": args.train_batch_tokens,
            "train_seq_len": args.train_seq_len,
            "val_loss_every": args.val_loss_every,
            "train_log_every": args.train_log_every,
            "min_val_seqs": args.min_val_seqs,
            "val_max_seqs": args.val_max_seqs,
            "val_batch_size": val_batch_size,
            "max_wallclock_seconds": args.max_wallclock_seconds,
            "compile": args.compile,
            "compile_strategy": args.compile_strategy,
            "distributed_mode": args.distributed_mode,
            "weight_decay": args.weight_decay,
            "torch_logs": args.torch_logs or None,
            "torch_trace": args.torch_trace or None,
            "torchinductor_max_autotune": args.torchinductor_max_autotune,
            "torchinductor_max_autotune_gemm": args.torchinductor_max_autotune_gemm,
            "data_path": str(args.data_path),
            "tokenizer_path": str(args.tokenizer_path),
            "vocab_size": args.vocab_size,
            "skip_completed_runs": args.skip_completed_runs,
            "ignore_incomplete_logs": args.ignore_incomplete_logs,
        },
        "runs": [run_manifest_entry(spec, args) for spec in selected_specs],
    }
    write_json(args.bundle_stage_dir / "bundle_manifest.json", manifest)


def run_manifest_entry(spec: RunSpec, args: argparse.Namespace) -> dict[str, Any]:
    """Return one manifest run entry.

    :param RunSpec spec: Run spec.
    :param argparse.Namespace args: Parsed arguments.
    :return dict[str, Any]: Manifest entry.
    """
    entry: dict[str, Any] = {
        "key": spec.key,
        "label": spec.label,
        "trainer": spec.trainer,
        "mode": spec.key,
        "run_id": spec.run_id,
    }
    if spec.key == "exact":
        entry.update(
            {
                "mode": "n/a",
                "data_path": str(args.data_path),
                "tokenizer_path": str(args.tokenizer_path),
                "vocab_size": args.vocab_size,
                "num_layers": 9,
                "model_dim": 512,
                "num_heads": 8,
                "num_kv_heads": 4,
                "mlp_mult": 2,
            }
        )
    else:
        entry.update(
            {
                "config": str(spec.config),
                "gdn_fla_recurrence_mode": spec.recurrence_mode,
            }
        )
    return entry


def build_bundle(
    args: argparse.Namespace,
    selected_specs: list[RunSpec],
    *,
    grad_accum_steps: int,
    val_batch_size: int,
    exit_status: int,
) -> tuple[list[str], list[str], list[str]]:
    """Stage logs, configs, manifest, and size screen outputs.

    :param argparse.Namespace args: Parsed arguments.
    :param list[RunSpec] selected_specs: Selected runs.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    :param int val_batch_size: Resolved validation batch size.
    :param int exit_status: Batch exit status.
    :return tuple[list[str], list[str], list[str]]: Complete, incomplete, missing IDs.
    """
    print()
    print(">>> stage bundle outputs")
    shutil.rmtree(args.bundle_stage_dir, ignore_errors=True)
    (args.bundle_stage_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.bundle_stage_dir / "configs").mkdir(parents=True, exist_ok=True)
    (args.bundle_stage_dir / "size_screen").mkdir(parents=True, exist_ok=True)

    copy_existing(
        [
            *(spec.config for spec in selected_specs if spec.config is not None),
            args.size_screen_config,
        ],
        args.bundle_stage_dir / "configs",
    )
    if args.command_log.is_file():
        shutil.copy2(args.command_log, args.bundle_stage_dir / "commands.sh")
    if args.size_screen_output.is_dir():
        shutil.copytree(
            args.size_screen_output,
            args.bundle_stage_dir / "size_screen",
            dirs_exist_ok=True,
        )

    completed: list[str] = []
    incomplete: list[str] = []
    missing: list[str] = []
    for spec in selected_specs:
        log_path = REPO_ROOT / "logs" / f"{spec.run_id}.txt"
        state = log_completion_state(log_path)
        if state == "complete":
            completed.append(spec.run_id)
            shutil.copy2(log_path, args.bundle_stage_dir / "logs" / log_path.name)
        elif state == "incomplete":
            if args.ignore_incomplete_logs:
                missing.append(spec.run_id)
            else:
                incomplete.append(spec.run_id)
                shutil.copy2(log_path, args.bundle_stage_dir / "logs" / log_path.name)
        else:
            missing.append(spec.run_id)

    write_manifest(
        args,
        selected_specs,
        grad_accum_steps=grad_accum_steps,
        val_batch_size=val_batch_size,
        exit_status=exit_status,
        completed_run_ids=completed,
        incomplete_run_ids=incomplete,
        missing_run_ids=missing,
    )
    print(f"bundle_dir={args.bundle_stage_dir}")
    return completed, incomplete, missing


def analyze_bundle(
    args: argparse.Namespace,
    selected_specs: list[RunSpec],
    *,
    incomplete: list[str],
    missing: list[str],
) -> None:
    """Run bundle analysis and write a decision only when selected rows complete.

    :param argparse.Namespace args: Parsed arguments.
    :param list[RunSpec] selected_specs: Selected runs.
    :param list[str] incomplete: Incomplete run IDs.
    :param list[str] missing: Missing run IDs.
    """
    print()
    print(">>> analyze local wallclock resolver")
    analysis_dir = args.bundle_stage_dir / "analysis"
    subprocess.run(
        [
            args.python_bin,
            "scripts/analyze_hgdn_experiment_bundle.py",
            "--bundle-dir",
            str(args.bundle_stage_dir),
            "--output-dir",
            str(analysis_dir),
            "--select",
            "none",
            "--metric",
            "final_roundtrip_bpb",
            "--top",
            "20",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    rows_path = analysis_dir / "rows.json"
    if rows_path.is_file():
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        has_hgdn = any(
            row.get("family") == "HGDN"
            and row.get("completed")
            and row.get("final_roundtrip_bpb") is not None
            for row in rows
        )
        if has_hgdn:
            subprocess.run(
                [
                    args.python_bin,
                    "scripts/analyze_hgdn_experiment_bundle.py",
                    "--bundle-dir",
                    str(args.bundle_stage_dir),
                    "--output-dir",
                    str(analysis_dir),
                    "--decision-json",
                    str(args.bundle_stage_dir / "selected_hgdn.json"),
                    "--select",
                    "config",
                    "--metric",
                    "final_roundtrip_bpb",
                    "--confirm-top-n",
                    "2",
                    "--top",
                    "20",
                ],
                cwd=REPO_ROOT,
                check=True,
            )
    if incomplete or missing:
        print(
            "wallclock_decision: skipped because selected run logs are incomplete or missing"
        )
        return
    selected_keys = ",".join(spec.key for spec in selected_specs)
    subprocess.run(
        [
            args.python_bin,
            "scripts/resolve_hgdn_wallclock_decision.py",
            "--rows-json",
            str(rows_path),
            "--output-json",
            str(args.bundle_stage_dir / "wallclock_decision.json"),
            "--primary-config",
            str(args.primary_hgdn_config),
            "--primary-control-config",
            str(args.primary_control_config),
            "--secondary-config",
            str(args.secondary_hgdn_config),
            "--secondary-control-config",
            str(args.secondary_control_config),
            "--primary-control-margin",
            str(args.primary_control_margin),
            "--secondary-primary-margin",
            str(args.secondary_primary_margin),
            "--run-plan",
            selected_keys,
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def finalize_paths(args: argparse.Namespace) -> None:
    """Fill path defaults that depend on the run prefix.

    :param argparse.Namespace args: Parsed arguments.
    """
    prefix = args.run_prefix_base
    args.bundle_stage_dir = args.bundle_stage_dir or Path(
        f"local-scratch/{prefix}_bundle"
    )
    args.archive_output = args.archive_output or Path(
        f"local-scratch/{prefix}_bundle.7z"
    )
    args.command_log = args.command_log or Path(f"local-scratch/{prefix}_commands.sh")
    args.size_screen_output = args.size_screen_output or Path(
        f"local-scratch/{prefix}_size_screen"
    )


def main() -> int:
    """Run the local wallclock resolver.

    :return int: Shell-style exit code.
    """
    enable_line_buffering()
    os.chdir(REPO_ROOT)
    args = parse_args()
    finalize_paths(args)
    if args.max_wallclock_seconds <= 0:
        raise SystemExit("--max-wallclock-seconds must be > 0 for wallclock resolver")
    ensure_py7zr_available()

    modes = recurrence_modes(args)
    selected_keys = resolve_run_keys(args.run_plan)
    specs_by_key = run_specs(args, modes)
    selected_specs = [specs_by_key[key] for key in selected_keys]
    grad_accum_steps = resolve_grad_accum_steps(args.ngpu, args.grad_accum_steps)
    val_batch_size = resolve_val_batch_size(
        args.ngpu,
        grad_accum_steps,
        args.train_seq_len,
        requested_tokens=args.val_batch_size,
        requested_seqs=args.val_batch_seqs,
    )
    base_env = training_env(args, grad_accum_steps, val_batch_size)
    command_specs = [command_for_spec(args, spec, base_env) for spec in selected_specs]
    prelude = [
        " ".join(
            [
                shlex_quote(args.python_bin),
                "scripts/screen_hgdn_arch_sizes.py",
                "--config",
                shlex_quote(str(args.size_screen_config)),
                "--gdn-fla-recurrence-mode",
                shlex_quote(
                    args.primary_gdn_fla_recurrence_mode
                    or args.gdn_fla_recurrence_mode
                    or "direct"
                ),
                "--output-dir",
                shlex_quote(str(args.size_screen_output)),
            ]
        )
    ]

    print_plan(args, selected_specs, grad_accum_steps, val_batch_size)
    check_cuda_jobs(args.check_cuda_idle, args.allow_active_cuda_jobs)
    write_command_log(args.command_log, command_specs, prelude)
    check_existing_logs(args, selected_specs)

    exit_status = 0
    try:
        run_size_screen(args)
        for spec, command_spec in zip(selected_specs, command_specs, strict=True):
            if should_skip_run(args, spec):
                continue
            code = run_command(command_spec)
            if code != 0:
                exit_status = code
                break
    except KeyboardInterrupt:
        exit_status = 130
    finally:
        completed, incomplete, missing = build_bundle(
            args,
            selected_specs,
            grad_accum_steps=grad_accum_steps,
            val_batch_size=val_batch_size,
            exit_status=exit_status,
        )
        try:
            analyze_bundle(args, selected_specs, incomplete=incomplete, missing=missing)
        finally:
            create_7z_archive(args.archive_output, args.bundle_stage_dir)
            print(f"bundle_archive={args.archive_output}")
    return exit_status


def shlex_quote(value: str) -> str:
    """Quote one shell token for the command prelude.

    :param str value: Token value.
    :return str: Quoted token.
    """
    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(main())
