#!/usr/bin/env python3
"""Run the H100 HGDN naive-baseline-contract comparison through argparse."""

from __future__ import annotations

import argparse
import os
import platform
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
    RECURRENCE_MODES,
    REPO_ROOT,
    bool_flag_value,
    common_thread_env,
    copy_existing,
    create_7z_archive,
    diagnostic_env,
    enable_line_buffering,
    ensure_py7zr_available,
    filtered_config_env,
    log_completion_state,
    resolve_grad_accum_steps,
    run_command,
    write_command_log,
)


@dataclass(frozen=True)
class H100RunSpec:
    """One H100 comparison leg."""

    key: str
    label: str
    trainer: str
    run_id: str
    config: Path | None = None


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
    parser.add_argument("--wandb-watch-log-freq", type=int, default=100)
    parser.add_argument("--run-prefix-base", default="h100naive1")
    parser.add_argument("--bundle-stage-dir", type=Path, default=None)
    parser.add_argument("--archive-output", type=Path, default=None)
    parser.add_argument("--command-log", type=Path, default=None)
    parser.add_argument("--allow-existing-logs", type=parse_bool_flag, default=False)
    parser.add_argument("--skip-completed-runs", type=parse_bool_flag, default=True)
    parser.add_argument("--torch-logs", default="")
    parser.add_argument("--torch-trace", default="")
    parser.add_argument("--omp-num-threads", type=int, default=1)
    parser.add_argument("--mkl-num-threads", type=int, default=1)
    parser.add_argument("--openblas-num-threads", type=int, default=1)
    parser.add_argument("--numexpr-num-threads", type=int, default=1)
    parser.add_argument("--nccl-ib-disable", type=int, default=1)
    parser.add_argument("--ngpu", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--train-batch-tokens", type=int, default=524_288)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--val-loss-every", type=int, default=200)
    parser.add_argument("--train-log-every", type=int, default=50)
    parser.add_argument("--val-batch-size", type=int, default=524_288)
    parser.add_argument("--min-val-seqs", type=int, default=512)
    parser.add_argument("--val-max-seqs", type=int, default=0)
    parser.add_argument("--max-wallclock-seconds", type=float, default=600.0)
    parser.add_argument("--compile", type=parse_bool_flag, default=True)
    parser.add_argument("--compile-strategy", default="hybrid")
    parser.add_argument("--attn-use-flash-attn3", type=parse_bool_flag, default=True)
    parser.add_argument("--distributed-mode", default="parallel_muon")
    parser.add_argument("--gdn-fla-recurrence-mode", default="direct")
    parser.add_argument("--muon-distributed-mode", default="packed_allreduce")
    parser.add_argument("--gdn-w-g-optimizer", default="matrix")
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
    parser.add_argument("--hgdn-config", type=Path, required=True)
    parser.add_argument("--attn-config", type=Path, default=None)
    parser.add_argument("--gpt-naive-run-id", default="")
    parser.add_argument("--hgdn-run-id", default="")
    parser.add_argument("--attn-run-id", default="")
    parser.add_argument("--naive-reference-name", default="2026-03-17_NaiveBaseline")
    parser.add_argument(
        "--naive-reference-roundtrip-bpb", type=float, default=1.22436570
    )
    parser.add_argument("--naive-reference-stop-bpb", type=float, default=1.2172)
    return parser.parse_args()


def infer_attention_control_config(hgdn_config: Path) -> Path:
    """Infer the matched attention-only diagnostic control config.

    :param Path hgdn_config: HGDN config path.
    :raises SystemExit: If the config shell cannot be inferred.
    :return Path: Matched attention-only config path.
    """
    name = hgdn_config.name
    mappings = (
        ("naive_contract_l8_d512_", "_m2.toml", "naive_contract_l8_d512_r0_m2.toml"),
        (
            "naive_contract_l8_d512_",
            "_m1p5.toml",
            "naive_contract_l8_d512_r0_m1p5.toml",
        ),
        (
            "naive_contract_l8_d512_",
            "_m1p25.toml",
            "naive_contract_l8_d512_r0_m1p25.toml",
        ),
        (
            "naive_contract_l8_d512_",
            "_m1p75.toml",
            "naive_contract_l8_d512_r0_m1p75.toml",
        ),
        ("naive_contract_l9_d512_", "_m2.toml", "naive_contract_l9_d512_r0_m2.toml"),
        (
            "naive_contract_l9_d512_",
            "_m1p75.toml",
            "naive_contract_l9_d512_r0_m1p75.toml",
        ),
    )
    for prefix, suffix, control in mappings:
        if name.startswith(prefix) and name.endswith(suffix):
            return Path("configs/hgdn") / control
    raise SystemExit(
        f"Could not infer matched attention-only control for {hgdn_config}"
    )


def finalize_args(args: argparse.Namespace) -> None:
    """Resolve path defaults and validate the H100 contract.

    :param argparse.Namespace args: Parsed arguments.
    :raises SystemExit: If the requested run is invalid.
    """
    prefix = args.run_prefix_base
    args.bundle_stage_dir = args.bundle_stage_dir or Path(
        f"local-scratch/{prefix}_bundle"
    )
    args.archive_output = args.archive_output or Path(
        f"local-scratch/{prefix}_bundle.7z"
    )
    args.command_log = args.command_log or Path(f"local-scratch/{prefix}_commands.sh")
    args.attn_config = args.attn_config or infer_attention_control_config(
        args.hgdn_config
    )
    if args.ngpu < 1:
        raise SystemExit("--ngpu must be >= 1")
    if args.gdn_fla_recurrence_mode not in RECURRENCE_MODES:
        raise SystemExit(
            f"Unsupported --gdn-fla-recurrence-mode {args.gdn_fla_recurrence_mode!r}; "
            f"expected one of {', '.join(RECURRENCE_MODES)}"
        )
    for path, name in (
        (args.hgdn_config, "--hgdn-config"),
        (args.attn_config, "--attn-config"),
    ):
        if not path.is_file():
            raise SystemExit(f"{name} not found: {path}")


def run_specs(args: argparse.Namespace) -> list[H100RunSpec]:
    """Build the three H100 comparison specs.

    :param argparse.Namespace args: Parsed arguments.
    :return list[H100RunSpec]: Ordered comparison specs.
    """
    prefix = args.run_prefix_base
    return [
        H100RunSpec(
            key="exact",
            label="exact repo naive baseline",
            trainer="train_gpt.py",
            run_id=args.gpt_naive_run_id
            or f"{prefix}_gpt_naive_baseline_seq{args.train_seq_len}",
        ),
        H100RunSpec(
            key="hgdn",
            label="naive-contract HGDN candidate",
            trainer="train_gpt_hybrid.py",
            run_id=args.hgdn_run_id
            or f"{prefix}_hybrid_naive_contract_seq{args.train_seq_len}",
            config=args.hgdn_config,
        ),
        H100RunSpec(
            key="attention_control",
            label="naive-contract attention-only baseline diagnostic control",
            trainer="train_gpt_hybrid.py",
            run_id=args.attn_run_id
            or f"{prefix}_attn_naive_contract_seq{args.train_seq_len}",
            config=args.attn_config,
        ),
    ]


def base_training_env(
    args: argparse.Namespace, grad_accum_steps: int
) -> dict[str, str]:
    """Build the common trainer environment overlay.

    :param argparse.Namespace args: Parsed arguments.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
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
            "GRAD_ACCUM_STEPS": str(grad_accum_steps),
            "PERF_SKIP_FINAL_EVAL": "0",
            "ITERATIONS": str(args.iterations),
            "MAX_WALLCLOCK_SECONDS": f"{args.max_wallclock_seconds:g}",
            "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
            "TRAIN_SEQ_LEN": str(args.train_seq_len),
            "VAL_LOSS_EVERY": str(args.val_loss_every),
            "TRAIN_LOG_EVERY": str(args.train_log_every),
            "VAL_BATCH_SIZE": str(args.val_batch_size),
            "MIN_VAL_SEQS": str(args.min_val_seqs),
            "VAL_MAX_SEQS": str(args.val_max_seqs),
            "DATA_PATH": str(args.data_path),
            "TOKENIZER_PATH": str(args.tokenizer_path),
            "VOCAB_SIZE": str(args.vocab_size),
        }
    )
    return env


def command_for_spec(
    args: argparse.Namespace, spec: H100RunSpec, base_env: dict[str, str]
) -> CommandSpec:
    """Build the command spec for one comparison leg.

    :param argparse.Namespace args: Parsed arguments.
    :param H100RunSpec spec: Run specification.
    :param dict[str, str] base_env: Common environment overlay.
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
                "ATTN_USE_FLASH_ATTN3": bool_flag_value(args.attn_use_flash_attn3),
                "DISTRIBUTED_MODE": args.distributed_mode,
                "MUON_DISTRIBUTED_MODE": args.muon_distributed_mode,
                "GDN_W_G_OPTIMIZER": args.gdn_w_g_optimizer,
                "WEIGHT_DECAY": f"{args.weight_decay:g}",
            }
        )
        env.update(filtered_config_env(spec.config, args.gdn_fla_recurrence_mode))
        command = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={args.ngpu}",
            "train_gpt_hybrid.py",
        ]
    return CommandSpec(spec.label, spec.run_id, env, command)


def print_plan(
    args: argparse.Namespace, specs: list[H100RunSpec], grad_accum_steps: int
) -> None:
    """Print the launch plan.

    :param argparse.Namespace args: Parsed arguments.
    :param list[H100RunSpec] specs: Run specs.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    """
    print()
    print(">>> H100 HGDN naive-baseline-contract sanity round")
    print(f"run_prefix_base={args.run_prefix_base}")
    print(f"ngpu={args.ngpu}")
    print(f"grad_accum_steps={grad_accum_steps}")
    print(f"iterations={args.iterations}")
    print(f"max_wallclock_seconds={args.max_wallclock_seconds:g}")
    print(f"train_batch_tokens={args.train_batch_tokens}")
    print(f"train_seq_len={args.train_seq_len}")
    print(f"val_batch_size={args.val_batch_size}")
    print(f"compile={int(args.compile)} compile_strategy={args.compile_strategy}")
    print(f"gdn_fla_recurrence_mode={args.gdn_fla_recurrence_mode}")
    print(f"hgdn_config={args.hgdn_config}")
    print(f"attn_config={args.attn_config}")
    print(f"wandb_mode={args.wandb_mode} use_wandb={int(args.use_wandb)}")
    for spec in specs:
        suffix = f" config={spec.config}" if spec.config else ""
        print(f"  - {spec.key}: {spec.run_id}{suffix}")


def check_existing_logs(args: argparse.Namespace, specs: list[H100RunSpec]) -> None:
    """Refuse accidental appends to existing logs.

    :param argparse.Namespace args: Parsed arguments.
    :param list[H100RunSpec] specs: Run specs.
    :raises SystemExit: If a conflicting log exists.
    """
    if args.allow_existing_logs:
        return
    conflicts: list[str] = []
    for spec in specs:
        log_path = REPO_ROOT / "logs" / f"{spec.run_id}.txt"
        state = log_completion_state(log_path)
        if state == "missing":
            continue
        if state == "complete" and args.skip_completed_runs:
            continue
        conflicts.append(f"{state}: {log_path}")
    if conflicts:
        raise SystemExit(
            "Refusing to append to existing run logs:\n"
            + "\n".join(conflicts)
            + "\nUse a fresh --run-prefix-base or --allow-existing-logs for an intentional append."
        )


def git_value(*args: str) -> str:
    """Return one git metadata value, or an empty string on failure.

    :param str args: Git command args.
    :return str: Git command output.
    """
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def run_manifest_entry(spec: H100RunSpec) -> dict[str, Any]:
    """Build one manifest run entry.

    :param H100RunSpec spec: Run spec.
    :return dict[str, Any]: Manifest entry.
    """
    entry: dict[str, Any] = {
        "key": spec.key,
        "label": spec.label,
        "trainer": spec.trainer,
        "run_id": spec.run_id,
    }
    if spec.config is not None:
        entry["config"] = str(spec.config)
    return entry


def build_bundle(
    args: argparse.Namespace,
    specs: list[H100RunSpec],
    *,
    grad_accum_steps: int,
    exit_status: int,
) -> None:
    """Stage logs, configs, manifest, and archive.

    :param argparse.Namespace args: Parsed arguments.
    :param list[H100RunSpec] specs: Run specs.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    :param int exit_status: Runner exit status.
    """
    print()
    print(">>> bundle outputs")
    shutil.rmtree(args.bundle_stage_dir, ignore_errors=True)
    (args.bundle_stage_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.bundle_stage_dir / "configs").mkdir(parents=True, exist_ok=True)
    copy_existing(
        [spec.config for spec in specs if spec.config is not None],
        args.bundle_stage_dir / "configs",
    )
    if args.command_log.is_file():
        shutil.copy2(args.command_log, args.bundle_stage_dir / "commands.sh")
    completed: list[str] = []
    incomplete: list[str] = []
    missing: list[str] = []
    for spec in specs:
        log_path = REPO_ROOT / "logs" / f"{spec.run_id}.txt"
        state = log_completion_state(log_path)
        if state == "complete":
            completed.append(spec.run_id)
            shutil.copy2(log_path, args.bundle_stage_dir / "logs" / log_path.name)
        elif state == "incomplete":
            incomplete.append(spec.run_id)
            shutil.copy2(log_path, args.bundle_stage_dir / "logs" / log_path.name)
        else:
            missing.append(spec.run_id)
    manifest = {
        "run_prefix_base": args.run_prefix_base,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "archive_output": str(args.archive_output),
        "command_log": str(args.command_log),
        "exit_status": exit_status,
        "matched_logs": not incomplete and not missing,
        "completed_log_count": len(completed),
        "completed_run_ids": completed,
        "incomplete_run_ids": incomplete,
        "missing_run_ids": missing,
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
            "val_batch_size": args.val_batch_size,
            "max_wallclock_seconds": args.max_wallclock_seconds,
            "compile": args.compile,
            "compile_strategy": args.compile_strategy,
            "attn_use_flash_attn3": args.attn_use_flash_attn3,
            "distributed_mode": args.distributed_mode,
            "gdn_fla_recurrence_mode": args.gdn_fla_recurrence_mode,
            "muon_distributed_mode": args.muon_distributed_mode,
            "gdn_w_g_optimizer": args.gdn_w_g_optimizer,
            "weight_decay": args.weight_decay,
            "data_path": str(args.data_path),
            "tokenizer_path": str(args.tokenizer_path),
            "vocab_size": args.vocab_size,
            "perf_skip_final_eval": False,
        },
        "provenance": {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "host_name": platform.node(),
        },
        "reference_record": {
            "name": args.naive_reference_name,
            "stop_step_bpb": args.naive_reference_stop_bpb,
            "roundtrip_bpb": args.naive_reference_roundtrip_bpb,
        },
        "runs": [run_manifest_entry(spec) for spec in specs],
    }
    write_json(args.bundle_stage_dir / "bundle_manifest.json", manifest)
    print(f"bundle_dir={args.bundle_stage_dir}")
    create_7z_archive(args.archive_output, args.bundle_stage_dir)
    print(f"bundle_archive={args.archive_output}")


def should_skip_run(args: argparse.Namespace, spec: H100RunSpec) -> bool:
    """Return whether an existing complete run should be skipped.

    :param argparse.Namespace args: Parsed arguments.
    :param H100RunSpec spec: Run spec.
    :return bool: Whether to skip this leg.
    """
    log_path = REPO_ROOT / "logs" / f"{spec.run_id}.txt"
    if args.skip_completed_runs and log_completion_state(log_path) == "complete":
        print()
        print(f">>> skip completed {spec.key}: {spec.run_id}")
        return True
    return False


def main() -> int:
    """Run the H100 comparison helper.

    :return int: Shell-style exit code.
    """
    enable_line_buffering()
    os.chdir(REPO_ROOT)
    args = parse_args()
    finalize_args(args)
    ensure_py7zr_available()
    grad_accum_steps = resolve_grad_accum_steps(args.ngpu, args.grad_accum_steps)
    specs = run_specs(args)
    base_env = base_training_env(args, grad_accum_steps)
    command_specs = [command_for_spec(args, spec, base_env) for spec in specs]
    prelude = [
        "# H100 comparison command log generated by scripts/run_h100_hgdn_naive_contract_round.py"
    ]
    print_plan(args, specs, grad_accum_steps)
    write_command_log(args.command_log, command_specs, prelude)
    check_existing_logs(args, specs)

    exit_status = 0
    try:
        for spec, command_spec in zip(specs, command_specs, strict=True):
            if should_skip_run(args, spec):
                continue
            code = run_command(command_spec)
            if code != 0:
                exit_status = code
                break
    except KeyboardInterrupt:
        exit_status = 130
    finally:
        build_bundle(
            args,
            specs,
            grad_accum_steps=grad_accum_steps,
            exit_status=exit_status,
        )
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
