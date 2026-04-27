#!/usr/bin/env python3
"""Probe local HGDN microbatch size through an argparse runner."""

from __future__ import annotations

import argparse
import os
import shutil
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
    csv_ints,
    csv_items,
    diagnostic_env,
    enable_line_buffering,
    ensure_py7zr_available,
    filtered_config_env,
    log_completion_state,
    parse_perf_log,
    resolve_grad_accum_steps,
    run_command,
    write_command_log,
    write_rows_csv,
)


@dataclass(frozen=True)
class LadderSpec:
    """One batch-ladder run."""

    key: str
    label: str
    run_id: str
    config: Path | None
    batch_tokens: int
    local_batch_size: int
    trainer: str
    recurrence_mode: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-prefix-base", default="localhgdn_batch_ladder1")
    parser.add_argument("--bundle-stage-dir", type=Path, default=None)
    parser.add_argument("--archive-output", type=Path, default=None)
    parser.add_argument("--command-log", type=Path, default=None)
    parser.add_argument(
        "--configs",
        default=(
            "configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml,"
            "configs/hgdn/naive_contract_l8_d512_r0_m2.toml"
        ),
        help="Comma-separated hybrid-trainer configs to probe.",
    )
    parser.add_argument(
        "--batch-tokens",
        default="65536,131072,196608,262144,327680",
        help="Comma-separated global TRAIN_BATCH_TOKENS values.",
    )
    parser.add_argument("--include-exact-baseline", type=parse_bool_flag, default=False)
    parser.add_argument("--gdn-fla-recurrence-mode", default="direct_fused")
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--val-loss-every", type=int, default=0)
    parser.add_argument("--train-log-every", type=int, default=20)
    parser.add_argument("--min-val-seqs", type=int, default=512)
    parser.add_argument("--val-max-seqs", type=int, default=512)
    parser.add_argument("--val-batch-seqs", type=int, default=8)
    parser.add_argument("--max-wallclock-seconds", type=float, default=0.0)
    parser.add_argument("--perf-ignore-steps", type=int, default=10)
    parser.add_argument("--compile", type=parse_bool_flag, default=True)
    parser.add_argument("--compile-strategy", default="hybrid")
    parser.add_argument("--distributed-mode", default="parallel_muon")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-wandb", type=parse_bool_flag, default=False)
    parser.add_argument(
        "--wandb-mode", default="offline", choices=["online", "offline"]
    )
    parser.add_argument("--wandb-project", default="pg-hgdn-ablations")
    parser.add_argument("--wandb-watch", default="none")
    parser.add_argument("--wandb-watch-log-freq", type=int, default=25)
    parser.add_argument("--torchinductor-max-autotune", type=int, default=0)
    parser.add_argument("--torchinductor-max-autotune-gemm", type=int, default=0)
    parser.add_argument("--torch-logs", default="")
    parser.add_argument("--torch-trace", default="")
    parser.add_argument("--artifact-limit-bytes", type=int, default=16_000_000)
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
    parser.add_argument("--allow-existing-logs", type=parse_bool_flag, default=False)
    parser.add_argument("--skip-completed-runs", type=parse_bool_flag, default=True)
    parser.add_argument("--check-cuda-idle", type=parse_bool_flag, default=True)
    parser.add_argument("--allow-active-cuda-jobs", type=parse_bool_flag, default=False)
    parser.add_argument(
        "--stop-config-after-failure", type=parse_bool_flag, default=True
    )
    parser.add_argument("--omp-num-threads", type=int, default=1)
    parser.add_argument("--mkl-num-threads", type=int, default=1)
    parser.add_argument("--openblas-num-threads", type=int, default=1)
    parser.add_argument("--numexpr-num-threads", type=int, default=1)
    parser.add_argument("--nccl-ib-disable", type=int, default=1)
    return parser.parse_args()


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


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments that need cross-field checks.

    :param argparse.Namespace args: Parsed arguments.
    """
    if args.gdn_fla_recurrence_mode not in RECURRENCE_MODES:
        raise SystemExit(
            f"Unsupported --gdn-fla-recurrence-mode {args.gdn_fla_recurrence_mode!r}; "
            f"expected one of {', '.join(RECURRENCE_MODES)}"
        )
    if args.iterations <= args.perf_ignore_steps:
        raise SystemExit("--iterations must be greater than --perf-ignore-steps")
    if args.ngpu < 1:
        raise SystemExit("--ngpu must be >= 1")


def local_batch_size(
    batch_tokens: int, ngpu: int, grad_accum_steps: int, seq_len: int
) -> int:
    """Return per-rank microbatch size for one global token batch.

    :param int batch_tokens: Global training batch tokens.
    :param int ngpu: Number of local GPU processes.
    :param int grad_accum_steps: Gradient accumulation steps.
    :param int seq_len: Sequence length.
    :raises SystemExit: If the token batch is incompatible with the contract.
    :return int: Per-rank microbatch size in sequences.
    """
    denom = ngpu * grad_accum_steps * seq_len
    if batch_tokens % denom != 0:
        raise SystemExit(
            f"--batch-tokens value {batch_tokens} is not divisible by "
            f"ngpu*grad_accum_steps*train_seq_len={denom}"
        )
    out = batch_tokens // denom
    if out < 1:
        raise SystemExit(
            f"--batch-tokens value {batch_tokens} gives local_batch_size < 1"
        )
    return out


def build_specs(args: argparse.Namespace, grad_accum_steps: int) -> list[LadderSpec]:
    """Build the ordered ladder run list.

    :param argparse.Namespace args: Parsed arguments.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    :return list[LadderSpec]: Run specs.
    """
    configs = [Path(item) for item in csv_items(args.configs)]
    batches = csv_ints(args.batch_tokens)
    specs: list[LadderSpec] = []
    for config in configs:
        for batch_tokens in batches:
            local_bs = local_batch_size(
                batch_tokens,
                args.ngpu,
                grad_accum_steps,
                args.train_seq_len,
            )
            run_id = (
                f"{args.run_prefix_base}_{config.stem}_bt{batch_tokens}"
                f"_lbs{local_bs}_seq{args.train_seq_len}"
            )
            specs.append(
                LadderSpec(
                    key=config.stem,
                    label=f"batch ladder {config.stem} batch_tokens={batch_tokens}",
                    run_id=run_id,
                    config=config,
                    batch_tokens=batch_tokens,
                    local_batch_size=local_bs,
                    trainer="train_gpt_hybrid.py",
                    recurrence_mode=args.gdn_fla_recurrence_mode,
                )
            )
    if args.include_exact_baseline:
        for batch_tokens in batches:
            local_bs = local_batch_size(
                batch_tokens,
                args.ngpu,
                grad_accum_steps,
                args.train_seq_len,
            )
            run_id = (
                f"{args.run_prefix_base}_gpt_naive_baseline_bt{batch_tokens}"
                f"_lbs{local_bs}_seq{args.train_seq_len}"
            )
            specs.append(
                LadderSpec(
                    key="gpt_naive_baseline",
                    label=f"batch ladder exact baseline batch_tokens={batch_tokens}",
                    run_id=run_id,
                    config=None,
                    batch_tokens=batch_tokens,
                    local_batch_size=local_bs,
                    trainer="train_gpt.py",
                )
            )
    return specs


def common_train_env(args: argparse.Namespace, grad_accum_steps: int) -> dict[str, str]:
    """Build common trainer adapter values.

    :param argparse.Namespace args: Parsed arguments.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    :return dict[str, str]: Environment overlay consumed by current trainers.
    """
    val_batch_size = args.val_batch_seqs * args.train_seq_len
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
            "TRAIN_SEQ_LEN": str(args.train_seq_len),
            "VAL_LOSS_EVERY": str(args.val_loss_every),
            "TRAIN_LOG_EVERY": str(args.train_log_every),
            "MIN_VAL_SEQS": str(args.min_val_seqs),
            "VAL_MAX_SEQS": str(args.val_max_seqs),
            "VAL_BATCH_SIZE": str(val_batch_size),
            "ARTIFACT_LIMIT_BYTES": str(args.artifact_limit_bytes),
            "PERF_TIMING": "1",
            "PERF_IGNORE_STEPS": str(args.perf_ignore_steps),
            "PERF_SKIP_FINAL_EVAL": "1",
        }
    )
    return env


def command_for_spec(
    args: argparse.Namespace,
    spec: LadderSpec,
    base_env: dict[str, str],
) -> CommandSpec:
    """Build one executable command.

    :param argparse.Namespace args: Parsed arguments.
    :param LadderSpec spec: Ladder spec.
    :param dict[str, str] base_env: Common trainer environment.
    :return CommandSpec: Command spec.
    """
    env = dict(base_env)
    env.update(
        {
            "RUN_ID": spec.run_id,
            "TRAIN_BATCH_TOKENS": str(spec.batch_tokens),
        }
    )
    if spec.config is None:
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
    args: argparse.Namespace, specs: list[LadderSpec], grad_accum_steps: int
) -> None:
    """Print the batch-ladder launch plan.

    :param argparse.Namespace args: Parsed arguments.
    :param list[LadderSpec] specs: Planned runs.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    """
    print()
    print(">>> Local HGDN batch ladder")
    print(f"run_prefix_base={args.run_prefix_base}")
    print(f"configs={args.configs}")
    print(f"batch_tokens={args.batch_tokens}")
    print(f"iterations={args.iterations}")
    print(f"train_seq_len={args.train_seq_len}")
    print(f"grad_accum_steps={grad_accum_steps}")
    print(f"compile={int(args.compile)} compile_strategy={args.compile_strategy}")
    print(f"gdn_fla_recurrence_mode={args.gdn_fla_recurrence_mode}")
    print(f"planned_runs={len(specs)}")
    for spec in specs:
        config = spec.config if spec.config is not None else "exact train_gpt.py"
        print(
            f"  - {spec.run_id}: batch_tokens={spec.batch_tokens} "
            f"local_batch_size={spec.local_batch_size} config={config}"
        )


def check_existing_logs(args: argparse.Namespace, specs: list[LadderSpec]) -> None:
    """Refuse unsafe log reuse before launching.

    :param argparse.Namespace args: Parsed arguments.
    :param list[LadderSpec] specs: Planned runs.
    :raises SystemExit: If existing logs would be appended.
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
            + "\nUse a fresh --run-prefix-base or pass --allow-existing-logs only intentionally."
        )


def should_skip_run(args: argparse.Namespace, spec: LadderSpec) -> bool:
    """Return whether a completed run should be skipped.

    :param argparse.Namespace args: Parsed arguments.
    :param LadderSpec spec: Ladder spec.
    :return bool: Whether launch should be skipped.
    """
    log_path = REPO_ROOT / "logs" / f"{spec.run_id}.txt"
    if log_completion_state(log_path) == "complete" and args.skip_completed_runs:
        print()
        print(f">>> skip completed {spec.run_id}")
        return True
    return False


def infer_error_hint(log_path: Path, returncode: int) -> str:
    """Infer a compact error hint from a failed run log.

    :param Path log_path: Trainer log path.
    :param int returncode: Process return code.
    :return str: Error hint.
    """
    if returncode == 0:
        return ""
    if not log_path.is_file():
        return "no_log"
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    if "out of memory" in text or "cuda error: out of memory" in text:
        return "cuda_oom"
    if "train_batch_tokens must be divisible" in text:
        return "invalid_batch_contract"
    return "failed"


def row_for_spec(spec: LadderSpec, returncode: int) -> dict[str, Any]:
    """Build one output row from a run spec and log.

    :param LadderSpec spec: Ladder spec.
    :param int returncode: Process return code.
    :return dict[str, Any]: Result row.
    """
    log_path = REPO_ROOT / "logs" / f"{spec.run_id}.txt"
    row: dict[str, Any] = {
        "run_id": spec.run_id,
        "key": spec.key,
        "trainer": spec.trainer,
        "config": str(spec.config) if spec.config is not None else "",
        "gdn_fla_recurrence_mode": spec.recurrence_mode or "",
        "batch_tokens": spec.batch_tokens,
        "local_batch_size": spec.local_batch_size,
        "returncode": returncode,
        "log_path": str(log_path),
        "error_hint": infer_error_hint(log_path, returncode),
    }
    row.update(parse_perf_log(log_path))
    return row


def write_manifest(
    args: argparse.Namespace,
    specs: list[LadderSpec],
    rows: list[dict[str, Any]],
    grad_accum_steps: int,
    exit_status: int,
) -> None:
    """Write the batch-ladder manifest.

    :param argparse.Namespace args: Parsed arguments.
    :param list[LadderSpec] specs: Planned runs.
    :param list[dict[str, Any]] rows: Result rows.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    :param int exit_status: Overall exit status.
    """
    manifest = {
        "run_prefix_base": args.run_prefix_base,
        "archive_output": str(args.archive_output),
        "command_log": str(args.command_log),
        "exit_status": exit_status,
        "contract": {
            "ngpu": args.ngpu,
            "grad_accum_steps": grad_accum_steps,
            "iterations": args.iterations,
            "train_seq_len": args.train_seq_len,
            "val_loss_every": args.val_loss_every,
            "train_log_every": args.train_log_every,
            "min_val_seqs": args.min_val_seqs,
            "val_max_seqs": args.val_max_seqs,
            "val_batch_seqs": args.val_batch_seqs,
            "max_wallclock_seconds": args.max_wallclock_seconds,
            "perf_ignore_steps": args.perf_ignore_steps,
            "compile": args.compile,
            "compile_strategy": args.compile_strategy,
            "distributed_mode": args.distributed_mode,
            "torch_logs": args.torch_logs or None,
            "torch_trace": args.torch_trace or None,
            "torchinductor_max_autotune": args.torchinductor_max_autotune,
            "torchinductor_max_autotune_gemm": args.torchinductor_max_autotune_gemm,
            "data_path": str(args.data_path),
            "tokenizer_path": str(args.tokenizer_path),
            "vocab_size": args.vocab_size,
        },
        "runs": [
            {
                "run_id": spec.run_id,
                "key": spec.key,
                "label": spec.label,
                "trainer": spec.trainer,
                "config": str(spec.config) if spec.config is not None else None,
                "batch_tokens": spec.batch_tokens,
                "local_batch_size": spec.local_batch_size,
                "gdn_fla_recurrence_mode": spec.recurrence_mode,
            }
            for spec in specs
        ],
        "rows": rows,
    }
    write_json(args.bundle_stage_dir / "bundle_manifest.json", manifest)


def stage_bundle(
    args: argparse.Namespace,
    specs: list[LadderSpec],
    rows: list[dict[str, Any]],
    *,
    grad_accum_steps: int,
    exit_status: int,
) -> None:
    """Stage logs, configs, rows, manifest, and command log.

    :param argparse.Namespace args: Parsed arguments.
    :param list[LadderSpec] specs: Planned runs.
    :param list[dict[str, Any]] rows: Result rows.
    :param int grad_accum_steps: Resolved gradient accumulation steps.
    :param int exit_status: Overall exit status.
    """
    print()
    print(">>> stage batch ladder outputs")
    shutil.rmtree(args.bundle_stage_dir, ignore_errors=True)
    (args.bundle_stage_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.bundle_stage_dir / "configs").mkdir(parents=True, exist_ok=True)
    copy_existing(
        [spec.config for spec in specs if spec.config is not None],
        args.bundle_stage_dir / "configs",
    )
    if args.command_log.is_file():
        shutil.copy2(args.command_log, args.bundle_stage_dir / "commands.sh")
    for spec in specs:
        log_path = REPO_ROOT / "logs" / f"{spec.run_id}.txt"
        if log_path.is_file():
            shutil.copy2(log_path, args.bundle_stage_dir / "logs" / log_path.name)
    write_json(args.bundle_stage_dir / "rows.json", rows)
    write_rows_csv(args.bundle_stage_dir / "rows.csv", rows)
    write_manifest(args, specs, rows, grad_accum_steps, exit_status)
    print(f"bundle_dir={args.bundle_stage_dir}")


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a compact ladder summary.

    :param list[dict[str, Any]] rows: Result rows.
    """
    print()
    print(">>> batch ladder summary")
    for row in rows:
        speed = row.get("perf_step_ms")
        mem = row.get("peak_reserved_mib")
        status = row.get("completion_state")
        hint = row.get("error_hint")
        speed_text = f"{speed:.2f} ms/step" if isinstance(speed, float) else "n/a"
        mem_text = f"{mem} MiB reserved" if isinstance(mem, int) else "n/a"
        print(
            f"{row['key']} batch_tokens={row['batch_tokens']} "
            f"local_batch_size={row['local_batch_size']} status={status} "
            f"returncode={row['returncode']} speed={speed_text} mem={mem_text} "
            f"hint={hint or 'ok'}"
        )


def main() -> int:
    """Run the batch ladder.

    :return int: Shell-style exit code.
    """
    enable_line_buffering()
    os.chdir(REPO_ROOT)
    args = parse_args()
    finalize_paths(args)
    validate_args(args)
    ensure_py7zr_available()
    grad_accum_steps = resolve_grad_accum_steps(args.ngpu, args.grad_accum_steps)
    specs = build_specs(args, grad_accum_steps)
    base_env = common_train_env(args, grad_accum_steps)
    command_specs = [command_for_spec(args, spec, base_env) for spec in specs]

    print_plan(args, specs, grad_accum_steps)
    check_cuda_jobs(args.check_cuda_idle, args.allow_active_cuda_jobs)
    write_command_log(args.command_log, command_specs, prelude=[])
    check_existing_logs(args, specs)

    rows: list[dict[str, Any]] = []
    exit_status = 0
    failed_keys: set[str] = set()
    try:
        for spec, command_spec in zip(specs, command_specs, strict=True):
            if spec.key in failed_keys:
                continue
            if should_skip_run(args, spec):
                rows.append(row_for_spec(spec, 0))
                continue
            code = run_command(command_spec)
            rows.append(row_for_spec(spec, code))
            if code != 0:
                exit_status = code
                if args.stop_config_after_failure:
                    failed_keys.add(spec.key)
    except KeyboardInterrupt:
        exit_status = 130
    finally:
        stage_bundle(
            args,
            specs,
            rows,
            grad_accum_steps=grad_accum_steps,
            exit_status=exit_status,
        )
        print_summary(rows)
        create_7z_archive(args.archive_output, args.bundle_stage_dir)
        print(f"bundle_archive={args.archive_output}")
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
