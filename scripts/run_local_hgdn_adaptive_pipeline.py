#!/usr/bin/env python3
"""Run the staged local HGDN adaptive pipeline through argparse."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _repo_bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path()

from hgdn_helper_cli import parse_bool_flag, write_json  # noqa: E402
from hgdn_local_runner import (  # noqa: E402
    RECURRENCE_MODES,
    REPO_ROOT,
    bool_flag_value,
    check_cuda_jobs,
    create_7z_archive,
    csv_items,
    enable_line_buffering,
    ensure_py7zr_available,
    resolve_grad_accum_steps,
    resolve_val_batch_size,
)

ADAPTER_ENV_KEYS = {
    "PYTHON_BIN",
    "USE_WANDB",
    "WANDB_MODE",
    "WANDB_PROJECT",
    "WANDB_WATCH",
    "WANDB_WATCH_LOG_FREQ",
    "RUN_PREFIX_BASE",
    "BUNDLE_STAGE_DIR",
    "PIPELINE_DIR",
    "ARCHIVE_OUTPUT",
    "COMMAND_LOG",
    "SIZE_SCREEN_OUTPUT",
    "SIZE_SCREEN_CONFIG",
    "TORCH_LOGS",
    "TORCH_TRACE",
    "ALLOW_EXISTING_LOGS",
    "CHECK_CUDA_IDLE",
    "ALLOW_ACTIVE_CUDA_JOBS",
    "NGPU",
    "TRAIN_BATCH_TOKENS",
    "TRAIN_SEQ_LEN",
    "GRAD_ACCUM_STEPS",
    "VAL_LOSS_EVERY",
    "TRAIN_LOG_EVERY",
    "MIN_VAL_SEQS",
    "VAL_MAX_SEQS",
    "VAL_BATCH_SIZE",
    "VAL_BATCH_SEQS",
    "MAX_WALLCLOCK_SECONDS",
    "COMPILE",
    "COMPILE_STRATEGY",
    "DISTRIBUTED_MODE",
    "WEIGHT_DECAY",
    "TORCHINDUCTOR_MAX_AUTOTUNE",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM",
    "DATA_PATH",
    "TOKENIZER_PATH",
    "VOCAB_SIZE",
    "ITERATIONS",
    "PERF_SKIP_FINAL_EVAL",
    "CANDIDATE_CONFIGS",
    "CONFIRM_CANDIDATE_CONFIGS",
    "CANDIDATE_INDEXES",
    "RUN_PREFIXES",
    "ALLOW_CUSTOM_CANDIDATE_CONFIGS",
    "GDN_FLA_RECURRENCE_MODE",
    "GDN_USE_DIRECT_FLA_LAYER_SEMANTICS",
}


@dataclass(frozen=True)
class RuntimePlan:
    """Resolved runtime fields derived from CLI arguments."""

    grad_accum_steps: int
    val_batch_size: int
    val_batch_seqs: int
    secondary_iterations: int
    secondary_perf_skip_final_eval: bool
    secondary_selection_metric: str
    pipeline_dir: Path
    archive_output: Path


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
    parser.add_argument("--run-prefix-base", default="localhgdnpipeline1")
    parser.add_argument("--pipeline-dir", type=Path, default=None)
    parser.add_argument("--archive-output", type=Path, default=None)
    parser.add_argument("--torch-logs", default="")
    parser.add_argument("--torch-trace", default="")
    parser.add_argument("--allow-existing-logs", type=parse_bool_flag, default=False)
    parser.add_argument("--check-cuda-idle", type=parse_bool_flag, default=True)
    parser.add_argument("--allow-active-cuda-jobs", type=parse_bool_flag, default=False)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--train-batch-tokens", type=int, default=131_072)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--val-loss-every", type=int, default=100)
    parser.add_argument("--train-log-every", type=int, default=25)
    parser.add_argument("--min-val-seqs", type=int, default=512)
    parser.add_argument("--val-max-seqs", type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument(
        "--val-batch-seqs",
        type=int,
        default=None,
        help="Validation batch in sequences. Defaults to 16 when --val-batch-size is unset.",
    )
    parser.add_argument("--max-wallclock-seconds", type=float, default=0.0)
    parser.add_argument("--compile", type=parse_bool_flag, default=True)
    parser.add_argument("--compile-strategy", default="hybrid")
    parser.add_argument("--distributed-mode", default="parallel_muon")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--torchinductor-max-autotune", type=int, default=0)
    parser.add_argument("--torchinductor-max-autotune-gemm", type=int, default=0)
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
    parser.add_argument("--run-stage0", type=parse_bool_flag, default=True)
    parser.add_argument("--run-stage1", type=parse_bool_flag, default=True)
    parser.add_argument("--run-stage2", type=parse_bool_flag, default=True)
    parser.add_argument("--run-stage3", type=parse_bool_flag, default=True)
    parser.add_argument("--recurrence-iterations", type=int, default=500)
    parser.add_argument("--screen-iterations", type=int, default=300)
    parser.add_argument("--confirm-iterations", type=int, default=500)
    parser.add_argument("--secondary-iterations", type=int, default=None)
    parser.add_argument(
        "--screen-perf-skip-final-eval", type=parse_bool_flag, default=True
    )
    parser.add_argument(
        "--confirm-perf-skip-final-eval", type=parse_bool_flag, default=False
    )
    parser.add_argument(
        "--secondary-perf-skip-final-eval", type=parse_bool_flag, default=None
    )
    parser.add_argument("--confirm-top-hgdn", type=int, default=2)
    parser.add_argument("--recurrence-selection-metric", default="equal_wallclock_bpb")
    parser.add_argument("--screen-selection-metric", default="auto")
    parser.add_argument("--confirm-selection-metric", default="final_roundtrip_bpb")
    parser.add_argument("--secondary-selection-metric", default=None)
    parser.add_argument("--secondary-force", type=parse_bool_flag, default=False)
    parser.add_argument(
        "--screen-candidate-configs",
        default=(
            "configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml,"
            "configs/hgdn/naive_contract_l8_d512_mid2_dk48_v2_m1p5.toml,"
            "configs/hgdn/naive_contract_l8_d512_r0_m1p5.toml,"
            "configs/hgdn/naive_contract_l8_d512_r0_m2.toml"
        ),
    )
    parser.add_argument(
        "--secondary-candidate-configs",
        default=(
            "configs/hgdn/naive_contract_l8_d512_olmoish_6g2a_v2_m1p25.toml,"
            "configs/hgdn/naive_contract_l8_d512_r0_m1p25.toml"
        ),
    )
    parser.add_argument("--confirm-candidate-configs", default="")
    parser.add_argument("--gdn-fla-recurrence-mode", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations before starting work.

    :param argparse.Namespace args: Parsed arguments.
    :raises SystemExit: If the requested plan is invalid.
    """
    if args.val_batch_size is not None and args.val_batch_seqs is not None:
        raise SystemExit("Set only one of --val-batch-size or --val-batch-seqs.")
    if args.ngpu < 1:
        raise SystemExit("--ngpu must be >= 1")
    for name in (
        "recurrence_iterations",
        "screen_iterations",
        "confirm_iterations",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if args.secondary_iterations is not None and args.secondary_iterations < 1:
        raise SystemExit("--secondary-iterations must be >= 1")
    if args.val_loss_every < 1:
        raise SystemExit("--val-loss-every must be >= 1 for adaptive decisions")
    if args.confirm_top_hgdn < 1:
        raise SystemExit("--confirm-top-hgdn must be >= 1")
    if not csv_items(args.screen_candidate_configs):
        raise SystemExit("--screen-candidate-configs resolved to zero configs")
    if not csv_items(args.secondary_candidate_configs):
        raise SystemExit("--secondary-candidate-configs resolved to zero configs")
    if (
        args.gdn_fla_recurrence_mode
        and args.gdn_fla_recurrence_mode not in RECURRENCE_MODES
    ):
        raise SystemExit(
            f"Unsupported --gdn-fla-recurrence-mode {args.gdn_fla_recurrence_mode!r}; "
            f"expected one of {', '.join(RECURRENCE_MODES)}"
        )
    if (
        args.screen_perf_skip_final_eval
        and args.screen_iterations % args.val_loss_every != 0
    ):
        raise SystemExit(
            "--screen-iterations must be divisible by --val-loss-every when "
            "--screen-perf-skip-final-eval is enabled."
        )
    if args.recurrence_iterations % args.val_loss_every != 0:
        raise SystemExit(
            "--recurrence-iterations must be divisible by --val-loss-every for clean promotion."
        )
    if (
        args.confirm_perf_skip_final_eval
        and args.confirm_iterations % args.val_loss_every != 0
    ):
        raise SystemExit(
            "--confirm-iterations must be divisible by --val-loss-every when "
            "--confirm-perf-skip-final-eval is enabled."
        )


def resolve_runtime(args: argparse.Namespace) -> RuntimePlan:
    """Resolve defaults that depend on multiple arguments.

    :param argparse.Namespace args: Parsed arguments.
    :return RuntimePlan: Resolved runtime plan.
    """
    grad_accum_steps = resolve_grad_accum_steps(args.ngpu, args.grad_accum_steps)
    requested_val_seqs = args.val_batch_seqs
    if requested_val_seqs is None and args.val_batch_size is None:
        requested_val_seqs = 16
    val_batch_size = resolve_val_batch_size(
        args.ngpu,
        grad_accum_steps,
        args.train_seq_len,
        requested_tokens=args.val_batch_size,
        requested_seqs=requested_val_seqs,
    )
    val_batch_seqs = val_batch_size // args.train_seq_len
    secondary_iterations = args.secondary_iterations or args.confirm_iterations
    secondary_perf_skip_final_eval = (
        args.secondary_perf_skip_final_eval
        if args.secondary_perf_skip_final_eval is not None
        else args.confirm_perf_skip_final_eval
    )
    if (
        secondary_perf_skip_final_eval
        and secondary_iterations % args.val_loss_every != 0
    ):
        raise SystemExit(
            "--secondary-iterations must be divisible by --val-loss-every when "
            "--secondary-perf-skip-final-eval is enabled."
        )
    secondary_selection_metric = (
        args.secondary_selection_metric or args.confirm_selection_metric
    )
    pipeline_dir = args.pipeline_dir or Path(
        f"local-scratch/{args.run_prefix_base}_pipeline"
    )
    archive_output = args.archive_output or Path(
        f"local-scratch/{args.run_prefix_base}_pipeline.7z"
    )
    return RuntimePlan(
        grad_accum_steps=grad_accum_steps,
        val_batch_size=val_batch_size,
        val_batch_seqs=val_batch_seqs,
        secondary_iterations=secondary_iterations,
        secondary_perf_skip_final_eval=secondary_perf_skip_final_eval,
        secondary_selection_metric=secondary_selection_metric,
        pipeline_dir=pipeline_dir,
        archive_output=archive_output,
    )


def stage_bundle_dir(stage_prefix: str) -> Path:
    """Return the bundle directory for one stage prefix.

    :param str stage_prefix: Stage run prefix.
    :return Path: Bundle directory.
    """
    return Path(f"local-scratch/{stage_prefix}_bundle")


def stage_archive(stage_prefix: str) -> Path:
    """Return the archive path for one stage prefix.

    :param str stage_prefix: Stage run prefix.
    :return Path: Archive path.
    """
    return Path(f"local-scratch/{stage_prefix}_bundle.7z")


def stage_command_log(stage_prefix: str) -> Path:
    """Return the command-log path for one stage prefix.

    :param str stage_prefix: Stage run prefix.
    :return Path: Command-log path.
    """
    return Path(f"local-scratch/{stage_prefix}_commands.sh")


def clean_env() -> dict[str, str]:
    """Return a subprocess environment without stale runner adapter keys.

    :return dict[str, str]: Cleaned environment.
    """
    env = os.environ.copy()
    for key in ADAPTER_ENV_KEYS:
        env.pop(key, None)
    return env


def common_stage_env(args: argparse.Namespace, runtime: RuntimePlan) -> dict[str, str]:
    """Build the common environment consumed by lower-level training helpers.

    :param argparse.Namespace args: Parsed arguments.
    :param RuntimePlan runtime: Resolved runtime plan.
    :return dict[str, str]: Common environment overlay.
    """
    env = {
        "PYTHON_BIN": args.python_bin,
        "USE_WANDB": bool_flag_value(args.use_wandb),
        "WANDB_MODE": args.wandb_mode,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_WATCH": args.wandb_watch,
        "WANDB_WATCH_LOG_FREQ": str(args.wandb_watch_log_freq),
        "TORCHINDUCTOR_MAX_AUTOTUNE": str(args.torchinductor_max_autotune),
        "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": str(args.torchinductor_max_autotune_gemm),
        "DATA_PATH": str(args.data_path),
        "TOKENIZER_PATH": str(args.tokenizer_path),
        "VOCAB_SIZE": str(args.vocab_size),
        "ALLOW_EXISTING_LOGS": bool_flag_value(args.allow_existing_logs),
        "NGPU": str(args.ngpu),
        "GRAD_ACCUM_STEPS": str(runtime.grad_accum_steps),
        "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
        "TRAIN_SEQ_LEN": str(args.train_seq_len),
        "VAL_LOSS_EVERY": str(args.val_loss_every),
        "TRAIN_LOG_EVERY": str(args.train_log_every),
        "MIN_VAL_SEQS": str(args.min_val_seqs),
        "VAL_MAX_SEQS": str(args.val_max_seqs),
        "VAL_BATCH_SIZE": str(runtime.val_batch_size),
        "VAL_BATCH_SEQS": "",
        "MAX_WALLCLOCK_SECONDS": f"{args.max_wallclock_seconds:g}",
        "COMPILE": bool_flag_value(args.compile),
        "COMPILE_STRATEGY": args.compile_strategy,
        "DISTRIBUTED_MODE": args.distributed_mode,
        "WEIGHT_DECAY": f"{args.weight_decay:g}",
    }
    if args.torch_logs:
        env["TORCH_LOGS"] = args.torch_logs
    if args.torch_trace:
        env["TORCH_TRACE"] = args.torch_trace
    return env


def run_stage_command(
    label: str, command: list[str], env_overlay: dict[str, str]
) -> None:
    """Run one required stage command.

    :param str label: Human-readable stage label.
    :param list[str] command: Command to execute.
    :param dict[str, str] env_overlay: Environment overlay.
    :raises SystemExit: If the command exits nonzero.
    """
    print()
    print(f">>> {label}")
    env = clean_env()
    env.update(env_overlay)
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def analyze_stage(
    args: argparse.Namespace,
    runtime: RuntimePlan,
    *,
    stage_name: str,
    bundle_dir: Path,
    select_kind: str,
    metric: str,
    decision_json: Path,
) -> None:
    """Analyze one stage bundle and write its decision JSON.

    :param argparse.Namespace args: Parsed arguments.
    :param RuntimePlan runtime: Resolved runtime plan.
    :param str stage_name: Stage name.
    :param Path bundle_dir: Bundle directory to analyze.
    :param str select_kind: Analyzer selection mode.
    :param str metric: Selection metric.
    :param Path decision_json: Output decision JSON.
    """
    output_dir = runtime.pipeline_dir / f"{stage_name}_analysis"
    run_stage_command(
        f"analyze {stage_name}",
        [
            args.python_bin,
            "scripts/analyze_hgdn_experiment_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--output-dir",
            str(output_dir),
            "--decision-json",
            str(decision_json),
            "--select",
            select_kind,
            "--metric",
            metric,
            "--confirm-top-n",
            str(args.confirm_top_hgdn),
            "--top",
            "20",
        ],
        {},
    )


def write_plan(args: argparse.Namespace, runtime: RuntimePlan) -> None:
    """Write the top-level pipeline plan JSON.

    :param argparse.Namespace args: Parsed arguments.
    :param RuntimePlan runtime: Resolved runtime plan.
    """
    runtime.pipeline_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        runtime.pipeline_dir / "pipeline_plan.json",
        {
            "run_prefix_base": args.run_prefix_base,
            "use_wandb": args.use_wandb,
            "wandb_mode": args.wandb_mode,
            "wandb_project": args.wandb_project,
            "ngpu": args.ngpu,
            "train_batch_tokens": args.train_batch_tokens,
            "train_seq_len": args.train_seq_len,
            "grad_accum_steps": runtime.grad_accum_steps,
            "val_loss_every": args.val_loss_every,
            "train_log_every": args.train_log_every,
            "min_val_seqs": args.min_val_seqs,
            "val_max_seqs": args.val_max_seqs,
            "val_batch_size": runtime.val_batch_size,
            "val_batch_seqs": runtime.val_batch_seqs,
            "compile": args.compile,
            "compile_strategy": args.compile_strategy,
            "distributed_mode": args.distributed_mode,
            "data_path": str(args.data_path),
            "tokenizer_path": str(args.tokenizer_path),
            "vocab_size": args.vocab_size,
            "recurrence_iterations": args.recurrence_iterations,
            "screen_iterations": args.screen_iterations,
            "confirm_iterations": args.confirm_iterations,
            "secondary_iterations": runtime.secondary_iterations,
            "confirm_top_hgdn": args.confirm_top_hgdn,
            "recurrence_selection_metric": args.recurrence_selection_metric,
            "screen_selection_metric": args.screen_selection_metric,
            "confirm_selection_metric": args.confirm_selection_metric,
            "secondary_selection_metric": runtime.secondary_selection_metric,
            "screen_candidate_configs": csv_items(args.screen_candidate_configs),
            "secondary_candidate_configs": csv_items(args.secondary_candidate_configs),
            "secondary_force": args.secondary_force,
        },
    )


def print_plan(args: argparse.Namespace, runtime: RuntimePlan) -> None:
    """Print the human-readable launch contract.

    :param argparse.Namespace args: Parsed arguments.
    :param RuntimePlan runtime: Resolved runtime plan.
    """
    print()
    print(">>> Local HGDN adaptive hierarchy")
    print(f"pipeline_dir={runtime.pipeline_dir}")
    print(f"archive_output={runtime.archive_output}")
    print(
        f"run_stage0={int(args.run_stage0)} recurrence_iterations={args.recurrence_iterations}"
    )
    print(
        f"run_stage1={int(args.run_stage1)} screen_iterations={args.screen_iterations}"
    )
    print(
        f"run_stage2={int(args.run_stage2)} confirm_iterations={args.confirm_iterations}"
    )
    print(
        f"run_stage3={int(args.run_stage3)} "
        f"secondary_iterations={runtime.secondary_iterations}"
    )
    print(f"screen_candidate_configs={args.screen_candidate_configs}")
    print(f"secondary_candidate_configs={args.secondary_candidate_configs}")
    print(f"confirm_top_hgdn={args.confirm_top_hgdn}")
    print(f"recurrence_selection_metric={args.recurrence_selection_metric}")
    print(f"screen_selection_metric={args.screen_selection_metric}")
    print(f"confirm_selection_metric={args.confirm_selection_metric}")
    print(f"secondary_selection_metric={runtime.secondary_selection_metric}")
    print(f"secondary_force={int(args.secondary_force)}")
    print(f"data_path={args.data_path}")
    print(f"tokenizer_path={args.tokenizer_path}")
    print(f"vocab_size={args.vocab_size}")
    print(f"grad_accum_steps={runtime.grad_accum_steps}")
    print(f"min_val_seqs={args.min_val_seqs}")
    print(f"val_max_seqs={args.val_max_seqs}")
    print(
        f"val_batch_size={runtime.val_batch_size} tokens "
        f"({runtime.val_batch_seqs} sequences)"
    )
    print("gates:")
    print("  stage0: recurrence mode matrix on v2_m1p5")
    print(
        "  stage1: bounded candidate/control screen using the selected recurrence mode"
    )
    print("  stage2: longer confirmation for top HGDN configs plus matched controls")
    print(
        "  stage3: conditional OLMo-ish 6G/2A sanity check when the primary beats control"
    )
    print("h100_handoff=none")


def run_recurrence_stage(args: argparse.Namespace, runtime: RuntimePlan) -> str:
    """Run stage 0 and return the selected recurrence mode.

    :param argparse.Namespace args: Parsed arguments.
    :param RuntimePlan runtime: Resolved runtime plan.
    :return str: Selected recurrence mode.
    """
    stage_prefix = f"{args.run_prefix_base}_s0_recur"
    check_cuda_jobs(args.check_cuda_idle, args.allow_active_cuda_jobs)
    env = common_stage_env(args, runtime)
    env.update(
        {
            "RUN_PREFIX_BASE": stage_prefix,
            "BUNDLE_STAGE_DIR": str(stage_bundle_dir(stage_prefix)),
            "ARCHIVE_OUTPUT": str(stage_archive(stage_prefix)),
            "COMMAND_LOG": str(stage_command_log(stage_prefix)),
            "CHECK_CUDA_IDLE": "0",
            "ITERATIONS": str(args.recurrence_iterations),
            "PERF_SKIP_FINAL_EVAL": "0",
        }
    )
    run_stage_command(
        "stage0 recurrence implementation matrix",
        ["bash", "scripts/run_local_hgdn_recurrence_matrix.sh"],
        env,
    )
    decision_json = runtime.pipeline_dir / "stage0_decision.json"
    analyze_stage(
        args,
        runtime,
        stage_name="stage0_recurrence",
        bundle_dir=stage_bundle_dir(stage_prefix),
        select_kind="mode",
        metric=args.recurrence_selection_metric,
        decision_json=decision_json,
    )
    decision = read_json(decision_json)
    selected = decision.get("selected_gdn_fla_recurrence_mode")
    if selected not in RECURRENCE_MODES:
        raise SystemExit(f"Stage 0 selected invalid recurrence mode: {selected!r}")
    return str(selected)


def run_search_stage(
    args: argparse.Namespace,
    runtime: RuntimePlan,
    *,
    stage_name: str,
    stage_prefix: str,
    iterations: int,
    perf_skip_final_eval: bool,
    selected_mode: str,
    candidate_configs: str,
    decision_kind: str,
    metric: str,
    decision_json: Path,
) -> None:
    """Run one config-search stage and analyze it.

    :param argparse.Namespace args: Parsed arguments.
    :param RuntimePlan runtime: Resolved runtime plan.
    :param str stage_name: Stage name.
    :param str stage_prefix: Stage run prefix.
    :param int iterations: Training iterations.
    :param bool perf_skip_final_eval: Whether trainers skip final roundtrip eval.
    :param str selected_mode: GDN recurrence mode.
    :param str candidate_configs: Comma-separated config list.
    :param str decision_kind: Analyzer selection mode.
    :param str metric: Analyzer selection metric.
    :param Path decision_json: Output decision JSON.
    """
    if selected_mode not in RECURRENCE_MODES:
        raise SystemExit(f"Unsupported recurrence mode: {selected_mode}")
    print()
    print(f">>> {stage_name}")
    print(f"selected_mode={selected_mode}")
    print(f"candidate_configs={candidate_configs}")
    check_cuda_jobs(args.check_cuda_idle, args.allow_active_cuda_jobs)
    env = common_stage_env(args, runtime)
    env.update(
        {
            "RUN_PREFIX_BASE": stage_prefix,
            "BUNDLE_STAGE_DIR": str(stage_bundle_dir(stage_prefix)),
            "ARCHIVE_OUTPUT": str(stage_archive(stage_prefix)),
            "COMMAND_LOG": str(stage_command_log(stage_prefix)),
            "SIZE_SCREEN_OUTPUT": f"local-scratch/{stage_prefix}_size_screen",
            "CANDIDATE_CONFIGS": candidate_configs,
            "GDN_FLA_RECURRENCE_MODE": selected_mode,
            "ITERATIONS": str(iterations),
            "PERF_SKIP_FINAL_EVAL": bool_flag_value(perf_skip_final_eval),
        }
    )
    run_stage_command(
        f"{stage_name} local naive-contract search",
        ["bash", "scripts/run_local_hgdn_naive_contract_search.sh"],
        env,
    )
    analyze_stage(
        args,
        runtime,
        stage_name=stage_name,
        bundle_dir=stage_bundle_dir(stage_prefix),
        select_kind=decision_kind,
        metric=metric,
        decision_json=decision_json,
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object.

    :param Path path: JSON path.
    :return dict[str, Any]: Parsed object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def decision_list(decision: dict[str, Any], key: str) -> str:
    """Return one decision list as a comma-separated string.

    :param dict[str, Any] decision: Decision JSON object.
    :param str key: List key.
    :return str: Comma-separated value.
    """
    value = decision.get(key)
    if isinstance(value, list):
        return ",".join(str(item) for item in value if item)
    return str(value or "")


def write_stage3_skip(
    runtime: RuntimePlan, reason: str, secondary_candidate_configs: str
) -> None:
    """Write the stage 3 skipped marker.

    :param RuntimePlan runtime: Resolved runtime plan.
    :param str reason: Skip reason.
    :param str secondary_candidate_configs: Secondary config CSV.
    """
    write_json(
        runtime.pipeline_dir / "stage3_skipped.json",
        {
            "stage3_skipped": True,
            "reason": reason,
            "secondary_candidate_configs": csv_items(secondary_candidate_configs),
        },
    )
    print(f"stage3_skipped={reason}")


def build_pipeline_bundle(runtime: RuntimePlan) -> None:
    """Archive the pipeline decision directory when it exists.

    :param RuntimePlan runtime: Resolved runtime plan.
    """
    if not runtime.pipeline_dir.is_dir():
        return
    print()
    print(">>> pipeline bundle")
    create_7z_archive(runtime.archive_output, runtime.pipeline_dir)
    print(f"pipeline_archive={runtime.archive_output}")


def run_pipeline(args: argparse.Namespace, runtime: RuntimePlan) -> None:
    """Run the staged pipeline.

    :param argparse.Namespace args: Parsed arguments.
    :param RuntimePlan runtime: Resolved runtime plan.
    """
    print_plan(args, runtime)
    write_plan(args, runtime)
    selected_mode = args.gdn_fla_recurrence_mode or "direct"
    if args.run_stage0:
        selected_mode = run_recurrence_stage(args, runtime)

    stage1_decision = runtime.pipeline_dir / "stage1_decision.json"
    if args.run_stage1:
        run_search_stage(
            args,
            runtime,
            stage_name="stage1_screen",
            stage_prefix=f"{args.run_prefix_base}_s1_screen",
            iterations=args.screen_iterations,
            perf_skip_final_eval=args.screen_perf_skip_final_eval,
            selected_mode=selected_mode,
            candidate_configs=args.screen_candidate_configs,
            decision_kind="config",
            metric=args.screen_selection_metric,
            decision_json=stage1_decision,
        )

    stage2_configs = args.confirm_candidate_configs
    if not stage2_configs and stage1_decision.is_file():
        stage2_configs = decision_list(
            read_json(stage1_decision), "selected_confirm_configs"
        )
    if args.run_stage2:
        if not stage2_configs:
            raise SystemExit(
                "No confirmation configs available. Run stage1 or set "
                "--confirm-candidate-configs."
            )
        run_search_stage(
            args,
            runtime,
            stage_name="stage2_confirm",
            stage_prefix=f"{args.run_prefix_base}_s2_confirm",
            iterations=args.confirm_iterations,
            perf_skip_final_eval=args.confirm_perf_skip_final_eval,
            selected_mode=selected_mode,
            candidate_configs=stage2_configs,
            decision_kind="config",
            metric=args.confirm_selection_metric,
            decision_json=runtime.pipeline_dir / "stage2_decision.json",
        )

    if args.run_stage3:
        stage2_decision = runtime.pipeline_dir / "stage2_decision.json"
        if not stage2_decision.is_file():
            write_stage3_skip(
                runtime, "missing_stage2_decision", args.secondary_candidate_configs
            )
        else:
            decision = read_json(stage2_decision)
            selected_beats_control = bool(decision.get("selected_beats_control"))
            if not args.secondary_force and not selected_beats_control:
                write_stage3_skip(
                    runtime,
                    "primary_did_not_beat_matched_control",
                    args.secondary_candidate_configs,
                )
            else:
                run_search_stage(
                    args,
                    runtime,
                    stage_name="stage3_secondary_sanity",
                    stage_prefix=f"{args.run_prefix_base}_s3_secondary",
                    iterations=runtime.secondary_iterations,
                    perf_skip_final_eval=runtime.secondary_perf_skip_final_eval,
                    selected_mode=selected_mode,
                    candidate_configs=args.secondary_candidate_configs,
                    decision_kind="config",
                    metric=runtime.secondary_selection_metric,
                    decision_json=runtime.pipeline_dir / "stage3_decision.json",
                )


def main() -> int:
    """Run the argparse adaptive pipeline.

    :return int: Shell-style exit code.
    """
    enable_line_buffering()
    os.chdir(REPO_ROOT)
    args = parse_args()
    validate_args(args)
    runtime = resolve_runtime(args)
    ensure_py7zr_available()
    status = 0
    try:
        run_pipeline(args, runtime)
    except KeyboardInterrupt:
        status = 130
    except SystemExit as exc:
        status = int(exc.code or 0)
    finally:
        try:
            build_pipeline_bundle(runtime)
        except Exception as exc:
            print(f"pipeline_bundle_error:{exc}", file=sys.stderr)
            if status == 0:
                status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
