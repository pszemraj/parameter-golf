#!/usr/bin/env python3
"""Structured launcher for HGDN helper workflows.

This launcher keeps the existing shell helpers as the execution backend, but
adds a real CLI surface so common HGDN runs do not require long inline env
blocks. It also supports optional TOML env configs for repeatable launch
setups.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

HGDN_PRESETS: dict[str, dict[str, str]] = {
    "default": {},
    "convcontig": {
        "GDN_CONV_OUTPUT_CONTIGUOUS": "1",
    },
    "packed-qkv": {
        "GDN_CONV_OUTPUT_CONTIGUOUS": "1",
        "GDN_USE_PACKED_QKV_CONV": "1",
        "GDN_USE_PACKED_QKV_PROJ": "1",
    },
    "current-winner": {
        "GDN_CONV_OUTPUT_CONTIGUOUS": "1",
        "GDN_USE_PACKED_QKV_CONV": "1",
        "GDN_USE_PACKED_QKV_PROJ": "1",
        "GDN_CONTROL_PROJ_FP32": "0",
    },
    "current-winner-cuda-fused": {
        "GDN_CONV_OUTPUT_CONTIGUOUS": "1",
        "GDN_USE_PACKED_QKV_CONV": "1",
        "GDN_USE_PACKED_QKV_PROJ": "1",
        "GDN_CONTROL_PROJ_FP32": "0",
        "GDN_OUTPUT_NORM_FP32": "1",
        "GDN_USE_CUDA_FUSED_FRONTEND": "1",
        "GDN_USE_CUDA_FUSED_OUTPUT": "1",
    },
    "current-winner-cuda-output-only": {
        "GDN_CONV_OUTPUT_CONTIGUOUS": "1",
        "GDN_USE_PACKED_QKV_CONV": "1",
        "GDN_USE_PACKED_QKV_PROJ": "1",
        "GDN_CONTROL_PROJ_FP32": "0",
        "GDN_OUTPUT_NORM_FP32": "1",
        "GDN_USE_CUDA_FUSED_OUTPUT": "1",
    },
}

COMMON_ENV_ARGS: tuple[tuple[str, str], ...] = (
    ("run_prefix", "RUN_PREFIX"),
    ("train_seq_len", "TRAIN_SEQ_LEN"),
    ("train_batch_tokens", "TRAIN_BATCH_TOKENS"),
    ("hybrid_gdn_ratio", "HYBRID_GDN_RATIO"),
    ("hybrid_mlp_mult", "HYBRID_MLP_MULT"),
    ("depth_mlp_mult", "DEPTH_MLP_MULT"),
    ("compile_strategy", "COMPILE_STRATEGY"),
    ("iterations", "ITERATIONS"),
    ("train_log_every", "TRAIN_LOG_EVERY"),
    ("profile_dir", "PROFILE_DIR"),
    ("profile_wait", "PROFILE_WAIT"),
    ("profile_warmup", "PROFILE_WARMUP"),
    ("profile_active", "PROFILE_ACTIVE"),
)


def scalar_to_env(value: Any) -> str:
    """Convert a Python scalar to an environment-string value.

    :param Any value: Value read from argparse or TOML.
    :raises TypeError: Raised when the value type cannot be serialized.
    :return str: Normalized environment string.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float, str)):
        return str(value)
    raise TypeError(f"Unsupported config value type: {type(value)!r}")


def parse_kv_assignment(raw: str) -> tuple[str, str]:
    """Parse a `KEY=VALUE` assignment string.

    :param str raw: Raw assignment passed on the command line.
    :raises ValueError: Raised when `raw` is not `KEY=VALUE`.
    :return tuple[str, str]: Parsed key and value.
    """
    if "=" not in raw:
        raise ValueError(f"Expected KEY=VALUE assignment, got: {raw!r}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Expected non-empty KEY in assignment: {raw!r}")
    return key, value


def load_env_config(path: Path | None) -> dict[str, str]:
    """Load a TOML env config.

    Supported shape:

    ```toml
    [env]
    GDN_USE_PACKED_QKV_CONV = 1
    GDN_USE_PACKED_QKV_PROJ = 1
    ```

    :param Path | None path: Optional config path.
    :raises TypeError: Raised when the TOML structure is not supported.
    :return dict[str, str]: Environment overrides loaded from TOML.
    """
    if path is None:
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    env_table = data.get("env", data)
    if not isinstance(env_table, dict):
        raise TypeError(f"Expected [env] table in {path}, got {type(env_table)!r}")
    return {str(k): scalar_to_env(v) for k, v in env_table.items()}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared launcher arguments to a subparser.

    :param argparse.ArgumentParser parser: Parser to extend.
    :return None: Mutates `parser` in place.
    """
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional TOML env config. Supports either a top-level env table or [env].",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(HGDN_PRESETS),
        default="default",
        help="Named HGDN preset. CLI flags override preset values.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force USE_WANDB=0 and WANDB_MODE=offline.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Force USE_WANDB=1 and WANDB_MODE=online.",
    )
    parser.add_argument("--run-prefix", help="Set RUN_PREFIX.")
    parser.add_argument("--train-seq-len", type=int, help="Set TRAIN_SEQ_LEN.")
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        help="Set TRAIN_BATCH_TOKENS.",
    )
    parser.add_argument(
        "--hybrid-gdn-ratio",
        type=int,
        help="Set HYBRID_GDN_RATIO.",
    )
    parser.add_argument(
        "--hybrid-mlp-mult",
        type=float,
        help="Set HYBRID_MLP_MULT.",
    )
    parser.add_argument(
        "--depth-mlp-mult",
        type=float,
        help="Set DEPTH_MLP_MULT.",
    )
    parser.add_argument(
        "--compile-strategy",
        choices=("model", "hybrid"),
        help="Set COMPILE_STRATEGY.",
    )
    parser.add_argument("--iterations", type=int, help="Set ITERATIONS.")
    parser.add_argument(
        "--train-log-every",
        type=int,
        help="Set TRAIN_LOG_EVERY.",
    )
    parser.add_argument("--profile-dir", help="Set PROFILE_DIR.")
    parser.add_argument("--profile-wait", type=int, help="Set PROFILE_WAIT.")
    parser.add_argument("--profile-warmup", type=int, help="Set PROFILE_WARMUP.")
    parser.add_argument("--profile-active", type=int, help="Set PROFILE_ACTIVE.")
    parser.add_argument(
        "--packed-qkv-conv",
        action="store_true",
        help="Set GDN_USE_PACKED_QKV_CONV=1.",
    )
    parser.add_argument(
        "--packed-qkv-proj",
        action="store_true",
        help="Set GDN_USE_PACKED_QKV_PROJ=1.",
    )
    parser.add_argument(
        "--conv-output-contiguous",
        action="store_true",
        help="Set GDN_CONV_OUTPUT_CONTIGUOUS=1.",
    )
    parser.add_argument(
        "--control-proj-bf16",
        action="store_true",
        help="Set GDN_CONTROL_PROJ_FP32=0.",
    )
    parser.add_argument(
        "--cuda-fused-frontend",
        action="store_true",
        help="Set GDN_USE_CUDA_FUSED_FRONTEND=1.",
    )
    parser.add_argument(
        "--cuda-fused-output",
        action="store_true",
        help="Set GDN_USE_CUDA_FUSED_OUTPUT=1.",
    )
    parser.add_argument(
        "--cuda-jit-build",
        action="store_true",
        help="Set GDN_CUDA_ALLOW_JIT_BUILD=1.",
    )
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Additional passthrough env override. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved env and backend command without executing it.",
    )


def parse_args() -> argparse.Namespace:
    """Parse launcher arguments.

    :return argparse.Namespace: Parsed launcher arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/hgdn.py h100-profile hybrid-eager --preset current-winner --run-prefix h100k5\n"
            "  python scripts/hgdn.py h100-perf perf --preset current-winner --run-prefix h100k5 --offline\n"
            "  conda run -s --name pg python scripts/hgdn.py arch-size-screen --config configs/hgdn/current_winner_retune.toml\n"
            "  conda run -s --name pg python scripts/hgdn.py fixed2k-compare --name h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048 --name h100k6_fixed2k_depth_mlp4.0_seq2048 --reference h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048\n"
            "  python scripts/hgdn.py local-phase1 --config configs/hgdn/current_winner.toml --run-prefix rtx4070_phase1\n"
            "  python scripts/hgdn.py preflight --preset current-winner\n"
            "  conda run -s --name pg python scripts/hgdn.py preflight --preset current-winner-cuda-fused"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="Run the direct CUDA HGDN preflight.",
    )
    add_common_args(preflight)

    local_phase1 = subparsers.add_parser(
        "local-phase1",
        help="Run the local RTX 4070 phase-1 bundle sequentially.",
    )
    add_common_args(local_phase1)

    arch_size = subparsers.add_parser(
        "arch-size-screen",
        help="Run the CPU-only HGDN architecture size screen.",
    )
    arch_size.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "hgdn" / "current_winner_retune.toml",
        help="TOML config for scripts/screen_hgdn_arch_sizes.py.",
    )
    arch_size.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory for the screen bundle.",
    )
    arch_size.add_argument(
        "--row-limit",
        type=int,
        default=20,
        help="Maximum number of candidates to print to stdout.",
    )
    arch_size.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved backend command without executing it.",
    )

    fixed2k_compare = subparsers.add_parser(
        "fixed2k-compare",
        help="Compare completed HGDN fixed-step W&B runs into a structured bundle.",
    )
    fixed2k_compare.add_argument("--entity", default="pszemraj")
    fixed2k_compare.add_argument("--project", default="pg-hconv-ablations")
    fixed2k_compare.add_argument(
        "--name",
        action="append",
        default=[],
        help="Exact run name to include. Repeatable.",
    )
    fixed2k_compare.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Substring filter for run names. Repeatable.",
    )
    fixed2k_compare.add_argument(
        "--reference",
        help="Exact run name to treat as the comparison reference.",
    )
    fixed2k_compare.add_argument(
        "--output-dir",
        type=Path,
        help="Optional bundle directory for the comparison report.",
    )
    fixed2k_compare.add_argument(
        "--eval-step",
        action="append",
        type=int,
        default=[],
        help="Sampled eval step to include. Repeatable.",
    )
    fixed2k_compare.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved backend command without executing it.",
    )

    h100_perf = subparsers.add_parser(
        "h100-perf",
        help="Run the 1xH100 perf/fixed-step helper.",
    )
    add_common_args(h100_perf)
    h100_perf.add_argument(
        "mode",
        nargs="?",
        default="perf",
        choices=("perf", "fixed2k", "fixed2k-hybrid", "all"),
        help="Backend mode for scripts/run_h100_single_gpu_hgdn.sh.",
    )

    h100_profile = subparsers.add_parser(
        "h100-profile",
        help="Run the 1xH100 profiling helper.",
    )
    add_common_args(h100_profile)
    h100_profile.add_argument(
        "mode",
        nargs="?",
        default="hybrid",
        choices=(
            "hybrid",
            "depth",
            "both",
            "hybrid-eager",
            "depth-eager",
            "both-eager",
        ),
        help="Backend mode for scripts/run_h100_single_gpu_hgdn_profile.sh.",
    )
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> dict[str, str]:
    """Resolve the final env override dictionary for a command.

    :param argparse.Namespace args: Parsed launcher arguments.
    :return dict[str, str]: Final environment overrides.
    """
    env: dict[str, str] = {}
    env.update(load_env_config(args.config))
    env.update(HGDN_PRESETS[args.preset])

    for arg_name, env_name in COMMON_ENV_ARGS:
        value = getattr(args, arg_name, None)
        if value is not None:
            env[env_name] = scalar_to_env(value)

    if args.offline and args.online:
        raise ValueError("Use only one of --offline or --online.")
    if args.offline:
        env["USE_WANDB"] = "0"
        env["WANDB_MODE"] = "offline"
    elif args.online:
        env["USE_WANDB"] = "1"
        env["WANDB_MODE"] = "online"

    if args.packed_qkv_conv:
        env["GDN_USE_PACKED_QKV_CONV"] = "1"
    if args.packed_qkv_proj:
        env["GDN_USE_PACKED_QKV_PROJ"] = "1"
    if args.conv_output_contiguous:
        env["GDN_CONV_OUTPUT_CONTIGUOUS"] = "1"
    if args.control_proj_bf16:
        env["GDN_CONTROL_PROJ_FP32"] = "0"
    if args.cuda_fused_frontend:
        env["GDN_USE_CUDA_FUSED_FRONTEND"] = "1"
    if args.cuda_fused_output:
        env["GDN_USE_CUDA_FUSED_OUTPUT"] = "1"
    if args.cuda_jit_build:
        env["GDN_CUDA_ALLOW_JIT_BUILD"] = "1"

    for raw in args.set:
        key, value = parse_kv_assignment(raw)
        env[key] = value
    return env


def command_for_args(args: argparse.Namespace) -> list[str]:
    """Build the backend command to execute.

    :param argparse.Namespace args: Parsed launcher arguments.
    :return list[str]: Backend command vector.
    """
    if args.command == "preflight":
        return ["bash", str(REPO_ROOT / "scripts" / "run_hgdn_cuda_preflight.sh")]
    if args.command == "local-phase1":
        return ["bash", str(REPO_ROOT / "scripts" / "run_hgdn_local_phase1.sh")]
    if args.command == "arch-size-screen":
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "screen_hgdn_arch_sizes.py"),
            "--config",
            str(args.config),
            "--row-limit",
            str(args.row_limit),
        ]
        if args.output_dir is not None:
            command.extend(["--output-dir", str(args.output_dir)])
        return command
    if args.command == "fixed2k-compare":
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "compare_hgdn_fixed2k.py"),
            "--entity",
            args.entity,
            "--project",
            args.project,
        ]
        for name in args.name:
            command.extend(["--name", name])
        for value in args.contains:
            command.extend(["--contains", value])
        if args.reference is not None:
            command.extend(["--reference", args.reference])
        if args.output_dir is not None:
            command.extend(["--output-dir", str(args.output_dir)])
        for step in args.eval_step:
            command.extend(["--eval-step", str(step)])
        return command
    if args.command == "h100-perf":
        return [
            "bash",
            str(REPO_ROOT / "scripts" / "run_h100_single_gpu_hgdn.sh"),
            args.mode,
        ]
    if args.command == "h100-profile":
        return [
            "bash",
            str(REPO_ROOT / "scripts" / "run_h100_single_gpu_hgdn_profile.sh"),
            args.mode,
        ]
    raise ValueError(f"Unsupported command: {args.command}")


def render_command(env: dict[str, str], command: list[str]) -> str:
    """Render a shell-style preview string.

    :param dict[str, str] env: Environment overrides.
    :param list[str] command: Backend command vector.
    :return str: Shell-like preview.
    """
    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
    cmd = " ".join(shlex.quote(part) for part in command)
    if env_prefix:
        return f"{env_prefix} {cmd}"
    return cmd


def main() -> int:
    """Run the launcher entrypoint.

    :return int: Process exit code.
    """
    args = parse_args()
    try:
        env_overrides = (
            {}
            if args.command in {"arch-size-screen", "fixed2k-compare"}
            else build_env(args)
        )
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"error: {exc}", file=sys.stderr)
        return 2
    command = command_for_args(args)
    preview = render_command(env_overrides, command)
    if args.dry_run:
        print(preview)
        return 0
    env = os.environ.copy()
    env.update(env_overrides)
    print(f">>> launcher: {preview}")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
