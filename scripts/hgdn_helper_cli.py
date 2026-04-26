#!/usr/bin/env python3
"""Shared CLI helpers for HGDN shell wrappers.

:param None: This module exposes subcommands through ``argparse``.
:return None: Process exit code indicates command success.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import tomllib
from pathlib import Path
from typing import Any

from _repo_bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path()


def parse_bool_flag(value: str) -> bool:
    """Parse a shell-style boolean flag.

    :param str value: Input string, typically ``0``/``1`` or ``true``/``false``.
    :raises argparse.ArgumentTypeError: Raised when the flag is not recognized.
    :return bool: Parsed boolean value.
    """
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected one of 0/1/true/false/yes/no/on/off, got {value!r}"
    )


def load_toml_env(path: Path, *, alias_aware: bool) -> dict[str, str]:
    """Load environment-style values from one TOML file.

    :param Path path: TOML file to read.
    :param bool alias_aware: Whether to follow the repo's ``alias`` indirection.
    :return dict[str, str]: Flattened environment mapping.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    alias = data.get("alias")
    if alias_aware and alias is not None:
        return load_toml_env((path.parent / alias).resolve(), alias_aware=True)
    env = data.get("env", data)
    merged: dict[str, str] = {}
    for key, value in env.items():
        if isinstance(value, bool):
            value = "1" if value else "0"
        elif isinstance(value, list):
            items = []
            for item in value:
                if isinstance(item, bool):
                    items.append("1" if item else "0")
                else:
                    items.append(str(item))
            value = ",".join(items)
        merged[str(key)] = str(value)
    return merged


def require_py7zr() -> Any:
    """Import and return ``py7zr``.

    :raises RuntimeError: Raised when ``py7zr`` is missing.
    :return Any: Imported ``py7zr`` module.
    """
    try:
        import py7zr
    except ModuleNotFoundError as exc:  # pragma: no cover - shell-helper runtime guard
        raise RuntimeError(
            "py7zr is required; install it with `python -m pip install py7zr`"
        ) from exc
    return py7zr


def collect_logs(bundle_dir: Path, *, recursive: bool) -> list[str]:
    """Collect log paths from one bundle directory.

    :param Path bundle_dir: Bundle directory to inspect.
    :param bool recursive: Whether to collect logs recursively.
    :return list[str]: Sorted log paths relative to the bundle root or flat names.
    """
    if recursive:
        return sorted(
            str(path.relative_to(bundle_dir))
            for path in bundle_dir.rglob("*.log")
            if path.is_file()
        )
    return sorted(path.name for path in bundle_dir.glob("*.log") if path.is_file())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON document with stable formatting.

    :param Path path: Output path.
    :param dict[str, Any] payload: JSON payload to serialize.
    """
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def cmd_module_exists(args: argparse.Namespace) -> int:
    """Check whether one Python module is importable.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    return 0 if importlib.util.find_spec(args.module) is not None else 1


def cmd_create_7z(args: argparse.Namespace) -> int:
    """Create one ``.7z`` archive from a source path.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    py7zr = require_py7zr()
    archive_output = args.archive_output
    source_path = args.source_path
    archive_output.parent.mkdir(parents=True, exist_ok=True)
    archive_output.unlink(missing_ok=True)
    with py7zr.SevenZipFile(archive_output, "w") as archive:
        archive.writeall(source_path, arcname=source_path.name)
    return 0


def cmd_load_env(args: argparse.Namespace) -> int:
    """Merge TOML env files and print ``KEY=VALUE`` lines.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    merged: dict[str, str] = {}
    for path in args.path:
        merged.update(load_toml_env(path, alias_aware=args.alias_aware))
    for key, value in merged.items():
        print(f"{key}={value}")
    return 0


def cmd_write_local_naive_contract_search_manifest(
    args: argparse.Namespace,
) -> int:
    """Write the local HGDN naive-contract-search manifest.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    if not (len(args.config) == len(args.label) == len(args.run_id)):
        raise SystemExit("config/label/run-id counts must match")
    manifest = {
        "run_prefix_base": args.run_prefix_base,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "archive_output": str(args.archive_output),
        "exit_status": args.exit_status,
        "matched_logs": args.matched_logs,
        "completed_log_count": args.completed_log_count,
        "missing_run_ids": args.missing_run_id,
        "size_screen": {
            "config": args.size_screen_config,
            "output_dir": args.size_screen_output,
        },
        "candidates": [
            {"config": config, "label": label, "run_id": run_id}
            for config, label, run_id in zip(
                args.config, args.label, args.run_id, strict=True
            )
        ],
        "contract": {
            "torch_logs": args.torch_logs or None,
            "torch_trace": args.torch_trace or None,
            "torchinductor_max_autotune": args.torchinductor_max_autotune,
            "torchinductor_max_autotune_gemm": args.torchinductor_max_autotune_gemm,
            "iterations": args.iterations,
            "train_batch_tokens": args.train_batch_tokens,
            "train_seq_len": args.train_seq_len,
            "val_loss_every": args.val_loss_every,
            "train_log_every": args.train_log_every,
            "val_batch_size": args.val_batch_size,
            "max_wallclock_seconds": args.max_wallclock_seconds,
            "weight_decay": args.weight_decay,
            "perf_skip_final_eval": args.perf_skip_final_eval,
            "compile": args.compile_enabled,
            "compile_strategy": args.compile_strategy,
            "distributed_mode": args.distributed_mode,
            "muon_distributed_mode": args.muon_distributed_mode,
            "gdn_w_g_optimizer": args.gdn_w_g_optimizer,
        },
    }
    write_json(args.output, manifest)
    return 0


def cmd_write_h100_naive_contract_manifest(args: argparse.Namespace) -> int:
    """Write the naive-baseline-contract manifest.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    manifest = {
        "run_prefix_base": args.run_prefix_base,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "archive_output": str(args.archive_output),
        "command_log": args.command_log,
        "matched_logs": args.matched_logs,
        "contract": {
            "ngpu": args.ngpu,
            "iterations": args.iterations,
            "train_batch_tokens": args.train_batch_tokens,
            "train_seq_len": args.train_seq_len,
            "val_loss_every": args.val_loss_every,
            "train_log_every": args.train_log_every,
            "val_batch_size": args.val_batch_size,
            "max_wallclock_seconds": args.max_wallclock_seconds,
            "compile": args.compile_enabled,
            "compile_strategy": args.compile_strategy,
            "weight_decay": args.weight_decay,
            "torch_logs": args.torch_logs or None,
            "torch_trace": args.torch_trace or None,
            "omp_num_threads": args.omp_num_threads,
            "mkl_num_threads": args.mkl_num_threads,
            "openblas_num_threads": args.openblas_num_threads,
            "numexpr_num_threads": args.numexpr_num_threads,
            "attn_use_flash_attn3": args.attn_use_flash_attn3,
            "distributed_mode": args.distributed_mode,
        },
        "provenance": {
            "git_commit": args.git_commit,
            "git_branch": args.git_branch,
            "host_name": args.host_name,
            "timestamp_utc": args.timestamp_utc,
        },
        "reference_record": {
            "name": args.naive_reference_name,
            "stop_step_bpb": args.naive_reference_stop_bpb,
            "roundtrip_bpb": args.naive_reference_roundtrip_bpb,
        },
        "runs": [
            {
                "label": "exact repo naive baseline on naive-baseline contract",
                "trainer": "train_gpt.py",
                "mode": "direct",
                "run_id": args.gpt_naive_run_id,
                "data_path": args.baseline_data_path,
                "tokenizer_path": args.baseline_tokenizer_path,
                "vocab_size": args.baseline_vocab_size,
                "num_layers": 9,
                "model_dim": 512,
                "num_heads": 8,
                "num_kv_heads": 4,
                "mlp_mult": 2,
            },
            {
                "label": "HGDN candidate on naive-baseline contract",
                "trainer": "train_gpt_hybrid.py",
                "mode": "single",
                "run_id": args.hgdn_run_id,
                "config": args.hgdn_config,
            },
            {
                "label": "attention-only baseline diagnostic control on naive-baseline contract",
                "trainer": "train_gpt_hybrid.py",
                "mode": "single",
                "run_id": args.attn_run_id,
                "config": args.attn_config,
            },
        ],
    }
    write_json(args.output, manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    :return argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Shared HGDN shell-helper utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    module_exists = subparsers.add_parser(
        "module-exists", help="return success when one module is importable"
    )
    module_exists.add_argument("--module", required=True)
    module_exists.set_defaults(func=cmd_module_exists)

    create_7z = subparsers.add_parser("create-7z", help="create one .7z archive")
    create_7z.add_argument("--archive-output", type=Path, required=True)
    create_7z.add_argument("--source-path", type=Path, required=True)
    create_7z.set_defaults(func=cmd_create_7z)

    load_env = subparsers.add_parser(
        "load-env", help="merge TOML env files and print KEY=VALUE lines"
    )
    load_env.add_argument("--alias-aware", action="store_true")
    load_env.add_argument("--path", type=Path, nargs="+", required=True)
    load_env.set_defaults(func=cmd_load_env)

    local_naive_search = subparsers.add_parser(
        "write-local-naive-contract-search-manifest",
        help="write the local HGDN naive-contract search manifest",
    )
    local_naive_search.add_argument("--output", type=Path, required=True)
    local_naive_search.add_argument("--run-prefix-base", required=True)
    local_naive_search.add_argument("--wandb-project", required=True)
    local_naive_search.add_argument("--wandb-mode", required=True)
    local_naive_search.add_argument("--archive-output", type=Path, required=True)
    local_naive_search.add_argument("--exit-status", type=int, required=True)
    local_naive_search.add_argument(
        "--matched-logs", type=parse_bool_flag, required=True
    )
    local_naive_search.add_argument("--completed-log-count", type=int, required=True)
    local_naive_search.add_argument("--size-screen-config", required=True)
    local_naive_search.add_argument("--size-screen-output", required=True)
    local_naive_search.add_argument("--torch-logs", default="")
    local_naive_search.add_argument("--torch-trace", default="")
    local_naive_search.add_argument(
        "--torchinductor-max-autotune", type=int, required=True
    )
    local_naive_search.add_argument(
        "--torchinductor-max-autotune-gemm", type=int, required=True
    )
    local_naive_search.add_argument("--iterations", type=int, required=True)
    local_naive_search.add_argument("--train-batch-tokens", type=int, required=True)
    local_naive_search.add_argument("--train-seq-len", type=int, required=True)
    local_naive_search.add_argument("--val-loss-every", type=int, required=True)
    local_naive_search.add_argument("--train-log-every", type=int, required=True)
    local_naive_search.add_argument("--val-batch-size", type=int, required=True)
    local_naive_search.add_argument(
        "--max-wallclock-seconds", type=float, required=True
    )
    local_naive_search.add_argument("--weight-decay", type=float, required=True)
    local_naive_search.add_argument(
        "--perf-skip-final-eval", type=parse_bool_flag, required=True
    )
    local_naive_search.add_argument(
        "--compile-enabled", type=parse_bool_flag, required=True
    )
    local_naive_search.add_argument("--compile-strategy", required=True)
    local_naive_search.add_argument("--distributed-mode", required=True)
    local_naive_search.add_argument("--muon-distributed-mode", required=True)
    local_naive_search.add_argument("--gdn-w-g-optimizer", required=True)
    local_naive_search.add_argument("--config", action="append", default=[])
    local_naive_search.add_argument("--label", action="append", default=[])
    local_naive_search.add_argument("--run-id", action="append", default=[])
    local_naive_search.add_argument("--missing-run-id", action="append", default=[])
    local_naive_search.set_defaults(func=cmd_write_local_naive_contract_search_manifest)

    h100_naive = subparsers.add_parser(
        "write-h100-naive-contract-manifest",
        help="write the naive-baseline-contract manifest",
    )
    h100_naive.add_argument("--output", type=Path, required=True)
    h100_naive.add_argument("--run-prefix-base", required=True)
    h100_naive.add_argument("--wandb-project", required=True)
    h100_naive.add_argument("--wandb-mode", required=True)
    h100_naive.add_argument("--archive-output", type=Path, required=True)
    h100_naive.add_argument("--matched-logs", type=parse_bool_flag, required=True)
    h100_naive.add_argument("--command-log", required=True)
    h100_naive.add_argument("--torch-logs", default="")
    h100_naive.add_argument("--torch-trace", default="")
    h100_naive.add_argument("--omp-num-threads", type=int, required=True)
    h100_naive.add_argument("--mkl-num-threads", type=int, required=True)
    h100_naive.add_argument("--openblas-num-threads", type=int, required=True)
    h100_naive.add_argument("--numexpr-num-threads", type=int, required=True)
    h100_naive.add_argument("--nccl-ib-disable", type=int, required=True)
    h100_naive.add_argument(
        "--attn-use-flash-attn3", type=parse_bool_flag, required=True
    )
    h100_naive.add_argument("--distributed-mode", required=True)
    h100_naive.add_argument("--ngpu", type=int, required=True)
    h100_naive.add_argument("--iterations", type=int, required=True)
    h100_naive.add_argument("--train-batch-tokens", type=int, required=True)
    h100_naive.add_argument("--train-seq-len", type=int, required=True)
    h100_naive.add_argument("--val-loss-every", type=int, required=True)
    h100_naive.add_argument("--train-log-every", type=int, required=True)
    h100_naive.add_argument("--val-batch-size", type=int, required=True)
    h100_naive.add_argument("--max-wallclock-seconds", type=float, required=True)
    h100_naive.add_argument("--compile-enabled", type=parse_bool_flag, required=True)
    h100_naive.add_argument("--compile-strategy", required=True)
    h100_naive.add_argument("--muon-distributed-mode", required=True)
    h100_naive.add_argument("--gdn-w-g-optimizer", required=True)
    h100_naive.add_argument("--weight-decay", type=float, required=True)
    h100_naive.add_argument("--baseline-data-path", required=True)
    h100_naive.add_argument("--baseline-tokenizer-path", required=True)
    h100_naive.add_argument("--baseline-vocab-size", type=int, required=True)
    h100_naive.add_argument("--gpt-naive-run-id", required=True)
    h100_naive.add_argument("--hgdn-config", required=True)
    h100_naive.add_argument("--attn-config", required=True)
    h100_naive.add_argument("--hgdn-run-id", required=True)
    h100_naive.add_argument("--attn-run-id", required=True)
    h100_naive.add_argument("--naive-reference-name", required=True)
    h100_naive.add_argument(
        "--naive-reference-roundtrip-bpb", type=float, required=True
    )
    h100_naive.add_argument("--naive-reference-stop-bpb", type=float, required=True)
    h100_naive.add_argument("--git-commit", default="")
    h100_naive.add_argument("--git-branch", default="")
    h100_naive.add_argument("--host-name", default="")
    h100_naive.add_argument("--timestamp-utc", default="")
    h100_naive.set_defaults(func=cmd_write_h100_naive_contract_manifest)

    return parser


def main() -> int:
    """Run the shared helper CLI.

    :return int: Shell-style exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
