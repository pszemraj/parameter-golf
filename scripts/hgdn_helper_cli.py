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


def load_extension_status(provider: str | None) -> dict[str, Any] | None:
    """Load one optional extension status provider.

    :param str | None provider: Import target in ``module:function`` form.
    :return dict[str, Any] | None: Provider payload or an error payload.
    """
    if not provider:
        return None
    try:
        module_name, func_name = provider.split(":", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return func()
    except Exception as exc:  # pragma: no cover - best-effort bundle metadata
        return {"error": str(exc), "loaded": False}


def append_bundle_metadata(
    metadata_path: Path,
    *,
    exit_status: int,
    extension_status: dict[str, Any] | None,
) -> None:
    """Append runtime bundle metadata to ``metadata.txt``.

    :param Path metadata_path: Metadata file path.
    :param int exit_status: Final exit status for the helper.
    :param dict[str, Any] | None extension_status: Optional extension status payload.
    """
    with metadata_path.open("a", encoding="utf-8") as fh:
        fh.write(f"bundle_exit_status={exit_status}\n")
        if extension_status is not None:
            fh.write(
                "extension_status_json="
                + json.dumps(extension_status, sort_keys=True)
                + "\n"
            )


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


def cmd_write_corekernel_bundle(args: argparse.Namespace) -> int:
    """Write the active HGDN core-kernel bundle manifest.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    extension_status = load_extension_status(args.extension_status_provider)
    append_bundle_metadata(
        args.metadata_path,
        exit_status=args.exit_status,
        extension_status=extension_status,
    )
    manifest = {
        "mode": args.mode,
        "run_prefix": args.run_prefix,
        "trainer_run_id": args.trainer_run_id,
        "archive_output": str(args.archive_output),
        "exit_status": args.exit_status,
        "paths": {
            "commands": args.commands_file.name,
            "metadata": args.metadata_path.name,
            "logs": collect_logs(args.bundle_dir, recursive=True),
        },
        "contract": {
            "torch_cuda_arch_list": args.torch_cuda_arch_list,
            "gdn_corekernel_rec_chunk_t": args.rec_chunk_t,
            "hk_timing_repeats": args.timing_repeats,
            "hk_trainer_launcher_mode": args.launcher_mode,
            "compile_strategy_default": args.compile_strategy_default,
            "hk_cases_dir": args.cases_dir,
            "torchinductor_cache_dir": args.trainer_cache_dir,
            "torch_logs": args.torch_logs or None,
            "iterations": args.iterations,
            "compile_warmup_steps": args.compile_warmup_steps,
            "train_seq_len": args.train_seq_len,
            "train_batch_tokens": args.train_batch_tokens,
            "val_loss_every": args.val_loss_every,
            "train_log_every": args.train_log_every,
        },
        "extension_status": extension_status,
        "provenance": {
            "git_commit": args.git_commit,
            "git_branch": args.git_branch,
            "host_name": args.host_name,
            "timestamp_utc": args.timestamp_utc,
        },
    }
    write_json(args.bundle_dir / "bundle_manifest.json", manifest)
    return 0


def cmd_write_megakernel_bundle(args: argparse.Namespace) -> int:
    """Write the HGDN megakernel bundle manifest.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    extension_status = load_extension_status(args.extension_status_provider)
    append_bundle_metadata(
        args.metadata_path,
        exit_status=args.exit_status,
        extension_status=extension_status,
    )
    manifest = {
        "mode": args.mode,
        "run_prefix": args.run_prefix,
        "trainer_run_id": args.trainer_run_id,
        "archive_output": str(args.archive_output),
        "exit_status": args.exit_status,
        "paths": {
            "commands": args.commands_file.name,
            "metadata": args.metadata_path.name,
            "logs": collect_logs(args.bundle_dir, recursive=False),
        },
        "contract": {
            "torch_cuda_arch_list": args.torch_cuda_arch_list,
            "gdn_megakernel_rec_chunk_t": args.rec_chunk_t,
            "mk_timing_repeats": args.timing_repeats,
            "mk_cases_dir": args.cases_dir,
            "torchinductor_cache_dir": args.trainer_cache_dir,
            "torch_logs": args.torch_logs or None,
            "iterations": args.iterations,
            "compile_warmup_steps": args.compile_warmup_steps,
            "train_seq_len": args.train_seq_len,
            "train_batch_tokens": args.train_batch_tokens,
            "val_loss_every": args.val_loss_every,
            "train_log_every": args.train_log_every,
        },
        "extension_status": extension_status,
        "provenance": {
            "git_commit": args.git_commit,
            "git_branch": args.git_branch,
            "host_name": args.host_name,
            "timestamp_utc": args.timestamp_utc,
        },
    }
    write_json(args.bundle_dir / "bundle_manifest.json", manifest)
    return 0


def cmd_write_local_resize_manifest(args: argparse.Namespace) -> int:
    """Write the local HGDN resize-round manifest.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    manifest = {
        "run_prefix_base": args.run_prefix_base,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "archive_output": str(args.archive_output),
        "matched_logs": args.matched_logs,
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
            "compile": args.compile_enabled,
            "compile_strategy": args.compile_strategy,
        },
        "run_ids": args.run_id,
    }
    write_json(args.output, manifest)
    return 0


def cmd_write_local_naive_contract_search_manifest(
    args: argparse.Namespace,
) -> int:
    """Write the local HGDN naive-contract-search manifest.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    manifest = {
        "run_prefix_base": args.run_prefix_base,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "archive_output": str(args.archive_output),
        "matched_logs": args.matched_logs,
        "size_screen": {
            "config": args.size_screen_config,
            "output_dir": args.size_screen_output,
        },
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
            "compile": args.compile_enabled,
            "compile_strategy": args.compile_strategy,
        },
        "run_ids": args.run_id,
    }
    write_json(args.output, manifest)
    return 0


def cmd_write_h100_bridge_manifest(args: argparse.Namespace) -> int:
    """Write the exact 8x H100 bridge-round manifest.

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
            "torch_logs": args.torch_logs or None,
            "torch_trace": args.torch_trace or None,
            "omp_num_threads": args.omp_num_threads,
            "mkl_num_threads": args.mkl_num_threads,
            "openblas_num_threads": args.openblas_num_threads,
            "numexpr_num_threads": args.numexpr_num_threads,
            "nccl_ib_disable": args.nccl_ib_disable,
            "attn_use_flash_attn3": args.attn_use_flash_attn3,
            "distributed_mode": args.distributed_mode,
        },
        "provenance": {
            "git_commit": args.git_commit,
            "git_branch": args.git_branch,
            "host_name": args.host_name,
            "timestamp_utc": args.timestamp_utc,
        },
        "runs": [
            {
                "label": "HGDN finalist",
                "mode": "single",
                "run_id": args.hgdn_run_id,
                "config": args.hgdn_config,
                "kernel_config": args.hgdn_kernel_config,
            },
            {
                "label": "attention-only baseline",
                "mode": "depth",
                "run_id": args.attn_run_id,
                "mlp_mult": args.depth_mlp_mult,
            },
        ],
    }
    write_json(args.output, manifest)
    return 0


def cmd_write_h100_resize_manifest(args: argparse.Namespace) -> int:
    """Write the H100 HGDN resize-round manifest.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    if not (
        len(args.config)
        == len(args.label)
        == len(args.batch_token)
        == len(args.grad_accum_steps)
        == len(args.run_prefix)
    ):
        raise SystemExit(
            "config/label/batch-token/grad-accum-steps/run-prefix counts must match"
        )
    plan = []
    for run_prefix, config, label, train_batch_tokens, grad_accum_steps in zip(
        args.run_prefix,
        args.config,
        args.label,
        args.batch_token,
        args.grad_accum_steps,
        strict=True,
    ):
        plan.append(
            {
                "run_prefix": run_prefix,
                "config": config,
                "label": label,
                "train_batch_tokens": train_batch_tokens,
                "grad_accum_steps": grad_accum_steps,
                "local_batch_size": train_batch_tokens
                // (grad_accum_steps * args.fixed2k_seq_len),
            }
        )
    manifest = {
        "run_prefix_base": args.run_prefix_base,
        "run_prefixes": args.run_prefix,
        "configs": args.config,
        "labels": args.label,
        "plan": plan,
        "compare_reference": args.compare_reference,
        "compare_reference_entity": args.compare_reference_entity,
        "compare_reference_project": args.compare_reference_project,
        "archive_output": str(args.archive_output),
        "command_log": args.command_log,
        "matched_logs": args.matched_logs,
        "torch_logs": args.torch_logs or None,
        "torch_trace": args.torch_trace or None,
        "contract": {
            "fixed2k_iterations": args.fixed2k_iterations,
            "train_batch_tokens_override": args.fixed2k_train_batch_tokens_override,
            "fixed2k_seq_len": args.fixed2k_seq_len,
            "fixed2k_val_loss_every": args.fixed2k_val_loss_every,
            "fixed2k_train_log_every": args.fixed2k_train_log_every,
            "grad_accum_steps_override": args.grad_accum_steps_override,
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
                "label": "HGDN finalist on naive-baseline contract",
                "trainer": "train_gpt_hybrid.py",
                "mode": "single",
                "run_id": args.hgdn_run_id,
                "config": args.hgdn_config,
                "kernel_config": args.hgdn_kernel_config,
            },
            {
                "label": "attention-only control on naive-baseline contract",
                "trainer": "train_gpt_hybrid.py",
                "mode": "depth",
                "run_id": args.attn_run_id,
                "num_layers": 9,
                "model_dim": 512,
                "num_heads": 8,
                "num_kv_heads": 4,
                "gdn_ratio": 0,
                "mlp_mult": 2,
            },
        ],
    }
    write_json(args.output, manifest)
    return 0


def cmd_write_h100_compile_tiebreak_manifest(args: argparse.Namespace) -> int:
    """Write the exact-8x compile-tiebreak manifest.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    if len(args.run_id) != len(args.compile_strategy):
        raise SystemExit("run-id and compile-strategy counts must match")
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
        "runs": [
            {
                "label": f"HGDN finalist COMPILE_STRATEGY={compile_strategy}",
                "trainer": "train_gpt_hybrid.py",
                "mode": "single-live14",
                "run_id": run_id,
                "compile_strategy": compile_strategy,
                "hgdn_config": args.hgdn_config,
                "hgdn_kernel_config": args.hgdn_kernel_config,
            }
            for run_id, compile_strategy in zip(
                args.run_id, args.compile_strategy, strict=True
            )
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

    core_bundle = subparsers.add_parser(
        "write-corekernel-bundle", help="write the HGDN core-kernel bundle manifest"
    )
    core_bundle.add_argument("--bundle-dir", type=Path, required=True)
    core_bundle.add_argument("--mode", required=True)
    core_bundle.add_argument("--run-prefix", required=True)
    core_bundle.add_argument("--trainer-run-id", required=True)
    core_bundle.add_argument("--torch-cuda-arch-list", required=True)
    core_bundle.add_argument("--rec-chunk-t", type=int, required=True)
    core_bundle.add_argument("--timing-repeats", type=int, required=True)
    core_bundle.add_argument("--launcher-mode", required=True)
    core_bundle.add_argument("--compile-strategy-default", required=True)
    core_bundle.add_argument("--cases-dir", required=True)
    core_bundle.add_argument("--trainer-cache-dir", required=True)
    core_bundle.add_argument("--archive-output", type=Path, required=True)
    core_bundle.add_argument("--commands-file", type=Path, required=True)
    core_bundle.add_argument("--metadata-path", type=Path, required=True)
    core_bundle.add_argument("--torch-logs", default="")
    core_bundle.add_argument("--iterations", type=int, required=True)
    core_bundle.add_argument("--compile-warmup-steps", type=int, required=True)
    core_bundle.add_argument("--train-seq-len", type=int, required=True)
    core_bundle.add_argument("--train-batch-tokens", type=int, required=True)
    core_bundle.add_argument("--val-loss-every", type=int, required=True)
    core_bundle.add_argument("--train-log-every", type=int, required=True)
    core_bundle.add_argument("--git-commit", required=True)
    core_bundle.add_argument("--git-branch", required=True)
    core_bundle.add_argument("--host-name", required=True)
    core_bundle.add_argument("--timestamp-utc", required=True)
    core_bundle.add_argument("--exit-status", type=int, required=True)
    core_bundle.add_argument(
        "--extension-status-provider",
        default="hgdn_megakernel:extension_status",
    )
    core_bundle.set_defaults(func=cmd_write_corekernel_bundle)

    mega_bundle = subparsers.add_parser(
        "write-megakernel-bundle", help="write the HGDN megakernel bundle manifest"
    )
    mega_bundle.add_argument("--bundle-dir", type=Path, required=True)
    mega_bundle.add_argument("--mode", required=True)
    mega_bundle.add_argument("--run-prefix", required=True)
    mega_bundle.add_argument("--trainer-run-id", required=True)
    mega_bundle.add_argument("--torch-cuda-arch-list", required=True)
    mega_bundle.add_argument("--rec-chunk-t", type=int, required=True)
    mega_bundle.add_argument("--timing-repeats", type=int, required=True)
    mega_bundle.add_argument("--cases-dir", required=True)
    mega_bundle.add_argument("--trainer-cache-dir", required=True)
    mega_bundle.add_argument("--archive-output", type=Path, required=True)
    mega_bundle.add_argument("--commands-file", type=Path, required=True)
    mega_bundle.add_argument("--metadata-path", type=Path, required=True)
    mega_bundle.add_argument("--torch-logs", default="")
    mega_bundle.add_argument("--iterations", type=int, required=True)
    mega_bundle.add_argument("--compile-warmup-steps", type=int, required=True)
    mega_bundle.add_argument("--train-seq-len", type=int, required=True)
    mega_bundle.add_argument("--train-batch-tokens", type=int, required=True)
    mega_bundle.add_argument("--val-loss-every", type=int, required=True)
    mega_bundle.add_argument("--train-log-every", type=int, required=True)
    mega_bundle.add_argument("--git-commit", required=True)
    mega_bundle.add_argument("--git-branch", required=True)
    mega_bundle.add_argument("--host-name", required=True)
    mega_bundle.add_argument("--timestamp-utc", required=True)
    mega_bundle.add_argument("--exit-status", type=int, required=True)
    mega_bundle.add_argument(
        "--extension-status-provider",
        default="hgdn_megakernel:extension_status",
    )
    mega_bundle.set_defaults(func=cmd_write_megakernel_bundle)

    local_resize = subparsers.add_parser(
        "write-local-resize-manifest",
        help="write the local HGDN resize-round manifest",
    )
    local_resize.add_argument("--output", type=Path, required=True)
    local_resize.add_argument("--run-prefix-base", required=True)
    local_resize.add_argument("--wandb-project", required=True)
    local_resize.add_argument("--wandb-mode", required=True)
    local_resize.add_argument("--archive-output", type=Path, required=True)
    local_resize.add_argument("--matched-logs", type=parse_bool_flag, required=True)
    local_resize.add_argument("--torch-logs", default="")
    local_resize.add_argument("--torch-trace", default="")
    local_resize.add_argument("--torchinductor-max-autotune", type=int, required=True)
    local_resize.add_argument(
        "--torchinductor-max-autotune-gemm", type=int, required=True
    )
    local_resize.add_argument("--iterations", type=int, required=True)
    local_resize.add_argument("--train-batch-tokens", type=int, required=True)
    local_resize.add_argument("--train-seq-len", type=int, required=True)
    local_resize.add_argument("--val-loss-every", type=int, required=True)
    local_resize.add_argument("--train-log-every", type=int, required=True)
    local_resize.add_argument("--val-batch-size", type=int, required=True)
    local_resize.add_argument("--compile-enabled", type=parse_bool_flag, required=True)
    local_resize.add_argument("--compile-strategy", required=True)
    local_resize.add_argument("--run-id", action="append", default=[])
    local_resize.set_defaults(func=cmd_write_local_resize_manifest)

    local_naive_search = subparsers.add_parser(
        "write-local-naive-contract-search-manifest",
        help="write the local HGDN naive-contract search manifest",
    )
    local_naive_search.add_argument("--output", type=Path, required=True)
    local_naive_search.add_argument("--run-prefix-base", required=True)
    local_naive_search.add_argument("--wandb-project", required=True)
    local_naive_search.add_argument("--wandb-mode", required=True)
    local_naive_search.add_argument("--archive-output", type=Path, required=True)
    local_naive_search.add_argument(
        "--matched-logs", type=parse_bool_flag, required=True
    )
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
        "--compile-enabled", type=parse_bool_flag, required=True
    )
    local_naive_search.add_argument("--compile-strategy", required=True)
    local_naive_search.add_argument("--run-id", action="append", default=[])
    local_naive_search.set_defaults(func=cmd_write_local_naive_contract_search_manifest)

    h100_bridge = subparsers.add_parser(
        "write-h100-bridge-manifest",
        help="write the exact-8x HGDN bridge-round manifest",
    )
    h100_bridge.add_argument("--output", type=Path, required=True)
    h100_bridge.add_argument("--run-prefix-base", required=True)
    h100_bridge.add_argument("--wandb-project", required=True)
    h100_bridge.add_argument("--wandb-mode", required=True)
    h100_bridge.add_argument("--archive-output", type=Path, required=True)
    h100_bridge.add_argument("--matched-logs", type=parse_bool_flag, required=True)
    h100_bridge.add_argument("--command-log", required=True)
    h100_bridge.add_argument("--torch-logs", default="")
    h100_bridge.add_argument("--torch-trace", default="")
    h100_bridge.add_argument("--omp-num-threads", type=int, required=True)
    h100_bridge.add_argument("--mkl-num-threads", type=int, required=True)
    h100_bridge.add_argument("--openblas-num-threads", type=int, required=True)
    h100_bridge.add_argument("--numexpr-num-threads", type=int, required=True)
    h100_bridge.add_argument(
        "--attn-use-flash-attn3", type=parse_bool_flag, required=True
    )
    h100_bridge.add_argument("--distributed-mode", required=True)
    h100_bridge.add_argument("--ngpu", type=int, required=True)
    h100_bridge.add_argument("--iterations", type=int, required=True)
    h100_bridge.add_argument("--train-batch-tokens", type=int, required=True)
    h100_bridge.add_argument("--train-seq-len", type=int, required=True)
    h100_bridge.add_argument("--val-loss-every", type=int, required=True)
    h100_bridge.add_argument("--train-log-every", type=int, required=True)
    h100_bridge.add_argument("--val-batch-size", type=int, required=True)
    h100_bridge.add_argument("--max-wallclock-seconds", type=float, required=True)
    h100_bridge.add_argument("--compile-enabled", type=parse_bool_flag, required=True)
    h100_bridge.add_argument("--compile-strategy", required=True)
    h100_bridge.add_argument("--depth-mlp-mult", type=float, required=True)
    h100_bridge.add_argument("--hgdn-config", required=True)
    h100_bridge.add_argument("--hgdn-kernel-config", required=True)
    h100_bridge.add_argument("--hgdn-run-id", required=True)
    h100_bridge.add_argument("--attn-run-id", required=True)
    h100_bridge.add_argument("--git-commit", default="")
    h100_bridge.add_argument("--git-branch", default="")
    h100_bridge.add_argument("--host-name", default="")
    h100_bridge.add_argument("--timestamp-utc", default="")
    h100_bridge.set_defaults(func=cmd_write_h100_bridge_manifest)

    h100_resize = subparsers.add_parser(
        "write-h100-resize-manifest",
        help="write the H100 HGDN resize-round manifest",
    )
    h100_resize.add_argument("--output", type=Path, required=True)
    h100_resize.add_argument("--run-prefix-base", required=True)
    h100_resize.add_argument("--compare-reference", required=True)
    h100_resize.add_argument("--compare-reference-entity", required=True)
    h100_resize.add_argument("--compare-reference-project", required=True)
    h100_resize.add_argument("--archive-output", type=Path, required=True)
    h100_resize.add_argument("--matched-logs", type=parse_bool_flag, required=True)
    h100_resize.add_argument("--torch-logs", default="")
    h100_resize.add_argument("--torch-trace", default="")
    h100_resize.add_argument("--command-log", required=True)
    h100_resize.add_argument("--fixed2k-iterations", type=int, required=True)
    h100_resize.add_argument(
        "--fixed2k-train-batch-tokens-override", type=int, default=None
    )
    h100_resize.add_argument("--fixed2k-seq-len", type=int, required=True)
    h100_resize.add_argument("--fixed2k-val-loss-every", type=int, required=True)
    h100_resize.add_argument("--fixed2k-train-log-every", type=int, required=True)
    h100_resize.add_argument("--grad-accum-steps-override", type=int, default=None)
    h100_resize.add_argument("--config", action="append", default=[])
    h100_resize.add_argument("--label", action="append", default=[])
    h100_resize.add_argument("--batch-token", type=int, action="append", default=[])
    h100_resize.add_argument(
        "--grad-accum-steps", type=int, action="append", default=[]
    )
    h100_resize.add_argument("--run-prefix", action="append", default=[])
    h100_resize.set_defaults(func=cmd_write_h100_resize_manifest)

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
    h100_naive.add_argument("--weight-decay", type=float, required=True)
    h100_naive.add_argument("--baseline-data-path", required=True)
    h100_naive.add_argument("--baseline-tokenizer-path", required=True)
    h100_naive.add_argument("--baseline-vocab-size", type=int, required=True)
    h100_naive.add_argument("--gpt-naive-run-id", required=True)
    h100_naive.add_argument("--hgdn-config", required=True)
    h100_naive.add_argument("--hgdn-kernel-config", required=True)
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

    h100_compile_tiebreak = subparsers.add_parser(
        "write-h100-compile-tiebreak-manifest",
        help="write the exact-8x packed HGDN compile-tiebreak manifest",
    )
    h100_compile_tiebreak.add_argument("--output", type=Path, required=True)
    h100_compile_tiebreak.add_argument("--run-prefix-base", required=True)
    h100_compile_tiebreak.add_argument("--wandb-project", required=True)
    h100_compile_tiebreak.add_argument("--wandb-mode", required=True)
    h100_compile_tiebreak.add_argument("--archive-output", type=Path, required=True)
    h100_compile_tiebreak.add_argument(
        "--matched-logs", type=parse_bool_flag, required=True
    )
    h100_compile_tiebreak.add_argument("--command-log", required=True)
    h100_compile_tiebreak.add_argument("--torch-logs", default="")
    h100_compile_tiebreak.add_argument("--torch-trace", default="")
    h100_compile_tiebreak.add_argument("--omp-num-threads", type=int, required=True)
    h100_compile_tiebreak.add_argument("--mkl-num-threads", type=int, required=True)
    h100_compile_tiebreak.add_argument(
        "--openblas-num-threads", type=int, required=True
    )
    h100_compile_tiebreak.add_argument("--numexpr-num-threads", type=int, required=True)
    h100_compile_tiebreak.add_argument(
        "--attn-use-flash-attn3", type=parse_bool_flag, required=True
    )
    h100_compile_tiebreak.add_argument("--distributed-mode", required=True)
    h100_compile_tiebreak.add_argument("--ngpu", type=int, required=True)
    h100_compile_tiebreak.add_argument("--iterations", type=int, required=True)
    h100_compile_tiebreak.add_argument("--train-batch-tokens", type=int, required=True)
    h100_compile_tiebreak.add_argument("--train-seq-len", type=int, required=True)
    h100_compile_tiebreak.add_argument("--val-loss-every", type=int, required=True)
    h100_compile_tiebreak.add_argument("--train-log-every", type=int, required=True)
    h100_compile_tiebreak.add_argument("--val-batch-size", type=int, required=True)
    h100_compile_tiebreak.add_argument(
        "--max-wallclock-seconds", type=float, required=True
    )
    h100_compile_tiebreak.add_argument(
        "--compile-enabled", type=parse_bool_flag, required=True
    )
    h100_compile_tiebreak.add_argument("--hgdn-config", required=True)
    h100_compile_tiebreak.add_argument("--hgdn-kernel-config", required=True)
    h100_compile_tiebreak.add_argument("--run-id", action="append", default=[])
    h100_compile_tiebreak.add_argument(
        "--compile-strategy", action="append", default=[]
    )
    h100_compile_tiebreak.add_argument("--git-commit", default="")
    h100_compile_tiebreak.add_argument("--git-branch", default="")
    h100_compile_tiebreak.add_argument("--host-name", default="")
    h100_compile_tiebreak.add_argument("--timestamp-utc", default="")
    h100_compile_tiebreak.set_defaults(func=cmd_write_h100_compile_tiebreak_manifest)

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
