#!/usr/bin/env python3
"""Shared utilities for local HGDN experiment runner CLIs."""

from __future__ import annotations

import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from _repo_bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path()

from hgdn_helper_cli import load_toml_env, require_py7zr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RECURRENCE_MODES = ("compile_visible", "direct", "direct_fused")
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PERF_RE = re.compile(
    rf"perf_summary ignore_steps:(?P<ignore>\d+) measured_steps:(?P<measured>\d+) "
    rf"step_ms:(?P<step_ms>{NUMBER_RE}) tokens_per_s:(?P<tokens_per_s>{NUMBER_RE})"
)
PEAK_RE = re.compile(
    r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB"
)
ARTIFACT_RE = re.compile(r"artifact_status:(?P<status>[A-Z_]+)")


@dataclass(frozen=True)
class CommandSpec:
    """Executable command plus environment overlay."""

    label: str
    run_id: str
    env: dict[str, str]
    command: list[str]


def csv_items(value: str) -> list[str]:
    """Split a comma-separated string into non-empty trimmed items.

    :param str value: Comma-separated value.
    :return list[str]: Items.
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def csv_ints(value: str) -> list[int]:
    """Split a comma-separated integer list.

    :param str value: Comma-separated integers.
    :return list[int]: Parsed integers.
    """
    return [int(item) for item in csv_items(value)]


def bool_flag_value(value: bool) -> str:
    """Return a trainer-compatible boolean string.

    :param bool value: Boolean value.
    :return str: ``1`` or ``0``.
    """
    return "1" if value else "0"


def enable_line_buffering() -> None:
    """Keep runner progress prints ordered when stdout is piped."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)


def resolve_grad_accum_steps(ngpu: int, override: int | None) -> int:
    """Resolve local gradient accumulation steps.

    :param int ngpu: Number of local GPU processes.
    :param int | None override: Optional explicit override.
    :raises ValueError: If the contract is invalid.
    :return int: Gradient accumulation steps.
    """
    if override is not None:
        if override < 1:
            raise ValueError(f"GRAD_ACCUM_STEPS must be >= 1, got {override}")
        return override
    if 8 % ngpu != 0:
        raise ValueError(
            f"NGPU must evenly divide 8 when GRAD_ACCUM_STEPS is unset: {ngpu}"
        )
    return 8 // ngpu


def resolve_val_batch_size(
    ngpu: int,
    grad_accum_steps: int,
    train_seq_len: int,
    *,
    requested_tokens: int | None,
    requested_seqs: int | None,
) -> int:
    """Resolve validation batch tokens from token or sequence inputs.

    :param int ngpu: Number of local GPU processes.
    :param int grad_accum_steps: Gradient accumulation steps.
    :param int train_seq_len: Training sequence length.
    :param int | None requested_tokens: Requested global validation tokens.
    :param int | None requested_seqs: Requested validation sequences.
    :raises ValueError: If incompatible values are requested.
    :return int: Global validation batch size in tokens.
    """
    min_tokens = ngpu * grad_accum_steps * train_seq_len
    min_seqs = ngpu * grad_accum_steps
    if requested_tokens is not None and requested_seqs is not None:
        raise ValueError("Set only one of VAL_BATCH_SIZE or VAL_BATCH_SEQS.")
    if requested_seqs is not None:
        if requested_seqs < min_seqs:
            raise ValueError(
                f"VAL_BATCH_SEQS must be at least {min_seqs} for NGPU={ngpu}, "
                f"GRAD_ACCUM_STEPS={grad_accum_steps}"
            )
        return requested_seqs * train_seq_len
    if requested_tokens is None:
        return min_tokens
    if requested_tokens < min_tokens:
        if requested_tokens >= min_seqs:
            print(
                f"Interpreting VAL_BATCH_SIZE={requested_tokens} as validation "
                f"sequences; prefer VAL_BATCH_SEQS={requested_tokens}.",
                file=sys.stderr,
            )
            return requested_tokens * train_seq_len
        raise ValueError(
            f"VAL_BATCH_SIZE must be at least {min_tokens} tokens for NGPU={ngpu}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={train_seq_len}"
        )
    return requested_tokens


def common_thread_env(
    *,
    omp_num_threads: int,
    mkl_num_threads: int,
    openblas_num_threads: int,
    numexpr_num_threads: int,
    nccl_ib_disable: int,
) -> dict[str, str]:
    """Build thread and NCCL environment defaults.

    :param int omp_num_threads: OMP thread count.
    :param int mkl_num_threads: MKL thread count.
    :param int openblas_num_threads: OpenBLAS thread count.
    :param int numexpr_num_threads: NumExpr thread count.
    :param int nccl_ib_disable: NCCL IB disable flag.
    :return dict[str, str]: Environment overlay.
    """
    return {
        "OMP_NUM_THREADS": str(omp_num_threads),
        "MKL_NUM_THREADS": str(mkl_num_threads),
        "OPENBLAS_NUM_THREADS": str(openblas_num_threads),
        "NUMEXPR_NUM_THREADS": str(numexpr_num_threads),
        "NCCL_IB_DISABLE": str(nccl_ib_disable),
    }


def diagnostic_env(torch_logs: str, torch_trace: str) -> dict[str, str]:
    """Build optional Torch diagnostics environment.

    :param str torch_logs: ``TORCH_LOGS`` value.
    :param str torch_trace: ``TORCH_TRACE`` value.
    :return dict[str, str]: Environment overlay.
    """
    out: dict[str, str] = {}
    if torch_logs:
        out["TORCH_LOGS"] = torch_logs
    if torch_trace:
        out["TORCH_TRACE"] = torch_trace
    return out


def filtered_config_env(config_path: Path, recurrence_mode: str) -> dict[str, str]:
    """Load a TOML config as environment and force recurrence mode.

    :param Path config_path: HGDN config path.
    :param str recurrence_mode: GDN FLA recurrence mode.
    :return dict[str, str]: Environment overlay.
    """
    env = load_toml_env(config_path, alias_aware=True)
    env.pop("GDN_FLA_RECURRENCE_MODE", None)
    env.pop("GDN_USE_DIRECT_FLA_LAYER_SEMANTICS", None)
    env["GDN_USE_DIRECT_FLA_LAYER_SEMANTICS"] = "0"
    env["GDN_FLA_RECURRENCE_MODE"] = recurrence_mode
    return env


def command_log_line(spec: CommandSpec) -> str:
    """Render one command as a shell-readable log line.

    :param CommandSpec spec: Command specification.
    :return str: Shell-quoted command line.
    """
    env_items = [f"{key}={value}" for key, value in spec.env.items()]
    return shlex.join([*env_items, *spec.command])


def write_command_log(
    path: Path, specs: Sequence[CommandSpec], prelude: Sequence[str]
) -> None:
    """Write a command manifest.

    :param Path path: Output command log path.
    :param Sequence[CommandSpec] specs: Planned commands.
    :param Sequence[str] prelude: Non-training prelude commands.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/bin/bash", "set -euo pipefail", *prelude]
    lines.extend(command_log_line(spec) for spec in specs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(spec: CommandSpec) -> int:
    """Run one command with its environment overlay.

    :param CommandSpec spec: Command specification.
    :return int: Process return code.
    """
    print()
    print(f">>> {spec.label}")
    print(f"run_id={spec.run_id}")
    env = os.environ.copy()
    env.update(spec.env)
    return subprocess.run(spec.command, cwd=REPO_ROOT, env=env, check=False).returncode


def log_completion_state(log_path: Path) -> str:
    """Classify one trainer log.

    :param Path log_path: Trainer log path.
    :return str: ``missing``, ``incomplete``, or ``complete``.
    """
    if not log_path.is_file():
        return "missing"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "perf_mode: skipping serialization and final roundtrip eval" in text:
        return "complete"
    if "final_int8_zlib_roundtrip_exact" in text and "artifact_status:" in text:
        return "complete"
    return "incomplete"


def parse_perf_log(log_path: Path) -> dict[str, Any]:
    """Parse perf-oriented terminal fields from one trainer log.

    :param Path log_path: Trainer log path.
    :return dict[str, Any]: Parsed fields.
    """
    out: dict[str, Any] = {"completion_state": log_completion_state(log_path)}
    if not log_path.is_file():
        return out
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if match := PERF_RE.search(text):
        out.update(
            {
                "perf_ignore_steps": int(match.group("ignore")),
                "perf_measured_steps": int(match.group("measured")),
                "perf_step_ms": float(match.group("step_ms")),
                "perf_tokens_per_s": float(match.group("tokens_per_s")),
            }
        )
    if match := PEAK_RE.search(text):
        out.update(
            {
                "peak_allocated_mib": int(match.group("allocated")),
                "peak_reserved_mib": int(match.group("reserved")),
            }
        )
    if match := ARTIFACT_RE.search(text):
        out["artifact_status"] = match.group("status")
    return out


def copy_existing(paths: Iterable[Path], output_dir: Path) -> None:
    """Copy existing files into one directory.

    :param Iterable[Path] paths: Candidate paths.
    :param Path output_dir: Destination directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)


def create_7z_archive(archive_output: Path, source_path: Path) -> None:
    """Create a py7zr archive from one source directory.

    :param Path archive_output: Archive output path.
    :param Path source_path: Source directory.
    """
    py7zr = require_py7zr()
    archive_output.parent.mkdir(parents=True, exist_ok=True)
    archive_output.unlink(missing_ok=True)
    with py7zr.SevenZipFile(archive_output, "w") as archive:
        archive.writeall(source_path, arcname=source_path.name)


def ensure_py7zr_available() -> None:
    """Ensure ``py7zr`` is importable before expensive work starts."""
    try:
        require_py7zr()
    except RuntimeError:
        print()
        print(">>> install python package: py7zr")
        subprocess.run([sys.executable, "-m", "pip", "install", "py7zr"], check=True)
        require_py7zr()


def check_cuda_jobs(check_idle: bool, allow_active_jobs: bool) -> None:
    """Refuse to start when other CUDA compute jobs are active.

    :param bool check_idle: Whether to check CUDA process state.
    :param bool allow_active_jobs: Whether active jobs are allowed.
    :raises SystemExit: If active jobs are found and not allowed.
    """
    if not check_idle or allow_active_jobs or shutil.which("nvidia-smi") is None:
        return
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    active = "\n".join(line for line in result.stdout.splitlines() if line.strip())
    if active:
        raise SystemExit(
            "Refusing to start while CUDA compute jobs are active:\n"
            f"{active}\n"
            "Set --allow-active-cuda-jobs only if this overlap is intentional."
        )


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV using the union of row fields.

    :param Path path: Output CSV path.
    :param list[dict[str, Any]] rows: Rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_seconds(value: float) -> str:
    """Format seconds compactly for run IDs.

    :param float value: Seconds.
    :return str: Compact wallclock tag.
    """
    return f"{value:g}".replace(".", "p")
