#!/usr/bin/env python3
"""Bundle one HGDN run into a portable `.7z` archive.

:param None: This module exposes a command-line entrypoint via ``argparse``.
:return None: Process exit code indicates bundling success.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _repo_bootstrap import ensure_repo_root_on_sys_path
from hgdn_helper_cli import require_py7zr

REPO_ROOT = ensure_repo_root_on_sys_path()


def repo_relative_path(value: str) -> Path:
    """Resolve one repo-relative or absolute path.

    :param str value: Raw user-provided path.
    :return Path: Absolute filesystem path.
    """
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def run_capture(
    command: list[str], *, allow_failure: bool = False
) -> tuple[int, str, str]:
    """Run one subprocess and capture text streams.

    :param list[str] command: Command vector to execute.
    :param bool allow_failure: Whether to tolerate a non-zero exit code.
    :return tuple[int, str, str]: Exit code, stdout, and stderr.
    :raises RuntimeError: Raised when the command fails and failure is not allowed.
    """
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.returncode, completed.stdout, completed.stderr


def copy_required(src: Path, stage_root: Path) -> str:
    """Copy one required path into the stage tree, preserving repo-relative layout.

    :param Path src: Source file to copy.
    :param Path stage_root: Bundle staging root.
    :return str: Relative path stored inside the stage tree.
    :raises FileNotFoundError: Raised when the source path does not exist.
    """
    if not src.exists():
        raise FileNotFoundError(f"required bundle input is missing: {src}")
    try:
        relative = src.relative_to(REPO_ROOT)
    except ValueError:
        relative = Path("external") / src.name
    dst = stage_root / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(relative)


def copy_optional(src: Path, stage_root: Path) -> str | None:
    """Copy one optional path into the stage tree if it exists.

    :param Path src: Source file to copy when present.
    :param Path stage_root: Bundle staging root.
    :return str | None: Relative stored path or ``None`` when absent.
    """
    if not src.exists():
        return None
    return copy_required(src, stage_root)


def write_text(path: Path, text: str) -> None:
    """Write one UTF-8 text file.

    :param Path path: Output file path.
    :param str text: Text payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file with stable formatting.

    :param Path path: Output file path.
    :param dict[str, Any] payload: JSON payload.
    """
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :return argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Bundle one HGDN run log, metadata, and selected artifacts."
    )
    parser.add_argument("--run-id", required=True, help="Logical trainer RUN_ID.")
    parser.add_argument(
        "--log-path",
        type=str,
        help="Repo-relative or absolute path to the primary run log. Defaults to logs/{RUN_ID}.txt.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Repo-relative or absolute config path to include. Repeatable.",
    )
    parser.add_argument(
        "--copy",
        action="append",
        default=[],
        help=(
            "Repo-relative or absolute extra file to include. Repeatable. "
            "Use this for optional artifacts such as final_model.pt."
        ),
    )
    parser.add_argument(
        "--command",
        default="",
        help="Optional exact shell command string to record in metadata.",
    )
    parser.add_argument(
        "--stage-dir",
        type=str,
        help="Stage directory. Defaults to artifacts/{RUN_ID}_bundle.",
    )
    parser.add_argument(
        "--archive-output",
        type=str,
        help="Output .7z path. Defaults to artifacts/{RUN_ID}_bundle.7z.",
    )
    parser.add_argument(
        "--skip-gpu-query",
        action="store_true",
        help="Skip the nvidia-smi hardware identity capture.",
    )
    return parser.parse_args()


def main() -> int:
    """Bundle one run and write a `.7z` archive.

    :return int: Shell-style exit code.
    """
    args = parse_args()
    run_id = args.run_id
    log_path = repo_relative_path(args.log_path or f"logs/{run_id}.txt")
    stage_dir = repo_relative_path(args.stage_dir or f"artifacts/{run_id}_bundle")
    archive_output = repo_relative_path(
        args.archive_output or f"artifacts/{run_id}_bundle.7z"
    )
    config_paths = [repo_relative_path(path) for path in args.config]
    extra_paths = [repo_relative_path(path) for path in args.copy]
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    archive_output.parent.mkdir(parents=True, exist_ok=True)
    archive_output.unlink(missing_ok=True)

    included_paths: list[str] = []
    included_paths.append(copy_required(log_path, stage_dir))
    for config_path in config_paths:
        included_paths.append(copy_required(config_path, stage_dir))
    for extra_path in extra_paths:
        included_paths.append(copy_required(extra_path, stage_dir))

    git_head_rc, git_head, _ = run_capture(["git", "rev-parse", "HEAD"])
    git_branch_rc, git_branch, _ = run_capture(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    )
    git_status_rc, git_status, _ = run_capture(["git", "status", "--short"])

    metadata_dir = stage_dir / "metadata"
    write_text(metadata_dir / "git_head.txt", git_head)
    write_text(metadata_dir / "git_branch.txt", git_branch)
    write_text(metadata_dir / "git_status.txt", git_status)

    gpu_query: dict[str, Any]
    if args.skip_gpu_query:
        gpu_query = {"skipped": True}
    else:
        gpu_rc, gpu_stdout, gpu_stderr = run_capture(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,pci.bus_id,power.limit",
                "--format=csv,noheader",
            ],
            allow_failure=True,
        )
        gpu_query = {
            "skipped": False,
            "returncode": gpu_rc,
            "stdout": gpu_stdout,
            "stderr": gpu_stderr,
        }
        write_text(metadata_dir / "nvidia_smi.txt", gpu_stdout or gpu_stderr)

    if args.command:
        write_text(metadata_dir / "command.txt", args.command + "\n")

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "host_name": socket.gethostname(),
        "paths": {
            "stage_dir": str(stage_dir),
            "archive_output": str(archive_output),
            "included": included_paths,
        },
        "git": {
            "head_returncode": git_head_rc,
            "branch_returncode": git_branch_rc,
            "status_returncode": git_status_rc,
            "head": git_head.strip(),
            "branch": git_branch.strip(),
        },
        "gpu_query": gpu_query,
        "command": args.command,
    }
    write_json(stage_dir / "bundle_manifest.json", manifest)

    py7zr = require_py7zr()
    with py7zr.SevenZipFile(archive_output, "w") as archive:
        archive.writeall(stage_dir, arcname=stage_dir.name)

    print(archive_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
