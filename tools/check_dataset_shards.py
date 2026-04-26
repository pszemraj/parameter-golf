#!/usr/bin/env python3
"""Report and validate FineWeb token-shard coverage for experiment launchers."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_amplifier_lm import training_token_file_fingerprint

SHARD_MAGIC = 20240520
HEADER_INTS = 256
DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int32": np.int32,
    "int64": np.int64,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", help="FineWeb token shard directory or file")
    ap.add_argument("--storage-dtype", default="uint16", choices=sorted(DTYPE_MAP))
    ap.add_argument("--expected-train-files", type=int, default=None)
    ap.add_argument("--expected-val-files", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    return ap.parse_args()


def _training_files(path: Path) -> list[Path]:
    """Return training token files, excluding validation shards.

    :param Path path: Token source.
    :return list[Path]: Ordered training files.
    """
    if path.is_file():
        return [path]
    train = sorted(path.glob("fineweb_train_*.bin"))
    if train:
        return train
    return [sp for sp in sorted(path.glob("*.bin")) if "val" not in sp.name]


def _validation_files(path: Path) -> list[Path]:
    """Return validation token files.

    :param Path path: Token source.
    :return list[Path]: Ordered validation files.
    """
    if path.is_file():
        return []
    val = sorted(path.glob("fineweb_val_*.bin"))
    if val:
        return val
    return [sp for sp in sorted(path.glob("*.bin")) if "val" in sp.name]


def _token_count(path: Path, *, storage_dtype: str) -> int:
    """Estimate token count from a token file.

    :param Path path: Token file.
    :param str storage_dtype: Storage dtype for raw token files.
    :return int: Token count.
    """
    dtype = np.dtype(DTYPE_MAP[storage_dtype])
    with path.open("rb") as f:
        header = f.read(HEADER_INTS * np.dtype("<i4").itemsize)
    if len(header) == HEADER_INTS * np.dtype("<i4").itemsize:
        header_arr = np.frombuffer(header, dtype="<i4", count=HEADER_INTS)
        if int(header_arr[0]) == SHARD_MAGIC:
            return int(header_arr[2])
    return int(path.stat().st_size // dtype.itemsize)


def _summarize_files(files: list[Path], *, storage_dtype: str) -> dict[str, int]:
    """Summarize a file list.

    :param list[Path] files: Token files.
    :param str storage_dtype: Storage dtype for raw token files.
    :return dict[str, int]: File, byte, and token counts.
    """
    return {
        "files": len(files),
        "bytes": sum(int(path.stat().st_size) for path in files),
        "tokens": sum(_token_count(path, storage_dtype=storage_dtype) for path in files),
    }


def main() -> None:
    """Validate and print shard coverage."""
    args = parse_args()
    data = Path(args.data).expanduser().resolve()
    if not data.exists():
        raise SystemExit(f"missing data path: {data}")
    train_files = _training_files(data)
    val_files = _validation_files(data)
    report: dict[str, Any] = {
        "data": str(data),
        "storage_dtype": args.storage_dtype,
        "train": _summarize_files(train_files, storage_dtype=args.storage_dtype),
        "validation": _summarize_files(val_files, storage_dtype=args.storage_dtype),
        "train_fingerprint": training_token_file_fingerprint(data),
    }

    failures: list[str] = []
    if (
        args.expected_train_files is not None
        and report["train"]["files"] != args.expected_train_files
    ):
        failures.append(
            f"expected {args.expected_train_files} train files, got {report['train']['files']}"
        )
    if (
        args.expected_val_files is not None
        and report["validation"]["files"] != args.expected_val_files
    ):
        failures.append(
            f"expected {args.expected_val_files} validation files, got "
            f"{report['validation']['files']}"
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"data={report['data']}")
        print(f"storage_dtype={report['storage_dtype']}")
        print(
            f"train_files={report['train']['files']} "
            f"train_tokens={report['train']['tokens']:,} "
            f"train_bytes={report['train']['bytes']:,}"
        )
        print(
            f"val_files={report['validation']['files']} "
            f"val_tokens={report['validation']['tokens']:,} "
            f"val_bytes={report['validation']['bytes']:,}"
        )
        print(f"train_fingerprint={report['train_fingerprint']['digest']}")

    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
