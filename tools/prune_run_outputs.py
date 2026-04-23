#!/usr/bin/env python3
"""Prune stale checkpoint files from completed experiment runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PruneCandidate:
    """Checkpoint file eligible for pruning."""

    path: Path
    size_bytes: int
    reason: str


def is_completed_run(run_dir: Path) -> bool:
    """Return whether a run directory is completed and finalized.

    :param Path run_dir: Candidate run directory.
    :return bool: ``True`` when the run has ``final.pt`` and ``run_results.json`` reports completion.
    """
    final_path = run_dir / "final.pt"
    results_path = run_dir / "run_results.json"
    if not final_path.exists() or not results_path.exists():
        return False
    try:
        payload = json.loads(results_path.read_text())
    except Exception:
        return False
    return bool(payload.get("completed"))


def collect_prune_candidates(root: Path) -> list[PruneCandidate]:
    """Collect stale checkpoint files under an experiment root.

    :param Path root: Root directory to scan.
    :return list[PruneCandidate]: Checkpoint files safe to prune.
    """
    candidates: list[PruneCandidate] = []
    for checkpoint_path in sorted(root.rglob("checkpoint_*.pt")):
        run_dir = checkpoint_path.parent
        if not is_completed_run(run_dir):
            continue
        candidates.append(
            PruneCandidate(
                path=checkpoint_path,
                size_bytes=checkpoint_path.stat().st_size,
                reason="completed_run_with_final",
            )
        )
    return candidates


def format_bytes(size_bytes: int) -> str:
    """Format a byte count with a GiB/MiB suffix.

    :param int size_bytes: Raw byte count.
    :return str: Human-readable size string.
    """
    gib = 1024**3
    mib = 1024**2
    if size_bytes >= gib:
        return f"{size_bytes / gib:.2f} GiB"
    return f"{size_bytes / mib:.2f} MiB"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=["experiments"],
        help="Root directories to scan",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files instead of printing a dry-run summary",
    )
    return parser.parse_args()


def main() -> None:
    """Run the pruning tool."""
    args = parse_args()
    roots = [Path(root).resolve() for root in args.roots]
    candidates: list[PruneCandidate] = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(collect_prune_candidates(root))

    total_bytes = sum(item.size_bytes for item in candidates)
    action = "Deleting" if args.apply else "Would delete"
    print(f"{action} {len(candidates)} checkpoint files ({format_bytes(total_bytes)})")
    for item in candidates:
        print(f"{item.path} [{format_bytes(item.size_bytes)}] {item.reason}")

    if not args.apply:
        return

    for item in candidates:
        item.path.unlink()

    print(f"Deleted {len(candidates)} checkpoint files ({format_bytes(total_bytes)})")


if __name__ == "__main__":
    main()
