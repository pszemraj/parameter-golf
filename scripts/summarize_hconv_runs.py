#!/usr/bin/env python3
"""Summarize completed GPT/hconv run directories into a markdown table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ORDER = ["GPT_REF", "B1", "C2", "T2", "T3", "I1", "I2", "I4", "I4H", "SMOKE_HCONV"]
VAL_RE = re.compile(r"^step:\d+/\d+ val_loss:\S+ val_bpb:(\S+) ")
STEP_RE = re.compile(r"^step:\d+/\d+ (?:train_loss|val_loss):.* step_avg:(\S+)ms")
SIZE_RE = re.compile(r"^Serialized model int8\+zlib: (\d+) bytes")


def parse_metrics(train_log: Path) -> tuple[str, str, str]:
    """Extract val_bpb, final step_avg, and int8+zlib bytes from one train log.

    :param Path train_log: Trainer log to parse.
    :return tuple[str, str, str]: Parsed val_bpb, step_avg_ms, and int8 payload bytes.
    """

    val_bpb = "-"
    step_avg_ms = "-"
    int8_bytes = "-"
    with train_log.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if match := VAL_RE.match(line):
                val_bpb = match.group(1)
            if match := STEP_RE.match(line):
                step_avg_ms = match.group(1)
            if match := SIZE_RE.match(line):
                int8_bytes = match.group(1)
    return val_bpb, step_avg_ms, int8_bytes


def iter_run_dirs(root: Path) -> list[Path]:
    """Return run directories in the preferred reporting order.

    :param Path root: Root directory containing per-target run subdirectories.
    :return list[Path]: Ordered run directories for reporting.
    """

    present = {child.name: child for child in root.iterdir() if child.is_dir()}
    ordered = [present[name] for name in ORDER if name in present]
    extras = sorted(name for name in present if name not in ORDER)
    ordered.extend(present[name] for name in extras)
    return ordered


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the run summarizer.

    :return argparse.ArgumentParser: Parser for summarizer CLI flags.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="runs_hconv_quality_5090",
        help="Root directory containing per-target run subdirectories.",
    )
    return parser


def main() -> None:
    """Load run directories from disk and print the markdown summary table."""

    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Run root does not exist: {root}")

    rows: list[tuple[str, str, str, str]] = []
    for run_dir in iter_run_dirs(root):
        train_log = run_dir / "train.log"
        if not train_log.is_file():
            rows.append((run_dir.name, "-", "-", "-"))
            continue
        rows.append((run_dir.name, *parse_metrics(train_log)))

    print("| Config | val_bpb | step_avg_ms | int8+zlib_bytes |")
    print("| --- | ---: | ---: | ---: |")
    for config, val_bpb, step_avg_ms, int8_bytes in rows:
        print(f"| {config} | {val_bpb} | {step_avg_ms} | {int8_bytes} |")


if __name__ == "__main__":
    main()
