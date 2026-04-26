#!/usr/bin/env python3
"""Check loss/BPB byte-accounting consistency in trainer logs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

NUM = r"[-+]?\d+(?:\.\d+)?"
EVAL_RE = re.compile(
    rf"^(?P<label>step:\d+/\d+|final_[a-z0-9_]+)"
    rf".*?val_loss:(?P<loss>{NUM})\b.*?val_bpb:(?P<bpb>{NUM})\b"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*", help="Log files to inspect.")
    parser.add_argument("--loss", type=float, help="One-off validation loss in nats.")
    parser.add_argument("--bpb", type=float, help="One-off bits-per-byte value.")
    parser.add_argument("--min-bytes-per-token", type=float, default=1.0)
    parser.add_argument("--max-bytes-per-token", type=float, default=8.0)
    parser.add_argument("--fail-on-warn", action="store_true")
    return parser.parse_args()


def sanity_row(
    *,
    source: str,
    label: str,
    loss_nats: float,
    bpb: float,
    min_bytes_per_token: float,
    max_bytes_per_token: float,
) -> tuple[str, bool]:
    """Build one sanity-check row.

    :param str source: Source file or CLI label.
    :param str label: Metric label.
    :param float loss_nats: Token cross-entropy in nats.
    :param float bpb: Bits per byte.
    :param float min_bytes_per_token: Lower plausible byte/token bound.
    :param float max_bytes_per_token: Upper plausible byte/token bound.
    :return tuple[str, bool]: Row string and whether it is suspicious.
    """
    bits_per_token = loss_nats / math.log(2.0)
    implied_bytes_per_token = bits_per_token / bpb if bpb else float("inf")
    warn = not (min_bytes_per_token <= implied_bytes_per_token <= max_bytes_per_token)
    status = "WARN" if warn else "OK"
    return (
        "| "
        + " | ".join(
            [
                source,
                label,
                f"{loss_nats:.8f}",
                f"{bpb:.8f}",
                f"{bits_per_token:.8f}",
                f"{implied_bytes_per_token:.4f}",
                status,
            ]
        )
        + " |",
        warn,
    )


def rows_from_log(
    path: Path, *, min_bytes_per_token: float, max_bytes_per_token: float
) -> list[tuple[str, bool]]:
    """Extract sanity rows from one trainer log.

    :param Path path: Log file.
    :param float min_bytes_per_token: Lower plausible byte/token bound.
    :param float max_bytes_per_token: Upper plausible byte/token bound.
    :return list[tuple[str, bool]]: Rendered rows and warning flags.
    """
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVAL_RE.search(line)
        if not match:
            continue
        rows.append(
            sanity_row(
                source=str(path),
                label=match.group("label"),
                loss_nats=float(match.group("loss")),
                bpb=float(match.group("bpb")),
                min_bytes_per_token=min_bytes_per_token,
                max_bytes_per_token=max_bytes_per_token,
            )
        )
    return rows


def main() -> int:
    """Run BPB sanity checks.

    :return int: Exit code.
    """
    args = parse_args()
    rows: list[tuple[str, bool]] = []
    if args.loss is not None or args.bpb is not None:
        if args.loss is None or args.bpb is None:
            raise SystemExit("--loss and --bpb must be provided together")
        rows.append(
            sanity_row(
                source="<cli>",
                label="manual",
                loss_nats=args.loss,
                bpb=args.bpb,
                min_bytes_per_token=args.min_bytes_per_token,
                max_bytes_per_token=args.max_bytes_per_token,
            )
        )
    for path in args.paths:
        rows.extend(
            rows_from_log(
                path,
                min_bytes_per_token=args.min_bytes_per_token,
                max_bytes_per_token=args.max_bytes_per_token,
            )
        )
    print(
        "| Source | Label | Nats/token | BPB | Bits/token | Implied bytes/token | Status |"
    )
    print("|---|---|---:|---:|---:|---:|---|")
    warned = False
    for row, warn in rows:
        print(row)
        warned = warned or warn
    if not rows:
        print("| <none> | no metrics found |  |  |  |  | WARN |")
        warned = True
    return 1 if warned and args.fail_on_warn else 0


if __name__ == "__main__":
    raise SystemExit(main())
