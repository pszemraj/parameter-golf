#!/usr/bin/env python3
"""Write a conservative next-step decision from wallclock resolver rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--primary-config", required=True)
    parser.add_argument("--primary-control-config", required=True)
    parser.add_argument("--secondary-config", required=True)
    parser.add_argument("--secondary-control-config", required=True)
    parser.add_argument("--primary-control-margin", type=float, default=0.003)
    parser.add_argument("--secondary-primary-margin", type=float, default=0.005)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load analyzer rows.

    :param Path path: Analyzer ``rows.json`` path.
    :return list[dict[str, Any]]: Rows.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def row_by_config(rows: list[dict[str, Any]], config: str) -> dict[str, Any] | None:
    """Return the first row matching a config path.

    :param list[dict[str, Any]] rows: Analyzer rows.
    :param str config: Config path.
    :return dict[str, Any] | None: Matching row.
    """
    for row in rows:
        if row.get("config") == config:
            return row
    return None


def exact_baseline_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the exact repo baseline row when present.

    :param list[dict[str, Any]] rows: Analyzer rows.
    :return dict[str, Any] | None: Baseline row.
    """
    for row in rows:
        if row.get("family") == "exact-baseline":
            return row
    return None


def metric(row: dict[str, Any] | None) -> float | None:
    """Return the true-wallclock decision BPB.

    :param dict[str, Any] | None row: Analyzer row.
    :return float | None: Metric value or None.
    """
    if row is None:
        return None
    if not row.get("completed"):
        return None
    if not row.get("wallclock_clean_stop"):
        return None
    value = row.get("final_roundtrip_bpb")
    return float(value) if value is not None else None


def main() -> int:
    """Run decision generation.

    :return int: Shell-style exit code.
    """
    args = parse_args()
    rows = load_rows(args.rows_json)
    primary = row_by_config(rows, args.primary_config)
    primary_control = row_by_config(rows, args.primary_control_config)
    secondary = row_by_config(rows, args.secondary_config)
    secondary_control = row_by_config(rows, args.secondary_control_config)
    exact = exact_baseline_row(rows)

    primary_bpb = metric(primary)
    primary_control_bpb = metric(primary_control)
    secondary_bpb = metric(secondary)
    secondary_control_bpb = metric(secondary_control)
    exact_bpb = metric(exact)

    required_values = [
        primary_bpb,
        primary_control_bpb,
        secondary_bpb,
        secondary_control_bpb,
    ]
    missing_required = any(value is None for value in required_values)
    primary_delta = (
        None
        if primary_bpb is None or primary_control_bpb is None
        else primary_control_bpb - primary_bpb
    )
    secondary_control_delta = (
        None
        if secondary_bpb is None or secondary_control_bpb is None
        else secondary_control_bpb - secondary_bpb
    )
    secondary_delta = (
        None
        if primary_bpb is None or secondary_bpb is None
        else primary_bpb - secondary_bpb
    )
    primary_supported = (
        primary_delta is not None and primary_delta >= args.primary_control_margin
    )
    secondary_beats_primary = (
        secondary_delta is not None and secondary_delta >= args.secondary_primary_margin
    )
    close_primary_secondary = (
        secondary_delta is None or abs(secondary_delta) < args.secondary_primary_margin
    )
    local_speed_inconclusive = missing_required or close_primary_secondary
    run_primary_h100 = True
    run_secondary_h100 = bool(local_speed_inconclusive or secondary_beats_primary)

    if missing_required:
        reason = "missing_or_non_wallclock_final_roundtrip_metric"
    elif secondary_beats_primary:
        reason = "secondary_beats_primary_by_margin"
    elif close_primary_secondary:
        reason = "primary_secondary_margin_too_close_for_laptop"
    else:
        reason = "primary_only_supported_by_local_wallclock"

    decision = {
        "primary_config": args.primary_config,
        "primary_control_config": args.primary_control_config,
        "secondary_config": args.secondary_config,
        "secondary_control_config": args.secondary_control_config,
        "exact_baseline_bpb": exact_bpb,
        "primary_bpb": primary_bpb,
        "primary_control_bpb": primary_control_bpb,
        "secondary_bpb": secondary_bpb,
        "secondary_control_bpb": secondary_control_bpb,
        "primary_control_delta": primary_delta,
        "secondary_control_delta": secondary_control_delta,
        "secondary_delta_vs_primary": secondary_delta,
        "primary_supported": primary_supported,
        "secondary_beats_primary": secondary_beats_primary,
        "local_speed_inconclusive": local_speed_inconclusive,
        "run_primary_h100": run_primary_h100,
        "run_secondary_h100": run_secondary_h100,
        "reason": reason,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
