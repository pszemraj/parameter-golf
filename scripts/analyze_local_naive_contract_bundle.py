#!/usr/bin/env python3
"""Analyze a local sparse HGDN exact-contract bundle."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from _repo_bootstrap import ensure_repo_root_on_sys_path

REPO_ROOT = ensure_repo_root_on_sys_path()

STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<planned>\d+) train_loss:(?P<loss>[0-9.]+) "
    r"train_time:(?P<time_ms>\d+)ms step_avg:(?P<step_ms>[0-9.]+)ms"
)
EVAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<planned>\d+) val_loss:(?P<loss>[0-9.]+) "
    r"val_bpb:(?P<bpb>[0-9.]+) train_time:(?P<time_ms>\d+)ms"
)
MODEL_RE = re.compile(r"model_params:(?P<params>\d+) blocks:(?P<blocks>\d+G\+\d+A)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=REPO_ROOT / "local-scratch" / "localnaivehgdn_sparse3_bundle",
        help="Unpacked bundle directory to analyze.",
    )
    parser.add_argument(
        "--speed-budget-ms",
        type=float,
        default=None,
        help="Common train-time budget for speed-aware ranking. Defaults to the fastest complete run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for rows.json, rows.csv, and summary.md.",
    )
    parser.add_argument("--top", type=int, default=15, help="Rows to print.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file.

    :param Path path: JSON path.
    :return dict[str, Any]: Parsed object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def load_size_rows(bundle_dir: Path) -> dict[str, dict[str, Any]]:
    """Load size-screen rows keyed by candidate/config stem.

    :param Path bundle_dir: Bundle directory.
    :return dict[str, dict[str, Any]]: Size rows by candidate name.
    """
    size_path = bundle_dir / "size_screen" / "rows.csv"
    if not size_path.is_file():
        return {}
    with size_path.open(newline="", encoding="utf-8") as fh:
        return {row["name"]: row for row in csv.DictReader(fh)}


def parse_log(path: Path) -> dict[str, Any]:
    """Parse one hybrid trainer log.

    :param Path path: Log file path.
    :return dict[str, Any]: Parsed run metrics.
    """
    train_points: list[dict[str, float]] = []
    eval_points: list[dict[str, float]] = []
    params: int | None = None
    blocks = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := STEP_RE.search(line):
            train_points.append(
                {
                    "step": float(match.group("step")),
                    "planned": float(match.group("planned")),
                    "train_time_ms": float(match.group("time_ms")),
                    "step_ms": float(match.group("step_ms")),
                }
            )
        if match := EVAL_RE.search(line):
            eval_points.append(
                {
                    "step": float(match.group("step")),
                    "planned": float(match.group("planned")),
                    "train_time_ms": float(match.group("time_ms")),
                    "val_loss": float(match.group("loss")),
                    "val_bpb": float(match.group("bpb")),
                }
            )
        if match := MODEL_RE.search(line):
            params = int(match.group("params"))
            blocks = match.group("blocks")
    final_train = train_points[-1] if train_points else {}
    final_eval = eval_points[-1] if eval_points else {}
    planned_step = int(final_train.get("planned", final_eval.get("planned", 0)))
    final_step = int(final_train.get("step", final_eval.get("step", 0)))
    return {
        "log_path": str(path),
        "run_id": path.stem,
        "params": params,
        "blocks": blocks,
        "family": "attention-only" if blocks.startswith("0G+") else "HGDN",
        "completed": bool(planned_step and final_step >= planned_step),
        "final_step": final_step,
        "planned_step": planned_step,
        "final_train_time_ms": final_train.get("train_time_ms"),
        "final_step_ms": final_train.get("step_ms"),
        "final_sampled_bpb": final_eval.get("val_bpb"),
        "final_sampled_loss": final_eval.get("val_loss"),
        "eval_points": eval_points,
    }


def interpolate_bpb(
    eval_points: list[dict[str, float]], budget_ms: float
) -> float | None:
    """Estimate BPB at a common train-time budget.

    :param list[dict[str, float]] eval_points: Validation points sorted by time.
    :param float budget_ms: Common wallclock budget in milliseconds.
    :return float | None: Interpolated BPB, or None when no eval is available.
    """
    if not eval_points:
        return None
    points = sorted(eval_points, key=lambda row: row["train_time_ms"])
    if budget_ms <= points[0]["train_time_ms"]:
        return points[0]["val_bpb"]
    if budget_ms >= points[-1]["train_time_ms"]:
        return points[-1]["val_bpb"]
    for left, right in zip(points, points[1:], strict=False):
        if left["train_time_ms"] <= budget_ms <= right["train_time_ms"]:
            span = right["train_time_ms"] - left["train_time_ms"]
            if span <= 0:
                return right["val_bpb"]
            alpha = (budget_ms - left["train_time_ms"]) / span
            return left["val_bpb"] + alpha * (right["val_bpb"] - left["val_bpb"])
    return points[-1]["val_bpb"]


def assign_rank(rows: list[dict[str, Any]], *, key: str, out_key: str) -> None:
    """Assign dense ascending ranks in place.

    :param list[dict[str, Any]] rows: Rows to rank.
    :param str key: Numeric metric key.
    :param str out_key: Rank output key.
    """
    ranked = sorted(
        [row for row in rows if row.get(key) is not None],
        key=lambda row: (float(row[key]), row["run_id"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row[out_key] = rank


def build_rows(
    bundle_dir: Path, speed_budget_ms: float | None
) -> tuple[list[dict[str, Any]], float]:
    """Build analyzer rows from an unpacked bundle.

    :param Path bundle_dir: Bundle directory.
    :param float | None speed_budget_ms: Optional common time budget.
    :return tuple[list[dict[str, Any]], float]: Rows and resolved speed budget.
    """
    manifest = load_json(bundle_dir / "bundle_manifest.json")
    size_rows = load_size_rows(bundle_dir)
    parsed_by_run_id = {
        row["run_id"]: row
        for row in map(parse_log, sorted((bundle_dir / "logs").glob("*.txt")))
    }
    rows: list[dict[str, Any]] = []
    for candidate in manifest.get("candidates", []):
        run_id = candidate["run_id"]
        config_stem = Path(candidate["config"]).stem
        row = dict(parsed_by_run_id.get(run_id, {"run_id": run_id, "completed": False}))
        row["label"] = candidate["label"]
        row["config"] = candidate["config"]
        row["candidate"] = config_stem.replace("naive_contract_", "")
        size_row = size_rows.get(config_stem.replace("naive_contract_", ""), {})
        if not size_row:
            size_row = size_rows.get(row["candidate"], {})
        row["size_status"] = size_row.get("artifact_status")
        row["headroom_bytes"] = (
            int(size_row["headroom_bytes"]) if size_row.get("headroom_bytes") else None
        )
        rows.append(row)
    complete_times = [
        float(row["final_train_time_ms"])
        for row in rows
        if row.get("completed") and row.get("final_train_time_ms") is not None
    ]
    resolved_budget_ms = float(speed_budget_ms or min(complete_times))
    for row in rows:
        row["speed_budget_ms"] = resolved_budget_ms
        row["speed_budget_bpb"] = interpolate_bpb(
            row.get("eval_points", []), resolved_budget_ms
        )
    assign_rank(rows, key="final_sampled_bpb", out_key="fixed_step_rank_all")
    assign_rank(rows, key="speed_budget_bpb", out_key="speed_rank_all")
    assign_rank(
        [row for row in rows if row.get("family") == "HGDN"],
        key="final_sampled_bpb",
        out_key="fixed_step_rank_hgdn",
    )
    assign_rank(
        [row for row in rows if row.get("family") == "HGDN"],
        key="speed_budget_bpb",
        out_key="speed_rank_hgdn",
    )
    return rows, resolved_budget_ms


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky eval points for tabular outputs.

    :param dict[str, Any] row: Full row.
    :return dict[str, Any]: Compact row.
    """
    return {key: value for key, value in row.items() if key != "eval_points"}


def write_outputs(
    output_dir: Path, rows: list[dict[str, Any]], speed_budget_ms: float
) -> None:
    """Write analyzer artifacts.

    :param Path output_dir: Output directory.
    :param list[dict[str, Any]] rows: Analyzer rows.
    :param float speed_budget_ms: Resolved speed budget.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    compact = [compact_row(row) for row in rows]
    (output_dir / "rows.json").write_text(
        json.dumps(compact, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "candidate",
        "family",
        "completed",
        "final_step",
        "planned_step",
        "final_step_ms",
        "final_sampled_bpb",
        "speed_budget_bpb",
        "fixed_step_rank_all",
        "speed_rank_all",
        "fixed_step_rank_hgdn",
        "speed_rank_hgdn",
        "size_status",
        "headroom_bytes",
        "config",
        "run_id",
    ]
    with (output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(compact)
    lines = [
        "# Local Sparse HGDN Analyzer",
        "",
        f"Speed-aware BPB is linearly interpolated at `{speed_budget_ms:.0f} ms` of training time.",
        "",
        "| Candidate | Family | Step | ms/step | BPB | Speed BPB | Fixed rank | Speed rank | Size |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(
        rows, key=lambda item: (item.get("speed_rank_all") or 999, item["candidate"])
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["candidate"],
                    row.get("family", ""),
                    f"{row.get('final_step', 0)}/{row.get('planned_step', 0)}",
                    f"{float(row['final_step_ms']):.2f}"
                    if row.get("final_step_ms") is not None
                    else "",
                    f"{float(row['final_sampled_bpb']):.4f}"
                    if row.get("final_sampled_bpb") is not None
                    else "",
                    f"{float(row['speed_budget_bpb']):.4f}"
                    if row.get("speed_budget_bpb") is not None
                    else "",
                    str(
                        row.get("fixed_step_rank_hgdn")
                        or row.get("fixed_step_rank_all")
                        or ""
                    ),
                    str(row.get("speed_rank_hgdn") or row.get("speed_rank_all") or ""),
                    str(row.get("size_status") or ""),
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(
    rows: list[dict[str, Any]], *, top: int, speed_budget_ms: float
) -> None:
    """Print a compact ranking summary.

    :param list[dict[str, Any]] rows: Analyzer rows.
    :param int top: Number of rows to print.
    :param float speed_budget_ms: Resolved speed budget.
    """
    print(f"speed_budget_ms:{speed_budget_ms:.0f}")
    print(
        "| Candidate | Family | Step | ms/step | BPB | Speed BPB | Fixed HGDN | Speed HGDN | Size |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted(
        rows, key=lambda item: (item.get("speed_rank_all") or 999, item["candidate"])
    )[:top]:
        print(
            f"| {row['candidate']} | {row.get('family', '')} | "
            f"{row.get('final_step', 0)}/{row.get('planned_step', 0)} | "
            f"{float(row['final_step_ms']):.2f} | "
            f"{float(row['final_sampled_bpb']):.4f} | "
            f"{float(row['speed_budget_bpb']):.4f} | "
            f"{row.get('fixed_step_rank_hgdn', '')} | "
            f"{row.get('speed_rank_hgdn', '')} | "
            f"{row.get('size_status') or ''} |"
        )


def main() -> None:
    """Run the bundle analyzer."""
    args = parse_args()
    rows, speed_budget_ms = build_rows(args.bundle_dir, args.speed_budget_ms)
    if args.output_dir is not None:
        write_outputs(args.output_dir, rows, speed_budget_ms)
    print_summary(rows, top=args.top, speed_budget_ms=speed_budget_ms)


if __name__ == "__main__":
    main()
