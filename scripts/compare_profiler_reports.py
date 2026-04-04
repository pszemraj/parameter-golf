"""Compare two structured profiler reports over selected buckets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiler_report import load_profile_report  # noqa: E402

DEFAULT_BUCKETS = (
    "aten::copy_",
    "aten::mul",
    "gdn.qkv_conv_packed",
    "gdn.q_conv",
    "gdn.k_conv",
    "gdn.v_conv",
    "gdn.recurrence",
    "aten::convolution_backward",
    "aten::_conv_depthwise2d",
    "gdn.q_norm",
    "gdn.k_norm",
    "gdn.g_proj",
    "gdn.g_pointwise",
    "gdn.beta_proj",
    "gdn.output_gate_proj",
    "gdn.output_norm",
    "gdn.output_gate_mul",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for report comparison.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=list(DEFAULT_BUCKETS),
        help="Exact profiler bucket names to compare.",
    )
    return parser.parse_args()


def find_row(report: dict[str, Any], bucket: str) -> dict[str, Any] | None:
    """Find one profiler row by exact name.

    :param dict[str, Any] report: Structured profiler report.
    :param str bucket: Exact bucket name.
    :return dict[str, Any] | None: Matching row, if present.
    """
    for row in report["rows"]:
        if row["name"] == bucket:
            return row
    return None


def row_ms(row: dict[str, Any] | None) -> float:
    """Extract self device time in milliseconds.

    :param dict[str, Any] row: Structured profiler row.
    :return float: Self device time in milliseconds.
    """
    if row is None:
        return 0.0
    return float(row["self_device_time_us"]) / 1000.0


def row_pct(row: dict[str, Any] | None) -> float:
    """Extract self device percent.

    :param dict[str, Any] row: Structured profiler row.
    :return float: Self device percentage.
    """
    if row is None:
        return 0.0
    return float(row["self_device_percent"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write comparison rows as CSV.

    :param Path path: Output CSV path.
    :param list[dict[str, Any]] rows: Rows to write.
    """
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_md(
    baseline_name: str,
    candidate_name: str,
    total_baseline_ms: float,
    total_candidate_ms: float,
    rows: list[dict[str, Any]],
) -> str:
    """Render a markdown summary.

    :param str baseline_name: Baseline label.
    :param str candidate_name: Candidate label.
    :param float total_baseline_ms: Baseline total self device time in ms.
    :param float total_candidate_ms: Candidate total self device time in ms.
    :param list[dict[str, Any]] rows: Comparison rows.
    :return str: Markdown summary.
    """
    delta_ms = total_candidate_ms - total_baseline_ms
    delta_pct = (
        0.0 if total_baseline_ms == 0.0 else 100.0 * delta_ms / total_baseline_ms
    )
    lines = [
        "# Profiler Comparison",
        "",
        f"- baseline: `{baseline_name}`",
        f"- candidate: `{candidate_name}`",
        f"- baseline total self device time: `{total_baseline_ms:.2f} ms`",
        f"- candidate total self device time: `{total_candidate_ms:.2f} ms`",
        f"- delta: `{delta_ms:+.2f} ms` (`{delta_pct:+.2f}%`)",
        "",
        "| Bucket | Baseline ms | Candidate ms | Delta ms | Baseline % | Candidate % | Delta % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['bucket']}` | {row['baseline_ms']:.2f} | "
            f"{row['candidate_ms']:.2f} | {row['delta_ms']:+.2f} | "
            f"{row['baseline_pct']:.2f} | {row['candidate_pct']:.2f} | "
            f"{row['delta_pct']:+.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Compare two structured profiler reports and write JSON/CSV/Markdown."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_profile_report(args.baseline)
    candidate = load_profile_report(args.candidate)
    rows: list[dict[str, Any]] = []
    for bucket in args.buckets:
        base_row = find_row(baseline, bucket)
        cand_row = find_row(candidate, bucket)
        rows.append(
            {
                "bucket": bucket,
                "baseline_ms": row_ms(base_row),
                "candidate_ms": row_ms(cand_row),
                "delta_ms": row_ms(cand_row) - row_ms(base_row),
                "baseline_pct": row_pct(base_row),
                "candidate_pct": row_pct(cand_row),
                "delta_pct": row_pct(cand_row) - row_pct(base_row),
            }
        )

    summary = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_total_self_device_ms": baseline["metadata"][
            "total_self_device_time_us"
        ]
        / 1000.0,
        "candidate_total_self_device_ms": candidate["metadata"][
            "total_self_device_time_us"
        ]
        / 1000.0,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "bucket_deltas.csv", rows)
    (args.output_dir / "bucket_deltas.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "comparison.md").write_text(
        render_md(
            str(args.baseline),
            str(args.candidate),
            summary["baseline_total_self_device_ms"],
            summary["candidate_total_self_device_ms"],
            rows,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
