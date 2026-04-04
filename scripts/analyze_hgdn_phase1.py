"""Analyze local HGDN phase-1 profiler outputs.

This script converts the phase-1 profiler bundle into two decision aids:

- a bucket-to-path attribution table
- an HGDN boundary dtype/layout table
"""

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

VIEW_LABELS = {
    "gdn": "bare_gdn",
    "hybrid_fwd_bwd": "hybrid_fwd_bwd",
    "hybrid_opt": "hybrid_opt",
    "trainer": "trainer_eager",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for phase-1 HGDN profile analysis.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gdn", type=Path, required=True, help="Bare GDN report dir/file."
    )
    parser.add_argument(
        "--hybrid-fwd-bwd",
        dest="hybrid_fwd_bwd",
        type=Path,
        required=True,
        help="Hybrid forward/backward report dir/file.",
    )
    parser.add_argument(
        "--hybrid-opt",
        dest="hybrid_opt",
        type=Path,
        required=True,
        help="Hybrid optimizer-only report dir/file.",
    )
    parser.add_argument(
        "--trainer",
        type=Path,
        required=True,
        help="Full trainer eager report dir/file.",
    )
    parser.add_argument(
        "--boundary-audit",
        dest="boundary_audit",
        type=Path,
        default=None,
        help="Optional JSONL boundary audit emitted by GDN_AUDIT_BOUNDARIES.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for markdown/csv/json analysis artifacts.",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=list(DEFAULT_BUCKETS),
        help="Exact profiler bucket names to compare.",
    )
    return parser.parse_args()


def find_row(report: dict[str, Any], bucket: str) -> dict[str, Any] | None:
    """Find one profiler row by exact event name.

    :param dict[str, Any] report: Structured report payload.
    :param str bucket: Exact bucket name.
    :return dict[str, Any] | None: Matching row, if present.
    """
    for row in report["rows"]:
        if row["name"] == bucket:
            return row
    return None


def bucket_cell(row: dict[str, Any] | None) -> str:
    """Format one report row for the markdown table.

    :param dict[str, Any] row: Structured row data.
    :return str: Compact `ms / % / calls` summary or `-`.
    """
    if row is None:
        return "-"
    return (
        f"{row['self_device_time_us'] / 1000.0:.2f}ms / "
        f"{row['self_device_percent']:.2f}% / "
        f"{row['count']}"
    )


def classify_bucket(view_rows: dict[str, dict[str, Any] | None]) -> tuple[str, str]:
    """Classify where a bucket primarily lives.

    :param dict[str, dict[str, Any] | None] view_rows: Rows keyed by view label.
    :return tuple[str, str]: Dominant view and suggested ownership label.
    """
    active_views = [
        label
        for label, row in view_rows.items()
        if row is not None
        and (row["self_device_percent"] >= 0.5 or row["self_device_time_us"] >= 1000.0)
    ]
    if not active_views:
        return "none", "not_captured"
    dominant_view = max(
        active_views,
        key=lambda label: view_rows[label]["self_device_percent"],  # type: ignore[index]
    )
    active_set = set(active_views)
    if active_set == {"trainer_eager"}:
        return dominant_view, "trainer_shell"
    if active_set == {"hybrid_opt"}:
        return dominant_view, "optimizer_only"
    if "bare_gdn" in active_set:
        if "trainer_eager" in active_set and len(active_set) > 1:
            return dominant_view, "gdn_path_plus_training_shell"
        return dominant_view, "gdn_path"
    if "hybrid_fwd_bwd" in active_set and "bare_gdn" not in active_set:
        if "trainer_eager" in active_set:
            return dominant_view, "hybrid_integration_plus_training_shell"
        return dominant_view, "hybrid_integration"
    if "hybrid_opt" in active_set:
        return dominant_view, "optimizer_or_update_path"
    return dominant_view, "mixed"


def render_bucket_table(
    reports: dict[str, dict[str, Any]],
    buckets: list[str],
) -> tuple[list[dict[str, Any]], str]:
    """Build the bucket attribution table in both data and markdown forms.

    :param dict[str, dict[str, Any]] reports: Structured reports keyed by short name.
    :param list[str] buckets: Exact bucket names to compare.
    :return tuple[list[dict[str, Any]], str]: Table rows and markdown rendering.
    """
    rows: list[dict[str, Any]] = []
    md_lines = [
        "| Bucket | bare_gdn | hybrid_fwd_bwd | hybrid_opt | trainer_eager | Dominant view | Suggested owner |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for bucket in buckets:
        view_rows = {
            VIEW_LABELS[name]: find_row(report, bucket)
            for name, report in reports.items()
        }
        dominant_view, owner = classify_bucket(view_rows)
        row = {
            "bucket": bucket,
            "bare_gdn": bucket_cell(view_rows["bare_gdn"]),
            "hybrid_fwd_bwd": bucket_cell(view_rows["hybrid_fwd_bwd"]),
            "hybrid_opt": bucket_cell(view_rows["hybrid_opt"]),
            "trainer_eager": bucket_cell(view_rows["trainer_eager"]),
            "dominant_view": dominant_view,
            "suggested_owner": owner,
        }
        rows.append(row)
        md_lines.append(
            f"| `{bucket}` | {row['bare_gdn']} | {row['hybrid_fwd_bwd']} | "
            f"{row['hybrid_opt']} | {row['trainer_eager']} | "
            f"`{dominant_view}` | `{owner}` |"
        )
    return rows, "\n".join(md_lines) + "\n"


def load_boundary_audit(path: Path | None) -> list[dict[str, Any]]:
    """Load structured HGDN boundary audit JSONL records.

    :param Path | None path: Audit path.
    :return list[dict[str, Any]]: Flat per-tensor boundary records.
    """
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for tensor in record["tensors"]:
            rows.append(
                {
                    "call_index": record["call_index"],
                    "boundary": record["boundary"],
                    "tensor": tensor["name"],
                    "dtype": tensor["dtype"],
                    "device": tensor["device"],
                    "shape": tuple(tensor["shape"]),
                    "stride": tuple(tensor["stride"]),
                    "contiguous": int(tensor["contiguous"]),
                }
            )
    return rows


def render_boundary_table(rows: list[dict[str, Any]]) -> str:
    """Render the boundary audit table as markdown.

    :param list[dict[str, Any]] rows: Flat boundary audit rows.
    :return str: Markdown table.
    """
    if not rows:
        return "_No boundary audit captured._\n"
    lines = [
        "| Call | Boundary | Tensor | Dtype | Shape | Stride | Contiguous | Device |",
        "|---:|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['call_index']} | `{row['boundary']}` | `{row['tensor']}` | "
            f"`{row['dtype']}` | `{row['shape']}` | `{row['stride']}` | "
            f"{row['contiguous']} | `{row['device']}` |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a row list as CSV.

    :param Path path: Output CSV path.
    :param list[dict[str, Any]] rows: Rows to write.
    """
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Load phase-1 outputs and write decision-complete analysis artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = {
        "gdn": load_profile_report(args.gdn),
        "hybrid_fwd_bwd": load_profile_report(args.hybrid_fwd_bwd),
        "hybrid_opt": load_profile_report(args.hybrid_opt),
        "trainer": load_profile_report(args.trainer),
    }

    bucket_rows, bucket_markdown = render_bucket_table(reports, args.buckets)
    boundary_rows = load_boundary_audit(args.boundary_audit)
    boundary_markdown = render_boundary_table(boundary_rows)

    summary = {
        "inputs": {
            "gdn": str(args.gdn),
            "hybrid_fwd_bwd": str(args.hybrid_fwd_bwd),
            "hybrid_opt": str(args.hybrid_opt),
            "trainer": str(args.trainer),
            "boundary_audit": str(args.boundary_audit) if args.boundary_audit else None,
        },
        "buckets": bucket_rows,
        "boundary_rows": boundary_rows,
    }
    (args.output_dir / "bucket_attribution.json").write_text(
        json.dumps(bucket_rows, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "bucket_attribution.csv", bucket_rows)
    (args.output_dir / "boundary_audit.json").write_text(
        json.dumps(boundary_rows, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "boundary_audit.csv", boundary_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    markdown = (
        "# HGDN Phase-1 Analysis\n\n"
        "## Bucket Attribution\n\n"
        f"{bucket_markdown}\n"
        "## Boundary Audit\n\n"
        f"{boundary_markdown}"
    )
    (args.output_dir / "analysis.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
