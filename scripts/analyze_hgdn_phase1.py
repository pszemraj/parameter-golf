"""Analyze local HGDN phase-1 profiler outputs.

This script converts the phase-1 profiler bundle into two decision aids:

- a bucket-to-path attribution table
- an HGDN boundary dtype/layout table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profiler_report import (  # noqa: E402
    HGDN_TRANSFER_BUCKETS,
    find_profile_row,
    format_profile_bucket_cell,
    load_boundary_audit_jsonl,
    load_profile_report,
    markdown_table,
    write_json,
    write_rows_csv,
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
        default=list(HGDN_TRANSFER_BUCKETS),
        help="Exact profiler bucket names to compare.",
    )
    return parser.parse_args()


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
    md_rows: list[tuple[str, str, str, str, str, str, str]] = []
    for bucket in buckets:
        view_rows = {
            VIEW_LABELS[name]: find_profile_row(report, bucket)
            for name, report in reports.items()
        }
        dominant_view, owner = classify_bucket(view_rows)
        row = {
            "bucket": bucket,
            "bare_gdn": format_profile_bucket_cell(view_rows["bare_gdn"]),
            "hybrid_fwd_bwd": format_profile_bucket_cell(view_rows["hybrid_fwd_bwd"]),
            "hybrid_opt": format_profile_bucket_cell(view_rows["hybrid_opt"]),
            "trainer_eager": format_profile_bucket_cell(view_rows["trainer_eager"]),
            "dominant_view": dominant_view,
            "suggested_owner": owner,
        }
        rows.append(row)
        md_rows.append(
            (
                f"`{bucket}`",
                row["bare_gdn"],
                row["hybrid_fwd_bwd"],
                row["hybrid_opt"],
                row["trainer_eager"],
                f"`{dominant_view}`",
                f"`{owner}`",
            )
        )
    return rows, markdown_table(
        [
            "Bucket",
            "bare_gdn",
            "hybrid_fwd_bwd",
            "hybrid_opt",
            "trainer_eager",
            "Dominant view",
            "Suggested owner",
        ],
        md_rows,
        aligns=["---", "---:", "---:", "---:", "---:", "---", "---"],
    )


def render_boundary_table(rows: list[dict[str, Any]]) -> str:
    """Render the boundary audit table as markdown.

    :param list[dict[str, Any]] rows: Flat boundary audit rows.
    :return str: Markdown table.
    """
    if not rows:
        return "_No boundary audit captured._\n"
    return markdown_table(
        [
            "Call",
            "Boundary",
            "Tensor",
            "Dtype",
            "Shape",
            "Stride",
            "Contiguous",
            "Device",
        ],
        [
            (
                row["call_index"],
                f"`{row['boundary']}`",
                f"`{row['tensor']}`",
                f"`{row['dtype']}`",
                f"`{row['shape']}`",
                f"`{row['stride']}`",
                row["contiguous"],
                f"`{row['device']}`",
            )
            for row in rows
        ],
        aligns=["---:", "---", "---", "---", "---", "---", "---:", "---"],
    )


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
    boundary_rows = load_boundary_audit_jsonl(args.boundary_audit)
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
    write_json(args.output_dir / "bucket_attribution.json", bucket_rows)
    write_rows_csv(args.output_dir / "bucket_attribution.csv", bucket_rows)
    write_json(args.output_dir / "boundary_audit.json", boundary_rows)
    write_rows_csv(args.output_dir / "boundary_audit.csv", boundary_rows)
    write_json(args.output_dir / "summary.json", summary)

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
