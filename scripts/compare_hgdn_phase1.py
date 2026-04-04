"""Compare two local HGDN phase-1 bundles.

This script is the structured before/after companion to
`scripts/run_hgdn_local_phase1.sh`. It compares the same transfer buckets across
baseline and candidate bundles and highlights whether the HGDN boundary layouts
actually changed.
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
    "gdn.conv_qkv",
    "gdn.recurrence",
    "aten::convolution_backward",
    "aten::_conv_depthwise2d",
    "gdn.norm_qkv",
    "gdn.gates",
    "gdn.output_gate",
)

VIEW_PATHS = {
    "bare_gdn": ("hotpath", "gdn_fwd_bwd.json"),
    "hybrid_fwd_bwd": ("hotpath", "hybrid_fwd_bwd.json"),
    "hybrid_opt": ("hotpath", "hybrid_optimizer.json"),
}

BOUNDARY_KEYS = (
    ("conv_qkv", "q"),
    ("conv_qkv", "k"),
    ("conv_qkv", "v"),
    ("norm_qkv", "q"),
    ("norm_qkv", "k"),
    ("norm_qkv", "v"),
    ("recurrence_inputs", "q"),
    ("recurrence_inputs", "k"),
    ("recurrence_inputs", "v"),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for phase-1 bundle comparison.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for markdown/json/csv output. Defaults to <candidate>/compare_vs_<baseline.name>/",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=list(DEFAULT_BUCKETS),
        help="Exact profiler bucket names to compare.",
    )
    return parser.parse_args()


def find_trainer_report(bundle_root: Path) -> Path:
    """Find the trainer eager report under one phase-1 bundle.

    :param Path bundle_root: Phase-1 bundle root.
    :raises FileNotFoundError: If no trainer report exists.
    :return Path: Trainer report directory.
    """
    matches = sorted(bundle_root.glob("trainer/*/key_averages.json"))
    if not matches:
        raise FileNotFoundError(
            f"No trainer report found under {bundle_root / 'trainer'}"
        )
    return matches[0].parent


def load_phase1_reports(bundle_root: Path) -> dict[str, dict[str, Any]]:
    """Load the four structured reports from one phase-1 bundle.

    :param Path bundle_root: Phase-1 bundle root.
    :return dict[str, dict[str, Any]]: Reports keyed by view label.
    """
    reports: dict[str, dict[str, Any]] = {}
    for label, parts in VIEW_PATHS.items():
        reports[label] = load_profile_report(bundle_root.joinpath(*parts))
    reports["trainer_eager"] = load_profile_report(find_trainer_report(bundle_root))
    return reports


def load_boundary_rows(bundle_root: Path) -> list[dict[str, Any]]:
    """Load flattened boundary-audit rows from one phase-1 bundle.

    :param Path bundle_root: Phase-1 bundle root.
    :return list[dict[str, Any]]: Boundary rows, or an empty list if missing.
    """
    path = bundle_root / "analysis" / "boundary_audit.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def find_row(report: dict[str, Any], bucket: str) -> dict[str, Any] | None:
    """Find one exact event row inside a structured profile report.

    :param dict[str, Any] report: Structured profile report.
    :param str bucket: Exact event label.
    :return dict[str, Any] | None: Matching row, if present.
    """
    for row in report["rows"]:
        if row["name"] == bucket:
            return row
    return None


def fmt_bucket(row: dict[str, Any] | None) -> str:
    """Format one profiler row into a compact cell.

    :param dict[str, Any] row: Structured row, or `None`.
    :return str: Compact display string.
    """
    if row is None:
        return "-"
    return (
        f"{row['self_device_time_us'] / 1000.0:.2f}ms / "
        f"{row['self_device_percent']:.2f}% / {row['count']}"
    )


def bucket_ms(row: dict[str, Any] | None) -> float:
    """Return self-device milliseconds for one profiler row.

    :param dict[str, Any] row: Structured profiler row.
    :return float: Self-device milliseconds.
    """
    if row is None:
        return 0.0
    return row["self_device_time_us"] / 1000.0


def bucket_pct(row: dict[str, Any] | None) -> float:
    """Return self-device percentage for one profiler row.

    :param dict[str, Any] row: Structured profiler row.
    :return float: Self-device percentage.
    """
    if row is None:
        return 0.0
    return row["self_device_percent"]


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


def build_bucket_rows(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    buckets: list[str],
) -> list[dict[str, Any]]:
    """Build before/after bucket deltas across all four profiler views.

    :param dict[str, dict[str, Any]] baseline: Baseline reports keyed by view label.
    :param dict[str, dict[str, Any]] candidate: Candidate reports keyed by view label.
    :param list[str] buckets: Exact bucket names to compare.
    :return list[dict[str, Any]]: Flat comparison rows.
    """
    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        for view in ("bare_gdn", "hybrid_fwd_bwd", "hybrid_opt", "trainer_eager"):
            base_row = find_row(baseline[view], bucket)
            cand_row = find_row(candidate[view], bucket)
            rows.append(
                {
                    "bucket": bucket,
                    "view": view,
                    "baseline": fmt_bucket(base_row),
                    "candidate": fmt_bucket(cand_row),
                    "delta_ms": round(bucket_ms(cand_row) - bucket_ms(base_row), 2),
                    "delta_pct_points": round(
                        bucket_pct(cand_row) - bucket_pct(base_row), 2
                    ),
                }
            )
    return rows


def render_bucket_markdown(rows: list[dict[str, Any]]) -> str:
    """Render the bucket comparison table as markdown.

    :param list[dict[str, Any]] rows: Flat comparison rows.
    :return str: Markdown table.
    """
    lines = [
        "| Bucket | View | Baseline | Candidate | Δ self CUDA ms | Δ self CUDA pp |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['bucket']}` | `{row['view']}` | {row['baseline']} | "
            f"{row['candidate']} | {row['delta_ms']:+.2f} | {row['delta_pct_points']:+.2f} |"
        )
    return "\n".join(lines) + "\n"


def boundary_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index boundary-audit rows by boundary/tensor pair for the first call.

    :param list[dict[str, Any]] rows: Flat boundary rows.
    :return dict[tuple[str, str], dict[str, Any]]: Indexed rows.
    """
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["boundary"], row["tensor"])
        if key not in indexed or row["call_index"] < indexed[key]["call_index"]:
            indexed[key] = row
    return indexed


def build_boundary_rows(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the boundary-layout comparison rows.

    :param list[dict[str, Any]] baseline: Baseline boundary rows.
    :param list[dict[str, Any]] candidate: Candidate boundary rows.
    :return list[dict[str, Any]]: Flat boundary comparison rows.
    """
    base_index = boundary_lookup(baseline)
    cand_index = boundary_lookup(candidate)
    rows: list[dict[str, Any]] = []
    for boundary, tensor in BOUNDARY_KEYS:
        base_row = base_index.get((boundary, tensor))
        cand_row = cand_index.get((boundary, tensor))
        rows.append(
            {
                "boundary": boundary,
                "tensor": tensor,
                "baseline_dtype": base_row["dtype"] if base_row else "-",
                "candidate_dtype": cand_row["dtype"] if cand_row else "-",
                "baseline_stride": tuple(base_row["stride"]) if base_row else "-",
                "candidate_stride": tuple(cand_row["stride"]) if cand_row else "-",
                "baseline_contiguous": (
                    int(base_row["contiguous"]) if base_row else -1
                ),
                "candidate_contiguous": (
                    int(cand_row["contiguous"]) if cand_row else -1
                ),
            }
        )
    return rows


def render_boundary_markdown(rows: list[dict[str, Any]]) -> str:
    """Render the boundary comparison table as markdown.

    :param list[dict[str, Any]] rows: Boundary comparison rows.
    :return str: Markdown table.
    """
    lines = [
        "| Boundary | Tensor | Baseline dtype | Candidate dtype | Baseline stride | Candidate stride | Baseline contig | Candidate contig |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['boundary']}` | `{row['tensor']}` | `{row['baseline_dtype']}` | "
            f"`{row['candidate_dtype']}` | `{row['baseline_stride']}` | "
            f"`{row['candidate_stride']}` | {row['baseline_contiguous']} | "
            f"{row['candidate_contiguous']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Compare two local HGDN phase-1 bundles and write structured deltas."""
    args = parse_args()
    output_dir = args.output_dir or (
        args.candidate / f"compare_vs_{args.baseline.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_reports = load_phase1_reports(args.baseline)
    candidate_reports = load_phase1_reports(args.candidate)
    baseline_boundaries = load_boundary_rows(args.baseline)
    candidate_boundaries = load_boundary_rows(args.candidate)

    bucket_rows = build_bucket_rows(
        baseline_reports,
        candidate_reports,
        args.buckets,
    )
    boundary_rows = build_boundary_rows(baseline_boundaries, candidate_boundaries)

    summary = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "bucket_rows": bucket_rows,
        "boundary_rows": boundary_rows,
    }
    (output_dir / "bucket_deltas.json").write_text(
        json.dumps(bucket_rows, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "bucket_deltas.csv", bucket_rows)
    (output_dir / "boundary_deltas.json").write_text(
        json.dumps(boundary_rows, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "boundary_deltas.csv", boundary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    markdown = (
        "# HGDN Phase-1 Comparison\n\n"
        f"- baseline: `{args.baseline}`\n"
        f"- candidate: `{args.candidate}`\n\n"
        "## Bucket Deltas\n\n"
        f"{render_bucket_markdown(bucket_rows)}\n"
        "## Boundary Deltas\n\n"
        f"{render_boundary_markdown(boundary_rows)}"
    )
    (output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
