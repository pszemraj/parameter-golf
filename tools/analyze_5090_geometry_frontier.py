#!/usr/bin/env python3
"""Analyze the final-three-day 5090 geometry frontier batch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = (
    "blocks0_d96_l6_i512",
    "blocks0_d64_l10_i512",
    "blocks0_d128_l4_i512",
    "blocks0_d128_l5_i512",
)
DEFAULT_BASELINE_BPB = 2.075171567328695
DEFAULT_BASELINE_TOK_S = 571660.2196885378


@dataclass(frozen=True)
class Geometry:
    """One geometry frontier point."""

    label: str
    core_dim: int
    layers: int
    inner_dim: int

    @property
    def recurrent_cells(self) -> int:
        """Return total stacked recurrent cells.

        :return int: ``layers * inner_dim``.
        """
        return int(self.layers * self.inner_dim)

    @property
    def benchmark_name(self) -> str:
        """Return the matching benchmark row name.

        :return str: Benchmark shape name.
        """
        return self.label.removeprefix("blocks0_")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--run-version", default="geom1")
    ap.add_argument("--seed", default="1337")
    ap.add_argument("--benchmark", type=Path, default=None)
    ap.add_argument("--baseline-bpb", type=float, default=DEFAULT_BASELINE_BPB)
    ap.add_argument("--baseline-tok-s", type=float, default=DEFAULT_BASELINE_TOK_S)
    ap.add_argument("--label", action="append", default=None, help="Geometry label to analyze")
    return ap.parse_args()


def parse_geometry(label: str) -> Geometry:
    """Parse a ``blocks0_d*_l*_i*`` geometry label.

    :param str label: Geometry label.
    :return Geometry: Parsed geometry.
    """
    match = re.search(r"d(?P<dim>\d+)_l(?P<layers>\d+)_i(?P<inner>\d+)", label)
    if match is None:
        raise SystemExit(f"cannot parse geometry label: {label}")
    return Geometry(
        label=label,
        core_dim=int(match.group("dim")),
        layers=int(match.group("layers")),
        inner_dim=int(match.group("inner")),
    )


def load_benchmark(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    """Load benchmark rows keyed by shape name.

    :param Optional[Path] path: Optional benchmark JSON path.
    :return dict[str, dict[str, Any]]: Benchmark rows.
    """
    if path is None or not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("name", "")): row for row in rows if row.get("name")}


def load_summary_row(
    repo_root: Path, geometry: Geometry, *, run_version: str, seed: str
) -> dict[str, str]:
    """Load the first summary row for a geometry run.

    :param Path repo_root: Repository root.
    :param Geometry geometry: Geometry to inspect.
    :param str run_version: Run version suffix.
    :param str seed: Seed string.
    :return dict[str, str]: Summary row, or an empty dict when missing.
    """
    summary_path = (
        repo_root
        / "experiments"
        / "5090_architecture"
        / f"{geometry.label}_trigram_seed{seed}_{run_version}"
        / "summary.tsv"
    )
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows[0] if rows else {}


def as_float(value: Any) -> Optional[float]:
    """Parse an optional float.

    :param Any value: Value to parse.
    :return Optional[float]: Parsed float or ``None``.
    """
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def screen_bpb(row: dict[str, str]) -> Optional[float]:
    """Return the screen score used for promotion decisions.

    :param dict[str, str] row: Summary row.
    :return Optional[float]: Last eval BPB, falling back to best BPB.
    """
    return as_float(row.get("last_val_bpb")) or as_float(row.get("best_val_bpb"))


def speed_ratio(
    row: dict[str, str],
    benchmark_row: dict[str, Any],
    *,
    baseline_tok_s: float,
    baseline_benchmark_tok_s: Optional[float],
) -> Optional[float]:
    """Return observed or benchmarked speed ratio versus current K2 baseline.

    :param dict[str, str] row: Summary row.
    :param dict[str, Any] benchmark_row: Benchmark row.
    :param float baseline_tok_s: Current baseline training throughput.
    :param Optional[float] baseline_benchmark_tok_s: Current benchmark throughput.
    :return Optional[float]: Speed ratio.
    """
    observed = as_float(row.get("steady_state_tokens_per_sec"))
    if observed is not None and baseline_tok_s > 0:
        return observed / baseline_tok_s
    bench_tok_s = as_float(benchmark_row.get("tokens_per_sec"))
    if bench_tok_s is not None and baseline_benchmark_tok_s and baseline_benchmark_tok_s > 0:
        return bench_tok_s / baseline_benchmark_tok_s
    return None


def decision(delta_bpb: Optional[float], ratio: Optional[float]) -> str:
    """Return the promotion decision for one row.

    :param Optional[float] delta_bpb: Geometry BPB minus baseline BPB.
    :param Optional[float] ratio: Speed ratio versus baseline.
    :return str: Decision label.
    """
    if delta_bpb is None:
        return "pending"
    if delta_bpb <= 0:
        return "promote_1b"
    ratio = ratio or 0.0
    if delta_bpb <= 0.020 and ratio >= 1.5:
        return "promote_time_matched_8192"
    if delta_bpb <= 0.035 and ratio >= 2.0:
        return "promote_time_matched_8192"
    if delta_bpb > 0.040:
        return "kill"
    return "inspect_curve"


def confirmation_command(geometry: Geometry, *, run_version: str, seed: str) -> str:
    """Build the next confirmation command for a promoted row.

    :param Geometry geometry: Geometry to confirm.
    :param str run_version: Source run version.
    :param str seed: Seed string.
    :return str: Shell command.
    """
    return (
        f"RUN_VERSION={run_version}_confirm SEEDS={seed} "
        f"GEOMETRY_LABEL={geometry.label} "
        f"GEOMETRY_CORE_DIM={geometry.core_dim} "
        f"GEOMETRY_CORE_LAYERS={geometry.layers} "
        f"GEOMETRY_CORE_INNER_DIM={geometry.inner_dim} "
        "GEOMETRY_NUM_STEPS=8192 GEOMETRY_LR_HOLD_STEPS=7000 "
        "FULL_VAL_FINAL=1 VAL_EVERY=512 LOG_EVERY=128 "
        "LOG_STATE_EVERY=512 SAVE_EVERY=4096 "
        "bash scripts/run_5090_trigram_aligned_geometry_screen.sh"
    )


def format_optional(value: Optional[float], *, digits: int = 6) -> str:
    """Format optional floats for Markdown tables.

    :param Optional[float] value: Optional value.
    :param int digits: Decimal places.
    :return str: Rendered value.
    """
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def main() -> None:
    """Analyze completed or pending geometry rows."""
    args = parse_args()
    repo_root = args.repo_root.resolve()
    labels = tuple(args.label or DEFAULT_LABELS)
    geometries = [parse_geometry(label) for label in labels]
    benchmark = load_benchmark(args.benchmark)
    baseline_bench = as_float(benchmark.get("current_d48_l12_i480", {}).get("tokens_per_sec"))

    rows: list[tuple[Geometry, dict[str, str], Optional[float], Optional[float], str]] = []
    for geometry in geometries:
        summary = load_summary_row(
            repo_root, geometry, run_version=args.run_version, seed=args.seed
        )
        bpb = screen_bpb(summary)
        delta = None if bpb is None else bpb - args.baseline_bpb
        ratio = speed_ratio(
            summary,
            benchmark.get(geometry.benchmark_name, {}),
            baseline_tok_s=args.baseline_tok_s,
            baseline_benchmark_tok_s=baseline_bench,
        )
        rows.append((geometry, summary, delta, ratio, decision(delta, ratio)))

    print("# 5090 Geometry Frontier Read")
    print()
    print(f"- run_version: `{args.run_version}`")
    print(f"- seed: `{args.seed}`")
    print(f"- baseline_bpb: `{args.baseline_bpb}`")
    print(f"- baseline_tok_s: `{args.baseline_tok_s}`")
    if args.benchmark:
        print(f"- benchmark: `{args.benchmark}`")
    print()
    print("| geometry | cells | status | screen bpb | delta | speed ratio | decision |")
    print("| --- | ---: | --- | ---: | ---: | ---: | --- |")
    for geometry, summary, delta, ratio, verdict in rows:
        status = summary.get("status", "missing")
        bpb = screen_bpb(summary)
        print(
            "| {label} | {cells} | {status} | {bpb} | {delta} | {ratio} | {verdict} |".format(
                label=geometry.label,
                cells=geometry.recurrent_cells,
                status=status,
                bpb=format_optional(bpb, digits=10),
                delta=format_optional(delta, digits=6),
                ratio=format_optional(ratio, digits=3),
                verdict=verdict,
            )
        )

    promoted = [
        (geometry, verdict) for geometry, _, _, _, verdict in rows if verdict.startswith("promote")
    ]
    print()
    if promoted:
        print("## Next Commands")
        print()
        for geometry, verdict in promoted:
            print(f"# {geometry.label}: {verdict}")
            print(confirmation_command(geometry, run_version=args.run_version, seed=args.seed))
            print()
    else:
        print("No geometry row currently clears the automatic promotion thresholds.")
        print()
        print("If all rows are completed and killed, return to K4 headroom on the current leader:")
        print(
            "RUN_VERSION=v2 TRIGRAM_TOP_K=4 SEEDS=1337 bash scripts/run_5090_trigram_memory_screen.sh"
        )


if __name__ == "__main__":
    main()
