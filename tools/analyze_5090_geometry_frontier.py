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
SCREEN_PLANNED_STEPS = 4096
SCREEN_EFFECTIVE_STEP_TOKENS = 131072
SCREEN_NUM_BLOCKS = 0
SCREEN_TRIGRAM_TOP_K = 2
TIME_MATCHED_STEP_GRANULARITY = 128
GEOMETRY_SUMMARY_FIELDS = ("core_dim", "core_inner_dim", "recurrent_cells")


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


@dataclass(frozen=True)
class AnalyzedRow:
    """One analyzer row with protocol validation state."""

    geometry: Geometry
    summary: dict[str, str]
    delta_bpb: Optional[float]
    speed_ratio: Optional[float]
    verdict: str
    eligibility_errors: tuple[str, ...]
    estimated_time_matched_steps: Optional[int]

    @property
    def is_valid_screen_row(self) -> bool:
        """Return whether the row is a completed screen row with matching geometry.

        :return bool: ``True`` when decisions and confirmation commands are allowed.
        """
        return not self.eligibility_errors


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


def load_summary_rows(
    repo_root: Path, geometry: Geometry, *, run_version: str, seed: str
) -> list[dict[str, str]]:
    """Load matching summary rows for a geometry run.

    :param Path repo_root: Repository root.
    :param Geometry geometry: Geometry to inspect.
    :param str run_version: Run version suffix.
    :param str seed: Seed string.
    :return list[dict[str, str]]: Matching summary rows.
    """
    summary_path = (
        repo_root
        / "experiments"
        / "5090_architecture"
        / f"{geometry.label}_trigram_seed{seed}_{run_version}"
        / "summary.tsv"
    )
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out: list[dict[str, str]] = []
    for row in rows:
        hydrate_summary_geometry(row)
        row_seed = str(row.get("seed", "")).strip()
        run_name = str(row.get("run_name", "")).strip()
        if row_seed == str(seed) or run_name.endswith(f"_s{seed}") or not row_seed:
            out.append(row)
    return out


def load_summary_row(
    repo_root: Path, geometry: Geometry, *, run_version: str, seed: str
) -> dict[str, str]:
    """Load the unique seed-matching summary row for a geometry run.

    :param Path repo_root: Repository root.
    :param Geometry geometry: Geometry to inspect.
    :param str run_version: Run version suffix.
    :param str seed: Seed string.
    :return dict[str, str]: Summary row, or an empty dict when missing.
    :raises SystemExit: If multiple matching rows make the run ambiguous.
    """
    rows = load_summary_rows(repo_root, geometry, run_version=run_version, seed=seed)
    if not rows:
        return {}
    if len(rows) == 1:
        return rows[0]
    completed = [row for row in rows if row.get("status") == "completed"]
    if len(completed) == 1:
        return completed[0]
    names = ", ".join(row.get("run_name", "<unnamed>") for row in rows)
    raise SystemExit(
        f"ambiguous summary rows for {geometry.label} seed={seed} run_version={run_version}: "
        f"{names}"
    )


def hydrate_summary_geometry(row: dict[str, str]) -> None:
    """Fill missing geometry fields from a run's resolved config when available.

    This keeps the frontier analyzer useful for runs completed before the
    summary TSV gained explicit geometry columns.

    :param dict[str, str] row: Summary row to update in place.
    """
    if not row or all(row.get(field) for field in GEOMETRY_SUMMARY_FIELDS):
        return
    run_dir_raw = row.get("run_dir")
    if not run_dir_raw:
        return
    resolved_path = Path(run_dir_raw) / "resolved_config.json"
    if not resolved_path.exists():
        return
    try:
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        model = resolved.get("model", {})
        core_dim = int(model["core_dim"])
        core_layers = int(model["core_layers"])
        core_inner_dim = int(core_dim * float(model["core_expansion"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return
    row.setdefault("core_dim", str(core_dim))
    row.setdefault("core_inner_dim", str(core_inner_dim))
    row.setdefault("recurrent_cells", str(core_layers * core_inner_dim))
    if not row["core_dim"]:
        row["core_dim"] = str(core_dim)
    if not row["core_inner_dim"]:
        row["core_inner_dim"] = str(core_inner_dim)
    if not row["recurrent_cells"]:
        row["recurrent_cells"] = str(core_layers * core_inner_dim)


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


def as_int(value: Any) -> Optional[int]:
    """Parse an optional integer.

    :param Any value: Value to parse.
    :return Optional[int]: Parsed integer or ``None``.
    """
    if value is None or value == "":
        return None
    try:
        out = int(str(value))
    except (TypeError, ValueError):
        return None
    return out


def screen_bpb(row: dict[str, str]) -> Optional[float]:
    """Return the screen score used for promotion decisions.

    :param dict[str, str] row: Summary row.
    :return Optional[float]: Last eval BPB, falling back to best BPB.
    """
    return as_float(row.get("last_val_bpb")) or as_float(row.get("best_val_bpb"))


def eligibility_errors(geometry: Geometry, row: dict[str, str]) -> tuple[str, ...]:
    """Return protocol violations that make a row decision-ineligible.

    :param Geometry geometry: Geometry parsed from the label.
    :param dict[str, str] row: Summary row.
    :return tuple[str, ...]: Human-readable validation errors.
    """
    if not row:
        return ("missing_summary",)

    errors: list[str] = []
    if row.get("status") != "completed":
        errors.append("not_completed")
    expected = {
        "planned_steps": SCREEN_PLANNED_STEPS,
        "effective_step_tokens": SCREEN_EFFECTIVE_STEP_TOKENS,
        "num_blocks": SCREEN_NUM_BLOCKS,
        "trigram_top_k": SCREEN_TRIGRAM_TOP_K,
        "core_dim": geometry.core_dim,
        "core_layers": geometry.layers,
        "core_inner_dim": geometry.inner_dim,
        "recurrent_cells": geometry.recurrent_cells,
    }
    for field, expected_value in expected.items():
        actual = as_int(row.get(field))
        if actual != expected_value:
            errors.append(f"{field}!={expected_value}")
    return tuple(errors)


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
    bench_tok_s = as_float(benchmark_row.get("tokens_per_sec"))
    if bench_tok_s is not None and baseline_benchmark_tok_s and baseline_benchmark_tok_s > 0:
        return bench_tok_s / baseline_benchmark_tok_s
    observed = as_float(row.get("steady_state_tokens_per_sec"))
    if observed is not None and baseline_tok_s > 0:
        return observed / baseline_tok_s
    return None


def decision(delta_bpb: Optional[float], ratio: Optional[float], *, valid_screen_row: bool) -> str:
    """Return the promotion decision for one row.

    :param Optional[float] delta_bpb: Geometry BPB minus baseline BPB.
    :param Optional[float] ratio: Speed ratio versus baseline.
    :param bool valid_screen_row: Whether the row is decision-eligible.
    :return str: Decision label.
    """
    if not valid_screen_row or delta_bpb is None:
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


def round_to_multiple(value: float, multiple: int) -> int:
    """Round a float to the nearest positive multiple.

    :param float value: Value to round.
    :param int multiple: Positive multiple.
    :return int: Rounded integer multiple.
    """
    return int(round(value / multiple) * multiple)


def estimated_time_matched_steps(
    row: dict[str, str], ratio: Optional[float], *, valid_screen_row: bool
) -> Optional[int]:
    """Estimate same-wallclock step budget from screen steps and speed ratio.

    :param dict[str, str] row: Summary row.
    :param Optional[float] ratio: Speed ratio versus current benchmark/baseline.
    :param bool valid_screen_row: Whether the row is a valid screen row.
    :return Optional[int]: Step estimate rounded to 128, or ``None``.
    """
    planned_steps = as_int(row.get("planned_steps"))
    if not valid_screen_row or ratio is None or planned_steps is None:
        return None
    return max(
        TIME_MATCHED_STEP_GRANULARITY,
        round_to_multiple(planned_steps * ratio, TIME_MATCHED_STEP_GRANULARITY),
    )


def confirmation_command(geometry: Geometry, *, run_version: str, seed: str) -> str:
    """Build the next confirmation command for a promoted row.

    :param Geometry geometry: Geometry to confirm.
    :param str run_version: Source run version.
    :param str seed: Seed string.
    :return str: Shell command.
    """
    return (
        "bash scripts/run_5090_trigram_aligned_geometry_screen.sh "
        f"--run-version {run_version}_confirm "
        f"--seeds {seed} "
        f"--geometry-label {geometry.label} "
        f"--geometry-core-dim {geometry.core_dim} "
        f"--geometry-core-layers {geometry.layers} "
        f"--geometry-core-inner-dim {geometry.inner_dim} "
        "--num-steps 8192 "
        "--lr-hold-steps 7000 "
        "--full-val-final "
        "--val-every 512 "
        "--log-every 128 "
        "--log-state-every 512 "
        "--save-every 4096"
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


def format_optional_int(value: Optional[int]) -> str:
    """Format optional integers for Markdown tables.

    :param Optional[int] value: Optional value.
    :return str: Rendered value.
    """
    return "" if value is None else str(value)


def main() -> None:
    """Analyze completed or pending geometry rows."""
    args = parse_args()
    repo_root = args.repo_root.resolve()
    labels = tuple(args.label or DEFAULT_LABELS)
    geometries = [parse_geometry(label) for label in labels]
    benchmark = load_benchmark(args.benchmark)
    baseline_bench = as_float(benchmark.get("current_d48_l12_i480", {}).get("tokens_per_sec"))

    rows: list[AnalyzedRow] = []
    for geometry in geometries:
        summary = load_summary_row(
            repo_root, geometry, run_version=args.run_version, seed=args.seed
        )
        errors = eligibility_errors(geometry, summary)
        valid_screen_row = not errors
        bpb = screen_bpb(summary)
        delta = None if bpb is None else bpb - args.baseline_bpb
        ratio = speed_ratio(
            summary,
            benchmark.get(geometry.benchmark_name, {}),
            baseline_tok_s=args.baseline_tok_s,
            baseline_benchmark_tok_s=baseline_bench,
        )
        verdict = decision(delta, ratio, valid_screen_row=valid_screen_row)
        rows.append(
            AnalyzedRow(
                geometry=geometry,
                summary=summary,
                delta_bpb=delta,
                speed_ratio=ratio,
                verdict=verdict,
                eligibility_errors=errors,
                estimated_time_matched_steps=estimated_time_matched_steps(
                    summary, ratio, valid_screen_row=valid_screen_row
                ),
            )
        )

    print("# 5090 Geometry Frontier Read")
    print()
    print(f"- run_version: `{args.run_version}`")
    print(f"- seed: `{args.seed}`")
    print(f"- baseline_bpb: `{args.baseline_bpb}`")
    print(f"- baseline_tok_s: `{args.baseline_tok_s}`")
    if baseline_bench is not None:
        print(f"- benchmark_baseline: `current_d48_l12_i480` `{baseline_bench}` tok/s")
    if args.benchmark:
        print(f"- benchmark: `{args.benchmark}`")
    print()
    print(
        "| geometry | cells | status | screen bpb | delta | speed ratio | est time-matched steps | decision | notes |"
    )
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        geometry = row.geometry
        status = row.summary.get("status", "missing")
        bpb = screen_bpb(row.summary)
        notes = ",".join(row.eligibility_errors)
        print(
            "| {label} | {cells} | {status} | {bpb} | {delta} | {ratio} | {steps} | {verdict} | {notes} |".format(
                label=geometry.label,
                cells=geometry.recurrent_cells,
                status=status,
                bpb=format_optional(bpb, digits=10),
                delta=format_optional(row.delta_bpb, digits=6),
                ratio=format_optional(row.speed_ratio, digits=3),
                steps=format_optional_int(row.estimated_time_matched_steps),
                verdict=row.verdict,
                notes=notes,
            )
        )

    promoted = [
        (row.geometry, row.verdict)
        for row in rows
        if row.is_valid_screen_row and row.verdict.startswith("promote")
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
        print(
            "If all rows are completed and killed, rerun the adaptive planner for the selected stage:"
        )
        print(
            "python tools/plan_5090_adaptive_closeout.py "
            "--stage k4 --run-version geom1 --seed 1337 --emit markdown"
        )


if __name__ == "__main__":
    main()
