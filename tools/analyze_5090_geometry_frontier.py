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
DECISION_PLANNED_STEPS = 8192
SCREEN_EFFECTIVE_STEP_TOKENS = 131072
SCREEN_NUM_BLOCKS = 0
SCREEN_TRIGRAM_TOP_K = 2
TIME_MATCHED_STEP_GRANULARITY = 128


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


@dataclass(frozen=True)
class ScreenContract:
    """Expected screen protocol for one analyzer invocation."""

    planned_steps: int
    effective_step_tokens: int
    num_blocks: int
    trigram_top_k: int
    seq_len: int
    batch_size: int
    bptt_chunks: int


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
    ap.add_argument("--screen-steps", type=int, default=None)
    ap.add_argument("--effective-step-tokens", type=int, default=SCREEN_EFFECTIVE_STEP_TOKENS)
    ap.add_argument("--screen-trigram-top-k", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--screen-batch-size", type=int, default=None)
    ap.add_argument("--screen-bptt-chunks", type=int, default=None)
    return ap.parse_args()


def infer_seq_len(run_version: str) -> int:
    """Infer the screen sequence length from a run-version suffix.

    :param str run_version: Run-version suffix.
    :return int: Inferred sequence length.
    """
    match = re.search(r"seq(?P<seq>\d+)", str(run_version))
    if match is not None:
        return int(match.group("seq"))
    return 512


def infer_bptt_chunks(run_version: str) -> int:
    """Infer the BPTT chunk count from a run-version suffix.

    :param str run_version: Run-version suffix.
    :return int: Inferred BPTT chunk count.
    """
    return 2 if "bptt2" in str(run_version) else 1


def infer_trigram_top_k(run_version: str) -> int:
    """Infer the trigram top-K contract from a run-version suffix.

    :param str run_version: Run-version suffix.
    :return int: Inferred trigram top-K.
    """
    run_version = str(run_version)
    return 4 if ("k4" in run_version or "_seq" in run_version) else SCREEN_TRIGRAM_TOP_K


def infer_planned_steps(run_version: str) -> int:
    """Infer planned steps from a run-version suffix.

    :param str run_version: Run-version suffix.
    :return int: Inferred planned steps.
    """
    run_version = str(run_version)
    if "confirm" in run_version or "1b" in run_version:
        return DECISION_PLANNED_STEPS
    return SCREEN_PLANNED_STEPS


def screen_contract(args: argparse.Namespace) -> ScreenContract:
    """Resolve the active screen contract from flags and run-version hints.

    :param argparse.Namespace args: Parsed arguments.
    :return ScreenContract: Expected screen contract.
    """
    seq_len = int(args.seq_len or infer_seq_len(str(args.run_version)))
    bptt_chunks = int(args.screen_bptt_chunks or infer_bptt_chunks(str(args.run_version)))
    batch_size = args.screen_batch_size
    if batch_size is None:
        denom = max(1, seq_len * bptt_chunks)
        batch_size = max(1, int(args.effective_step_tokens) // denom)
    top_k = int(args.screen_trigram_top_k or infer_trigram_top_k(str(args.run_version)))
    return ScreenContract(
        planned_steps=int(args.screen_steps or infer_planned_steps(str(args.run_version))),
        effective_step_tokens=int(args.effective_step_tokens),
        num_blocks=SCREEN_NUM_BLOCKS,
        trigram_top_k=top_k,
        seq_len=seq_len,
        batch_size=int(batch_size),
        bptt_chunks=bptt_chunks,
    )


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
        hydrate_partial_metrics(row)
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
    if not row:
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
        training = resolved.get("training", {})
        core_dim = int(model["core_dim"])
        core_layers = int(model["core_layers"])
        core_inner_dim = int(core_dim * float(model["core_expansion"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return

    def set_if_empty(field: str, value: Any) -> None:
        """Set a summary field only when it is absent or blank.

        :param str field: Summary field name.
        :param Any value: Value to stringify into the row.
        """
        row.setdefault(field, "")
        if row[field] == "" and value is not None:
            row[field] = str(value)

    set_if_empty("core_dim", core_dim)
    set_if_empty("core_layers", core_layers)
    set_if_empty("core_inner_dim", core_inner_dim)
    set_if_empty("recurrent_cells", core_layers * core_inner_dim)
    set_if_empty("num_blocks", model.get("num_blocks"))
    set_if_empty("trigram_top_k", model.get("trigram_top_k"))
    set_if_empty("seq_len", training.get("seq_len"))
    set_if_empty("batch_size", training.get("batch_size"))
    set_if_empty("bptt_chunks", training.get("bptt_chunks"))
    set_if_empty("effective_step_tokens", training.get("effective_step_tokens"))
    set_if_empty("planned_steps", training.get("num_steps"))
    set_if_empty("learning_rate", training.get("learning_rate"))
    set_if_empty("lr_hold_steps", training.get("lr_hold_steps"))


def hydrate_partial_metrics(row: dict[str, str]) -> None:
    """Fill partial/spec-only rows with last observed metrics when available.

    :param dict[str, str] row: Summary row to update in place.
    """
    if not row or row.get("status") == "completed":
        return
    run_dir_raw = row.get("run_dir")
    if not run_dir_raw:
        return
    metrics_path = Path(run_dir_raw) / "metrics.jsonl"
    if not metrics_path.exists():
        return
    best_bpb: Optional[float] = None
    best_step: Optional[int] = None
    last_eval: dict[str, Any] = {}
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("kind") != "eval":
                    continue
                bpb = as_float(payload.get("val_bpb"))
                step = as_int(payload.get("step"))
                last_eval = payload
                if bpb is not None and (best_bpb is None or bpb < best_bpb):
                    best_bpb = bpb
                    best_step = step
    except (OSError, json.JSONDecodeError):
        return
    if not last_eval:
        return
    if row.get("status") == "spec_only":
        row["status"] = "partial"
    if best_bpb is not None and not row.get("best_val_bpb"):
        row["best_val_bpb"] = str(best_bpb)
    if best_step is not None and not row.get("best_step"):
        row["best_step"] = str(best_step)
    if not row.get("last_val_bpb") and last_eval.get("val_bpb") is not None:
        row["last_val_bpb"] = str(last_eval["val_bpb"])
    metric_map = {
        "last_eval_step": "step",
        "processed_tokens": "processed_tokens",
        "last_eval_tokens": "eval_tokens",
        "last_eval_bytes": "eval_bytes",
        "last_eval_coverage_denominator_tokens": "eval_coverage_denominator_tokens",
        "last_eval_coverage_frac": "eval_coverage_frac",
        "last_eval_full_coverage": "eval_full_coverage",
    }
    for row_key, metric_key in metric_map.items():
        if not row.get(row_key) and last_eval.get(metric_key) is not None:
            row[row_key] = str(last_eval[metric_key])


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


def full_eval_accounting_errors(row: dict[str, str]) -> tuple[str, ...]:
    """Return stale or missing full-validation accounting errors.

    :param dict[str, str] row: Summary row.
    :return tuple[str, ...]: Full-validation accounting errors.
    """
    if str(row.get("last_eval_full_coverage", "")).strip().lower() not in {"1", "true", "yes"}:
        return ()
    errors: list[str] = []
    eval_tokens = as_int(row.get("last_eval_tokens"))
    denom_tokens = as_int(row.get("last_eval_coverage_denominator_tokens"))
    if eval_tokens is None or denom_tokens is None:
        errors.append("missing_full_val_target_accounting")
    elif eval_tokens != denom_tokens:
        errors.append("stale_full_val_target_mismatch")
    if str(row.get("exact_val_bpb", "")).strip().lower() in {"1", "true", "yes"}:
        positive = as_int(row.get("exact_bpb_positive_target_count"))
        zero = as_int(row.get("exact_bpb_zero_byte_target_count"))
        if positive is None or zero is None:
            errors.append("missing_exact_bpb_target_accounting")
        elif denom_tokens is not None and positive + zero != denom_tokens:
            errors.append("exact_bpb_target_count_mismatch")
    return tuple(errors)


def eligibility_errors(
    geometry: Geometry,
    row: dict[str, str],
    contract: Optional[ScreenContract] = None,
) -> tuple[str, ...]:
    """Return protocol violations that make a row decision-ineligible.

    :param Geometry geometry: Geometry parsed from the label.
    :param dict[str, str] row: Summary row.
    :param Optional[ScreenContract] contract: Expected screen contract.
    :return tuple[str, ...]: Human-readable validation errors.
    """
    if not row:
        return ("missing_summary",)

    contract = contract or ScreenContract(
        planned_steps=SCREEN_PLANNED_STEPS,
        effective_step_tokens=SCREEN_EFFECTIVE_STEP_TOKENS,
        num_blocks=SCREEN_NUM_BLOCKS,
        trigram_top_k=SCREEN_TRIGRAM_TOP_K,
        seq_len=512,
        batch_size=256,
        bptt_chunks=1,
    )
    errors: list[str] = []
    if row.get("status") != "completed":
        errors.append("not_completed")
    expected = {
        "planned_steps": contract.planned_steps,
        "effective_step_tokens": contract.effective_step_tokens,
        "num_blocks": contract.num_blocks,
        "trigram_top_k": contract.trigram_top_k,
        "seq_len": contract.seq_len,
        "batch_size": contract.batch_size,
        "bptt_chunks": contract.bptt_chunks,
        "core_dim": geometry.core_dim,
        "core_layers": geometry.layers,
        "core_inner_dim": geometry.inner_dim,
        "recurrent_cells": geometry.recurrent_cells,
    }
    for field, expected_value in expected.items():
        actual = as_int(row.get(field))
        if actual != expected_value:
            errors.append(f"{field}!={expected_value}")
    errors.extend(full_eval_accounting_errors(row))
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


def decision(
    delta_bpb: Optional[float],
    ratio: Optional[float],
    *,
    valid_screen_row: bool,
    decision_grade: bool = True,
) -> str:
    """Return the promotion decision for one row.

    :param Optional[float] delta_bpb: Geometry BPB minus baseline BPB.
    :param Optional[float] ratio: Speed ratio versus baseline.
    :param bool valid_screen_row: Whether the row is decision-eligible.
    :param bool decision_grade: Whether non-promotion verdicts may be final.
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
    if not decision_grade:
        return "probe_only"
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


def confirmation_train_label(contract: ScreenContract) -> str:
    """Return the confirmation train-label for a screen contract.

    :param ScreenContract contract: Expected screen contract.
    :return str: Train-label suffix.
    """
    parts = ["1b"]
    if contract.seq_len != 512:
        parts.append(f"seq{contract.seq_len}")
    if contract.bptt_chunks != 1:
        parts.append(f"bptt{contract.bptt_chunks}")
    if contract.trigram_top_k != 2:
        parts.append(f"k{contract.trigram_top_k}")
    return "_".join(parts)


def confirmation_command(
    geometry: Geometry,
    *,
    run_version: str,
    seed: str,
    contract: Optional[ScreenContract] = None,
) -> str:
    """Build the next confirmation command for a promoted row.

    :param Geometry geometry: Geometry to confirm.
    :param str run_version: Source run version.
    :param str seed: Seed string.
    :param Optional[ScreenContract] contract: Screen contract to preserve.
    :return str: Shell command.
    """
    contract = contract or ScreenContract(
        planned_steps=SCREEN_PLANNED_STEPS,
        effective_step_tokens=SCREEN_EFFECTIVE_STEP_TOKENS,
        num_blocks=SCREEN_NUM_BLOCKS,
        trigram_top_k=SCREEN_TRIGRAM_TOP_K,
        seq_len=512,
        batch_size=256,
        bptt_chunks=1,
    )
    parts = [
        "bash scripts/run_5090_trigram_aligned_geometry_screen.sh",
        f"--run-version {run_version}_confirm",
        f"--seeds {seed}",
        f"--geometry-label {geometry.label}",
        f"--geometry-core-dim {geometry.core_dim}",
        f"--geometry-core-layers {geometry.layers}",
        f"--geometry-core-inner-dim {geometry.inner_dim}",
        "--num-steps 8192",
        "--lr-hold-steps 7000",
        f"--trigram-top-k {contract.trigram_top_k}",
        f"--geometry-seq-len {contract.seq_len}",
        f"--geometry-batch-size {contract.batch_size}",
        f"--geometry-bptt-chunks {contract.bptt_chunks}",
        f"--target-effective-step-tokens {contract.effective_step_tokens}",
        f"--geometry-train-label {confirmation_train_label(contract)}",
        "--full-val-final",
        "--val-every 512",
        "--log-every 128",
        "--log-state-every 512",
        "--save-every 4096",
    ]
    return " ".join(parts)


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


def format_eval_targets(row: dict[str, str]) -> str:
    """Format sampled/full eval target counts for tables.

    :param dict[str, str] row: Summary row.
    :return str: Eval target count summary.
    """
    targets = as_int(row.get("last_eval_tokens"))
    denom = as_int(row.get("last_eval_coverage_denominator_tokens"))
    full = str(row.get("last_eval_full_coverage", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if targets is None:
        return ""
    if full and denom is not None:
        return f"{targets}/{denom}"
    return str(targets)


def main() -> None:
    """Analyze completed or pending geometry rows."""
    args = parse_args()
    repo_root = args.repo_root.resolve()
    contract = screen_contract(args)
    decision_grade = contract.planned_steps >= DECISION_PLANNED_STEPS
    labels = tuple(args.label or DEFAULT_LABELS)
    geometries = [parse_geometry(label) for label in labels]
    benchmark = load_benchmark(args.benchmark)
    baseline_bench = as_float(benchmark.get("current_d48_l12_i480", {}).get("tokens_per_sec"))

    rows: list[AnalyzedRow] = []
    for geometry in geometries:
        summary = load_summary_row(
            repo_root, geometry, run_version=args.run_version, seed=args.seed
        )
        errors = eligibility_errors(geometry, summary, contract)
        valid_screen_row = not errors
        bpb = screen_bpb(summary)
        delta = None if bpb is None else bpb - args.baseline_bpb
        ratio = speed_ratio(
            summary,
            benchmark.get(geometry.benchmark_name, {}),
            baseline_tok_s=args.baseline_tok_s,
            baseline_benchmark_tok_s=baseline_bench,
        )
        verdict = decision(
            delta,
            ratio,
            valid_screen_row=valid_screen_row,
            decision_grade=decision_grade,
        )
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
    print(
        "- screen_contract: "
        f"`steps={contract.planned_steps} eff_tokens={contract.effective_step_tokens} "
        f"k={contract.trigram_top_k} seq={contract.seq_len} "
        f"batch={contract.batch_size} bptt={contract.bptt_chunks}`"
    )
    print(f"- evidence_tier: `{'decision' if decision_grade else 'probe'}`")
    if baseline_bench is not None:
        print(f"- benchmark_baseline: `current_d48_l12_i480` `{baseline_bench}` tok/s")
    if args.benchmark:
        print(f"- benchmark: `{args.benchmark}`")
    print()
    print(
        "| geometry | k | seq | bptt | batch | steps | eval targets | full val | status | eval bpb | delta | speed ratio | est time-matched steps | decision | notes |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |"
    )
    for row in rows:
        geometry = row.geometry
        status = row.summary.get("status", "missing")
        bpb = screen_bpb(row.summary)
        notes = ",".join(row.eligibility_errors)
        full_val = (
            str(row.summary.get("last_eval_full_coverage", "") if row.summary else "")
            .strip()
            .lower()
        )
        full_val = "yes" if full_val in {"1", "true", "yes"} else "no"
        print(
            "| {label} | {top_k} | {seq_len} | {bptt} | {batch} | {planned_steps} | {eval_targets} | {full_val} | {status} | {bpb} | {delta} | {ratio} | {steps} | {verdict} | {notes} |".format(
                label=geometry.label,
                top_k=row.summary.get("trigram_top_k", ""),
                seq_len=row.summary.get("seq_len", ""),
                bptt=row.summary.get("bptt_chunks", ""),
                batch=row.summary.get("batch_size", ""),
                planned_steps=row.summary.get("planned_steps", ""),
                eval_targets=format_eval_targets(row.summary),
                full_val=full_val,
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
            print(
                confirmation_command(
                    geometry,
                    run_version=args.run_version,
                    seed=args.seed,
                    contract=contract,
                )
            )
            print()
    else:
        print("No geometry row currently clears the automatic promotion thresholds.")
        print()
        print("For adaptive follow-ups, rerun the planner for the selected stage:")
        print(
            "python tools/plan_5090_adaptive_closeout.py "
            "--stage k4 --run-version geom1 --seed 1337 --emit markdown"
        )


if __name__ == "__main__":
    main()
