#!/usr/bin/env python3
"""Analyze HGDN experiment bundles and write promotion decisions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from pathlib import Path
from typing import Any

from _repo_bootstrap import ensure_repo_root_on_sys_path

REPO_ROOT = ensure_repo_root_on_sys_path()

NUMBER = r"[-+0-9.eE]+"
STEP_RE = re.compile(
    rf"step:(?P<step>\d+)/(?P<planned>\d+) "
    rf"train_loss:(?P<loss>{NUMBER}) train_time:(?P<time_ms>{NUMBER})ms "
    rf"step_avg:(?P<step_ms>{NUMBER})ms"
)
EVAL_RE = re.compile(
    rf"step:(?P<step>\d+)/(?P<planned>\d+) "
    rf"val_loss:(?P<loss>{NUMBER}) val_bpb:(?P<bpb>{NUMBER}) "
    rf"train_time:(?P<time_ms>{NUMBER})ms"
)
MODEL_RE = re.compile(r"model_params:(?P<params>\d+) blocks:(?P<blocks>\d+G\+\d+A)")
MODE_RE = re.compile(r"gdn_fla_recurrence_mode:(?P<mode>[a-z_]+)")
PERF_RE = re.compile(
    rf"perf_summary ignore_steps:(?P<ignore>\d+) measured_steps:(?P<measured>\d+) "
    rf"step_ms:(?P<step_ms>{NUMBER}) tokens_per_s:(?P<tokens_per_s>{NUMBER})"
)
ROUNDTRIP_RE = re.compile(
    rf"final_int8_zlib_roundtrip_exact val_loss:(?P<loss>{NUMBER}) "
    rf"val_bpb:(?P<bpb>{NUMBER})"
)
ARTIFACT_RE = re.compile(
    r"artifact_status:(?P<status>[A-Z_]+) "
    r"artifact_warning:(?P<warning>\S+) headroom_bytes:(?P<headroom>-?\d+)"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--decision-env", type=Path)
    parser.add_argument(
        "--select",
        choices=["none", "mode", "config"],
        default="none",
        help="Promotion decision to write when --decision-env is set.",
    )
    parser.add_argument(
        "--metric",
        choices=[
            "auto",
            "final_step_ms",
            "speed_budget_bpb",
            "final_sampled_bpb",
            "final_roundtrip_bpb",
        ],
        default="auto",
        help="Ranking metric. auto prefers speed_budget_bpb, then final_sampled_bpb.",
    )
    parser.add_argument("--speed-budget-ms", type=float, default=None)
    parser.add_argument("--confirm-top-n", type=int, default=2)
    parser.add_argument(
        "--include-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include matched attention-only baseline controls in confirm configs.",
    )
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def parse_float(value: str) -> float | None:
    """Parse a float, returning None for invalid values.

    :param str value: Float-like text.
    :return float | None: Parsed float or None.
    """
    try:
        return float(value)
    except ValueError:
        return None


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file.

    :param Path path: JSON path.
    :return dict[str, Any]: Parsed object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def load_size_rows(bundle_dir: Path) -> dict[str, dict[str, str]]:
    """Load size-screen rows keyed by candidate name.

    :param Path bundle_dir: Bundle directory.
    :return dict[str, dict[str, str]]: Size rows.
    """
    path = bundle_dir / "size_screen" / "rows.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["name"]: row for row in csv.DictReader(fh)}


def strip_config_prefix(config: str) -> str:
    """Return the short candidate name for a config path.

    :param str config: Config path.
    :return str: Short candidate name.
    """
    stem = Path(config).stem
    return stem.removeprefix("naive_contract_")


def infer_family(config: str, blocks: str, trainer: str | None = None) -> str:
    """Infer experiment family.

    :param str config: Config path.
    :param str blocks: Parsed block summary.
    :param str | None trainer: Trainer name from manifest.
    :return str: Family label.
    """
    if trainer == "train_gpt.py":
        return "exact-baseline"
    if blocks.startswith("0G+"):
        return "attention-only"
    if "_r0_" in Path(config).stem:
        return "attention-only"
    return "HGDN"


def parse_log(path: Path) -> dict[str, Any]:
    """Parse one trainer log.

    :param Path path: Log path.
    :return dict[str, Any]: Parsed metrics.
    """
    train_points: list[dict[str, float]] = []
    eval_points: list[dict[str, float]] = []
    params: int | None = None
    blocks = ""
    mode = ""
    perf_step_ms: float | None = None
    perf_tokens_per_s: float | None = None
    final_roundtrip_bpb: float | None = None
    final_roundtrip_loss: float | None = None
    artifact_status = ""
    artifact_warning = ""
    artifact_headroom_bytes: int | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := STEP_RE.search(line):
            train_points.append(
                {
                    "step": float(match.group("step")),
                    "planned": float(match.group("planned")),
                    "train_loss": float(match.group("loss")),
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
        if match := MODE_RE.search(line):
            mode = match.group("mode")
        if match := PERF_RE.search(line):
            perf_step_ms = float(match.group("step_ms"))
            perf_tokens_per_s = float(match.group("tokens_per_s"))
        if match := ROUNDTRIP_RE.search(line):
            final_roundtrip_loss = float(match.group("loss"))
            final_roundtrip_bpb = float(match.group("bpb"))
        if match := ARTIFACT_RE.search(line):
            artifact_status = match.group("status")
            artifact_warning = match.group("warning")
            artifact_headroom_bytes = int(match.group("headroom"))

    final_train = train_points[-1] if train_points else {}
    final_eval = eval_points[-1] if eval_points else {}
    planned_step = int(final_train.get("planned", final_eval.get("planned", 0)))
    final_step = int(final_train.get("step", final_eval.get("step", 0)))
    final_step_ms = (
        perf_step_ms
        if perf_step_ms is not None
        else final_train.get("step_ms")
        if final_train
        else None
    )
    return {
        "log_path": str(path),
        "run_id": path.stem,
        "params": params,
        "blocks": blocks,
        "completed": bool(planned_step and final_step >= planned_step),
        "final_step": final_step,
        "planned_step": planned_step,
        "final_train_time_ms": final_train.get("train_time_ms"),
        "final_step_ms": final_step_ms,
        "final_train_loss": final_train.get("train_loss"),
        "final_sampled_bpb": final_eval.get("val_bpb"),
        "final_sampled_loss": final_eval.get("val_loss"),
        "final_roundtrip_bpb": final_roundtrip_bpb,
        "final_roundtrip_loss": final_roundtrip_loss,
        "perf_tokens_per_s": perf_tokens_per_s,
        "artifact_status": artifact_status,
        "artifact_warning": artifact_warning,
        "artifact_headroom_bytes": artifact_headroom_bytes,
        "log_recurrence_mode": mode,
        "eval_points": eval_points,
    }


def interpolate_bpb(
    eval_points: list[dict[str, float]], budget_ms: float
) -> float | None:
    """Estimate BPB at a common train-time budget.

    :param list[dict[str, float]] eval_points: Validation points.
    :param float budget_ms: Common training time in milliseconds.
    :return float | None: Interpolated BPB.
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


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return run entries from any supported manifest shape.

    :param dict[str, Any] manifest: Bundle manifest.
    :return list[dict[str, Any]]: Entries.
    """
    if isinstance(manifest.get("candidates"), list):
        return list(manifest["candidates"])
    if isinstance(manifest.get("runs"), list):
        return list(manifest["runs"])
    raise KeyError("Manifest must contain either 'candidates' or 'runs'.")


def build_rows(
    bundle_dir: Path, speed_budget_ms: float | None
) -> tuple[list[dict[str, Any]], float | None]:
    """Build analyzer rows.

    :param Path bundle_dir: Unpacked bundle directory.
    :param float | None speed_budget_ms: Optional common time budget.
    :return tuple[list[dict[str, Any]], float | None]: Rows and resolved budget.
    """
    manifest = load_json(bundle_dir / "bundle_manifest.json")
    entries = manifest_entries(manifest)
    size_rows = load_size_rows(bundle_dir)
    parsed_logs = {
        log.stem: parse_log(log) for log in sorted((bundle_dir / "logs").glob("*.txt"))
    }
    contract = manifest.get("contract", {})
    train_batch_tokens = contract.get("train_batch_tokens")
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        run_id = entry["run_id"]
        config = entry.get("config", "")
        parsed = parsed_logs.get(run_id, {"run_id": run_id, "completed": False})
        row = dict(parsed)
        row["manifest_index"] = index
        row["label"] = entry.get("label", "")
        row["trainer"] = entry.get("trainer", "train_gpt_hybrid.py")
        row["config"] = config
        row["candidate"] = strip_config_prefix(config) if config else run_id
        row["gdn_fla_recurrence_mode"] = (
            entry.get("gdn_fla_recurrence_mode")
            or row.get("log_recurrence_mode")
            or contract.get("gdn_fla_recurrence_mode")
            or entry.get("mode", "")
        )
        row["family"] = infer_family(config, row.get("blocks", ""), row["trainer"])
        size_row = size_rows.get(row["candidate"], {})
        row["size_status"] = size_row.get("artifact_status")
        row["size_headroom_bytes"] = (
            int(size_row["headroom_bytes"]) if size_row.get("headroom_bytes") else None
        )
        if row.get("perf_tokens_per_s") is not None:
            row["tokens_per_s"] = row["perf_tokens_per_s"]
        elif train_batch_tokens is not None and row.get("final_step_ms") is not None:
            row["tokens_per_s"] = (
                float(train_batch_tokens) * 1000.0 / float(row["final_step_ms"])
            )
        else:
            row["tokens_per_s"] = None
        rows.append(row)

    complete_times = [
        float(row["final_train_time_ms"])
        for row in rows
        if row.get("completed")
        and row.get("final_train_time_ms") is not None
        and row.get("eval_points")
    ]
    resolved_budget = (
        float(speed_budget_ms or min(complete_times)) if complete_times else None
    )
    for row in rows:
        row["speed_budget_ms"] = resolved_budget
        row["speed_budget_bpb"] = (
            interpolate_bpb(row.get("eval_points", []), resolved_budget)
            if resolved_budget is not None
            else None
        )
    assign_ranks(rows)
    return rows, resolved_budget


def assign_ranks(rows: list[dict[str, Any]]) -> None:
    """Assign dense ranks for common metrics.

    :param list[dict[str, Any]] rows: Rows to mutate.
    """
    for metric, rank_key in (
        ("final_sampled_bpb", "fixed_step_rank_all"),
        ("speed_budget_bpb", "speed_rank_all"),
    ):
        ranked = sorted(
            [row for row in rows if row.get(metric) is not None],
            key=lambda row: (float(row[metric]), row["run_id"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row[rank_key] = rank
        ranked_hgdn = [row for row in ranked if row.get("family") == "HGDN"]
        for rank, row in enumerate(ranked_hgdn, start=1):
            row[rank_key.replace("_all", "_hgdn")] = rank


def metric_value(row: dict[str, Any], metric: str) -> tuple[str, float | None]:
    """Return the selected metric value for one row.

    :param dict[str, Any] row: Analyzer row.
    :param str metric: Metric name or `auto`.
    :return tuple[str, float | None]: Resolved metric name and value.
    """
    if metric == "auto":
        for key in ("speed_budget_bpb", "final_sampled_bpb", "final_roundtrip_bpb"):
            if row.get(key) is not None:
                return key, float(row[key])
        return "auto", None
    value = row.get(metric)
    return metric, float(value) if value is not None else None


def is_legal_size(row: dict[str, Any]) -> bool:
    """Return whether a row is not known to exceed the artifact cap.

    :param dict[str, Any] row: Analyzer row.
    :return bool: True when legal or unknown.
    """
    statuses = [row.get("artifact_status"), row.get("size_status")]
    return "OVER_LIMIT" not in statuses


def eligible_rows(
    rows: list[dict[str, Any]], *, select: str, metric: str
) -> list[dict[str, Any]]:
    """Filter rows eligible for promotion.

    :param list[dict[str, Any]] rows: Analyzer rows.
    :param str select: Selection kind.
    :param str metric: Ranking metric.
    :return list[dict[str, Any]]: Eligible rows.
    """
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if select in {"mode", "config"} and row.get("family") != "HGDN":
            continue
        if not row.get("completed"):
            continue
        if not is_legal_size(row):
            continue
        _metric_name, value = metric_value(row, metric)
        if value is None:
            continue
        eligible.append(row)
    return eligible


def sort_for_promotion(
    rows: list[dict[str, Any]], *, metric: str
) -> list[dict[str, Any]]:
    """Sort rows by promotion score.

    :param list[dict[str, Any]] rows: Eligible rows.
    :param str metric: Ranking metric.
    :return list[dict[str, Any]]: Sorted rows.
    """
    return sorted(
        rows,
        key=lambda row: (
            metric_value(row, metric)[1],
            row.get("final_sampled_bpb")
            if row.get("final_sampled_bpb") is not None
            else 999.0,
            row.get("final_step_ms")
            if row.get("final_step_ms") is not None
            else 999999.0,
            row["run_id"],
        ),
    )


def matched_control_config(config: str) -> str:
    """Return the same-shell attention-only baseline config when it exists.

    :param str config: HGDN config path.
    :return str: Control config path or empty string.
    """
    stem = Path(config).stem
    layer_match = re.match(r"naive_contract_(l\d+)_d512_", stem)
    mult_match = re.search(r"_(m\d+(?:p\d+)?)$", stem)
    if not layer_match or not mult_match:
        return ""
    candidate = (
        REPO_ROOT
        / "configs"
        / "hgdn"
        / f"naive_contract_{layer_match.group(1)}_d512_r0_{mult_match.group(1)}.toml"
    )
    if not candidate.is_file():
        return ""
    return str(candidate.relative_to(REPO_ROOT))


def unique_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate strings while preserving order.

    :param list[str] values: Values.
    :return list[str]: Unique values.
    """
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky fields for serialized outputs.

    :param dict[str, Any] row: Full row.
    :return dict[str, Any]: Compact row.
    """
    return {key: value for key, value in row.items() if key != "eval_points"}


def summary_sort_key(
    row: dict[str, Any], metric: str
) -> tuple[float, float, float, str]:
    """Return a stable sort key for human summaries.

    :param dict[str, Any] row: Analyzer row.
    :param str metric: Primary metric.
    :return tuple[float, float, float, str]: Sort key.
    """
    _metric_name, value = metric_value(row, metric)
    primary = value if value is not None else 999999.0
    sampled_bpb = (
        float(row["final_sampled_bpb"])
        if row.get("final_sampled_bpb") is not None
        else 999.0
    )
    step_ms = (
        float(row["final_step_ms"])
        if row.get("final_step_ms") is not None
        else 999999.0
    )
    return (primary, sampled_bpb, step_ms, row["run_id"])


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    speed_budget_ms: float | None,
    *,
    metric: str,
) -> None:
    """Write rows, CSV, and markdown summary.

    :param Path output_dir: Output directory.
    :param list[dict[str, Any]] rows: Analyzer rows.
    :param float | None speed_budget_ms: Resolved speed budget.
    :param str metric: Primary summary metric.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    compact = [compact_row(row) for row in rows]
    (output_dir / "rows.json").write_text(json.dumps(compact, indent=2) + "\n")
    fields = [
        "manifest_index",
        "candidate",
        "family",
        "gdn_fla_recurrence_mode",
        "completed",
        "final_step",
        "planned_step",
        "final_step_ms",
        "tokens_per_s",
        "final_train_loss",
        "final_sampled_bpb",
        "speed_budget_bpb",
        "final_roundtrip_bpb",
        "fixed_step_rank_all",
        "speed_rank_all",
        "fixed_step_rank_hgdn",
        "speed_rank_hgdn",
        "size_status",
        "artifact_status",
        "artifact_headroom_bytes",
        "config",
        "run_id",
        "label",
    ]
    with (output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(compact)
    lines = [
        "# HGDN Experiment Bundle Analysis",
        "",
        f"Speed budget ms: {speed_budget_ms:.0f}"
        if speed_budget_ms is not None
        else "Speed budget ms: n/a",
        f"Selection metric: {metric}",
        "",
        "| Candidate | Mode | Family | Step | ms/step | toks/s | BPB | Speed BPB | Roundtrip | Size |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: summary_sort_key(item, metric)):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate"]),
                    str(row.get("gdn_fla_recurrence_mode") or ""),
                    str(row.get("family") or ""),
                    f"{row.get('final_step', 0)}/{row.get('planned_step', 0)}",
                    format_optional(row.get("final_step_ms"), digits=2),
                    format_optional(row.get("tokens_per_s"), digits=1),
                    format_optional(row.get("final_sampled_bpb"), digits=4),
                    format_optional(row.get("speed_budget_bpb"), digits=4),
                    format_optional(row.get("final_roundtrip_bpb"), digits=4),
                    str(row.get("artifact_status") or row.get("size_status") or ""),
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def format_optional(value: Any, *, digits: int) -> str:
    """Format an optional numeric value.

    :param Any value: Value.
    :param int digits: Decimal places.
    :return str: Formatted value or empty string.
    """
    return "" if value is None else f"{float(value):.{digits}f}"


def shell_assign(name: str, value: str) -> str:
    """Format one shell assignment.

    :param str name: Variable name.
    :param str value: Variable value.
    :return str: Shell assignment line.
    """
    return f"{name}={shlex.quote(value)}"


def write_decision_env(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    select: str,
    metric: str,
    confirm_top_n: int,
    include_controls: bool,
) -> dict[str, Any]:
    """Write a shell-safe promotion decision file.

    :param Path path: Output env file.
    :param list[dict[str, Any]] rows: Analyzer rows.
    :param str select: Selection kind.
    :param str metric: Ranking metric.
    :param int confirm_top_n: Number of HGDN configs to promote.
    :param bool include_controls: Whether to append matched controls.
    :raises SystemExit: When no eligible row is available for a requested selection.
    :return dict[str, Any]: Decision payload.
    """
    decision: dict[str, Any] = {"select": select}
    ranked = sort_for_promotion(
        eligible_rows(rows, select=select, metric=metric), metric=metric
    )
    if select != "none" and not ranked:
        raise SystemExit(f"No eligible rows found for --select {select}.")

    env_lines = ["# Generated by scripts/analyze_hgdn_experiment_bundle.py"]
    if ranked:
        winner = ranked[0]
        metric_name, metric_score = metric_value(winner, metric)
        top_configs = unique_preserve_order(
            [str(row["config"]) for row in ranked[:confirm_top_n]]
        )
        controls = (
            [matched_control_config(config) for config in top_configs]
            if include_controls
            else []
        )
        selected_control_config = matched_control_config(str(winner.get("config", "")))
        confirm_configs = unique_preserve_order(top_configs + controls)
        decision.update(
            {
                "selected_run_id": winner["run_id"],
                "selected_label": winner.get("label", ""),
                "selected_config": winner.get("config", ""),
                "selected_candidate": winner.get("candidate", ""),
                "selected_gdn_fla_recurrence_mode": winner.get(
                    "gdn_fla_recurrence_mode", ""
                ),
                "selected_control_config": selected_control_config,
                "selected_metric": metric_name,
                "selected_metric_value": metric_score,
                "selected_top_configs": top_configs,
                "selected_confirm_configs": confirm_configs,
            }
        )
        env_lines.extend(
            [
                shell_assign("SELECTED_RUN_ID", str(decision["selected_run_id"])),
                shell_assign("SELECTED_LABEL", str(decision["selected_label"])),
                shell_assign("SELECTED_CONFIG", str(decision["selected_config"])),
                shell_assign("SELECTED_CANDIDATE", str(decision["selected_candidate"])),
                shell_assign(
                    "SELECTED_GDN_FLA_RECURRENCE_MODE",
                    str(decision["selected_gdn_fla_recurrence_mode"]),
                ),
                shell_assign(
                    "SELECTED_CONTROL_CONFIG",
                    str(decision["selected_control_config"]),
                ),
                shell_assign("SELECTED_METRIC", str(decision["selected_metric"])),
                shell_assign(
                    "SELECTED_METRIC_VALUE",
                    ""
                    if decision["selected_metric_value"] is None
                    else str(decision["selected_metric_value"]),
                ),
                shell_assign("SELECTED_TOP_CONFIGS_CSV", ",".join(top_configs)),
                shell_assign("SELECTED_CONFIRM_CONFIGS_CSV", ",".join(confirm_configs)),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(env_lines) + "\n")
    (path.with_suffix(".json")).write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def print_summary(
    rows: list[dict[str, Any]],
    *,
    top: int,
    speed_budget_ms: float | None,
    metric: str,
) -> None:
    """Print a compact analysis table.

    :param list[dict[str, Any]] rows: Analyzer rows.
    :param int top: Row limit.
    :param float | None speed_budget_ms: Resolved speed budget.
    :param str metric: Primary summary metric.
    """
    budget = "n/a" if speed_budget_ms is None else f"{speed_budget_ms:.0f}"
    print(f"speed_budget_ms:{budget}")
    print(f"selection_metric:{metric}")
    print(
        "| Candidate | Mode | Family | Step | ms/step | toks/s | BPB | Speed BPB | Size |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda item: summary_sort_key(item, metric))[:top]:
        print(
            f"| {row['candidate']} | {row.get('gdn_fla_recurrence_mode') or ''} | "
            f"{row.get('family') or ''} | "
            f"{row.get('final_step', 0)}/{row.get('planned_step', 0)} | "
            f"{format_optional(row.get('final_step_ms'), digits=2)} | "
            f"{format_optional(row.get('tokens_per_s'), digits=1)} | "
            f"{format_optional(row.get('final_sampled_bpb'), digits=4)} | "
            f"{format_optional(row.get('speed_budget_bpb'), digits=4)} | "
            f"{row.get('artifact_status') or row.get('size_status') or ''} |"
        )


def main() -> None:
    """Run bundle analysis."""
    args = parse_args()
    rows, speed_budget_ms = build_rows(args.bundle_dir, args.speed_budget_ms)
    if args.output_dir is not None:
        write_outputs(args.output_dir, rows, speed_budget_ms, metric=args.metric)
    if args.decision_env is not None:
        write_decision_env(
            args.decision_env,
            rows=rows,
            select=args.select,
            metric=args.metric,
            confirm_top_n=args.confirm_top_n,
            include_controls=args.include_controls,
        )
    print_summary(
        rows, top=args.top, speed_budget_ms=speed_budget_ms, metric=args.metric
    )


if __name__ == "__main__":
    main()
