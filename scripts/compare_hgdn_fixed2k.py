#!/usr/bin/env python3
"""Compare fixed-step HGDN W&B runs and write a structured bundle.

:param None: This module is intended to be executed as a script.
:return None: Uses ``SystemExit`` from :func:`main`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb

from export_wandb_hgdn_runs import flatten_config, matches
from profiler_report import write_json, write_rows_csv

DEFAULT_EVAL_STEPS = (500, 1000, 1500, 2000)
DEFAULT_HISTORY_KEYS = ("_step", "train/step_ms", "eval/loss", "eval/bpb")
DEFAULT_PROJECT = "pg-hgdn-ablations"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the fixed-step HGDN comparator.

    :return argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare fixed-step HGDN W&B runs into a structured bundle.",
    )
    parser.add_argument("--entity", default="pszemraj")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="Exact run name to include. Repeatable.",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Substring filter for run names. Repeatable.",
    )
    parser.add_argument(
        "--reference",
        help="Exact run name to treat as the comparison reference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to profiles/fixed2k_compare/<timestamp>.",
    )
    parser.add_argument(
        "--eval-step",
        action="append",
        type=int,
        default=[],
        help="Sampled eval step to include. Repeatable. Defaults to 500/1000/1500/2000.",
    )
    return parser.parse_args()


def pick(mapping: dict[str, Any], *keys: str) -> Any:
    """Return the first present non-``None`` value for the provided keys.

    :param dict[str, Any] mapping: Source mapping.
    :param str keys: Candidate lookup keys.
    :return Any: First non-``None`` value, or ``None`` if none are present.
    """
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def coerce_int(value: Any) -> int | None:
    """Convert a config or summary scalar to ``int`` when possible.

    :param Any value: Value to normalize.
    :return int | None: Normalized integer or ``None``.
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> float | None:
    """Convert a config or summary scalar to ``float`` when possible.

    :param Any value: Value to normalize.
    :return float | None: Normalized float or ``None``.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_family(run_name: str, config: dict[str, Any]) -> str:
    """Infer the trainer family for a run.

    :param str run_name: W&B run name.
    :param dict[str, Any] config: Flattened W&B config.
    :return str: Human-readable family label.
    """
    gdn_ratio = coerce_int(pick(config, "GDN_RATIO", "gdn_ratio"))
    if "_depth_" in run_name or gdn_ratio == 0:
        return "attention_only_baseline"
    if "_hybrid_" in run_name or gdn_ratio is not None:
        return "hybrid"
    return "unknown"


def collect_history_metrics(run: Any, eval_steps: tuple[int, ...]) -> dict[str, Any]:
    """Collect sampled eval checkpoints and the last logged step time.

    :param Any run: W&B run object.
    :param tuple[int, ...] eval_steps: Eval checkpoints to retain.
    :return dict[str, Any]: Structured history-derived metrics.
    """
    wanted_steps = set(eval_steps)
    eval_rows: dict[int, dict[str, float | None]] = {}
    last_eval_step: int | None = None
    last_train_step_ms: float | None = None

    for row in run.scan_history(keys=list(DEFAULT_HISTORY_KEYS), page_size=500):
        step = coerce_int(row.get("_step"))
        train_step_ms = coerce_float(row.get("train/step_ms"))
        eval_bpb = coerce_float(row.get("eval/bpb"))
        eval_loss = coerce_float(row.get("eval/loss"))

        if train_step_ms is not None:
            last_train_step_ms = train_step_ms
        if step is None or eval_bpb is None:
            continue
        if step in wanted_steps:
            eval_rows[step] = {"eval_bpb": eval_bpb, "eval_loss": eval_loss}
        if last_eval_step is None or step >= last_eval_step:
            last_eval_step = step

    last_eval_bpb = None
    last_eval_loss = None
    if last_eval_step is not None:
        last_eval = eval_rows.get(last_eval_step)
        if last_eval is not None:
            last_eval_bpb = last_eval["eval_bpb"]
            last_eval_loss = last_eval["eval_loss"]

    return {
        "eval_rows": eval_rows,
        "last_eval_step": last_eval_step,
        "last_eval_bpb": last_eval_bpb,
        "last_eval_loss": last_eval_loss,
        "last_train_step_ms": last_train_step_ms,
    }


def build_run_row(run: Any, eval_steps: tuple[int, ...]) -> dict[str, Any]:
    """Build one normalized comparison row for a W&B run.

    :param Any run: W&B run object.
    :param tuple[int, ...] eval_steps: Eval checkpoints to retain.
    :return dict[str, Any]: Normalized comparison row.
    """
    summary = run.summary._json_dict
    config = flatten_config(run.config)
    history = collect_history_metrics(run, eval_steps)
    row: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url,
        "state": run.state,
        "family": infer_family(run.name, config),
        "num_layers": coerce_int(pick(config, "NUM_LAYERS", "num_layers")),
        "model_dim": coerce_int(pick(config, "MODEL_DIM", "model_dim")),
        "mlp_mult": coerce_float(pick(config, "MLP_MULT", "mlp_mult")),
        "gdn_ratio": coerce_int(pick(config, "GDN_RATIO", "gdn_ratio")),
        "n_params": coerce_int(pick(config, "N_PARAMS", "n_params")),
        "n_gdn_blocks": coerce_int(pick(config, "CONV_BLOCKS", "n_gdn_blocks")),
        "n_attn_blocks": coerce_int(pick(config, "ATTN_BLOCKS", "n_attn_blocks")),
        "gdn_use_packed_qkv_conv": pick(
            config,
            "GDN_USE_PACKED_QKV_CONV",
            "gdn_use_packed_qkv_conv",
        ),
        "gdn_use_packed_qkv_proj": pick(
            config,
            "GDN_USE_PACKED_QKV_PROJ",
            "gdn_use_packed_qkv_proj",
        ),
        "gdn_conv_output_contiguous": pick(
            config,
            "GDN_CONV_OUTPUT_CONTIGUOUS",
            "gdn_conv_output_contiguous",
        ),
        "gdn_control_proj_fp32": pick(
            config,
            "GDN_CONTROL_PROJ_FP32",
            "gdn_control_proj_fp32",
        ),
        "sampled_eval_step_last": history["last_eval_step"],
        "sampled_eval_bpb_last": history["last_eval_bpb"],
        "sampled_eval_loss_last": history["last_eval_loss"],
        "train_step_ms_last": history["last_train_step_ms"],
        "roundtrip_val_bpb_final": coerce_float(summary.get("roundtrip_val_bpb_final")),
        "roundtrip_val_loss_final": coerce_float(
            summary.get("roundtrip_val_loss_final")
        ),
        "artifact_bytes_final": coerce_int(summary.get("artifact_bytes_final")),
        "artifact_headroom_bytes_final": coerce_int(
            summary.get("artifact_headroom_bytes_final")
        ),
        "artifact_status_final": summary.get("artifact_status_final"),
        "artifact_warning_final": summary.get("artifact_warning_final"),
        "artifact_code_bytes_final": coerce_int(
            summary.get("artifact/code_bytes_final")
        ),
        "artifact_int8_payload_zlib_bytes_final": coerce_int(
            summary.get("artifact/int8_payload_zlib_bytes_final")
        ),
        "peak_mem_alloc_mib": coerce_float(summary.get("system/peak_mem_alloc_mib")),
        "peak_mem_reserved_mib": coerce_float(
            summary.get("system/peak_mem_reserved_mib")
        ),
    }
    for step in eval_steps:
        eval_row = history["eval_rows"].get(step, {})
        row[f"eval_bpb_step_{step}"] = eval_row.get("eval_bpb")
        row[f"eval_loss_step_{step}"] = eval_row.get("eval_loss")
    return row


def add_reference_deltas(
    rows: list[dict[str, Any]], reference_name: str | None
) -> dict[str, Any] | None:
    """Add delta columns versus a named reference run.

    :param list[dict[str, Any]] rows: Comparison rows to annotate.
    :param str | None reference_name: Exact reference run name, if any.
    :return dict[str, Any] | None: Reference row when found.
    """
    if reference_name is None:
        return None
    reference = next((row for row in rows if row["run_name"] == reference_name), None)
    if reference is None:
        raise SystemExit(f"Reference run not found in selection: {reference_name!r}")

    delta_fields = (
        "sampled_eval_bpb_last",
        "roundtrip_val_bpb_final",
        "train_step_ms_last",
        "artifact_bytes_final",
        "artifact_headroom_bytes_final",
    )
    for row in rows:
        for field in delta_fields:
            lhs = coerce_float(row.get(field))
            rhs = coerce_float(reference.get(field))
            row[f"delta_{field}"] = None if lhs is None or rhs is None else lhs - rhs
    return reference


def sort_rows(
    rows: list[dict[str, Any]], reference_name: str | None
) -> list[dict[str, Any]]:
    """Sort rows for human-readable output.

    :param list[dict[str, Any]] rows: Rows to sort.
    :param str | None reference_name: Optional reference run name.
    :return list[dict[str, Any]]: Sorted rows.
    """
    reference = None
    others = rows
    if reference_name is not None:
        reference = next(
            (row for row in rows if row["run_name"] == reference_name), None
        )
        others = [row for row in rows if row["run_name"] != reference_name]
    others = sorted(
        others,
        key=lambda row: (
            float("inf")
            if row.get("roundtrip_val_bpb_final") is None
            else row["roundtrip_val_bpb_final"],
            row["run_name"],
        ),
    )
    return ([reference] if reference is not None else []) + others


def format_float(value: Any, decimals: int = 4) -> str:
    """Render a float-like value for markdown output.

    :param Any value: Value to format.
    :param int decimals: Decimal precision.
    :return str: Formatted scalar or ``-``.
    """
    normalized = coerce_float(value)
    if normalized is None:
        return "-"
    return f"{normalized:.{decimals}f}"


def format_int(value: Any) -> str:
    """Render an int-like value for markdown output.

    :param Any value: Value to format.
    :return str: Formatted integer or ``-``.
    """
    normalized = coerce_int(value)
    if normalized is None:
        return "-"
    return f"{normalized:,}"


def render_markdown(
    rows: list[dict[str, Any]],
    reference: dict[str, Any] | None,
    eval_steps: tuple[int, ...],
    output_dir: Path,
) -> str:
    """Render a compact markdown comparison report.

    :param list[dict[str, Any]] rows: Comparison rows.
    :param dict[str, Any] | None reference: Optional reference row.
    :param tuple[int, ...] eval_steps: Eval checkpoints included in the report.
    :param Path output_dir: Output bundle directory.
    :return str: Markdown report body.
    """
    lines = ["# HGDN Fixed2k Comparison", ""]
    lines.append(f"- output_dir: `{output_dir}`")
    if reference is not None:
        lines.append(f"- reference: `{reference['run_name']}`")
    lines.append("")
    header = [
        "run_name",
        "family",
        "layers",
        "dim",
        "mlp",
        "sampled_eval_bpb_last",
        "roundtrip_bpb",
        "step_ms_last",
        "artifact_bytes",
        "headroom_bytes",
        "status",
    ]
    if reference is not None:
        header.extend(
            [
                "delta_roundtrip_bpb",
                "delta_step_ms",
                "delta_artifact_bytes",
            ]
        )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        cells = [
            row["run_name"],
            row["family"],
            format_int(row.get("num_layers")),
            format_int(row.get("model_dim")),
            format_float(row.get("mlp_mult"), decimals=2),
            format_float(row.get("sampled_eval_bpb_last")),
            format_float(row.get("roundtrip_val_bpb_final")),
            format_float(row.get("train_step_ms_last"), decimals=2),
            format_int(row.get("artifact_bytes_final")),
            format_int(row.get("artifact_headroom_bytes_final")),
            str(row.get("artifact_status_final") or "-"),
        ]
        if reference is not None:
            cells.extend(
                [
                    format_float(row.get("delta_roundtrip_val_bpb_final")),
                    format_float(row.get("delta_train_step_ms_last"), decimals=2),
                    format_int(row.get("delta_artifact_bytes_final")),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Eval checkpoints")
    lines.append("")
    eval_header = ["run_name"] + [f"eval_bpb_{step}" for step in eval_steps]
    lines.append("| " + " | ".join(eval_header) + " |")
    lines.append("|" + "|".join(["---"] * len(eval_header)) + "|")
    for row in rows:
        cells = [row["run_name"]]
        cells.extend(
            format_float(row.get(f"eval_bpb_step_{step}")) for step in eval_steps
        )
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def default_output_dir() -> Path:
    """Build the default output directory for this report.

    :return Path: Timestamped output path under `profiles/fixed2k_compare`.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("profiles") / "fixed2k_compare" / f"compare_{timestamp}"


def main() -> int:
    """Run the fixed-step comparison export.

    :return int: Process exit code.
    """
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")
    exact_names = set(args.name)
    substrings = args.contains
    selected = [run for run in runs if matches(run.name, exact_names, substrings)]
    if not selected:
        raise SystemExit("No matching runs found.")

    eval_steps = tuple(args.eval_step) if args.eval_step else DEFAULT_EVAL_STEPS
    rows = [build_run_row(run, eval_steps) for run in selected]
    reference = add_reference_deltas(rows, args.reference)
    rows = sort_rows(rows, args.reference)

    manifest = {
        "entity": args.entity,
        "project": args.project,
        "selected_runs": [row["run_name"] for row in rows],
        "reference_run": None if reference is None else reference["run_name"],
        "eval_steps": list(eval_steps),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "rows.json", rows)
    write_rows_csv(output_dir / "rows.csv", rows)
    markdown = render_markdown(rows, reference, eval_steps, output_dir)
    (output_dir / "comparison.md").write_text(markdown, encoding="utf-8")

    print(f"compared_runs:{len(rows)} output_dir:{output_dir}")
    print(f"comparison_md:{output_dir / 'comparison.md'}")
    print(f"rows_csv:{output_dir / 'rows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
