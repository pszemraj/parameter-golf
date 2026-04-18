"""Export selected HGDN W&B runs into structured JSON/CSV tables."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb

from _repo_bootstrap import ensure_repo_root_on_sys_path

REPO_ROOT = ensure_repo_root_on_sys_path()

from hgdn_wandb_utils import (  # noqa: E402
    DEFAULT_PROJECT,
    flatten_config,
    matches,
)
from profiler_report import write_json, write_rows_csv  # noqa: E402


DEFAULT_HISTORY_KEYS = [
    "_step",
    "train/loss",
    "train/step_ms",
    "train/tokens_per_s",
    "eval/loss",
    "eval/bpb",
]

DEFAULT_SUMMARY_KEYS = [
    "perf_step_ms_final",
    "perf_tokens_per_s_final",
    "roundtrip_val_loss_final",
    "roundtrip_val_bpb_final",
    "artifact_bytes_final",
    "artifact_headroom_bytes_final",
    "artifact_status_final",
    "artifact/quant_baseline_tensor_bytes_final",
    "artifact/quant_int8_payload_bytes_final",
    "artifact/quant_raw_torch_bytes_final",
    "artifact/int8_payload_zlib_bytes_final",
    "artifact/code_bytes_final",
    "system/peak_mem_alloc_mib",
    "system/peak_mem_reserved_mib",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for W&B run export.

    :return argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export selected W&B runs for HGDN analysis.",
    )
    parser.add_argument("--entity", default="pszemraj")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="Exact run name to export. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Substring filter for run names. Repeat for multiple filters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to wandb_exports/<timestamp>.",
    )
    parser.add_argument(
        "--max-history-rows",
        type=int,
        default=0,
        help="Optional cap per run when exporting history. 0 means no cap.",
    )
    parser.add_argument(
        "--history-key",
        action="append",
        default=[],
        help="Additional history key to export. Repeat as needed.",
    )
    parser.add_argument(
        "--summary-key",
        action="append",
        default=[],
        help="Additional summary key to export. Repeat as needed.",
    )
    return parser.parse_args()


def iter_history_rows(
    run: Any,
    keys: list[str],
    max_rows: int,
) -> Iterable[dict[str, Any]]:
    """Yield selected history rows from a W&B run.

    :param Any run: W&B run object.
    :param list[str] keys: History keys to request.
    :param int max_rows: Optional row cap. ``0`` means unlimited.
    :return Iterable[dict[str, Any]]: Stream of selected history rows.
    """
    count = 0
    for row in run.scan_history(keys=keys, page_size=500):
        yield row
        count += 1
        if max_rows and count >= max_rows:
            break


def main() -> int:
    """Export summary and history tables for selected runs.

    :return int: Process exit code.
    """
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("wandb_exports") / f"hgdn_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    exact_names = set(args.name)
    substrings = args.contains
    history_keys = list(dict.fromkeys(DEFAULT_HISTORY_KEYS + args.history_key))
    summary_keys = list(dict.fromkeys(DEFAULT_SUMMARY_KEYS + args.summary_key))

    selected_runs = [run for run in runs if matches(run.name, exact_names, substrings)]
    if not selected_runs:
        raise SystemExit("No matching runs found.")

    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for run in selected_runs:
        summary = run.summary._json_dict
        config = flatten_config(run.config)
        summary_row: dict[str, Any] = {
            "run_id": run.id,
            "run_name": run.name,
            "run_url": run.url,
            "state": run.state,
            "created_at": getattr(run, "created_at", None),
        }
        for key in summary_keys:
            summary_row[key] = summary.get(key)
        for key, value in config.items():
            summary_row[f"config/{key}"] = value
        summary_rows.append(summary_row)

        history_count = 0
        for row in iter_history_rows(run, history_keys, args.max_history_rows):
            history_count += 1
            history_rows.append({"run_id": run.id, "run_name": run.name, **row})

        manifest.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "run_url": run.url,
                "state": run.state,
                "history_rows": history_count,
            }
        )

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary_rows)
    write_rows_csv(output_dir / "summary.csv", summary_rows)
    write_rows_csv(output_dir / "history.csv", history_rows)

    print(f"exported_runs:{len(selected_runs)} output_dir:{output_dir}")
    print(f"summary_csv:{output_dir / 'summary.csv'}")
    print(f"history_csv:{output_dir / 'history.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
