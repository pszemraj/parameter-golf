#!/usr/bin/env python3
"""Rebuild a sweep summary TSV from completed run directories.

This is useful when a long sweep gets interrupted or when a launcher appended
partial / duplicate rows while resuming. The script scans run directories under
`--root`, reads `config.json`, `metrics.jsonl`, and optional `spec.pt`, and
writes a deterministic TSV.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _float_or_blank(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(float(x))
    except Exception:
        return ""


def _int_or_blank(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(int(x))
    except Exception:
        return ""


def _status(run_dir: Path) -> str:
    metrics = run_dir / "metrics.jsonl"
    spec = run_dir / "spec.pt"
    if metrics.exists() and metrics.stat().st_size > 0:
        return "done"
    if spec.exists():
        return "spec_only"
    return "missing"


def _run_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if run_dir.name.startswith("_") or run_dir.name == "data":
            continue
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue

        cfg = _read_json(cfg_path)
        model = cfg.get("model", {})
        training = cfg.get("training", {})
        metrics_path = run_dir / "metrics.jsonl"
        metrics = _read_metrics(metrics_path)
        best: dict[str, Any] = {}
        last: dict[str, Any] = {}
        if metrics:

            def key(row: dict[str, Any]) -> float:
                try:
                    return float(row.get("val_loss", math.inf))
                except Exception:
                    return math.inf

            best = min(metrics, key=key)
            last = metrics[-1]

        spec_path = run_dir / "spec.pt"
        spec_bytes = ""
        gzip_spec_bytes = ""
        if spec_path.exists():
            data = spec_path.read_bytes()
            spec_bytes = str(spec_path.stat().st_size)
            gzip_spec_bytes = str(len(gzip.compress(data, compresslevel=9)))

        rows.append(
            {
                "run_name": run_dir.name,
                "status": _status(run_dir),
                "core_layers": _int_or_blank(model.get("core_layers")),
                "core_expansion": _float_or_blank(model.get("core_expansion")),
                "carry_chunks": _int_or_blank(training.get("carry_chunks")),
                "bptt_chunks": _int_or_blank(training.get("bptt_chunks")),
                "residual_core": str(model.get("residual_core", "")),
                "residual_core_init": _float_or_blank(model.get("residual_core_init")),
                "branch_lags": ",".join(str(x) for x in model.get("branch_lags", [])),
                "num_blocks": _int_or_blank(model.get("num_blocks")),
                "readout_rank": ""
                if model.get("readout_rank") is None
                else _int_or_blank(model.get("readout_rank")),
                "best_step": _int_or_blank(best.get("step")),
                "best_val_loss": _float_or_blank(best.get("val_loss")),
                "best_val_bpb": _float_or_blank(best.get("val_bpb")),
                "last_step": _int_or_blank(last.get("step")),
                "last_val_loss": _float_or_blank(last.get("val_loss")),
                "last_val_bpb": _float_or_blank(last.get("val_bpb")),
                "spec_bytes": spec_bytes,
                "gzip_spec_bytes": gzip_spec_bytes,
                "run_dir": str(run_dir),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Directory containing per-run subdirectories")
    ap.add_argument(
        "--out", type=str, default=None, help="Output TSV path (default: <root>/summary.tsv)"
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else (root / "summary.tsv")
    rows = _run_rows(root)
    header = [
        "run_name",
        "status",
        "core_layers",
        "core_expansion",
        "carry_chunks",
        "bptt_chunks",
        "residual_core",
        "residual_core_init",
        "branch_lags",
        "num_blocks",
        "readout_rank",
        "best_step",
        "best_val_loss",
        "best_val_bpb",
        "last_step",
        "last_val_loss",
        "last_val_bpb",
        "spec_bytes",
        "gzip_spec_bytes",
        "run_dir",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row.get(k, "") for k in header) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
