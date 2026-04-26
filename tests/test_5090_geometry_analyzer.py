"""Focused tests for 5090 geometry summary and analyzer protocol gates."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from core_amplifier_lm.experiment import SUMMARY_FIELDS, summarize_run_dir
from tools.analyze_5090_geometry_frontier import (
    SCREEN_EFFECTIVE_STEP_TOKENS,
    SCREEN_NUM_BLOCKS,
    SCREEN_PLANNED_STEPS,
    SCREEN_TRIGRAM_TOP_K,
    decision,
    eligibility_errors,
    estimated_time_matched_steps,
    parse_geometry,
    speed_ratio,
)

PKG_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_row(label: str, **overrides: str) -> dict[str, str]:
    geometry = parse_geometry(label)
    row = {
        "run_name": label,
        "status": "completed",
        "planned_steps": str(SCREEN_PLANNED_STEPS),
        "effective_step_tokens": str(SCREEN_EFFECTIVE_STEP_TOKENS),
        "num_blocks": str(SCREEN_NUM_BLOCKS),
        "trigram_top_k": str(SCREEN_TRIGRAM_TOP_K),
        "core_dim": str(geometry.core_dim),
        "core_layers": str(geometry.layers),
        "core_inner_dim": str(geometry.inner_dim),
        "recurrent_cells": str(geometry.recurrent_cells),
        "last_val_bpb": "2.06",
        "steady_state_tokens_per_sec": "900000",
    }
    row.update(overrides)
    return row


def _write_summary(repo_root: Path, label: str, row: dict[str, str]) -> None:
    summary_dir = (
        repo_root / "experiments" / "5090_architecture" / f"{label}_trigram_seed1337_geom1"
    )
    summary_dir.mkdir(parents=True)
    with (summary_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def test_summarize_run_dir_adds_resolved_geometry_fields(tmp_path: Path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    _write_json(
        run_dir / "resolved_config.json",
        {
            "model": {
                "core_dim": 64,
                "core_layers": 10,
                "core_expansion": 8.0,
                "branch_lags": [1, 2],
            },
            "runtime": {"compile": {"enabled": False}},
            "training": {"num_steps": SCREEN_PLANNED_STEPS},
        },
    )
    _write_json(run_dir / "run_results.json", {"completed": True})

    row = summarize_run_dir(run_dir)

    assert row["core_dim"] == "64"
    assert row["core_inner_dim"] == "512"
    assert row["recurrent_cells"] == "5120"


def test_invalid_or_incomplete_rows_are_pending_even_with_good_bpb():
    geometry = parse_geometry("blocks0_d96_l6_i512")
    row = _summary_row(geometry.label, status="partial", last_val_bpb="2.0")
    errors = eligibility_errors(geometry, row)

    assert "not_completed" in errors
    assert decision(-0.1, 2.5, valid_screen_row=not errors) == "pending"
    assert estimated_time_matched_steps(row, 2.5, valid_screen_row=not errors) is None


def test_geometry_consistency_is_required_for_decisions():
    geometry = parse_geometry("blocks0_d96_l6_i512")
    row = _summary_row(geometry.label, core_inner_dim="480", recurrent_cells="2880")

    errors = eligibility_errors(geometry, row)

    assert "core_inner_dim!=512" in errors
    assert "recurrent_cells!=3072" in errors
    assert decision(-0.1, 2.5, valid_screen_row=not errors) == "pending"


def test_benchmark_ratio_prefers_current_d48_l12_i480_and_rounds_steps():
    row = _summary_row("blocks0_d96_l6_i512", steady_state_tokens_per_sec="100")
    ratio = speed_ratio(
        row,
        {"tokens_per_sec": 246.0},
        baseline_tok_s=50.0,
        baseline_benchmark_tok_s=100.0,
    )

    assert ratio == 2.46
    assert estimated_time_matched_steps(row, ratio, valid_screen_row=True) == 10112


def test_cli_prints_confirmation_only_for_valid_completed_screen_rows(tmp_path: Path):
    valid_label = "blocks0_d96_l6_i512"
    invalid_label = "blocks0_d64_l10_i512"
    valid_run_dir = tmp_path / "runs" / valid_label
    valid_run_dir.mkdir(parents=True)
    _write_json(
        valid_run_dir / "resolved_config.json",
        {
            "model": {
                "core_dim": 96,
                "core_layers": 6,
                "core_expansion": 5.333333333333333,
            }
        },
    )
    _write_summary(
        tmp_path,
        valid_label,
        _summary_row(
            valid_label,
            last_val_bpb="2.0",
            core_dim="",
            core_inner_dim="",
            recurrent_cells="",
            run_dir=str(valid_run_dir),
        ),
    )
    _write_summary(
        tmp_path,
        invalid_label,
        _summary_row(invalid_label, status="partial", last_val_bpb="2.0"),
    )
    benchmark_path = tmp_path / "benchmark.json"
    _write_json(
        benchmark_path,
        [
            {"name": "current_d48_l12_i480", "tokens_per_sec": 100.0},
            {"name": "d96_l6_i512", "tokens_per_sec": 246.0},
            {"name": "d64_l10_i512", "tokens_per_sec": 300.0},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PKG_ROOT / "tools" / "analyze_5090_geometry_frontier.py"),
            "--repo-root",
            str(tmp_path),
            "--benchmark",
            str(benchmark_path),
            "--label",
            valid_label,
            "--label",
            invalid_label,
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "benchmark_baseline: `current_d48_l12_i480` `100.0` tok/s" in result.stdout
    assert f"# {valid_label}: promote_1b" in result.stdout
    assert f"# {invalid_label}:" not in result.stdout
    assert (
        "| blocks0_d96_l6_i512 | 3072 | completed | 2.0000000000 | -0.075172 | 2.460 | 10112 | promote_1b |  |"
        in result.stdout
    )
    assert "not_completed" in result.stdout
