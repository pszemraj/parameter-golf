"""Tests for adaptive 5090 closeout planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from core_amplifier_lm.experiment import SUMMARY_FIELDS
from tools.plan_5090_adaptive_closeout import (
    BPTT_STEPS,
    CONFIRM_STEPS,
    DEFAULT_BPTT_BATCH_SIZE,
    DEFAULT_BPTT_CHUNKS,
    geometry_command,
    parse_args,
    plan_bptt,
    plan_confirmations,
    plan_k4,
)


def _write_summary(
    repo_root: Path,
    label: str,
    run_version: str,
    *,
    run_name: str | None = None,
    seed: str = "1337",
    status: str = "completed",
    steps: int = 4096,
    bpb: float = 2.0,
    top_k: int = 2,
    batch_size: int = 256,
    bptt_chunks: int = 1,
    full_coverage: bool = False,
    artifact_estimate_bytes: int = 8_000_000,
) -> None:
    """Write one summary TSV row for planner tests."""
    geometry_parts = label.split("_")
    core_dim = geometry_parts[1].removeprefix("d")
    layers = geometry_parts[2].removeprefix("l")
    inner = geometry_parts[3].removeprefix("i")
    summary_dir = (
        repo_root
        / "experiments"
        / "5090_architecture"
        / f"{label}_trigram_seed{seed}_{run_version}"
    )
    summary_dir.mkdir(parents=True)
    row = {
        "run_name": run_name or label,
        "status": status,
        "seed": seed,
        "core_dim": core_dim,
        "core_layers": layers,
        "core_inner_dim": inner,
        "recurrent_cells": str(int(layers) * int(inner)),
        "trigram_top_k": str(top_k),
        "num_blocks": "0",
        "batch_size": str(batch_size),
        "bptt_chunks": str(bptt_chunks),
        "effective_step_tokens": "131072",
        "planned_steps": str(steps),
        "last_val_bpb": str(bpb),
        "best_val_bpb": str(bpb),
        "steady_state_tokens_per_sec": "900000",
        "last_eval_full_coverage": str(full_coverage).lower(),
        "exact_val_bpb": "true",
        "artifact_estimate_bytes": str(artifact_estimate_bytes),
    }
    with (summary_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _write_benchmark(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {"name": "current_d48_l12_i480", "tokens_per_sec": 100.0},
                {"name": "d96_l6_i512", "tokens_per_sec": 200.0},
                {"name": "d64_l10_i512", "tokens_per_sec": 120.0},
            ]
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path, stage: str):
    benchmark = tmp_path / "bench.json"
    _write_benchmark(benchmark)
    return parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "--stage",
            stage,
            "--run-version",
            "geom1",
            "--benchmark",
            str(benchmark),
        ]
    )


def test_confirmation_planner_limits_promoted_rows(tmp_path: Path) -> None:
    """Only valid promoted geometry screens should become confirmations."""
    _write_summary(tmp_path, "blocks0_d96_l6_i512", "geom1", bpb=2.0)
    _write_summary(tmp_path, "blocks0_d64_l10_i512", "geom1", bpb=2.04)
    args = _args(tmp_path, "confirmations")
    args.max_confirmations = 1

    commands = plan_confirmations(args)

    assert len(commands) == 1
    assert commands[0].label == "blocks0_d96_l6_i512"
    assert "--full-val-final" in commands[0].command
    assert str(CONFIRM_STEPS) in commands[0].command


def test_bptt_planner_uses_best_completed_confirmation(tmp_path: Path) -> None:
    """BPTT2 should run only for the best completed confirmation."""
    _write_summary(
        tmp_path,
        "blocks0_d96_l6_i512",
        "geom1_confirm",
        steps=8192,
        bpb=2.03,
        full_coverage=True,
    )
    _write_summary(
        tmp_path,
        "blocks0_d64_l10_i512",
        "geom1_confirm",
        steps=8192,
        bpb=2.04,
        full_coverage=True,
    )
    args = _args(tmp_path, "bptt")

    commands = plan_bptt(args)

    assert len(commands) == 1
    assert commands[0].label == "blocks0_d96_l6_i512"
    assert "--geometry-bptt-chunks" in commands[0].command
    assert str(DEFAULT_BPTT_CHUNKS) in commands[0].command
    assert str(DEFAULT_BPTT_BATCH_SIZE) in commands[0].command
    assert str(BPTT_STEPS) in commands[0].command


def test_k4_planner_combines_bptt_only_when_it_wins(tmp_path: Path) -> None:
    """K4 should combine with BPTT2 only after the BPTT screen improves."""
    label = "blocks0_d96_l6_i512"
    _write_summary(tmp_path, label, "geom1", bpb=2.06)
    _write_summary(
        tmp_path,
        label,
        "geom1_confirm",
        steps=8192,
        bpb=2.03,
        full_coverage=True,
    )
    _write_summary(
        tmp_path,
        label,
        "geom1_bptt2",
        bpb=2.05,
        batch_size=DEFAULT_BPTT_BATCH_SIZE,
        bptt_chunks=DEFAULT_BPTT_CHUNKS,
    )
    args = _args(tmp_path, "k4")

    commands = plan_k4(args)

    assert len(commands) == 1
    assert "--trigram-top-k" in commands[0].command
    assert "4" in commands[0].command
    assert "--geometry-bptt-chunks" in commands[0].command


def test_k4_planner_waits_for_completed_bptt_read(tmp_path: Path) -> None:
    """K4 should not run before the selected geometry has a BPTT result."""
    label = "blocks0_d96_l6_i512"
    _write_summary(tmp_path, label, "geom1", bpb=2.06)
    _write_summary(
        tmp_path,
        label,
        "geom1_confirm",
        steps=8192,
        bpb=2.03,
        full_coverage=True,
    )
    args = _args(tmp_path, "k4")

    assert plan_k4(args) == []


def test_geometry_command_uses_flag_protocol() -> None:
    """Generated commands should use launcher flags rather than env assignment."""
    command = geometry_command(
        "blocks0_d96_l6_i512",
        run_version="geom1_confirm",
        seed="1337",
        num_steps=8192,
        hold_steps=7000,
        trigram_top_k=2,
        full_val_final=True,
        val_every=512,
        log_every=128,
        log_state_every=512,
        save_every=4096,
    )

    assert command[:2] == ["bash", "scripts/run_5090_trigram_aligned_geometry_screen.sh"]
    assert "--run-version" in command
    assert "RUN_VERSION=" not in " ".join(command)
