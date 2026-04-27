"""Tests for adaptive 5090 closeout planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core_amplifier_lm.experiment import SUMMARY_FIELDS
from tools.plan_5090_adaptive_closeout import (
    CONFIRM_STEPS,
    DECISION_STEPS,
    DEFAULT_BPTT_BATCH_SIZE,
    DEFAULT_BPTT_CHUNKS,
    DEFAULT_BPTT_IMPROVEMENT_BPB,
    geometry_command,
    parse_args,
    plan_bptt,
    plan_confirmations,
    plan_k4,
    plan_stage,
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
    seq_len: int = 512,
    bptt_chunks: int = 1,
    effective_step_tokens: int = 131072,
    full_coverage: bool = False,
    last_eval_tokens: int | None = None,
    last_eval_denominator_tokens: int = 62_021_845,
    exact_bpb_positive_target_count: int = 61_971_846,
    exact_bpb_zero_byte_target_count: int = 49_999,
    artifact_estimate_bytes: int = 8_000_000,
    artifact_status: str = "LEFT_ON_TABLE",
    validation_source: str = "explicit_val_shard",
    lr_hold_steps: int | None = None,
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
        "seq_len": str(seq_len),
        "bptt_chunks": str(bptt_chunks),
        "effective_step_tokens": str(effective_step_tokens),
        "planned_steps": str(steps),
        "learning_rate": "0.0035",
        "lr_hold_steps": str(
            lr_hold_steps if lr_hold_steps is not None else (7000 if steps == 8192 else 3500)
        ),
        "last_val_bpb": str(bpb),
        "best_val_bpb": str(bpb),
        "steady_state_tokens_per_sec": "900000",
        "last_eval_full_coverage": str(full_coverage).lower(),
        "last_eval_tokens": str(
            last_eval_tokens
            if last_eval_tokens is not None
            else (last_eval_denominator_tokens if full_coverage else 1_048_576)
        ),
        "last_eval_coverage_denominator_tokens": str(last_eval_denominator_tokens),
        "exact_val_bpb": "true",
        "exact_bpb_positive_target_count": str(exact_bpb_positive_target_count),
        "exact_bpb_zero_byte_target_count": str(exact_bpb_zero_byte_target_count),
        "artifact_estimate_bytes": str(artifact_estimate_bytes),
        "artifact_status": artifact_status,
        "validation_source": validation_source,
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
    assert "--count-workers" not in commands[0].command
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
    assert str(DECISION_STEPS) in commands[0].command
    assert "--full-val-final" in commands[0].command


def test_bptt_planner_preserves_k4_long_context_contract(tmp_path: Path) -> None:
    """BPTT follow-up after a K4 context confirmation must not fall back to K2."""
    label = "blocks0_d128_l5_i512"
    _write_summary(
        tmp_path,
        label,
        "geom1_seq2048_confirm",
        steps=8192,
        bpb=1.973,
        top_k=4,
        batch_size=64,
        seq_len=2048,
        full_coverage=True,
    )
    args = _args(tmp_path, "bptt")
    args.run_version = "geom1_seq2048"
    args.screen_trigram_top_k = 4
    args.seq_len = 2048
    args.screen_batch_size = 64
    args.screen_bptt_chunks = 1

    commands = plan_bptt(args)

    assert len(commands) == 1
    command = commands[0].command
    assert "--trigram-top-k" in command
    assert "4" in command
    assert "--geometry-seq-len" in command
    assert "2048" in command
    assert "--geometry-train-label" in command
    assert "1b_seq2048_bptt2_k4" in command
    assert "--geometry-batch-size" in command
    assert "32" in command


def test_finalist_stage_runs_only_missing_seeds(tmp_path: Path) -> None:
    """Explicit stability checks may skip completed seeds and emit missing ones."""
    label = "blocks0_d128_l5_i512"
    _write_summary(
        tmp_path,
        label,
        "geom1_seq2048_bptt2_k6_final",
        steps=8192,
        bpb=1.957,
        top_k=6,
        batch_size=32,
        seq_len=2048,
        bptt_chunks=2,
        full_coverage=True,
    )
    args = _args(tmp_path, "finalist")
    args.label = [label]
    args.finalist_run_version = "geom1_seq2048_bptt2_k6_final"
    args.finalist_seeds = "1337 2027 3141"
    args.finalist_trigram_top_k = 6
    args.finalist_seq_len = 2048
    args.finalist_batch_size = 32
    args.finalist_bptt_chunks = 2
    args.finalist_train_label = "1b_seq2048_bptt2_k6"
    args.finalist_stability_check = True

    stage = plan_stage(args)

    assert stage.status == "commands"
    assert len(stage.commands) == 2
    shells = "\n".join(command.shell for command in stage.commands)
    assert "--seeds 2027" in shells
    assert "--seeds 3141" in shells
    assert "--seeds 1337" not in shells
    assert "--trigram-top-k 6" in shells
    assert "--geometry-seq-len 2048" in shells
    assert "--geometry-bptt-chunks 2" in shells
    assert "--full-val-final" in shells


def test_finalist_stage_blocks_multiple_seeds_by_default(tmp_path: Path) -> None:
    """Normal finalist closeout should not silently become seed replication."""
    label = "blocks0_d128_l5_i512"
    args = _args(tmp_path, "finalist")
    args.label = [label]
    args.finalist_run_version = "geom1_seq2048_bptt2_k6_final"
    args.finalist_seeds = "1337 2027"
    args.finalist_trigram_top_k = 6
    args.finalist_seq_len = 2048
    args.finalist_batch_size = 32
    args.finalist_bptt_chunks = 2

    stage = plan_stage(args)

    assert stage.status == "blocked"
    assert "multiple finalist seeds require --finalist-stability-check" in stage.reason
    assert stage.commands == []


def test_finalist_stage_reports_already_complete(tmp_path: Path) -> None:
    """A fully completed finalist seed set should become a no-op."""
    label = "blocks0_d128_l5_i512"
    _write_summary(
        tmp_path,
        label,
        "geom1_seq2048_bptt2_k6_final",
        seed="1337",
        steps=8192,
        bpb=1.957,
        top_k=6,
        batch_size=32,
        seq_len=2048,
        bptt_chunks=2,
        full_coverage=True,
    )
    args = _args(tmp_path, "finalist")
    args.label = [label]
    args.finalist_run_version = "geom1_seq2048_bptt2_k6_final"
    args.finalist_seeds = "1337"
    args.finalist_trigram_top_k = 6
    args.finalist_seq_len = 2048
    args.finalist_batch_size = 32
    args.finalist_bptt_chunks = 2

    stage = plan_stage(args)

    assert stage.status == "already_complete"
    assert stage.commands == []


def test_finalist_stage_reruns_when_validation_source_is_not_explicit(tmp_path: Path) -> None:
    """Finalist rows must prove they used the explicit validation shard."""
    label = "blocks0_d128_l5_i512"
    _write_summary(
        tmp_path,
        label,
        "geom1_seq2048_bptt2_k6_final",
        seed="1337",
        steps=8192,
        bpb=1.957,
        top_k=6,
        batch_size=32,
        seq_len=2048,
        bptt_chunks=2,
        full_coverage=True,
        validation_source="train_split_fallback",
    )
    args = _args(tmp_path, "finalist")
    args.label = [label]
    args.finalist_run_version = "geom1_seq2048_bptt2_k6_final"
    args.finalist_seeds = "1337"
    args.finalist_trigram_top_k = 6
    args.finalist_seq_len = 2048
    args.finalist_batch_size = 32
    args.finalist_bptt_chunks = 2

    stage = plan_stage(args)

    assert stage.status == "commands"
    assert len(stage.commands) == 1


def test_finalist_preflight_emits_preflight_only_command(tmp_path: Path) -> None:
    """Artifact boundary probes should stop after shared-spec preflight."""
    label = "blocks0_d128_l5_i512"
    args = _args(tmp_path, "finalist")
    args.label = [label]
    args.finalist_run_version = "geom1_seq2048_bptt2_k7_preflight"
    args.finalist_seeds = "1337 2027"
    args.finalist_trigram_top_k = 7
    args.finalist_seq_len = 2048
    args.finalist_batch_size = 32
    args.finalist_bptt_chunks = 2
    args.finalist_train_label = "preflight_seq2048_bptt2_k7"
    args.finalist_preflight_only = True
    args.preflight_trainable_payload_bytes = 1_267_367
    args.spec_max_tokens = 200_000

    stage = plan_stage(args)

    assert stage.status == "commands"
    assert len(stage.commands) == 1
    command = stage.commands[0].command
    assert "--preflight-only" in command
    assert "--spec-max-tokens" in command
    assert "200000" in command
    assert "--trigram-top-k" in command
    assert "7" in command
    assert "--preflight-trainable-payload-bytes" in command
    assert "1267367" in command
    assert "--seeds" in command
    assert "1337" in command


def test_finalist_stage_requires_explicit_contract(tmp_path: Path) -> None:
    """Finalist planning should not infer serious contracts from names."""
    args = _args(tmp_path, "finalist")
    args.label = ["blocks0_d128_l5_i512"]

    stage = plan_stage(args)

    assert stage.status == "blocked"
    assert "--finalist-trigram-top-k" in stage.reason


def test_finalist_stage_rejects_non_divisible_effective_tokens(tmp_path: Path) -> None:
    """Finalist contracts should fail before launching impossible batch shapes."""
    args = _args(tmp_path, "finalist")
    args.label = ["blocks0_d128_l5_i512"]
    args.finalist_run_version = "bad_contract"
    args.finalist_trigram_top_k = 6
    args.finalist_seq_len = 1536
    args.finalist_batch_size = 42
    args.finalist_bptt_chunks = 2

    with pytest.raises(SystemExit, match="finalist batch contract"):
        plan_stage(args)


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
        steps=8192,
        bpb=2.05,
        batch_size=DEFAULT_BPTT_BATCH_SIZE,
        bptt_chunks=DEFAULT_BPTT_CHUNKS,
        full_coverage=True,
    )
    args = _args(tmp_path, "k4")

    commands = plan_k4(args)

    assert len(commands) == 1
    assert "--trigram-top-k" in commands[0].command
    assert "4" in commands[0].command
    assert "--geometry-bptt-chunks" in commands[0].command
    assert "--full-val-final" in commands[0].command
    assert str(DECISION_STEPS) in commands[0].command


def test_confirmation_planner_preserves_k4_long_context_contract(tmp_path: Path) -> None:
    """K4 context screens should promote without reverting to K2/seq512."""
    label = "blocks0_d128_l5_i512"
    _write_summary(
        tmp_path,
        label,
        "geom1_seq2048",
        bpb=2.017,
        top_k=4,
        batch_size=64,
        seq_len=2048,
    )
    args = _args(tmp_path, "confirmations")
    args.run_version = "geom1_seq2048"
    args.screen_trigram_top_k = 4

    commands = plan_confirmations(args)

    assert len(commands) == 1
    command = commands[0].command
    assert "--trigram-top-k" in command
    assert "4" in command
    assert "--geometry-seq-len" in command
    assert "2048" in command
    assert "--geometry-batch-size" in command
    assert "64" in command
    assert "--geometry-train-label" in command
    assert "1b_seq2048_k4" in command


def test_confirmation_planner_selects_best_context_candidate(tmp_path: Path) -> None:
    """Context closeout should choose the best K4 screen, not every promoted one."""
    label = "blocks0_d128_l5_i512"
    _write_summary(
        tmp_path,
        label,
        "geom1_seq1024",
        bpb=2.018,
        top_k=4,
        batch_size=128,
        seq_len=1024,
    )
    _write_summary(
        tmp_path,
        label,
        "geom1_seq2048",
        bpb=2.017,
        top_k=4,
        batch_size=64,
        seq_len=2048,
    )
    _write_summary(
        tmp_path,
        label,
        "geom1_seq4096",
        bpb=2.047,
        top_k=4,
        batch_size=32,
        seq_len=4096,
    )
    args = _args(tmp_path, "confirmations")
    args.run_version = "geom1_seq1024"
    args.candidate_run_version = ["geom1_seq2048", "geom1_seq4096"]
    args.screen_trigram_top_k = 4
    args.max_confirmations = 1

    commands = plan_confirmations(args)

    assert len(commands) == 1
    assert "source=geom1_seq2048" in commands[0].reason
    assert "--run-version" in commands[0].command
    assert "geom1_seq2048_confirm" in commands[0].command
    assert "2048" in commands[0].command
    assert "geom1_seq4096_confirm" not in commands[0].command
    assert "1b_seq4096_k4" not in commands[0].command


def test_k4_planner_uses_bptt1_when_bptt_gain_is_noise(tmp_path: Path) -> None:
    """Tiny BPTT2 gains should not be compounded into K4."""
    label = "blocks0_d96_l6_i512"
    _write_summary(tmp_path, label, "geom1", bpb=2.0600)
    _write_summary(
        tmp_path,
        label,
        "geom1_confirm",
        steps=8192,
        bpb=2.0300,
        full_coverage=True,
    )
    _write_summary(
        tmp_path,
        label,
        "geom1_bptt2",
        steps=8192,
        bpb=2.0600 - (DEFAULT_BPTT_IMPROVEMENT_BPB / 2.0),
        batch_size=DEFAULT_BPTT_BATCH_SIZE,
        bptt_chunks=DEFAULT_BPTT_CHUNKS,
        full_coverage=True,
    )
    args = _args(tmp_path, "k4")

    commands = plan_k4(args)

    assert len(commands) == 1
    assert "--trigram-top-k" in commands[0].command
    assert "--geometry-bptt-chunks" not in commands[0].command


def test_stale_full_validation_confirmation_is_rejected(tmp_path: Path) -> None:
    """Rows from the old replaying full-val path must not satisfy confirmations."""
    label = "blocks0_d96_l6_i512"
    _write_summary(
        tmp_path,
        label,
        "geom1_confirm",
        steps=8192,
        bpb=2.03,
        full_coverage=True,
        last_eval_tokens=62_128_128,
        last_eval_denominator_tokens=62_021_845,
    )
    args = _args(tmp_path, "bptt")

    stage = plan_stage(args)

    assert stage.status == "blocked"
    assert "confirmation" in stage.reason.lower()


def test_k4_planner_ignores_bptt_gain_when_base_screen_invalid(tmp_path: Path) -> None:
    """BPTT selection should not compare against a stale/mismatched base screen."""
    label = "blocks0_d96_l6_i512"
    _write_summary(tmp_path, label, "geom1", bpb=2.20, steps=1024)
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
        steps=8192,
        bpb=2.00,
        batch_size=DEFAULT_BPTT_BATCH_SIZE,
        bptt_chunks=DEFAULT_BPTT_CHUNKS,
        full_coverage=True,
    )
    args = _args(tmp_path, "k4")

    commands = plan_k4(args)

    assert len(commands) == 1
    assert "--geometry-bptt-chunks" not in commands[0].command
    assert "base screen invalid" in commands[0].reason


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


def test_k4_stage_reports_blocked_without_bptt_read(tmp_path: Path) -> None:
    """A missing BPTT prerequisite should be a blocked stage, not success."""
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

    stage = plan_stage(args)

    assert stage.status == "blocked"
    assert stage.commands == []
    assert "BPTT" in stage.reason


def test_bptt_stage_reports_already_complete(tmp_path: Path) -> None:
    """Completed BPTT should be a distinct no-op, not a blocked stage."""
    label = "blocks0_d96_l6_i512"
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
        steps=8192,
        bpb=2.02,
        batch_size=DEFAULT_BPTT_BATCH_SIZE,
        bptt_chunks=DEFAULT_BPTT_CHUNKS,
        full_coverage=True,
    )
    args = _args(tmp_path, "bptt")

    stage = plan_stage(args)

    assert stage.status == "already_complete"
    assert stage.commands == []


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


def test_geometry_command_propagates_count_workers() -> None:
    """Generated variant commands should preserve trigram count-worker settings."""
    command = geometry_command(
        "blocks0_d96_l6_i512",
        run_version="geom1_k4",
        seed="1337",
        num_steps=4096,
        hold_steps=3500,
        trigram_top_k=4,
        full_val_final=False,
        val_every=256,
        log_every=64,
        log_state_every=256,
        save_every=2048,
        count_workers=4,
    )

    assert "--count-workers" in command
    assert "4" in command


def test_smoke_contract_uses_tiny_steps_and_disables_wandb(tmp_path: Path) -> None:
    """Smoke planning should use the explicit tiny contract, not serious defaults."""
    _write_summary(
        tmp_path,
        "blocks0_d96_l6_i512",
        "smoke_adaptive",
        steps=2,
        bpb=3.0,
        batch_size=2,
        seq_len=64,
        effective_step_tokens=128,
        lr_hold_steps=1,
    )
    args = _args(tmp_path, "confirmations")
    args.run_version = "smoke_adaptive"
    args.baseline_bpb = 99.0
    args.screen_steps = 2
    args.lr_hold_steps = 1
    args.effective_step_tokens = 128
    args.confirm_steps = 3
    args.confirm_hold_steps = 1
    args.confirm_full_val_final = False
    args.screen_batch_size = 2
    args.seq_len = 64
    args.val_steps = 1
    args.trigram_max_tokens = 200_000
    args.data_max_tokens = 131_072
    args.no_wandb = True
    args.smoke_test = True

    commands = plan_confirmations(args)

    assert len(commands) == 1
    command = commands[0].command
    assert "--num-steps" in command
    assert "3" in command
    assert "--target-effective-step-tokens" in command
    assert "128" in command
    assert "--no-wandb" in command
    assert "--smoke-test" in command


def test_adaptive_runner_has_status_aware_execution() -> None:
    """The adaptive runner must fail loudly on blocked or unknown planner output."""
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_5090_adaptive_closeout.sh"
    ).read_text(encoding="utf-8")

    assert "planned_${stage} status=${status:-unknown}" in script
    assert "Planner blocked ${stage}; refusing to continue." in script
    assert "Unknown adaptive_status for ${stage}: ${status:-empty}" in script
