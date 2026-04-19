"""Tests for local experiment helpers and summary rebuilds."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from core_amplifier_lm.experiment import (
    ARTIFACT_LIMIT_BYTES,
    artifact_headroom_bytes,
    artifact_status,
    collect_summary_rows,
    compute_steady_state_tokens_per_sec,
    summarize_run_dir,
)
from tools.run_core_amp_sweep import structure_preset_defaults

PKG_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )


def test_compute_steady_state_tokens_per_sec_prefers_post_compile_rows():
    rows = [
        {"kind": "train", "step": 0, "tokens_per_sec": 100.0},
        {"kind": "train", "step": 1, "tokens_per_sec": 200.0},
        {"kind": "train", "step": 2, "tokens_per_sec": 300.0},
        {"kind": "train", "step": 3, "tokens_per_sec": 800.0},
        {"kind": "train", "step": 4, "tokens_per_sec": 900.0},
        {"kind": "train", "step": 5, "tokens_per_sec": 1000.0},
    ]
    steady = compute_steady_state_tokens_per_sec(rows, compile_trigger_step=2)
    assert steady == 900.0


def test_artifact_budget_helpers_report_left_on_table_under_cap():
    artifact_bytes = ARTIFACT_LIMIT_BYTES - 123
    assert artifact_headroom_bytes(artifact_bytes) == 123
    assert artifact_status(artifact_bytes) == "LEFT_ON_TABLE"


def test_structure_default_preset_uses_real_5090_budget():
    defaults = structure_preset_defaults("structure_default")
    assert defaults["CORE_DIM"] == "48"
    assert defaults["SPEC_MAX_TOKENS"] == "5000000"
    assert defaults["SEQ_LEN"] == "512"
    assert defaults["NUM_STEPS"] == "192"
    assert defaults["DATA_MAX_TOKENS"] == ""
    assert defaults["FORCE_DEVICE"] == ""
    assert defaults["NO_MMAP"] is False


def test_summarize_run_dir_reads_structured_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    _write_json(
        run_dir / "resolved_config.json",
        {
            "run_name": "run_a",
            "phase": "5090_controller_screening",
            "seed": 1337,
            "model": {
                "core_layers": 5,
                "core_expansion": 2.0,
                "residual_core": True,
                "residual_core_init": -2.0,
                "branch_lags": [1, 2, 4],
                "num_blocks": 3,
                "readout_rank": None,
            },
            "training": {
                "carry_chunks": 16,
                "bptt_chunks": 2,
                "seq_len": 512,
                "batch_size": 256,
                "grad_accum": 1,
                "num_steps": 192,
                "planned_train_tokens": 50331648,
            },
            "data": {"source": "/tmp/data.bin"},
            "runtime": {
                "exact_val_bpb": True,
                "compile": {
                    "enabled": True,
                    "compile_after": 200,
                    "compile_mode": "reduce-overhead",
                },
            },
            "tokenizer_path": "/tmp/tok.model",
            "spec": {"spec_bytes": 1234, "gzip_spec_bytes": 234},
        },
    )
    _write_json(
        run_dir / "run_metadata.json",
        {
            "git_commit": "abc123",
            "system": {
                "tf32_matmul": True,
                "tf32_cudnn": True,
                "float32_matmul_precision": "high",
            },
        },
    )
    _write_json(
        run_dir / "run_results.json",
        {
            "completed": True,
            "final_step": 192,
            "seen_train_tokens": 50331648,
            "elapsed_sec": 12.5,
            "steady_state_tokens_per_sec": 950000.0,
            "compile_duration_sec": 1.25,
            "peak_mem_alloc_mib": 2048.0,
            "peak_mem_reserved_mib": 3072.0,
            "artifact_estimate_bytes": 999999,
            "artifact_headroom_bytes": 15000001,
            "artifact_status": "LEFT_ON_TABLE",
            "spec_bytes": 1234,
            "gzip_spec_bytes": 234,
        },
    )
    _write_jsonl(
        run_dir / "metrics.jsonl",
        [
            {"kind": "train", "step": 0, "tokens_per_sec": 400000.0},
            {"kind": "eval", "step": 100, "val_loss": 4.2, "val_bpb": 2.4},
            {"kind": "eval", "step": 191, "val_loss": 4.1, "val_bpb": 2.3},
        ],
    )
    row = summarize_run_dir(run_dir)
    assert row["status"] == "completed"
    assert row["run_name"] == "run_a"
    assert row["git_commit"] == "abc123"
    assert row["best_val_bpb"] == "2.3"
    assert row["steady_state_tokens_per_sec"] == "950000.0"
    assert row["artifact_estimate_bytes"] == "999999"
    assert row["artifact_status"] == "LEFT_ON_TABLE"


def test_rebuild_summary_cli_writes_tsv_and_markdown(tmp_path: Path):
    run_dir = tmp_path / "run_b"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    _write_json(
        run_dir / "resolved_config.json",
        {
            "run_name": "run_b",
            "model": {"branch_lags": [1, 2], "num_blocks": 1, "readout_rank": 128},
            "training": {"num_steps": 10},
            "runtime": {"compile": {"enabled": False}},
            "spec": {"spec_bytes": 10, "gzip_spec_bytes": 5},
        },
    )
    _write_json(run_dir / "run_metadata.json", {"git_commit": "def456", "system": {}})
    _write_json(run_dir / "run_results.json", {"completed": True, "final_step": 10})
    _write_jsonl(
        run_dir / "metrics.jsonl", [{"kind": "eval", "step": 9, "val_loss": 4.0, "val_bpb": 2.0}]
    )

    out_tsv = tmp_path / "summary.tsv"
    out_md = tmp_path / "summary.md"
    subprocess.run(
        [
            sys.executable,
            str(PKG_ROOT / "tools" / "rebuild_summary.py"),
            str(tmp_path),
            "--out",
            str(out_tsv),
            "--md-out",
            str(out_md),
            "--title",
            "Test Summary",
        ],
        check=True,
    )
    rows = collect_summary_rows(tmp_path)
    assert len(rows) == 1
    assert "run_b" in out_tsv.read_text(encoding="utf-8")
    assert "Test Summary" in out_md.read_text(encoding="utf-8")
