"""Tests for local experiment helpers and summary rebuilds."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from core_amplifier_lm.experiment import (
    ARTIFACT_LIMIT_BYTES,
    artifact_headroom_bytes,
    artifact_estimate_bytes,
    artifact_status,
    collect_summary_rows,
    compute_steady_state_tokens_per_sec,
    export_trainable_int8_zlib,
    serialize_trainable_int8_zlib,
    summarize_run_dir,
    trainable_int8_zlib_bytes,
)
from tools.run_core_amp_sweep import (
    controller_spec_max_tokens_default,
    parse_controller_specs,
    resolve_step_token_contract,
    structure_preset_defaults,
)

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


def test_trainable_int8_zlib_bytes_grows_with_more_parameters():
    small = {"w": torch.ones(8)}
    big = {"w": torch.ones(64)}
    assert trainable_int8_zlib_bytes(big) > trainable_int8_zlib_bytes(small)


def test_export_trainable_int8_zlib_matches_serialized_length(tmp_path: Path):
    state = {
        "proj.weight": torch.randn(128, 64),
        "proj.bias": torch.randn(128),
    }
    blob, stats = serialize_trainable_int8_zlib(state)
    out_path = tmp_path / "trainable.int8.ptz"
    export_stats = export_trainable_int8_zlib(out_path, state)
    assert len(blob) == stats["int8_zlib_bytes"] == export_stats["int8_zlib_bytes"]
    assert out_path.stat().st_size == len(blob)
    assert trainable_int8_zlib_bytes(state) == len(blob)


def test_artifact_estimate_includes_trainable_payload(tmp_path: Path):
    spec_path = tmp_path / "spec.pt"
    spec_path.write_bytes(b"x" * 32)
    base = artifact_estimate_bytes(repo_root=tmp_path, spec_path=spec_path)
    bigger = artifact_estimate_bytes(
        repo_root=tmp_path,
        spec_path=spec_path,
        trainable_payload_bytes=123,
    )
    assert base is not None
    assert bigger == base + 123


def test_structure_default_preset_uses_full_frozen_spec_budget():
    defaults = structure_preset_defaults("structure_default")
    assert defaults["CORE_DIM"] == "48"
    assert defaults["SPEC_MAX_TOKENS"] == ""
    assert defaults["SEQ_LEN"] == "512"
    assert defaults["NUM_STEPS"] == "192"
    assert defaults["GRADIENT_CHECKPOINTING"] is True
    assert defaults["DATA_MAX_TOKENS"] == ""
    assert defaults["FORCE_DEVICE"] == ""
    assert defaults["NO_MMAP"] is False


def test_controller_default_preset_uses_full_frozen_spec_budget():
    assert controller_spec_max_tokens_default("controller_default") == ""
    assert controller_spec_max_tokens_default("cpu_smoke") == "500000"


def test_parse_controller_specs_accepts_optional_per_run_warmup():
    legacy = parse_controller_specs("plain3_e20 3 2.0 8 1 0 -2.0 0.003 1500 0.0003 384 256 512")
    assert legacy[0].warmup_steps is None
    assert legacy[0].lr_hold_steps == 1500

    explicit = parse_controller_specs(
        "plain4_e20_c8t4 4 2.0 8 4 0 -2.0 0.003 25 375 0.0003 96 256 512"
    )
    assert explicit[0].warmup_steps == 25
    assert explicit[0].lr_hold_steps == 375


def test_resolve_step_token_contract_supports_fixed_effective_tokens(monkeypatch):
    monkeypatch.setenv("LOCAL_BATCH_SIZE_OVERRIDE", "128")
    monkeypatch.setenv("TARGET_EFFECTIVE_STEP_TOKENS", "131072")
    batch_size, seq_len, grad_accum, local_step_tokens, effective_step_tokens = (
        resolve_step_token_contract(
            batch_size=256,
            seq_len=512,
            bptt_chunks=1,
            grad_accum=1,
        )
    )
    assert batch_size == 128
    assert seq_len == 512
    assert grad_accum == 2
    assert local_step_tokens == 65536
    assert effective_step_tokens == 131072


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
                "residual_token_gate_mode": "base",
                "branch_router_mode": "softmax",
                "base_bigram_delta": "full",
                "trigram_memory": "frozen",
                "trigram_top_k": 2,
                "residual_readout_delta_rank": 128,
                "residual_readout_delta_init_std": 0.02,
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
                "local_step_tokens": 262144,
                "effective_step_tokens": 262144,
                "learning_rate": 0.003,
                "min_lr": 0.0003,
                "warmup_steps": 50,
                "lr_hold_steps": 750,
                "weight_decay": 0.001,
                "num_steps": 192,
                "planned_train_tokens": 50331648,
                "gradient_checkpointing": True,
            },
            "data": {"source": "/tmp/data.bin"},
            "runtime": {
                "gradient_checkpointing": True,
                "exact_val_bpb": True,
                "scan_backend_active": "assoc",
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
            "env": {"TORCH_BLAS_PREFER_CUBLASLT": "1"},
            "system": {
                "torch_version": "2.11.0+cu128",
                "cuda_version": "12.8",
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "driver_version": "580.126.20",
                "gpu_total_memory_mib": 32109,
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
            "repo_code_bytes": 654321,
            "spec_bytes": 1234,
            "gzip_spec_bytes": 234,
            "trainable_int8_zlib_bytes": 345,
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
    assert row["trainable_int8_zlib_bytes"] == "345"
    assert row["residual_token_gate_mode"] == "base"
    assert row["branch_router_mode"] == "softmax"
    assert row["base_bigram_delta"] == "full"
    assert row["trigram_memory"] == "frozen"
    assert row["trigram_top_k"] == "2"
    assert row["residual_readout_delta_rank"] == "128"
    assert row["residual_readout_delta_init_std"] == "0.02"
    assert row["scan_backend"] == "assoc"
    assert row["gradient_checkpointing"] == "true"
    assert row["repo_code_bytes"] == "654321"
    assert row["local_step_tokens"] == "262144"
    assert row["effective_step_tokens"] == "262144"
    assert row["warmup_steps"] == "50"
    assert row["lr_hold_steps"] == "750"
    assert row["torch_version"] == "2.11.0+cu128"
    assert row["cuda_version"] == "12.8"
    assert row["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert row["driver_version"] == "580.126.20"
    assert row["gpu_total_memory_mib"] == "32109"
    assert row["blas_prefer_cublaslt"] == "true"


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
