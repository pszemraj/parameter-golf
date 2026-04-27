"""Tests for 5090 shared-spec rebuild and launcher contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tools import run_core_amp_sweep


PKG_ROOT = Path(__file__).resolve().parents[1]


def test_sweep_refuses_to_rebuild_explicit_shared_spec(tmp_path: Path) -> None:
    """The sweep layer must not rebuild an explicit shared spec with defaults."""
    result = subprocess.run(
        [
            sys.executable,
            "tools/run_core_amp_sweep.py",
            "controller",
            "--shared-spec-dir",
            str(tmp_path / "shared"),
            "--model-root",
            str(tmp_path / "runs"),
            "--run-spec",
            "test 5 4.0 8 2 1 -3.0 0.0035 100 7000 0.0003 8192 32 2048",
            "--rebuild-shared",
            "--dry-run",
        ],
        cwd=PKG_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "refusing to rebuild an explicit --shared-spec-dir" in result.stderr


def test_sweep_clears_ambient_rebuild_for_explicit_shared_spec(tmp_path: Path, monkeypatch) -> None:
    """Ambient REBUILD_SHARED must not leak into explicit shared-spec consumers."""
    monkeypatch.setenv("REBUILD_SHARED", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_core_amp_sweep.py",
            "controller",
            "--shared-spec-dir",
            str(tmp_path / "shared"),
        ],
    )

    args = run_core_amp_sweep.parse_args()
    run_core_amp_sweep.apply_cli_overrides(args)

    assert os.environ["REBUILD_SHARED"] == "0"


def test_aligned_launcher_dry_run_keeps_spec_rebuild_at_launcher_layer() -> None:
    """The maintained launcher should build the spec once, then sweep no-rebuild."""
    env = {
        **os.environ,
        "WANDB_MODE": "",
        "SPEC_MAX_TOKENS": "",
        "DATA_MAX_TOKENS": "",
        "TRAIN_FRAC": "",
        "LOCAL_BATCH_SIZE_OVERRIDE": "",
        "BATCH_SIZE_OVERRIDE": "",
        "SEQ_LEN_OVERRIDE": "",
        "TARGET_STEP_TOKENS": "",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/run_5090_trigram_aligned_geometry_screen.sh",
            "--run-version",
            "geom1_seq2048_bptt2_k6",
            "--seeds",
            "1337",
            "--geometry-label",
            "blocks0_d128_l5_i512",
            "--geometry-core-dim",
            "128",
            "--geometry-core-layers",
            "5",
            "--geometry-core-inner-dim",
            "512",
            "--geometry-batch-size",
            "32",
            "--geometry-seq-len",
            "2048",
            "--geometry-bptt-chunks",
            "2",
            "--num-steps",
            "8192",
            "--lr-hold-steps",
            "7000",
            "--geometry-train-label",
            "1b_seq2048_bptt2_k6",
            "--trigram-top-k",
            "6",
            "--count-workers",
            "4",
            "--full-val-final",
            "--rebuild-shared",
            "--dry-run",
        ],
        cwd=PKG_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--suppress-config-summary" in result.stdout
    assert "--trigram-top-k 6" in result.stdout
    assert "--trigram-count-workers 4" in result.stdout
    assert "+ REBUILD_SHARED=0 " in result.stdout
    downstream = result.stdout.split("run_core_amp_sweep.py controller", 1)[1]
    assert "blocks0_d128_l5_i512_trigram_1b_seq2048_bptt2_k6\\ 5\\ 4.0" in downstream
    assert "blocks0_d128_l5_i512_trigramk6_lr0035" not in downstream
    assert "_s1337" not in downstream
    assert "--no-rebuild-shared" in result.stdout
    assert "--rebuild-shared" not in downstream
