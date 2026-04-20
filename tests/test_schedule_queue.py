"""Tests for the first 5090 schedule-screen launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "run_core_amp_schedule_sweeps.py"


def load_module():
    """Load the queue script as a module for inspection."""
    spec = importlib.util.spec_from_file_location("schedule_queue", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_queue_contains_expected_schedule_families():
    """The launcher should expose the intended first schedule families."""
    module = load_module()
    names = module.family_names(module.build_queue(REPO_ROOT))
    assert names == [
        "blocks0_12x10_hold_screen_v1",
        "blocks0_10x12_hold_screen_v1",
    ]


def test_schedule_queue_uses_blocks0_shared_spec_and_fixed_token_contract():
    """The launcher should keep schedule screening on the agreed blocks0 contract."""
    module = load_module()
    queue = {launch.name: launch for launch in module.build_queue(REPO_ROOT)}

    top = queue["blocks0_12x10_hold_screen_v1"]
    env = top.merged_env()
    assert "fullspec_blocks0_radical_v1" in top.shared_spec_dir
    assert env["NUM_BLOCKS"] == "0"
    assert env["GRADIENT_CHECKPOINTING"] == "0"
    assert env["TARGET_EFFECTIVE_STEP_TOKENS"] == "131072"
    assert "blocks0_resid12_e10_h1500_512m" in env["RUN_SPECS"]

    control = queue["blocks0_10x12_hold_screen_v1"]
    assert "blocks0_resid10_e12_h2500_512m" in control.merged_env()["RUN_SPECS"]
