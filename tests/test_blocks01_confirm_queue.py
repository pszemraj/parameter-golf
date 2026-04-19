"""Tests for the blocks0/blocks1 1B confirmation launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "run_core_amp_blocks01_confirm1b.py"


def load_module():
    """Load the queue script as a module for inspection."""
    spec = importlib.util.spec_from_file_location("blocks01_confirm_queue", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_queue_contains_expected_confirmation_families():
    """The launcher should expose the intended 1B confirmation families."""
    module = load_module()
    names = module.family_names(module.build_queue(REPO_ROOT))
    assert names == [
        "blocks0_confirm1b_v1",
        "blocks1_confirm1b_v1",
        "blocks0_large_checkpointed_confirm1b_v1",
    ]


def test_confirmation_queue_uses_correct_shared_specs():
    """The launcher should point at the completed shared-spec roots."""
    module = load_module()
    queue = {launch.name: launch for launch in module.build_queue(REPO_ROOT)}

    blocks0 = queue["blocks0_confirm1b_v1"]
    assert "fullspec_blocks0_confirm1b_v1" in blocks0.model_root
    assert "fullspec_blocks0_radical_v1" in blocks0.shared_spec_dir
    assert blocks0.env_overrides["NUM_BLOCKS"] == "0"
    assert "blocks0_resid12_e10_c8t1_r3_current_1b" in blocks0.env_overrides["RUN_SPECS"]

    blocks1 = queue["blocks1_confirm1b_v1"]
    assert "fullspec_blocks1_confirm1b_v1" in blocks1.model_root
    assert "fullspec_blocks1_radical_v1" in blocks1.shared_spec_dir
    assert blocks1.env_overrides["NUM_BLOCKS"] == "1"

    large = queue["blocks0_large_checkpointed_confirm1b_v1"]
    assert large.env_overrides["GRADIENT_CHECKPOINTING"] == "1"
