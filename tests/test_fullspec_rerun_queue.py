"""Tests for the corrected full-spec rerun launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "run_core_amp_fullspec_reruns.py"


def load_module():
    """Load the queue script as a module for inspection."""
    spec = importlib.util.spec_from_file_location("fullspec_reruns", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_queue_contains_pending_fullspec_families():
    """The corrected queue should cover the remaining invalidated families."""
    module = load_module()
    queue = module.build_queue(REPO_ROOT)
    names = module.family_names(queue)
    assert "structure_round1" in names
    assert "blocks3_followup_clean" in names
    assert "blocks3_neighborhood_v1" in names
    assert "blocks3_bptt_v2" in names
    assert "blocks3_carry_v1" in names
    assert "blocks1_radical_v1" in names
    assert "blocks2_confirm1b_v1" in names
    assert "blocks3_confirm1b_v1" in names


def test_corrected_queue_uses_uncapped_spec_roots():
    """The launcher should point at fresh fullspec roots without a spec token cap."""
    module = load_module()
    queue = {launch.name: launch for launch in module.build_queue(REPO_ROOT)}
    structure = queue["structure_round1"]
    assert "fullspec_round1" in structure.model_root
    assert structure.env_overrides["PRESET"] == "structure_default"
    assert "SPEC_MAX_TOKENS" not in structure.env_overrides

    followup = queue["blocks3_followup_clean"]
    assert "fullspec_blocks3_followup_clean" in followup.model_root
    assert followup.env_overrides["PRESET"] == "controller_default"
    assert "SPEC_MAX_TOKENS" not in followup.env_overrides
    assert followup.env_overrides["RUN_SPECS"].startswith("plain3_e20")
