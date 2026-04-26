"""Guard the maintained 5090 W&B project contract."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROJECT = "pg-core-amp"
RETIRED_PROJECT = "pg-" + "hconv-ablations"
PROJECT_CONTRACT_FILES = (
    "README.md",
    "docs/5090_final_week_plan.md",
    "docs/5090_next_experiments.md",
    "scripts/5090_common.sh",
    "scripts/run_5090_trigram_aligned_geometry_screen.sh",
    "tools/run_core_amp_sweep.py",
    "train_core_amplifier.py",
)


def test_5090_wandb_project_contract_uses_core_amp() -> None:
    """Maintained 5090 paths should default and guard to pg-core-amp."""
    combined = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8") for path in PROJECT_CONTRACT_FILES
    )

    assert EXPECTED_PROJECT in combined
    assert RETIRED_PROJECT not in combined
