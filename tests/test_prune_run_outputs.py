"""Tests for stale run-output pruning."""

from __future__ import annotations

import json
from pathlib import Path

from tools.prune_run_outputs import collect_prune_candidates, is_completed_run


def _write_run(run_dir: Path, *, completed: bool, with_final: bool = True) -> None:
    """Create a minimal fake run directory for pruning tests.

    :param Path run_dir: Run directory to populate.
    :param bool completed: Whether the run is marked completed.
    :param bool with_final: Whether to create ``final.pt``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint_2048.pt").write_bytes(b"x" * 32)
    if with_final:
        (run_dir / "final.pt").write_bytes(b"y" * 16)
    (run_dir / "run_results.json").write_text(json.dumps({"completed": completed}))


def test_is_completed_run_requires_final_and_completed_flag(tmp_path: Path) -> None:
    """Completed-run detection should be conservative."""
    good = tmp_path / "good"
    _write_run(good, completed=True, with_final=True)
    assert is_completed_run(good) is True

    incomplete = tmp_path / "incomplete"
    _write_run(incomplete, completed=False, with_final=True)
    assert is_completed_run(incomplete) is False

    missing_final = tmp_path / "missing_final"
    _write_run(missing_final, completed=True, with_final=False)
    assert is_completed_run(missing_final) is False


def test_collect_prune_candidates_only_returns_completed_runs(tmp_path: Path) -> None:
    """Only stale checkpoints from completed finalized runs should be pruned."""
    keep = tmp_path / "keep"
    _write_run(keep, completed=False, with_final=True)

    prune = tmp_path / "prune"
    _write_run(prune, completed=True, with_final=True)

    missing_final = tmp_path / "missing_final"
    _write_run(missing_final, completed=True, with_final=False)

    candidates = collect_prune_candidates(tmp_path)
    assert [item.path for item in candidates] == [prune / "checkpoint_2048.pt"]
    assert candidates[0].size_bytes == 32
