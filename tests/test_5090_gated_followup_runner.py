"""Tests for the gated follow-up shell runner."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parents[1]


def test_gated_followup_runner_propagates_planned_command_failure(tmp_path: Path) -> None:
    """A selected failing command must make the wrapper fail."""
    fake_planner = tmp_path / "fake_planner.py"
    fake_planner.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

out = Path(sys.argv[sys.argv.index("--write-script") + 1])
out.write_text(
    "#!/usr/bin/env bash\\n"
    "set -euo pipefail\\n"
    "# adaptive_status=commands\\n"
    "# adaptive_reason=test failure propagation\\n"
    "bash -c 'exit 23'\\n",
    encoding="utf-8",
)
print("Stage status: commands")
print("Reason: test failure propagation")
""",
        encoding="utf-8",
    )
    fake_planner.chmod(0o755)
    log_dir = tmp_path / "logs"

    result = subprocess.run(
        [
            "bash",
            str(PKG_ROOT / "scripts" / "run_5090_gated_followup.sh"),
            "--run-id",
            "failure_test",
            "--log-dir",
            str(log_dir),
            "--",
            "--run-version",
            "fake",
        ],
        cwd=PKG_ROOT,
        env={**os.environ, "PYTHON": str(fake_planner)},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 23
    assert "execute_pass1" in result.stdout
    assert "Gated follow-up runner complete" not in result.stdout
    assert (log_dir / "execute_pass1.log").exists()
