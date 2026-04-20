#!/usr/bin/env python3
"""Launch restart-safe 5090 schedule sweeps for the current core-amplifier frontier.

The goal is to keep schedule work as disciplined as the structure/controller
queues: fixed token budgets, explicit family names, and delegated execution via
``tools/run_core_amp_sweep.py`` so per-run metadata and summaries stay aligned.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"
DEFAULT_WANDB_PROJECT = "pg-core-amp"
DEFAULT_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "TORCH_BLAS_PREFER_CUBLASLT": "1",
    "WANDB": "1",
    "WANDB_PROJECT": DEFAULT_WANDB_PROJECT,
    "SKIP_DONE": "1",
    "COMPILE": "0",
    "TRAIN_FRAC": "0.98",
    "DATA_PATH": str(DEFAULT_DATA_PATH),
    # Keep schedule screening apples-to-apples even if the local microbatch is
    # reduced later for memory reasons.
    "TARGET_EFFECTIVE_STEP_TOKENS": "131072",
}


@dataclass(frozen=True)
class SweepLaunch:
    """One schedule family delegated through the canonical sweep tool."""

    name: str
    model_root: str
    shared_spec_dir: str
    env_overrides: dict[str, str]
    description: str

    def merged_env(self) -> dict[str, str]:
        """Return the full environment contract for this family."""
        merged = dict(DEFAULT_ENV)
        merged.update(self.env_overrides)
        merged["MODEL_ROOT"] = self.model_root
        merged["SHARED_SPEC_DIR"] = self.shared_spec_dir
        return merged


def _controller_family(
    *,
    name: str,
    model_root: str,
    shared_spec_dir: str,
    run_specs: str,
    description: str,
    num_blocks: int = 0,
    val_every: int = 256,
    val_steps: int = 8,
    log_every: int = 64,
    log_state_every: int = 256,
    save_every: int = 2048,
) -> SweepLaunch:
    return SweepLaunch(
        name=name,
        model_root=model_root,
        shared_spec_dir=shared_spec_dir,
        description=description,
        env_overrides={
            "PRESET": "controller_default",
            "NUM_BLOCKS": str(num_blocks),
            "VAL_EVERY": str(val_every),
            "VAL_STEPS": str(val_steps),
            "LOG_EVERY": str(log_every),
            "LOG_STATE_EVERY": str(log_state_every),
            "SAVE_EVERY": str(save_every),
            "BRANCH_TEMPORAL_MODE": "current",
            "RUN_SPECS": run_specs.strip(),
            "WANDB_TAGS": "core_amp,5090,schedule,screening,hold",
        },
    )


def build_queue(repo_root: Path = REPO_ROOT) -> list[SweepLaunch]:
    """Return the first disciplined schedule-screen families."""
    schedule_root = repo_root / "experiments" / "5090_schedule"
    blocks0_shared = str(
        repo_root
        / "experiments"
        / "5090_structure"
        / "fullspec_blocks0_radical_v1"
        / "blocks0_resid12_e6_c8t1_r3_current_512m"
    )
    return [
        _controller_family(
            name="blocks0_12x10_hold_screen_v1",
            model_root=str(schedule_root / "blocks0_12x10_hold_screen_v1"),
            shared_spec_dir=blocks0_shared,
            description=(
                "Isolated lr_hold_steps sweep on the best completed pure-quality "
                "blocks0 controller."
            ),
            run_specs="""
blocks0_resid12_e10_h0_512m     12 10.0 8 1 1 -3.0 0.003 100    0 0.0003 4096 256 512
blocks0_resid12_e10_h500_512m   12 10.0 8 1 1 -3.0 0.003 100  500 0.0003 4096 256 512
blocks0_resid12_e10_h1500_512m  12 10.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512
blocks0_resid12_e10_h2500_512m  12 10.0 8 1 1 -3.0 0.003 100 2500 0.0003 4096 256 512
""",
        ),
        _controller_family(
            name="blocks0_10x12_hold_screen_v1",
            model_root=str(schedule_root / "blocks0_10x12_hold_screen_v1"),
            shared_spec_dir=blocks0_shared,
            description=(
                "Matching lr_hold_steps sweep on the near-tied geometry-control blocks0 controller."
            ),
            run_specs="""
blocks0_resid10_e12_h0_512m     10 12.0 8 1 1 -3.0 0.003 100    0 0.0003 4096 256 512
blocks0_resid10_e12_h500_512m   10 12.0 8 1 1 -3.0 0.003 100  500 0.0003 4096 256 512
blocks0_resid10_e12_h1500_512m  10 12.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512
blocks0_resid10_e12_h2500_512m  10 12.0 8 1 1 -3.0 0.003 100 2500 0.0003 4096 256 512
""",
        ),
    ]


def family_names(queue: Iterable[SweepLaunch]) -> list[str]:
    """Return queue family names in launch order."""
    return [launch.name for launch in queue]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--family",
        action="append",
        default=[],
        help="Launch only the named family. Repeat to select multiple families.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print available family names and exit.",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue through the queue after a family failure.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print delegated sweep commands without executing them.",
    )
    return ap.parse_args()


def launch_family(launch: SweepLaunch, *, dry_run: bool = False) -> int:
    """Run one schedule family through the canonical sweep tool."""
    env = os.environ.copy()
    env.update(launch.merged_env())
    cmd = [
        "conda",
        "run",
        "-s",
        "--name",
        "train",
        "python",
        str(REPO_ROOT / "tools" / "run_core_amp_sweep.py"),
        "controller",
    ]
    print(f"\n=== {launch.name} ===", flush=True)
    print(f"root: {launch.model_root}", flush=True)
    print(f"spec: {launch.shared_spec_dir}", flush=True)
    print(f"desc: {launch.description}", flush=True)
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False).returncode


def main() -> None:
    """Entry point."""
    args = parse_args()
    queue = build_queue(REPO_ROOT)
    known = {launch.name for launch in queue}
    if args.list:
        for launch in queue:
            print(f"{launch.name}\t{launch.model_root}\t{launch.shared_spec_dir}")
        return
    if args.family:
        missing = sorted(set(args.family) - known)
        if missing:
            raise SystemExit(f"unknown family name(s): {', '.join(missing)}")
        wanted = set(args.family)
        queue = [launch for launch in queue if launch.name in wanted]

    for launch in queue:
        rc = launch_family(launch, dry_run=args.dry_run)
        if rc == 0:
            continue
        print(f"Family failed: {launch.name} (exit {rc})", file=sys.stderr, flush=True)
        if not args.continue_on_error:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
