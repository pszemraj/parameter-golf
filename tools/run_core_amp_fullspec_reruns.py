#!/usr/bin/env python3
"""Launch the remaining corrected full-spec 5090 rerun queue.

This script exists to cleanly replay the historical controller / structure /
temporal families that were invalidated when the shared frozen spec was built
with a capped token budget. It delegates the actual work to
``tools/run_core_amp_sweep.py`` so the per-run metadata, commands, summaries,
and restart checks remain identical to the canonical local harness.
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
}


@dataclass(frozen=True)
class SweepLaunch:
    """One sweep-tool invocation within the corrected rerun queue."""

    name: str
    kind: str
    model_root: str
    env_overrides: dict[str, str]
    description: str

    def merged_env(self) -> dict[str, str]:
        """Return the full environment contract for this family."""
        merged = dict(DEFAULT_ENV)
        merged.update(self.env_overrides)
        merged["MODEL_ROOT"] = self.model_root
        return merged


def _controller_family(
    *,
    name: str,
    model_root: str,
    run_specs: str,
    description: str,
    num_blocks: int,
    val_every: int,
    val_steps: int,
    log_every: int,
    log_state_every: int,
    save_every: int,
    branch_temporal_mode: str = "current",
    branch_temporal_lag_scale: str = "1.0",
    gradient_checkpointing: bool = False,
) -> SweepLaunch:
    return SweepLaunch(
        name=name,
        kind="controller",
        model_root=model_root,
        description=description,
        env_overrides={
            "PRESET": "controller_default",
            "NUM_BLOCKS": str(num_blocks),
            "VAL_EVERY": str(val_every),
            "VAL_STEPS": str(val_steps),
            "LOG_EVERY": str(log_every),
            "LOG_STATE_EVERY": str(log_state_every),
            "SAVE_EVERY": str(save_every),
            "BRANCH_TEMPORAL_MODE": branch_temporal_mode,
            "BRANCH_TEMPORAL_LAG_SCALE": branch_temporal_lag_scale,
            "GRADIENT_CHECKPOINTING": "1" if gradient_checkpointing else "0",
            "RUN_SPECS": run_specs.strip(),
        },
    )


def _structure_family(
    *,
    name: str,
    model_root: str,
    run_specs: str,
    description: str,
    gradient_checkpointing: bool = False,
) -> SweepLaunch:
    return SweepLaunch(
        name=name,
        kind="structure",
        model_root=model_root,
        description=description,
        env_overrides={
            "PRESET": "structure_default",
            "GRADIENT_CHECKPOINTING": "1" if gradient_checkpointing else "0",
            "RUN_SPECS": run_specs.strip(),
        },
    )


def build_queue(repo_root: Path = REPO_ROOT) -> list[SweepLaunch]:
    """Return the corrected full-spec rerun families in launch order."""
    controller_root = repo_root / "experiments" / "5090_controller"
    structure_root = repo_root / "experiments" / "5090_structure"
    temporal_root = repo_root / "experiments" / "5090_temporal"

    return [
        _structure_family(
            name="structure_round1",
            model_root=str(structure_root / "fullspec_round1"),
            description="Corrected full-dataset structure ablation screen.",
            gradient_checkpointing=False,
            run_specs="""
blocks0 1,2,3,4,6,8,12,16,24,32,48,64 0 0
blocks3 1,2,3,4,6,8,12,16,24,32,48,64 3 0
blocks6 1,2,3,4,6,8,12,16,24,32,48,64 6 0
blocks9 1,2,3,4,6,8,12,16,24,32,48,64 9 0
branches8_pow2 1,2,4,8,16,32,64,128 9 0
readout256 1,2,3,4,6,8,12,16,24,32,48,64 9 256
readout128 1,2,3,4,6,8,12,16,24,32,48,64 9 128
""",
        ),
        _controller_family(
            name="blocks3_followup_clean",
            model_root=str(controller_root / "fullspec_blocks3_followup_clean"),
            description="Corrected blocks3 plain-vs-residual screening rerun.",
            num_blocks=3,
            val_every=64,
            val_steps=8,
            log_every=16,
            log_state_every=64,
            save_every=1000,
            run_specs="""
plain3_e20 3 2.0 8 1 0 -2.0 0.003 100 1500 0.0003 384 256 512
resid5_e20 5 2.0 16 2 1 -2.0 0.003 100 1500 0.0003 192 256 512
""",
        ),
        _controller_family(
            name="blocks3_neighborhood_v1",
            model_root=str(controller_root / "fullspec_blocks3_neighborhood_v1"),
            description="Corrected controller-capacity neighborhood screen on blocks3.",
            num_blocks=3,
            val_every=64,
            val_steps=8,
            log_every=16,
            log_state_every=64,
            save_every=1000,
            run_specs="""
plain4_e20_c8t1 4 2.0 8 1 0 -2.0 0.003 100 1500 0.0003 384 256 512
plain3_e25_c8t1 3 2.5 8 1 0 -2.0 0.003 100 1500 0.0003 384 256 512
resid4_e20_c8t1 4 2.0 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512
resid4_e25_c8t1 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512
""",
        ),
        _controller_family(
            name="blocks3_bptt_v2",
            model_root=str(controller_root / "fullspec_blocks3_bptt_v2"),
            description="Corrected BPTT horizon sweep on blocks3 controllers.",
            num_blocks=3,
            val_every=64,
            val_steps=8,
            log_every=16,
            log_state_every=64,
            save_every=1000,
            run_specs="""
plain4_e20_c8t1 4 2.0 8 1 0 -2.0 0.003 100 1500 0.0003 384 256 512
plain4_e20_c8t2 4 2.0 8 2 0 -2.0 0.003 50 750 0.0003 192 256 512
plain4_e20_c8t4 4 2.0 8 4 0 -2.0 0.003 25 375 0.0003 96 256 512
resid4_e25_c8t1 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512
resid4_e25_c8t2 4 2.5 8 2 1 -2.0 0.003 50 750 0.0003 192 256 512
resid4_e25_c8t4 4 2.5 8 4 1 -2.0 0.003 25 375 0.0003 96 256 512
""",
        ),
        _controller_family(
            name="blocks3_carry_v1",
            model_root=str(controller_root / "fullspec_blocks3_carry_v1"),
            description="Corrected carry horizon sweep on blocks3 controllers.",
            num_blocks=3,
            val_every=64,
            val_steps=8,
            log_every=16,
            log_state_every=64,
            save_every=1000,
            run_specs="""
resid4_e20_c8t1 4 2.0 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512
resid4_e20_c16t1 4 2.0 16 1 1 -2.0 0.003 100 1500 0.0003 384 256 512
resid4_e20_c32t1 4 2.0 32 1 1 -2.0 0.003 100 1500 0.0003 384 256 512
resid4_e25_c8t1 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512
resid4_e25_c16t1 4 2.5 16 1 1 -2.0 0.003 100 1500 0.0003 384 256 512
resid4_e25_c32t1 4 2.5 32 1 1 -2.0 0.003 100 1500 0.0003 384 256 512
""",
        ),
        _controller_family(
            name="blocks3_temporal_current_v1",
            model_root=str(temporal_root / "fullspec_blocks3_resid4e25_current_v1"),
            description="Corrected current-branch temporal baseline on blocks3.",
            num_blocks=3,
            val_every=256,
            val_steps=8,
            log_every=64,
            log_state_every=256,
            save_every=2048,
            branch_temporal_mode="current",
            run_specs="""
resid4_e25_c8t1_current_512m 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 4096 256 512
""",
        ),
        _controller_family(
            name="blocks3_temporal_lagged_v1",
            model_root=str(temporal_root / "fullspec_blocks3_resid4e25_lagged_v1"),
            description="Corrected lagged-branch temporal variant on blocks3.",
            num_blocks=3,
            val_every=256,
            val_steps=8,
            log_every=64,
            log_state_every=256,
            save_every=2048,
            branch_temporal_mode="lagged",
            run_specs="""
resid4_e25_c8t1_lagged_512m 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 4096 256 512
""",
        ),
        _controller_family(
            name="blocks3_temporal_hybrid_v1",
            model_root=str(temporal_root / "fullspec_blocks3_resid4e25_hybrid_v1"),
            description="Corrected hybrid-branch temporal variant on blocks3.",
            num_blocks=3,
            val_every=256,
            val_steps=8,
            log_every=64,
            log_state_every=256,
            save_every=2048,
            branch_temporal_mode="hybrid",
            branch_temporal_lag_scale="1.0",
            run_specs="""
resid4_e25_c8t1_hybrid_512m 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 4096 256 512
""",
        ),
        _controller_family(
            name="blocks1_radical_v1",
            model_root=str(controller_root / "fullspec_blocks1_radical_v1"),
            description="Corrected blocks1 radical controller guardrail rerun.",
            num_blocks=1,
            val_every=256,
            val_steps=8,
            log_every=64,
            log_state_every=256,
            save_every=2048,
            branch_temporal_mode="current",
            run_specs="""
blocks1_resid12_e6_c8t1_r3_current_512m 12 6.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512
""",
        ),
        _controller_family(
            name="blocks2_confirm1b_v1",
            model_root=str(controller_root / "fullspec_blocks2_confirm1b_v1"),
            description="Corrected 1B-token confirmation rerun on blocks2 residual 6x2.5.",
            num_blocks=2,
            val_every=512,
            val_steps=8,
            log_every=128,
            log_state_every=512,
            save_every=4096,
            branch_temporal_mode="current",
            run_specs="""
blocks2_resid6_e25_c8t1_1b 6 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 8192 256 512
""",
        ),
        _controller_family(
            name="blocks3_confirm1b_v1",
            model_root=str(controller_root / "fullspec_blocks3_confirm1b_v1"),
            description="Corrected 1B-token confirmation rerun on top blocks3 contenders.",
            num_blocks=3,
            val_every=512,
            val_steps=8,
            log_every=128,
            log_state_every=512,
            save_every=4096,
            run_specs="""
resid4_e20_c16t1_1b 4 2.0 16 1 1 -2.0 0.003 100 1500 0.0003 8192 256 512
resid4_e25_c8t1_1b 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 8192 256 512
""",
        ),
    ]


def family_names(queue: Iterable[SweepLaunch]) -> list[str]:
    """Return the queue names in launch order."""
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
        help="Print the available family names and exit.",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue through the queue after a family failure.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the delegated sweep commands without executing them.",
    )
    return ap.parse_args()


def launch_family(launch: SweepLaunch, *, dry_run: bool = False) -> int:
    """Run one corrected rerun family through the canonical sweep tool."""
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
        launch.kind,
    ]
    print(f"\n=== {launch.name} ===", flush=True)
    print(f"root: {launch.model_root}", flush=True)
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
            print(f"{launch.name}\t{launch.kind}\t{launch.model_root}")
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
