#!/usr/bin/env python3
"""Canonical local sweep runner for the root Core/Amplifier experiments.

The bash scripts remain as convenience entrypoints, but this file owns the
behavior, metadata layout, restart checks, and summary rebuild flow.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_amplifier_lm import ModelConfig
from core_amplifier_lm.experiment import (
    collect_summary_rows,
    read_json,
    shell_join,
    write_summary_markdown,
    write_summary_tsv,
)


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_command(path: Path, cmd: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(shell_join(cmd) + "\n")


def maybe_add_wandb_args(
    cmd: list[str],
    *,
    run_name: str,
    group_default: str,
    tags_default: list[str],
    enabled_default: bool,
) -> None:
    """Append consistent W&B flags when the current sweep should log online."""
    if not env_bool("WANDB", enabled_default):
        return
    cmd.append("--wandb")
    cmd += ["--wandb-project", env("WANDB_PROJECT", "pg-core-amp")]
    cmd += ["--wandb-run-name", run_name]
    entity = env("WANDB_ENTITY", "")
    if entity:
        cmd += ["--wandb-entity", entity]
    group = env("WANDB_GROUP", group_default)
    if group:
        cmd += ["--wandb-group", group]
    tags = env("WANDB_TAGS", ",".join(tags_default))
    if tags:
        cmd += ["--wandb-tags", tags]
    watch = env("WANDB_WATCH", "gradients")
    if watch:
        cmd += ["--wandb-watch", watch]
    watch_log_freq = env("WANDB_WATCH_LOG_FREQ", "25")
    if watch_log_freq:
        cmd += ["--wandb-watch-log-freq", watch_log_freq]


def stream_command(
    cmd: list[str], *, log_path: Optional[Path] = None, dry_run: bool = False
) -> int:
    print("+", shell_join(cmd), flush=True)
    if dry_run:
        return 0
    if log_path is None:
        return subprocess.run(cmd, check=False).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def ensure_data_path(
    *,
    repo_root: Path,
    data_path: str,
    storage_dtype: str,
    auto_convert_parquet: bool,
    model_root: Path,
    python_bin: str,
    dry_run: bool,
) -> str:
    path = Path(data_path)
    if path.suffix.lower() != ".parquet":
        return str(path)
    if not auto_convert_parquet:
        raise SystemExit("DATA_PATH points to a parquet file and AUTO_CONVERT_PARQUET=0")
    out_dir = model_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.{storage_dtype}.bin"
    if out_path.exists():
        return str(out_path)
    cmd = [
        python_bin,
        str(repo_root / "tools" / "parquet_tokens_to_bin.py"),
        str(path),
        str(out_path),
        "--dtype",
        storage_dtype,
    ]
    rc = stream_command(cmd, dry_run=dry_run)
    if rc != 0:
        raise SystemExit(f"parquet conversion failed with exit {rc}")
    return str(out_path)


def run_complete(run_dir: Path, planned_steps: int) -> bool:
    results = read_json(run_dir / "run_results.json")
    return bool(results.get("completed")) and int(results.get("final_step", -1)) >= int(
        planned_steps
    )


def rebuild_summaries(model_root: Path, title: str) -> None:
    rows = collect_summary_rows(model_root)
    write_summary_tsv(rows, model_root / "summary.tsv")
    write_summary_markdown(rows, model_root / "summary.md", title=title)


def copy_shared_model(shared_dir: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shared_dir / "spec.pt", run_dir / "spec.pt")
    shutil.copy2(shared_dir / "config.json", run_dir / "config.json")
    for ext in ("*.model", "*.vocab"):
        for src in shared_dir.glob(ext):
            shutil.copy2(src, run_dir / src.name)


@dataclass(frozen=True)
class ControllerRunSpec:
    name: str
    core_layers: int
    core_expansion: float
    carry_chunks: int
    bptt_chunks: int
    residual_core: bool
    residual_core_init: float
    learning_rate: float
    warmup_steps: Optional[int]
    lr_hold_steps: int
    min_lr: float
    num_steps: int
    batch_size: int
    seq_len: int


@dataclass(frozen=True)
class StructureRunSpec:
    name: str
    branch_lags: str
    num_blocks: int
    readout_rank: Optional[int]


def controller_default_specs(preset: str) -> list[ControllerRunSpec]:
    if preset == "controller_default":
        return [
            ControllerRunSpec(
                "d5_e20", 5, 2.0, 16, 2, True, -2.0, 0.003, 100, 1500, 0.0003, 7000, 256, 512
            ),
            ControllerRunSpec(
                "d4_e25", 4, 2.5, 16, 2, True, -2.0, 0.003, 100, 1500, 0.0003, 7000, 256, 512
            ),
            ControllerRunSpec(
                "d5_e30", 5, 3.0, 16, 2, True, -2.0, 0.003, 100, 1500, 0.0003, 7000, 256, 512
            ),
            ControllerRunSpec(
                "d6_e25", 6, 2.5, 16, 2, True, -2.0, 0.003, 100, 1500, 0.0003, 7000, 256, 512
            ),
        ]
    if preset == "cpu_smoke":
        return [
            ControllerRunSpec(
                "plain3_e20", 3, 2.0, 8, 1, False, -2.0, 0.003, 20, 20, 0.0003, 80, 8, 128
            ),
            ControllerRunSpec(
                "resid5_e20", 5, 2.0, 16, 1, True, -2.0, 0.003, 20, 20, 0.0003, 80, 8, 128
            ),
            ControllerRunSpec(
                "resid5_e20_tbptt2",
                5,
                2.0,
                16,
                2,
                True,
                -2.0,
                0.003,
                20,
                20,
                0.0003,
                80,
                8,
                128,
            ),
        ]
    raise SystemExit(f"unknown controller preset: {preset}")


def controller_spec_max_tokens_default(preset: str) -> str:
    """Return the default frozen-spec build budget for a controller preset."""
    if preset == "controller_default":
        return "5000000"
    if preset == "cpu_smoke":
        return "500000"
    raise SystemExit(f"unknown controller preset: {preset}")


def parse_controller_specs(raw: str) -> list[ControllerRunSpec]:
    specs: list[ControllerRunSpec] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) not in {13, 14}:
            raise SystemExit(f"invalid controller run spec: {line}")
        if len(fields) == 13:
            warmup_steps = None
            lr_hold_idx = 8
        else:
            warmup_steps = int(fields[8])
            lr_hold_idx = 9
        specs.append(
            ControllerRunSpec(
                name=fields[0],
                core_layers=int(fields[1]),
                core_expansion=float(fields[2]),
                carry_chunks=int(fields[3]),
                bptt_chunks=int(fields[4]),
                residual_core=bool(int(fields[5])),
                residual_core_init=float(fields[6]),
                learning_rate=float(fields[7]),
                warmup_steps=warmup_steps,
                lr_hold_steps=int(fields[lr_hold_idx]),
                min_lr=float(fields[lr_hold_idx + 1]),
                num_steps=int(fields[lr_hold_idx + 2]),
                batch_size=int(fields[lr_hold_idx + 3]),
                seq_len=int(fields[lr_hold_idx + 4]),
            )
        )
    return specs


def structure_default_specs(preset: str) -> list[StructureRunSpec]:
    if preset not in {"structure_default", "cpu_structure"}:
        raise SystemExit(f"unknown structure preset: {preset}")
    return [
        StructureRunSpec("blocks0", "1,2,3,4,6,8,12,16,24,32,48,64", 0, None),
        StructureRunSpec("blocks3", "1,2,3,4,6,8,12,16,24,32,48,64", 3, None),
        StructureRunSpec("blocks6", "1,2,3,4,6,8,12,16,24,32,48,64", 6, None),
        StructureRunSpec("blocks9", "1,2,3,4,6,8,12,16,24,32,48,64", 9, None),
        StructureRunSpec("branches8_pow2", "1,2,4,8,16,32,64,128", 9, None),
        StructureRunSpec("readout256", "1,2,3,4,6,8,12,16,24,32,48,64", 9, 256),
        StructureRunSpec("readout128", "1,2,3,4,6,8,12,16,24,32,48,64", 9, 128),
    ]


def structure_preset_defaults(preset: str) -> dict[str, str | bool]:
    """Return stable structure defaults for real 5090 runs vs CPU smoke."""
    if preset == "structure_default":
        return {
            "CORE_DIM": "48",
            "FIXED_DTYPE": "bfloat16",
            "EMBEDDING_INIT": "spectral",
            "SPEC_STRATEGY": "auto",
            "SPEC_MAX_TOKENS": "5000000",
            "SEQ_LEN": "512",
            "BATCH_SIZE": "256",
            "NUM_STEPS": "192",
            "LEARNING_RATE": "0.003",
            "WARMUP_STEPS": "100",
            "LR_HOLD_STEPS": "1500",
            "MIN_LR": "0.0003",
            "WEIGHT_DECAY": "0.001",
            "HARD_LOSS_GAMMA": "0.5",
            "HARD_LOSS_CAP": "5.0",
            "CARRY_CHUNKS": "16",
            "BPTT_CHUNKS": "2",
            "CORE_LAYERS": "5",
            "CORE_EXPANSION": "2.0",
            "RESIDUAL_CORE": "1",
            "RESIDUAL_CORE_INIT": "-2.0",
            "VAL_EVERY": "64",
            "VAL_STEPS": "8",
            "LOG_EVERY": "16",
            "LOG_STATE_EVERY": "64",
            "SAVE_EVERY": "1000",
            "DATA_MAX_TOKENS": "",
            "FORCE_DEVICE": "",
            "NO_MMAP": False,
            "COMPILE": False,
            "TRAIN_FRAC": "0.98",
        }
    if preset == "cpu_structure":
        return {
            "CORE_DIM": "16",
            "FIXED_DTYPE": "float16",
            "EMBEDDING_INIT": "svd",
            "SPEC_STRATEGY": "stream",
            "SPEC_MAX_TOKENS": "500000",
            "SEQ_LEN": "128",
            "BATCH_SIZE": "8",
            "NUM_STEPS": "80",
            "LEARNING_RATE": "0.003",
            "WARMUP_STEPS": "20",
            "LR_HOLD_STEPS": "20",
            "MIN_LR": "0.0003",
            "WEIGHT_DECAY": "0.001",
            "HARD_LOSS_GAMMA": "0.5",
            "HARD_LOSS_CAP": "5.0",
            "CARRY_CHUNKS": "16",
            "BPTT_CHUNKS": "2",
            "CORE_LAYERS": "5",
            "CORE_EXPANSION": "2.0",
            "RESIDUAL_CORE": "1",
            "RESIDUAL_CORE_INIT": "-2.0",
            "VAL_EVERY": "20",
            "VAL_STEPS": "4",
            "LOG_EVERY": "20",
            "LOG_STATE_EVERY": "20",
            "SAVE_EVERY": "1000",
            "DATA_MAX_TOKENS": "500000",
            "FORCE_DEVICE": "cpu",
            "NO_MMAP": True,
            "COMPILE": False,
            "TRAIN_FRAC": "0.98",
        }
    raise SystemExit(f"unknown structure preset: {preset}")


def parse_structure_specs(raw: str) -> list[StructureRunSpec]:
    specs: list[StructureRunSpec] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise SystemExit(f"invalid structure run spec: {line}")
        rank = int(fields[3])
        specs.append(
            StructureRunSpec(
                name=fields[0],
                branch_lags=fields[1],
                num_blocks=int(fields[2]),
                readout_rank=None if rank <= 0 else rank,
            )
        )
    return specs


def update_controller_config(
    run_dir: Path,
    *,
    run_name: str,
    phase: str,
    data_path: str,
    storage_dtype: str,
    spec: ControllerRunSpec,
    train_defaults: dict[str, str],
) -> None:
    cfg = ModelConfig.load(run_dir)
    branch_temporal_mode = env(
        "BRANCH_TEMPORAL_MODE", cfg.model.get("branch_temporal_mode", "current")
    )
    warmup_steps = (
        int(spec.warmup_steps)
        if spec.warmup_steps is not None
        else int(train_defaults["WARMUP_STEPS"])
    )
    cfg.meta["run_name"] = run_name
    cfg.meta["phase"] = phase
    cfg.data["source"] = data_path
    cfg.data["storage_dtype"] = storage_dtype
    cfg.model["core_layers"] = spec.core_layers
    cfg.model["core_expansion"] = spec.core_expansion
    cfg.model["residual_core"] = spec.residual_core
    cfg.model["residual_core_init"] = spec.residual_core_init
    cfg.model["branch_temporal_mode"] = branch_temporal_mode
    cfg.model["branch_temporal_lag_scale"] = float(env("BRANCH_TEMPORAL_LAG_SCALE", "1.0"))
    cfg.training["seq_len"] = spec.seq_len
    cfg.training["batch_size"] = spec.batch_size
    cfg.training["grad_accum"] = int(train_defaults["GRAD_ACCUM"])
    cfg.training["carry_chunks"] = spec.carry_chunks
    cfg.training["bptt_chunks"] = spec.bptt_chunks
    cfg.training["num_steps"] = spec.num_steps
    cfg.training["learning_rate"] = spec.learning_rate
    cfg.training["lr_schedule"] = train_defaults["LR_SCHEDULE"]
    cfg.training["min_lr"] = spec.min_lr
    cfg.training["warmup_steps"] = warmup_steps
    cfg.training["lr_hold_steps"] = spec.lr_hold_steps
    cfg.training["weight_decay"] = float(train_defaults["WEIGHT_DECAY"])
    cfg.training["grad_clip"] = float(train_defaults["GRAD_CLIP"])
    cfg.training["hard_loss_gamma"] = float(train_defaults["HARD_LOSS_GAMMA"])
    cfg.training["hard_loss_cap"] = float(train_defaults["HARD_LOSS_CAP"])
    cfg.training["val_every"] = int(train_defaults["VAL_EVERY"])
    cfg.training["val_steps"] = int(train_defaults["VAL_STEPS"])
    cfg.training["log_every"] = int(train_defaults["LOG_EVERY"])
    cfg.training["log_state_every"] = int(train_defaults["LOG_STATE_EVERY"])
    cfg.training["save_every"] = int(train_defaults["SAVE_EVERY"])
    cfg.data["train_frac"] = float(train_defaults["TRAIN_FRAC"])
    if train_defaults["DATA_MAX_TOKENS"]:
        cfg.data["max_tokens"] = int(train_defaults["DATA_MAX_TOKENS"])
    cfg.save()


def run_controller_sweep(repo_root: Path) -> None:
    python_bin = sys.executable
    preset = env("PRESET", "controller_default")
    dry_run = env_bool("DRY_RUN", False)
    skip_done = env_bool("SKIP_DONE", True)
    rebuild_shared = env_bool("REBUILD_SHARED", False)
    auto_convert_parquet = env_bool("AUTO_CONVERT_PARQUET", True)
    run_filter = env("RUN_FILTER", "")

    default_data_path = repo_root / "data" / "datasets" / "fineweb10B_sp1024"
    model_root = Path(
        env(
            "MODEL_ROOT",
            str(repo_root / "experiments" / "5090_controller" / f"{now_stamp()}_{preset}"),
        )
    )
    model_root.mkdir(parents=True, exist_ok=True)
    data_path = ensure_data_path(
        repo_root=repo_root,
        data_path=env("DATA_PATH", str(default_data_path)),
        storage_dtype=env("STORAGE_DTYPE", "uint16"),
        auto_convert_parquet=auto_convert_parquet,
        model_root=model_root,
        python_bin=python_bin,
        dry_run=dry_run,
    )
    shared_spec_dir = Path(env("SHARED_SPEC_DIR", str(model_root / "_shared_spec")))
    commands_txt = Path(env("COMMANDS_TXT", str(model_root / "commands.txt")))

    train_defaults = {
        "LR_SCHEDULE": env("LR_SCHEDULE", "cosine"),
        "WARMUP_STEPS": env("WARMUP_STEPS", "100"),
        "WEIGHT_DECAY": env("WEIGHT_DECAY", "0.001"),
        "GRAD_CLIP": env("GRAD_CLIP", "1.0"),
        "HARD_LOSS_GAMMA": env("HARD_LOSS_GAMMA", "0.5"),
        "HARD_LOSS_CAP": env("HARD_LOSS_CAP", "5.0"),
        "VAL_EVERY": env("VAL_EVERY", "200"),
        "VAL_STEPS": env("VAL_STEPS", "20"),
        "LOG_EVERY": env("LOG_EVERY", "20"),
        "LOG_STATE_EVERY": env("LOG_STATE_EVERY", "200"),
        "SAVE_EVERY": env("SAVE_EVERY", "1000"),
        "GRAD_ACCUM": env("GRAD_ACCUM", "1"),
        "DATA_MAX_TOKENS": env("DATA_MAX_TOKENS", ""),
        "TRAIN_FRAC": env("TRAIN_FRAC", "0.9"),
    }

    run_specs_raw = os.environ.get("RUN_SPECS")
    specs = (
        parse_controller_specs(run_specs_raw)
        if run_specs_raw is not None
        else controller_default_specs(preset)
    )

    compile_enabled = env_bool("COMPILE", preset != "cpu_smoke")
    compile_after = env("COMPILE_AFTER", "200")
    compile_mode = env("COMPILE_MODE", "reduce-overhead")
    compile_base_path = env_bool("COMPILE_BASE_PATH", True)
    no_mmap = env_bool("NO_MMAP", False)
    tokens_on_device = env_bool("TOKENS_ON_DEVICE", False)
    no_autocast = env_bool("NO_AUTOCAST", False)
    force_device = env("FORCE_DEVICE", "")
    wandb_enabled_default = preset != "cpu_smoke"

    init_cmd = [
        python_bin,
        str(repo_root / "inspect_model.py"),
        "init",
        str(shared_spec_dir),
        "--data",
        data_path,
        "--storage-dtype",
        env("STORAGE_DTYPE", "uint16"),
        "--vocab-size",
        env("VOCAB_SIZE", "1024"),
        "--core-dim",
        env("CORE_DIM", "48"),
        "--branch-lags",
        env("BRANCH_LAGS", "1,2,3,4,6,8,12,16,24,32,48,64"),
        "--branch-temporal-mode",
        env("BRANCH_TEMPORAL_MODE", "current"),
        "--branch-temporal-lag-scale",
        env("BRANCH_TEMPORAL_LAG_SCALE", "1.0"),
        "--num-blocks",
        env("NUM_BLOCKS", "9"),
        "--fixed-dtype",
        env("FIXED_DTYPE", "bfloat16"),
        "--embedding-init",
        env("EMBEDDING_INIT", "spectral"),
        "--spectral-neighbors",
        env("SPECTRAL_NEIGHBORS", "64"),
        "--lag-identity-base",
        env("LAG_IDENTITY_BASE", "0.15"),
        "--spec-strategy",
        env("SPEC_STRATEGY", "auto"),
        "--spec-workers",
        env("SPEC_WORKERS", "-1"),
        "--core-layers",
        "5",
        "--core-expansion",
        "2.0",
        "--residual-core",
        "1",
        "--residual-core-init",
        "-2.0",
    ]
    spec_max_tokens = env("SPEC_MAX_TOKENS", controller_spec_max_tokens_default(preset))
    if spec_max_tokens:
        init_cmd += ["--max-tokens", spec_max_tokens]

    if (
        rebuild_shared
        or not (shared_spec_dir / "spec.pt").exists()
        or not (shared_spec_dir / "config.json").exists()
    ):
        append_command(commands_txt, init_cmd)
        rc = stream_command(init_cmd, dry_run=dry_run)
        if rc != 0:
            raise SystemExit(f"shared spec init failed with exit {rc}")
    else:
        print(f"Using existing shared spec: {shared_spec_dir}")

    for spec in specs:
        if run_filter and run_filter not in spec.name:
            continue
        run_dir = model_root / spec.name
        log_path = run_dir / "train.log"
        warmup_steps = (
            int(spec.warmup_steps)
            if spec.warmup_steps is not None
            else int(train_defaults["WARMUP_STEPS"])
        )
        if skip_done and run_complete(run_dir, spec.num_steps):
            print(f"Skipping completed run: {spec.name}")
            continue

        if not dry_run and run_dir.exists():
            shutil.rmtree(run_dir)
        if not dry_run:
            copy_shared_model(shared_spec_dir, run_dir)
            update_controller_config(
                run_dir,
                run_name=spec.name,
                phase="5090_controller_screening",
                data_path=data_path,
                storage_dtype=env("STORAGE_DTYPE", "uint16"),
                spec=spec,
                train_defaults=train_defaults,
            )

        cmd = [
            python_bin,
            str(repo_root / "train_core_amplifier.py"),
            str(run_dir),
            "--data",
            data_path,
            "--storage-dtype",
            env("STORAGE_DTYPE", "uint16"),
            "--seq-len",
            str(spec.seq_len),
            "--batch-size",
            str(spec.batch_size),
            "--grad-accum",
            train_defaults["GRAD_ACCUM"],
            "--carry-chunks",
            str(spec.carry_chunks),
            "--bptt-chunks",
            str(spec.bptt_chunks),
            "--num-steps",
            str(spec.num_steps),
            "--learning-rate",
            str(spec.learning_rate),
            "--lr-schedule",
            train_defaults["LR_SCHEDULE"],
            "--min-lr",
            str(spec.min_lr),
            "--warmup-steps",
            str(warmup_steps),
            "--lr-hold-steps",
            str(spec.lr_hold_steps),
            "--weight-decay",
            train_defaults["WEIGHT_DECAY"],
            "--hard-loss-gamma",
            train_defaults["HARD_LOSS_GAMMA"],
            "--hard-loss-cap",
            train_defaults["HARD_LOSS_CAP"],
            "--grad-clip",
            train_defaults["GRAD_CLIP"],
            "--core-layers",
            str(spec.core_layers),
            "--core-expansion",
            str(spec.core_expansion),
            "--residual-core",
            "1" if spec.residual_core else "0",
            "--residual-core-init",
            str(spec.residual_core_init),
            "--branch-temporal-mode",
            env("BRANCH_TEMPORAL_MODE", "current"),
            "--branch-temporal-lag-scale",
            env("BRANCH_TEMPORAL_LAG_SCALE", "1.0"),
            "--val-every",
            train_defaults["VAL_EVERY"],
            "--val-steps",
            train_defaults["VAL_STEPS"],
            "--save-every",
            train_defaults["SAVE_EVERY"],
            "--log-every",
            train_defaults["LOG_EVERY"],
            "--log-state-every",
            train_defaults["LOG_STATE_EVERY"],
            "--train-frac",
            train_defaults["TRAIN_FRAC"],
            "--seed",
            env("SEED", "1337"),
        ]
        if train_defaults["DATA_MAX_TOKENS"]:
            cmd += ["--data-max-tokens", train_defaults["DATA_MAX_TOKENS"]]
        if no_mmap:
            cmd.append("--no-mmap")
        if tokens_on_device:
            cmd.append("--tokens-on-device")
        if force_device:
            cmd += ["--force-device", force_device]
        if no_autocast:
            cmd.append("--no-autocast")
        if compile_enabled:
            cmd += ["--compile", "--compile-after", compile_after, "--compile-mode", compile_mode]
            if compile_base_path:
                cmd.append("--compile-base-path")
        maybe_add_wandb_args(
            cmd,
            run_name=spec.name,
            group_default=model_root.name,
            tags_default=["core_amp", "5090", "controller", "screening"],
            enabled_default=wandb_enabled_default,
        )

        append_command(commands_txt, cmd)
        rc = stream_command(cmd, log_path=log_path, dry_run=dry_run)
        if rc != 0:
            print(f"Run failed: {spec.name} (exit {rc})", file=sys.stderr)
            if not dry_run:
                rebuild_summaries(model_root, "5090 Controller Sweep")
            raise SystemExit(rc)

    if not dry_run:
        rebuild_summaries(model_root, "5090 Controller Sweep")
    print(f"Done. Summary: {model_root / 'summary.tsv'}")


def run_structure_sweep(repo_root: Path) -> None:
    python_bin = sys.executable
    preset = env("PRESET", "structure_default")
    dry_run = env_bool("DRY_RUN", False)
    skip_done = env_bool("SKIP_DONE", True)
    auto_convert_parquet = env_bool("AUTO_CONVERT_PARQUET", True)
    run_filter = env("RUN_FILTER", "")

    default_data_path = repo_root / "data" / "datasets" / "fineweb10B_sp1024"
    model_root = Path(
        env(
            "MODEL_ROOT",
            str(repo_root / "experiments" / "5090_structure" / f"{now_stamp()}_{preset}"),
        )
    )
    model_root.mkdir(parents=True, exist_ok=True)
    data_path = ensure_data_path(
        repo_root=repo_root,
        data_path=env("DATA_PATH", str(default_data_path)),
        storage_dtype=env("STORAGE_DTYPE", "uint16"),
        auto_convert_parquet=auto_convert_parquet,
        model_root=model_root,
        python_bin=python_bin,
        dry_run=dry_run,
    )
    commands_txt = Path(env("COMMANDS_TXT", str(model_root / "commands.txt")))

    run_specs_raw = os.environ.get("RUN_SPECS")
    specs = (
        parse_structure_specs(run_specs_raw)
        if run_specs_raw is not None
        else structure_default_specs(preset)
    )
    defaults = structure_preset_defaults(preset)
    compile_enabled = env_bool("COMPILE", bool(defaults["COMPILE"]))
    no_mmap = env_bool("NO_MMAP", bool(defaults["NO_MMAP"]))
    force_device = env("FORCE_DEVICE", str(defaults["FORCE_DEVICE"]))
    wandb_enabled_default = preset != "cpu_structure"
    planned_steps = int(env("NUM_STEPS", str(defaults["NUM_STEPS"])))

    for spec in specs:
        if run_filter and run_filter not in spec.name:
            continue
        run_dir = model_root / spec.name
        log_path = run_dir / "train.log"
        if skip_done and run_complete(run_dir, planned_steps):
            print(f"Skipping completed run: {spec.name}")
            continue
        if not dry_run and run_dir.exists():
            shutil.rmtree(run_dir)

        init_cmd = [
            python_bin,
            str(repo_root / "inspect_model.py"),
            "init",
            str(run_dir),
            "--data",
            data_path,
            "--storage-dtype",
            env("STORAGE_DTYPE", "uint16"),
            "--vocab-size",
            env("VOCAB_SIZE", "1024"),
            "--core-dim",
            env("CORE_DIM", str(defaults["CORE_DIM"])),
            "--branch-lags",
            spec.branch_lags,
            "--branch-temporal-mode",
            env("BRANCH_TEMPORAL_MODE", "current"),
            "--branch-temporal-lag-scale",
            env("BRANCH_TEMPORAL_LAG_SCALE", "1.0"),
            "--num-blocks",
            str(spec.num_blocks),
            "--fixed-dtype",
            env("FIXED_DTYPE", str(defaults["FIXED_DTYPE"])),
            "--embedding-init",
            env("EMBEDDING_INIT", str(defaults["EMBEDDING_INIT"])),
            "--spectral-neighbors",
            env("SPECTRAL_NEIGHBORS", "64"),
            "--lag-identity-base",
            env("LAG_IDENTITY_BASE", "0.15"),
            "--spec-strategy",
            env("SPEC_STRATEGY", str(defaults["SPEC_STRATEGY"])),
            "--spec-workers",
            env("SPEC_WORKERS", "-1"),
            "--core-layers",
            env("CORE_LAYERS", str(defaults["CORE_LAYERS"])),
            "--core-expansion",
            env("CORE_EXPANSION", str(defaults["CORE_EXPANSION"])),
            "--residual-core",
            env("RESIDUAL_CORE", str(defaults["RESIDUAL_CORE"])),
            "--residual-core-init",
            env("RESIDUAL_CORE_INIT", str(defaults["RESIDUAL_CORE_INIT"])),
        ]
        spec_max_tokens = env("SPEC_MAX_TOKENS", str(defaults["SPEC_MAX_TOKENS"]))
        if spec_max_tokens:
            init_cmd += ["--max-tokens", spec_max_tokens]
        if spec.readout_rank is not None:
            init_cmd += ["--readout-rank", str(spec.readout_rank)]

        append_command(commands_txt, init_cmd)
        rc = stream_command(init_cmd, dry_run=dry_run)
        if rc != 0:
            raise SystemExit(f"structure init failed with exit {rc}")

        if not dry_run:
            cfg = ModelConfig.load(run_dir)
            cfg.meta["run_name"] = spec.name
            cfg.meta["phase"] = "5090_structure_screening"
            cfg.save()

        train_cmd = [
            python_bin,
            str(repo_root / "train_core_amplifier.py"),
            str(run_dir),
            "--data",
            data_path,
            "--storage-dtype",
            env("STORAGE_DTYPE", "uint16"),
            "--seq-len",
            env("SEQ_LEN", str(defaults["SEQ_LEN"])),
            "--batch-size",
            env("BATCH_SIZE", str(defaults["BATCH_SIZE"])),
            "--num-steps",
            str(planned_steps),
            "--learning-rate",
            env("LEARNING_RATE", str(defaults["LEARNING_RATE"])),
            "--lr-schedule",
            "cosine",
            "--warmup-steps",
            env("WARMUP_STEPS", str(defaults["WARMUP_STEPS"])),
            "--lr-hold-steps",
            env("LR_HOLD_STEPS", str(defaults["LR_HOLD_STEPS"])),
            "--min-lr",
            env("MIN_LR", str(defaults["MIN_LR"])),
            "--weight-decay",
            env("WEIGHT_DECAY", str(defaults["WEIGHT_DECAY"])),
            "--hard-loss-gamma",
            env("HARD_LOSS_GAMMA", str(defaults["HARD_LOSS_GAMMA"])),
            "--hard-loss-cap",
            env("HARD_LOSS_CAP", str(defaults["HARD_LOSS_CAP"])),
            "--carry-chunks",
            env("CARRY_CHUNKS", str(defaults["CARRY_CHUNKS"])),
            "--bptt-chunks",
            env("BPTT_CHUNKS", str(defaults["BPTT_CHUNKS"])),
            "--core-layers",
            env("CORE_LAYERS", str(defaults["CORE_LAYERS"])),
            "--core-expansion",
            env("CORE_EXPANSION", str(defaults["CORE_EXPANSION"])),
            "--residual-core",
            env("RESIDUAL_CORE", str(defaults["RESIDUAL_CORE"])),
            "--residual-core-init",
            env("RESIDUAL_CORE_INIT", str(defaults["RESIDUAL_CORE_INIT"])),
            "--branch-temporal-mode",
            env("BRANCH_TEMPORAL_MODE", "current"),
            "--branch-temporal-lag-scale",
            env("BRANCH_TEMPORAL_LAG_SCALE", "1.0"),
            "--val-every",
            env("VAL_EVERY", str(defaults["VAL_EVERY"])),
            "--val-steps",
            env("VAL_STEPS", str(defaults["VAL_STEPS"])),
            "--log-every",
            env("LOG_EVERY", str(defaults["LOG_EVERY"])),
            "--log-state-every",
            env("LOG_STATE_EVERY", str(defaults["LOG_STATE_EVERY"])),
            "--save-every",
            env("SAVE_EVERY", str(defaults["SAVE_EVERY"])),
            "--train-frac",
            env("TRAIN_FRAC", str(defaults["TRAIN_FRAC"])),
            "--seed",
            env("SEED", "1337"),
        ]
        data_max_tokens = env("DATA_MAX_TOKENS", str(defaults["DATA_MAX_TOKENS"]))
        if data_max_tokens:
            train_cmd += ["--data-max-tokens", data_max_tokens]
        if force_device:
            train_cmd += ["--force-device", force_device]
        if no_mmap:
            train_cmd.append("--no-mmap")
        if compile_enabled:
            train_cmd += [
                "--compile",
                "--compile-after",
                env("COMPILE_AFTER", "200"),
                "--compile-mode",
                env("COMPILE_MODE", "reduce-overhead"),
            ]
            if env_bool("COMPILE_BASE_PATH", True):
                train_cmd.append("--compile-base-path")
        maybe_add_wandb_args(
            train_cmd,
            run_name=spec.name,
            group_default=model_root.name,
            tags_default=["core_amp", "5090", "structure", "screening"],
            enabled_default=wandb_enabled_default,
        )

        append_command(commands_txt, train_cmd)
        rc = stream_command(train_cmd, log_path=log_path, dry_run=dry_run)
        if rc != 0:
            print(f"Run failed: {spec.name} (exit {rc})", file=sys.stderr)
            if not dry_run:
                rebuild_summaries(model_root, "5090 Structure Sweep")
            raise SystemExit(rc)

    if not dry_run:
        rebuild_summaries(model_root, "5090 Structure Sweep")
    print(f"Done. Summary: {model_root / 'summary.tsv'}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["controller", "structure"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "controller":
        run_controller_sweep(REPO_ROOT)
    else:
        run_structure_sweep(REPO_ROOT)


if __name__ == "__main__":
    main()
