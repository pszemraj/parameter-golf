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
from dataclasses import dataclass, replace
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
    """Return an environment variable or a fallback string.

    :param str name: Environment variable name.
    :param str default: Fallback value when unset.
    :return str: Resolved string value.
    """
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    """Return a boolean environment variable with common truthy parsing.

    :param str name: Environment variable name.
    :param bool default: Fallback value when unset.
    :return bool: Parsed boolean value.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def now_stamp() -> str:
    """Return a timestamp suitable for run-directory names.

    :return str: Timestamp in ``YYYYMMDD_HHMMSS`` format.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def env_optional_int(*names: str) -> Optional[int]:
    """Return the first non-empty integer environment variable.

    :param str names: Candidate environment variable names.
    :return Optional[int]: Parsed integer value, or ``None`` when unset.
    """
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        return int(raw)
    return None


def append_command(path: Path, cmd: list[str]) -> None:
    """Append a shell-escaped command line to a command log.

    :param Path path: Destination text file.
    :param list[str] cmd: Command vector to record.
    """
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
    """Append consistent W&B flags when the current sweep should log online.

    :param list[str] cmd: Command vector to extend in place.
    :param str run_name: Concrete W&B run name.
    :param str group_default: Default W&B group when the environment does not override it.
    :param list[str] tags_default: Default W&B tags.
    :param bool enabled_default: Whether W&B is enabled by default for this sweep kind.
    """
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
    """Run a command, optionally teeing combined output to a log file.

    :param list[str] cmd: Command vector to execute.
    :param Optional[Path] log_path: Optional output log path.
    :param bool dry_run: Whether to print without executing.
    :return int: Process exit code.
    """
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
    """Resolve the effective training-data path, converting parquet when requested.

    :param Path repo_root: Repository root.
    :param str data_path: Requested data path.
    :param str storage_dtype: Output integer dtype for converted token binaries.
    :param bool auto_convert_parquet: Whether parquet inputs may be converted automatically.
    :param Path model_root: Root directory for sweep artifacts.
    :param str python_bin: Python interpreter used for conversion.
    :param bool dry_run: Whether to print without executing the conversion.
    :return str: Resolved binary token path.
    """
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
    """Return whether a run directory contains a completed run at the expected step.

    :param Path run_dir: Candidate run directory.
    :param int planned_steps: Required terminal step count.
    :return bool: ``True`` when the run finished the planned contract.
    """
    results = read_json(run_dir / "run_results.json")
    return bool(results.get("completed")) and int(results.get("final_step", -1)) >= int(
        planned_steps
    )


def rebuild_summaries(model_root: Path, title: str) -> None:
    """Rebuild summary TSV and Markdown files for a sweep root.

    :param Path model_root: Sweep root directory.
    :param str title: Markdown title for the summary report.
    """
    rows = collect_summary_rows(model_root)
    write_summary_tsv(rows, model_root / "summary.tsv")
    write_summary_markdown(rows, model_root / "summary.md", title=title)


def copy_shared_model(shared_dir: Path, run_dir: Path) -> None:
    """Copy a shared spec/config bundle into a concrete run directory.

    :param Path shared_dir: Shared model directory containing the frozen spec and tokenizer.
    :param Path run_dir: Destination run directory.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shared_dir / "spec.pt", run_dir / "spec.pt")
    shutil.copy2(shared_dir / "config.json", run_dir / "config.json")
    for ext in ("*.model", "*.vocab"):
        for src in shared_dir.glob(ext):
            shutil.copy2(src, run_dir / src.name)


@dataclass(frozen=True)
class ControllerRunSpec:
    """Immutable controller sweep point."""

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
    """Immutable frozen-structure sweep point."""

    name: str
    branch_lags: str
    num_blocks: int
    readout_rank: Optional[int]


def resolve_step_token_contract(
    *,
    batch_size: int,
    seq_len: int,
    bptt_chunks: int,
    grad_accum: int,
) -> tuple[int, int, int, int, int]:
    """Resolve local batch overrides and a fixed effective tokens-per-step contract.

    The local 5090 can use a smaller microbatch than the eventual H100 run, but
    comparisons stay apples-to-apples only if the effective processed tokens per
    optimizer step stay fixed. When ``TARGET_EFFECTIVE_STEP_TOKENS`` is set, we
    derive ``grad_accum`` from the resolved microbatch contract.

    :param int batch_size: Requested local batch size.
    :param int seq_len: Requested sequence length.
    :param int bptt_chunks: Number of BPTT chunks per optimizer step.
    :param int grad_accum: Requested gradient accumulation steps.
    :return tuple[int, int, int, int, int]: Resolved batch size, sequence length,
        gradient accumulation, local step tokens, and effective step tokens.
    """
    resolved_batch_size = env_optional_int("LOCAL_BATCH_SIZE_OVERRIDE", "BATCH_SIZE_OVERRIDE")
    resolved_seq_len = env_optional_int("SEQ_LEN_OVERRIDE")
    batch_size = int(resolved_batch_size or batch_size)
    seq_len = int(resolved_seq_len or seq_len)
    local_step_tokens = int(batch_size * seq_len * bptt_chunks)
    target_effective_step_tokens = env_optional_int(
        "TARGET_EFFECTIVE_STEP_TOKENS", "TARGET_STEP_TOKENS"
    )
    if target_effective_step_tokens is not None:
        if target_effective_step_tokens < local_step_tokens:
            raise SystemExit(
                "TARGET_EFFECTIVE_STEP_TOKENS is smaller than one local microbatch "
                f"({target_effective_step_tokens} < {local_step_tokens})"
            )
        if target_effective_step_tokens % local_step_tokens != 0:
            raise SystemExit(
                "TARGET_EFFECTIVE_STEP_TOKENS must divide evenly by "
                "batch_size * seq_len * bptt_chunks "
                f"({target_effective_step_tokens} vs {local_step_tokens})"
            )
        grad_accum = target_effective_step_tokens // local_step_tokens
    grad_accum = max(1, int(grad_accum))
    effective_step_tokens = int(local_step_tokens * grad_accum)
    return batch_size, seq_len, grad_accum, local_step_tokens, effective_step_tokens


def controller_default_specs(preset: str) -> list[ControllerRunSpec]:
    """Return the built-in controller sweep preset.

    :param str preset: Preset name.
    :return list[ControllerRunSpec]: Default controller run specs.
    """
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
    """Return the default frozen-spec build budget for a controller preset.

    Real 5090 controller sweeps should build the frozen spec from the full
    available training shard set unless the caller explicitly requests a cap.

    :param str preset: Controller preset name.
    :return str: Default spec-token cap, or ``""`` for the full dataset.
    """
    if preset == "controller_default":
        return ""
    if preset == "cpu_smoke":
        return "500000"
    raise SystemExit(f"unknown controller preset: {preset}")


def parse_controller_specs(raw: str) -> list[ControllerRunSpec]:
    """Parse controller run specs from a newline-delimited environment payload.

    :param str raw: Raw spec text.
    :return list[ControllerRunSpec]: Parsed run specs.
    """
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
    """Return the built-in structure sweep preset.

    :param str preset: Preset name.
    :return list[StructureRunSpec]: Default structure run specs.
    """
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
    """Return stable structure defaults for real 5090 runs vs CPU smoke.

    :param str preset: Structure preset name.
    :return dict[str, str | bool]: Default environment-style values for that preset.
    """
    if preset == "structure_default":
        return {
            "CORE_DIM": "48",
            "FIXED_DTYPE": "bfloat16",
            "EMBEDDING_INIT": "spectral",
            "SPEC_STRATEGY": "auto",
            "SPEC_MAX_TOKENS": "",
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
            "GRADIENT_CHECKPOINTING": True,
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
            "GRADIENT_CHECKPOINTING": False,
            "TRAIN_FRAC": "0.98",
        }
    raise SystemExit(f"unknown structure preset: {preset}")


def parse_structure_specs(raw: str) -> list[StructureRunSpec]:
    """Parse frozen-structure run specs from a newline-delimited payload.

    :param str raw: Raw spec text.
    :return list[StructureRunSpec]: Parsed run specs.
    """
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
    batch_size: int,
    seq_len: int,
    grad_accum: int,
    local_step_tokens: int,
    effective_step_tokens: int,
) -> None:
    """Write a per-run controller config snapshot before training starts.

    :param Path run_dir: Concrete run directory.
    :param str run_name: Run name stored in metadata.
    :param str phase: Experiment phase label.
    :param str data_path: Training-data path.
    :param str storage_dtype: Token storage dtype.
    :param ControllerRunSpec spec: Controller sweep point.
    :param dict[str, str] train_defaults: Sweep-level training defaults.
    :param int batch_size: Resolved local batch size.
    :param int seq_len: Resolved sequence length.
    :param int grad_accum: Resolved gradient accumulation steps.
    :param int local_step_tokens: Tokens processed by one local microbatch step.
    :param int effective_step_tokens: Tokens processed per optimizer step.
    """
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
    cfg.model["residual_token_gate_mode"] = env("RESIDUAL_TOKEN_GATE_MODE", "none")
    cfg.model["branch_router_mode"] = env("BRANCH_ROUTER_MODE", "none")
    cfg.model["base_bigram_delta"] = env("BASE_BIGRAM_DELTA", "none")
    cfg.model["trigram_sidecar"] = env("TRIGRAM_SIDECAR", "none")
    cfg.model["trigram_log_scale_init"] = float(env("TRIGRAM_LOG_SCALE_INIT", "0.0"))
    cfg.model["residual_readout_delta_rank"] = int(env("RESIDUAL_READOUT_DELTA_RANK", "0"))
    cfg.model["residual_readout_delta_init_std"] = float(
        env("RESIDUAL_READOUT_DELTA_INIT_STD", "0.02")
    )
    cfg.training["seq_len"] = int(seq_len)
    cfg.training["batch_size"] = int(batch_size)
    cfg.training["grad_accum"] = int(grad_accum)
    cfg.training["local_step_tokens"] = int(local_step_tokens)
    cfg.training["effective_step_tokens"] = int(effective_step_tokens)
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
    cfg.training["scan_backend"] = env("SCAN_BACKEND", "auto")
    cfg.training["gradient_checkpointing"] = env_bool(
        "GRADIENT_CHECKPOINTING",
        bool(int(train_defaults["GRADIENT_CHECKPOINTING"])),
    )
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
    """Run the controller sweep protocol rooted at ``repo_root``.

    :param Path repo_root: Repository root.
    """
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
        "GRADIENT_CHECKPOINTING": env(
            "GRADIENT_CHECKPOINTING", "0" if preset == "cpu_smoke" else "1"
        ),
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
    gradient_checkpointing = env_bool(
        "GRADIENT_CHECKPOINTING",
        bool(int(train_defaults["GRADIENT_CHECKPOINTING"])),
    )
    force_device = env("FORCE_DEVICE", "")
    wandb_enabled_default = preset != "cpu_smoke"
    core_amp_phase = env("CORE_AMP_PHASE", "5090_controller_screening")

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
        "--residual-token-gate-mode",
        env("RESIDUAL_TOKEN_GATE_MODE", "none"),
        "--branch-router-mode",
        env("BRANCH_ROUTER_MODE", "none"),
        "--base-bigram-delta",
        env("BASE_BIGRAM_DELTA", "none"),
        "--trigram-sidecar",
        env("TRIGRAM_SIDECAR", "none"),
        "--trigram-log-scale-init",
        env("TRIGRAM_LOG_SCALE_INIT", "0.0"),
        "--residual-readout-delta-rank",
        env("RESIDUAL_READOUT_DELTA_RANK", "0"),
        "--residual-readout-delta-init-std",
        env("RESIDUAL_READOUT_DELTA_INIT_STD", "0.02"),
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
        env("CORE_LAYERS", "5"),
        "--core-expansion",
        env("CORE_EXPANSION", "2.0"),
        "--residual-core",
        env("RESIDUAL_CORE", "1"),
        "--residual-core-init",
        env("RESIDUAL_CORE_INIT", "-2.0"),
        "--scan-backend",
        env("SCAN_BACKEND", "auto"),
    ]
    spec_max_tokens = env("SPEC_MAX_TOKENS", controller_spec_max_tokens_default(preset))
    spec_budget_label = spec_max_tokens if spec_max_tokens else "full_available_train_shards"
    print(f"Frozen spec budget: {spec_budget_label}", flush=True)
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
        resolved_spec = replace(spec)
        batch_size, seq_len, grad_accum, local_step_tokens, effective_step_tokens = (
            resolve_step_token_contract(
                batch_size=resolved_spec.batch_size,
                seq_len=resolved_spec.seq_len,
                bptt_chunks=resolved_spec.bptt_chunks,
                grad_accum=int(train_defaults["GRAD_ACCUM"]),
            )
        )
        resolved_spec = replace(resolved_spec, batch_size=batch_size, seq_len=seq_len)
        run_dir = model_root / spec.name
        log_path = run_dir / "train.log"
        warmup_steps = (
            int(resolved_spec.warmup_steps)
            if resolved_spec.warmup_steps is not None
            else int(train_defaults["WARMUP_STEPS"])
        )
        if skip_done and run_complete(run_dir, resolved_spec.num_steps):
            print(f"Skipping completed run: {spec.name}")
            continue

        if not dry_run and run_dir.exists():
            shutil.rmtree(run_dir)
        if not dry_run:
            copy_shared_model(shared_spec_dir, run_dir)
            update_controller_config(
                run_dir,
                run_name=spec.name,
                phase=core_amp_phase,
                data_path=data_path,
                storage_dtype=env("STORAGE_DTYPE", "uint16"),
                spec=resolved_spec,
                train_defaults=train_defaults,
                batch_size=batch_size,
                seq_len=seq_len,
                grad_accum=grad_accum,
                local_step_tokens=local_step_tokens,
                effective_step_tokens=effective_step_tokens,
            )
        print(
            f"Batch contract for {spec.name}: local_batch_size={batch_size} seq_len={seq_len} "
            f"bptt_chunks={resolved_spec.bptt_chunks} grad_accum={grad_accum} "
            f"local_step_tokens={local_step_tokens:,} "
            f"effective_step_tokens={effective_step_tokens:,}",
            flush=True,
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
            str(seq_len),
            "--batch-size",
            str(batch_size),
            "--grad-accum",
            str(grad_accum),
            "--carry-chunks",
            str(resolved_spec.carry_chunks),
            "--bptt-chunks",
            str(resolved_spec.bptt_chunks),
            "--num-steps",
            str(resolved_spec.num_steps),
            "--learning-rate",
            str(resolved_spec.learning_rate),
            "--lr-schedule",
            train_defaults["LR_SCHEDULE"],
            "--min-lr",
            str(resolved_spec.min_lr),
            "--warmup-steps",
            str(warmup_steps),
            "--lr-hold-steps",
            str(resolved_spec.lr_hold_steps),
            "--weight-decay",
            train_defaults["WEIGHT_DECAY"],
            "--hard-loss-gamma",
            train_defaults["HARD_LOSS_GAMMA"],
            "--hard-loss-cap",
            train_defaults["HARD_LOSS_CAP"],
            "--grad-clip",
            train_defaults["GRAD_CLIP"],
            "--core-layers",
            str(resolved_spec.core_layers),
            "--core-expansion",
            str(resolved_spec.core_expansion),
            "--residual-core",
            "1" if resolved_spec.residual_core else "0",
            "--residual-core-init",
            str(resolved_spec.residual_core_init),
            "--branch-temporal-mode",
            env("BRANCH_TEMPORAL_MODE", "current"),
            "--branch-temporal-lag-scale",
            env("BRANCH_TEMPORAL_LAG_SCALE", "1.0"),
            "--residual-token-gate-mode",
            env("RESIDUAL_TOKEN_GATE_MODE", "none"),
            "--branch-router-mode",
            env("BRANCH_ROUTER_MODE", "none"),
            "--base-bigram-delta",
            env("BASE_BIGRAM_DELTA", "none"),
            "--trigram-sidecar",
            env("TRIGRAM_SIDECAR", "none"),
            "--trigram-log-scale-init",
            env("TRIGRAM_LOG_SCALE_INIT", "0.0"),
            "--residual-readout-delta-rank",
            env("RESIDUAL_READOUT_DELTA_RANK", "0"),
            "--residual-readout-delta-init-std",
            env("RESIDUAL_READOUT_DELTA_INIT_STD", "0.02"),
            "--scan-backend",
            env("SCAN_BACKEND", "auto"),
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
        if env_bool("FULL_VAL_FINAL", False):
            cmd.append("--full-val-final")
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
        if gradient_checkpointing:
            cmd.append("--gradient-checkpointing")
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
    """Run the frozen-structure sweep protocol rooted at ``repo_root``.

    :param Path repo_root: Repository root.
    """
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
    gradient_checkpointing = env_bool(
        "GRADIENT_CHECKPOINTING", bool(defaults["GRADIENT_CHECKPOINTING"])
    )
    force_device = env("FORCE_DEVICE", str(defaults["FORCE_DEVICE"]))
    wandb_enabled_default = preset != "cpu_structure"
    planned_steps = int(env("NUM_STEPS", str(defaults["NUM_STEPS"])))
    structure_bptt_chunks = int(env("BPTT_CHUNKS", str(defaults["BPTT_CHUNKS"])))
    structure_batch_size = int(env("BATCH_SIZE", str(defaults["BATCH_SIZE"])))
    structure_seq_len = int(env("SEQ_LEN", str(defaults["SEQ_LEN"])))
    structure_grad_accum = int(env("GRAD_ACCUM", "1"))
    (
        resolved_structure_batch_size,
        resolved_structure_seq_len,
        resolved_structure_grad_accum,
        structure_local_step_tokens,
        structure_effective_step_tokens,
    ) = resolve_step_token_contract(
        batch_size=structure_batch_size,
        seq_len=structure_seq_len,
        bptt_chunks=structure_bptt_chunks,
        grad_accum=structure_grad_accum,
    )

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
            "--residual-token-gate-mode",
            env("RESIDUAL_TOKEN_GATE_MODE", "none"),
            "--branch-router-mode",
            env("BRANCH_ROUTER_MODE", "none"),
            "--base-bigram-delta",
            env("BASE_BIGRAM_DELTA", "none"),
            "--trigram-sidecar",
            env("TRIGRAM_SIDECAR", "none"),
            "--trigram-log-scale-init",
            env("TRIGRAM_LOG_SCALE_INIT", "0.0"),
            "--residual-readout-delta-rank",
            env("RESIDUAL_READOUT_DELTA_RANK", "0"),
            "--residual-readout-delta-init-std",
            env("RESIDUAL_READOUT_DELTA_INIT_STD", "0.02"),
            "--scan-backend",
            env("SCAN_BACKEND", "auto"),
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
        spec_budget_label = spec_max_tokens if spec_max_tokens else "full_available_train_shards"
        print(f"Frozen spec budget: {spec_budget_label}", flush=True)
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
            cfg.meta["phase"] = env("CORE_AMP_PHASE", "5090_structure_screening")
            cfg.model["base_bigram_delta"] = env("BASE_BIGRAM_DELTA", "none")
            cfg.model["trigram_sidecar"] = env("TRIGRAM_SIDECAR", "none")
            cfg.model["trigram_log_scale_init"] = float(env("TRIGRAM_LOG_SCALE_INIT", "0.0"))
            cfg.model["residual_readout_delta_rank"] = int(env("RESIDUAL_READOUT_DELTA_RANK", "0"))
            cfg.model["residual_readout_delta_init_std"] = float(
                env("RESIDUAL_READOUT_DELTA_INIT_STD", "0.02")
            )
            cfg.training["seq_len"] = int(resolved_structure_seq_len)
            cfg.training["batch_size"] = int(resolved_structure_batch_size)
            cfg.training["grad_accum"] = int(resolved_structure_grad_accum)
            cfg.training["local_step_tokens"] = int(structure_local_step_tokens)
            cfg.training["effective_step_tokens"] = int(structure_effective_step_tokens)
            cfg.training["gradient_checkpointing"] = gradient_checkpointing
            cfg.save()
        print(
            f"Batch contract for {spec.name}: local_batch_size={resolved_structure_batch_size} "
            f"seq_len={resolved_structure_seq_len} bptt_chunks={structure_bptt_chunks} "
            f"grad_accum={resolved_structure_grad_accum} "
            f"local_step_tokens={structure_local_step_tokens:,} "
            f"effective_step_tokens={structure_effective_step_tokens:,}",
            flush=True,
        )

        train_cmd = [
            python_bin,
            str(repo_root / "train_core_amplifier.py"),
            str(run_dir),
            "--data",
            data_path,
            "--storage-dtype",
            env("STORAGE_DTYPE", "uint16"),
            "--seq-len",
            str(resolved_structure_seq_len),
            "--batch-size",
            str(resolved_structure_batch_size),
            "--grad-accum",
            str(resolved_structure_grad_accum),
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
            str(structure_bptt_chunks),
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
            "--residual-token-gate-mode",
            env("RESIDUAL_TOKEN_GATE_MODE", "none"),
            "--branch-router-mode",
            env("BRANCH_ROUTER_MODE", "none"),
            "--base-bigram-delta",
            env("BASE_BIGRAM_DELTA", "none"),
            "--trigram-sidecar",
            env("TRIGRAM_SIDECAR", "none"),
            "--trigram-log-scale-init",
            env("TRIGRAM_LOG_SCALE_INIT", "0.0"),
            "--residual-readout-delta-rank",
            env("RESIDUAL_READOUT_DELTA_RANK", "0"),
            "--residual-readout-delta-init-std",
            env("RESIDUAL_READOUT_DELTA_INIT_STD", "0.02"),
            "--scan-backend",
            env("SCAN_BACKEND", "auto"),
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
        if env_bool("FULL_VAL_FINAL", False):
            train_cmd.append("--full-val-final")
        data_max_tokens = env("DATA_MAX_TOKENS", str(defaults["DATA_MAX_TOKENS"]))
        if data_max_tokens:
            train_cmd += ["--data-max-tokens", data_max_tokens]
        if force_device:
            train_cmd += ["--force-device", force_device]
        if no_mmap:
            train_cmd.append("--no-mmap")
        if gradient_checkpointing:
            train_cmd.append("--gradient-checkpointing")
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
    """Parse CLI arguments for the sweep runner.

    :return argparse.Namespace: Parsed CLI arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["controller", "structure"])
    return ap.parse_args()


def main() -> None:
    """Dispatch to the requested sweep kind."""
    args = parse_args()
    if args.kind == "controller":
        run_controller_sweep(REPO_ROOT)
    else:
        run_structure_sweep(REPO_ROOT)


if __name__ == "__main__":
    main()
