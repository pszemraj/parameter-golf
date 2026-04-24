#!/usr/bin/env python3
"""Root training entrypoint for the Core/Amplifier LM.

This fork-level script intentionally makes the root training path the
Core/Amplifier language model, so experimentation happens in the repo core
rather than only inside `/records`.

The original transformer baseline from upstream is preserved as
`train_gpt_transformer_baseline.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from core_amplifier_lm.config import DEFAULTS


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return "1" if default else "0"
    return "1" if raw.lower() in {"1", "true", "yes", "on"} else "0"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _to_env_default(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    model_defaults = DEFAULTS["model"]
    train_defaults = DEFAULTS["training"]

    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 or (local_rank_env is not None and int(local_rank_env) != 0):
        raise SystemExit(
            "Core/Amplifier root train_gpt.py is single-process only; DDP/torchrun "
            "sharding is not implemented for this path yet."
        )

    default_data = repo_root / "data" / "datasets" / "fineweb10B_sp1024"
    model_dir = Path(env("MODEL_DIR", "core_amp_run"))
    data_path = env("DATA_PATH", str(default_data))

    python = sys.executable

    init_cmd = [
        python,
        str(repo_root / "inspect_model.py"),
        "init",
        str(model_dir),
        "--data",
        data_path,
        "--storage-dtype",
        env("STORAGE_DTYPE", "uint16"),
        "--vocab-size",
        env("VOCAB_SIZE", _to_env_default(model_defaults["vocab_size"])),
        "--core-dim",
        env("CORE_DIM", _to_env_default(model_defaults["core_dim"])),
        "--branch-lags",
        env("BRANCH_LAGS", _to_env_default(model_defaults["branch_lags"])),
        "--num-blocks",
        env("NUM_BLOCKS", _to_env_default(model_defaults["num_blocks"])),
        "--fixed-dtype",
        env("FIXED_DTYPE", _to_env_default(model_defaults["fixed_dtype"])),
        "--embedding-init",
        env("EMBEDDING_INIT", _to_env_default(model_defaults["embedding_init"])),
        "--spectral-neighbors",
        env("SPECTRAL_NEIGHBORS", _to_env_default(model_defaults["spectral_neighbors"])),
        "--lag-identity-base",
        env("LAG_IDENTITY_BASE", _to_env_default(model_defaults["lag_identity_base"])),
        "--spec-strategy",
        env("SPEC_STRATEGY", "auto"),
        "--spec-workers",
        env("SPEC_WORKERS", "-1"),
        "--core-layers",
        env("CORE_LAYERS", _to_env_default(model_defaults["core_layers"])),
        "--core-expansion",
        env("CORE_EXPANSION", _to_env_default(model_defaults["core_expansion"])),
        "--residual-core",
        env_bool("RESIDUAL_CORE", bool(model_defaults["residual_core"])),
        "--residual-core-init",
        env("RESIDUAL_CORE_INIT", _to_env_default(model_defaults["residual_core_init"])),
        "--base-bigram-delta",
        env("BASE_BIGRAM_DELTA", _to_env_default(model_defaults["base_bigram_delta"])),
        "--trigram-sidecar",
        env("TRIGRAM_SIDECAR", _to_env_default(model_defaults["trigram_sidecar"])),
        "--trigram-log-scale-init",
        env(
            "TRIGRAM_LOG_SCALE_INIT",
            _to_env_default(model_defaults["trigram_log_scale_init"]),
        ),
    ]
    readout_rank = os.environ.get("READOUT_RANK")
    if readout_rank is not None:
        init_cmd += ["--readout-rank", readout_rank]
    spec_max_tokens = os.environ.get("SPEC_MAX_TOKENS")
    if spec_max_tokens:
        init_cmd += ["--max-tokens", spec_max_tokens]

    if not (model_dir / "config.json").exists() or not (model_dir / "spec.pt").exists():
        run(init_cmd)
    elif env_bool("REUSE_MODEL_DIR", False) != "1":
        raise SystemExit(
            f"Refusing to reuse existing model dir without REUSE_MODEL_DIR=1: {model_dir}"
        )
    else:
        print(f"Using existing model dir: {model_dir}", flush=True)

    train_cmd = [
        python,
        str(repo_root / "train_core_amplifier.py"),
        str(model_dir),
        "--data",
        data_path,
        "--storage-dtype",
        env("STORAGE_DTYPE", "uint16"),
        "--seq-len",
        env("TRAIN_SEQ_LEN", env("SEQ_LEN", _to_env_default(train_defaults["seq_len"]))),
        "--batch-size",
        env("BATCH_SIZE", _to_env_default(train_defaults["batch_size"])),
        "--num-steps",
        env("NUM_STEPS", env("ITERATIONS", _to_env_default(train_defaults["num_steps"]))),
        "--learning-rate",
        env("LEARNING_RATE", env("LR", _to_env_default(train_defaults["learning_rate"]))),
        "--lr-schedule",
        env("LR_SCHEDULE", _to_env_default(train_defaults["lr_schedule"])),
        "--min-lr",
        env("MIN_LR", _to_env_default(train_defaults["min_lr"])),
        "--warmup-steps",
        env("WARMUP_STEPS", _to_env_default(train_defaults["warmup_steps"])),
        "--lr-hold-steps",
        env("LR_HOLD_STEPS", _to_env_default(train_defaults["lr_hold_steps"])),
        "--weight-decay",
        env("WEIGHT_DECAY", _to_env_default(train_defaults["weight_decay"])),
        "--grad-clip",
        env("GRAD_CLIP", _to_env_default(train_defaults["grad_clip"])),
        "--hard-loss-gamma",
        env("HARD_LOSS_GAMMA", _to_env_default(train_defaults["hard_loss_gamma"])),
        "--hard-loss-cap",
        env("HARD_LOSS_CAP", _to_env_default(train_defaults["hard_loss_cap"])),
        "--carry-chunks",
        env("CARRY_CHUNKS", _to_env_default(train_defaults["carry_chunks"])),
        "--bptt-chunks",
        env("BPTT_CHUNKS", _to_env_default(train_defaults["bptt_chunks"])),
        "--core-layers",
        env("CORE_LAYERS", _to_env_default(model_defaults["core_layers"])),
        "--core-expansion",
        env("CORE_EXPANSION", _to_env_default(model_defaults["core_expansion"])),
        "--residual-core",
        env_bool("RESIDUAL_CORE", bool(model_defaults["residual_core"])),
        "--residual-core-init",
        env("RESIDUAL_CORE_INIT", _to_env_default(model_defaults["residual_core_init"])),
        "--base-bigram-delta",
        env("BASE_BIGRAM_DELTA", _to_env_default(model_defaults["base_bigram_delta"])),
        "--trigram-sidecar",
        env("TRIGRAM_SIDECAR", _to_env_default(model_defaults["trigram_sidecar"])),
        "--trigram-log-scale-init",
        env(
            "TRIGRAM_LOG_SCALE_INIT",
            _to_env_default(model_defaults["trigram_log_scale_init"]),
        ),
        "--val-every",
        env("VAL_EVERY", "200"),
        "--val-steps",
        env("VAL_STEPS", "20"),
        "--log-every",
        env("LOG_EVERY", "20"),
        "--log-state-every",
        env("LOG_STATE_EVERY", _to_env_default(train_defaults["log_state_every"])),
        "--save-every",
        env("SAVE_EVERY", "1000"),
    ]
    if env_bool("FULL_VAL_FINAL", bool(train_defaults.get("full_val_final", False))) == "1":
        train_cmd.append("--full-val-final")
    data_max_tokens = os.environ.get("DATA_MAX_TOKENS")
    if data_max_tokens:
        train_cmd += ["--data-max-tokens", data_max_tokens]
    if env_bool("NO_MMAP", False) == "1":
        train_cmd.append("--no-mmap")
    if env_bool("TOKENS_ON_DEVICE", False) == "1":
        train_cmd.append("--tokens-on-device")
    if env_bool("COMPILE", True) == "1":
        train_cmd += [
            "--compile",
            "--compile-after",
            env("COMPILE_AFTER", "200"),
            "--compile-mode",
            env("COMPILE_MODE", "reduce-overhead"),
        ]
        if env_bool("COMPILE_BASE_PATH", True) == "1":
            train_cmd.append("--compile-base-path")
    force_device = os.environ.get("FORCE_DEVICE")
    if force_device:
        train_cmd += ["--force-device", force_device]

    run(train_cmd)


if __name__ == "__main__":
    main()
