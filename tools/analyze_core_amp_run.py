#!/usr/bin/env python3
"""Diagnose a trained Core/Amplifier run against its frozen base path."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_amplifier_lm import (
    AmplifierSpec,
    CoreAmplifierLM,
    ModelConfig,
    trigram_memory_config_value,
)
from train_core_amplifier import (
    SequentialStreamBatcher,
    _list_train_val_shards,
    _memmap_token_file,
    build_byte_count_lut,
    cross_entropy_per_token,
    get_device,
    load_trainable_state,
    mmap_train_val,
    resolve_runtime_amplifier_dtype,
)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object if it exists.

    :param Path path: JSON path.
    :return dict[str, Any]: Parsed object or an empty dict.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    """Load a Torch object across old and new PyTorch defaults.

    :param Path path: Torch checkpoint path.
    :param str | torch.device map_location: Load target.
    :return Any: Loaded object.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _as_bool(value: Any) -> bool:
    """Normalize common config boolean encodings.

    :param Any value: Raw value.
    :return bool: Normalized boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _merged_config(
    cfg: ModelConfig, resolved: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge saved config and resolved run payload.

    :param ModelConfig cfg: Base model config.
    :param dict[str, Any] resolved: Resolved training payload.
    :return tuple[dict[str, Any], dict[str, Any]]: Model and training maps.
    """
    model = dict(cfg.model)
    model.update(resolved.get("model", {}))
    training = dict(cfg.training)
    training.update(resolved.get("training", {}))
    return model, training


def _checkpoint_path(cfg: ModelConfig, explicit: Optional[str]) -> Path:
    """Resolve the checkpoint to analyze.

    :param ModelConfig cfg: Run configuration wrapper.
    :param Optional[str] explicit: Optional checkpoint override.
    :return Path: Existing checkpoint path.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        latest = cfg.latest_checkpoint()
        if latest is None:
            raise FileNotFoundError(
                f"no final.pt/checkpoint.pt/checkpoint_*.pt under {cfg.model_dir}"
            )
        path = latest.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _validation_tokens(
    source: str | Path,
    *,
    storage_dtype: str,
    train_frac: float,
    allow_train_frac_val_split: bool,
    cache_dir: Optional[str],
) -> np.ndarray:
    """Load or map validation tokens without touching train shards when possible.

    :param str | Path source: Dataset source.
    :param str storage_dtype: Token storage dtype.
    :param float train_frac: Split ratio for single-file fallback.
    :param bool allow_train_frac_val_split: Whether train split fallback is allowed.
    :param Optional[str] cache_dir: Optional mmap cache directory.
    :return np.ndarray: Validation token view.
    """
    source_path = Path(source)
    train_shards, val_shards = _list_train_val_shards(source_path)
    if source_path.is_dir() and val_shards:
        if len(val_shards) == 1:
            return _memmap_token_file(val_shards[0], storage_dtype=storage_dtype)
        return np.concatenate(
            [
                np.asarray(_memmap_token_file(path, storage_dtype=storage_dtype))
                for path in val_shards
            ]
        )
    if source_path.is_dir() and train_shards and not allow_train_frac_val_split:
        raise FileNotFoundError(
            f"{source_path} has train shards but no fineweb_val_* shard. "
            "Diagnostics do not silently fall back to train-frac validation; "
            "pass --allow-train-frac-val-split for an explicit local debug run."
        )
    _, val = mmap_train_val(
        source_path,
        storage_dtype=storage_dtype,
        train_frac=train_frac,
        allow_train_frac_val_split=allow_train_frac_val_split,
        cache_dir=cache_dir,
        verbose=False,
    )
    return val


def _find_tokenizer(
    cfg: ModelConfig, resolved: dict[str, Any], explicit: Optional[str]
) -> Optional[Path]:
    """Resolve the tokenizer path used for exact byte-normalized metrics.

    :param ModelConfig cfg: Run configuration wrapper.
    :param dict[str, Any] resolved: Resolved training payload.
    :param Optional[str] explicit: Optional tokenizer override.
    :return Optional[Path]: Tokenizer path if present.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    resolved_path = resolved.get("tokenizer_path")
    if resolved_path:
        path = Path(str(resolved_path))
        if path.exists():
            return path.resolve()
    if cfg.tokenizer_path is not None and cfg.tokenizer_path.exists():
        return cfg.tokenizer_path.resolve()
    return None


@contextmanager
def _zero_base_bigram_delta(model: CoreAmplifierLM) -> Iterator[bool]:
    """Temporarily disable the optional trainable base-bigram delta.

    :param CoreAmplifierLM model: Model to modify in-place and restore.
    :return Iterator[bool]: Whether a base-bigram delta table was present.
    """
    delta = model.base_bigram_delta
    if delta is None:
        yield False
        return
    saved = delta.weight.detach().clone()
    with torch.no_grad():
        delta.weight.zero_()
    try:
        yield True
    finally:
        with torch.no_grad():
            delta.weight.copy_(saved)


@contextmanager
def _zero_readout_delta(model: CoreAmplifierLM) -> Iterator[bool]:
    """Temporarily disable the optional trainable readout delta.

    :param CoreAmplifierLM model: Model to modify in-place and restore.
    :return Iterator[bool]: Whether a delta output matrix was present.
    """
    delta_out = model.residual_readout_delta_out
    if delta_out is None:
        yield False
        return
    saved = delta_out.weight.detach().clone()
    with torch.no_grad():
        delta_out.weight.zero_()
    try:
        yield True
    finally:
        with torch.no_grad():
            delta_out.weight.copy_(saved)


@contextmanager
def _disable_trigram_memory(model: CoreAmplifierLM) -> Iterator[bool]:
    """Temporarily disable the optional frozen trigram memory.

    :param CoreAmplifierLM model: Model to modify in-place and restore.
    :return Iterator[bool]: Whether a trigram memory was active.
    """
    old_mode = getattr(model, "trigram_memory_mode", "none")
    if old_mode == "none":
        yield False
        return
    model.trigram_memory_mode = "none"
    try:
        yield True
    finally:
        model.trigram_memory_mode = old_mode


def _bucket_rows(
    base_losses: torch.Tensor,
    full_losses: torch.Tensor,
    no_delta_losses: Optional[torch.Tensor],
) -> list[dict[str, float | int | str]]:
    """Summarize improvements by frozen-base loss quantile.

    :param torch.Tensor base_losses: Per-token frozen-base losses.
    :param torch.Tensor full_losses: Per-token full-model losses.
    :param Optional[torch.Tensor] no_delta_losses: Per-token losses without the
        readout delta, when applicable.
    :return list[dict[str, float | int | str]]: Bucket summaries.
    """
    quantiles = torch.quantile(base_losses.float(), torch.tensor([0.25, 0.5, 0.75]))
    edges = [
        (None, float(quantiles[0].item()), "q00_q25"),
        (float(quantiles[0].item()), float(quantiles[1].item()), "q25_q50"),
        (float(quantiles[1].item()), float(quantiles[2].item()), "q50_q75"),
        (float(quantiles[2].item()), None, "q75_q100"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for lo, hi, label in edges:
        mask = torch.ones_like(base_losses, dtype=torch.bool)
        if lo is not None:
            mask &= base_losses > lo
        if hi is not None:
            mask &= base_losses <= hi
        if not bool(mask.any()):
            continue
        base_mean = float(base_losses[mask].mean().item())
        full_mean = float(full_losses[mask].mean().item())
        row: dict[str, float | int | str] = {
            "bucket": label,
            "tokens": int(mask.sum().item()),
            "base_loss": base_mean,
            "full_loss": full_mean,
            "full_gain_nats": base_mean - full_mean,
        }
        if no_delta_losses is not None:
            no_delta_mean = float(no_delta_losses[mask].mean().item())
            row["no_readout_delta_loss"] = no_delta_mean
            row["readout_delta_gain_nats"] = no_delta_mean - full_mean
        rows.append(row)
    return rows


def _format_float(value: Any) -> str:
    """Format optional floats for Markdown.

    :param Any value: Value to format.
    :return str: Display string.
    """
    if value is None:
        return "-"
    return f"{float(value):.6f}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown diagnostic report.

    :param Path path: Output path.
    :param dict[str, Any] payload: Diagnostic payload.
    """
    losses = payload["losses"]
    lines = [
        "# Core/Amplifier Run Diagnostics",
        "",
        f"- run: `{payload['run_dir']}`",
        f"- checkpoint: `{payload['checkpoint_path']}`",
        f"- device: `{payload['device']}`",
        f"- sampled tokens: `{payload['validation']['tokens']}`",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| base_loss | {_format_float(losses['base_loss'])} |",
        f"| full_loss | {_format_float(losses['full_loss'])} |",
        f"| full_gain_nats | {_format_float(losses['full_gain_nats'])} |",
        f"| base_bpb | {_format_float(losses.get('base_bpb'))} |",
        f"| full_bpb | {_format_float(losses.get('full_bpb'))} |",
        f"| full_gain_bpb | {_format_float(losses.get('full_gain_bpb'))} |",
    ]
    if "no_readout_delta_loss" in losses:
        lines.extend(
            [
                f"| no_readout_delta_loss | {_format_float(losses['no_readout_delta_loss'])} |",
                f"| readout_delta_gain_nats | {_format_float(losses['readout_delta_gain_nats'])} |",
                f"| readout_delta_gain_bpb | {_format_float(losses.get('readout_delta_gain_bpb'))} |",
            ]
        )
    if "no_base_bigram_delta_loss" in losses:
        lines.extend(
            [
                f"| no_base_bigram_delta_loss | {_format_float(losses['no_base_bigram_delta_loss'])} |",
                f"| base_bigram_delta_gain_nats | {_format_float(losses['base_bigram_delta_gain_nats'])} |",
                f"| base_bigram_delta_gain_bpb | {_format_float(losses.get('base_bigram_delta_gain_bpb'))} |",
            ]
        )
    if "no_trigram_memory_loss" in losses:
        lines.extend(
            [
                f"| no_trigram_memory_loss | {_format_float(losses['no_trigram_memory_loss'])} |",
                f"| trigram_memory_gain_nats | {_format_float(losses['trigram_memory_gain_nats'])} |",
                f"| trigram_memory_gain_bpb | {_format_float(losses.get('trigram_memory_gain_bpb'))} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Base-Loss Buckets",
            "",
            "| Bucket | Tokens | Base Loss | Full Loss | Full Gain | No Delta Loss | Delta Gain |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["base_loss_buckets"]:
        lines.append(
            "| {bucket} | {tokens} | {base} | {full} | {gain} | {no_delta} | {delta_gain} |".format(
                bucket=row["bucket"],
                tokens=row["tokens"],
                base=_format_float(row["base_loss"]),
                full=_format_float(row["full_loss"]),
                gain=_format_float(row["full_gain_nats"]),
                no_delta=_format_float(row.get("no_readout_delta_loss")),
                delta_gain=_format_float(row.get("readout_delta_gain_nats")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=str, help="Run directory containing config/spec/checkpoint")
    ap.add_argument("--checkpoint", type=str, default=None, help="Checkpoint override")
    ap.add_argument("--data", type=str, default=None, help="Validation data source override")
    ap.add_argument("--tokenizer", type=str, default=None, help="Tokenizer override")
    ap.add_argument("--steps", type=int, default=16, help="Sequential validation batches")
    ap.add_argument("--batch-size", type=int, default=8, help="Validation batch size")
    ap.add_argument("--seq-len", type=int, default=None, help="Validation sequence length")
    ap.add_argument("--device", type=str, default="cpu", help="Runtime device, default cpu")
    ap.add_argument("--mmap-cache-dir", type=str, default=None, help="Optional mmap cache dir")
    ap.add_argument(
        "--allow-train-frac-val-split",
        action="store_true",
        help="Allow validation split fallback for local debug only",
    )
    ap.add_argument(
        "--allow-approx-bpb",
        action="store_true",
        help="Allow loss-only diagnostics when tokenizer bytes are unavailable",
    )
    ap.add_argument("--out", type=str, default=None, help="JSON output path")
    ap.add_argument("--md-out", type=str, default=None, help="Markdown output path")
    return ap.parse_args()


def main() -> None:
    """Run diagnostics and write JSON/Markdown reports."""
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = ModelConfig.load(run_dir)
    resolved = _load_json(run_dir / "resolved_config.json")
    model_cfg, training_cfg = _merged_config(cfg, resolved)

    spec = AmplifierSpec.load(cfg.spec_path)
    device, device_type, default_amp_dtype = get_device(force=args.device)
    amplifier_dtype = resolve_runtime_amplifier_dtype(
        str(training_cfg.get("amplifier_dtype", "auto")),
        device_type=device_type,
        default_amp_dtype=default_amp_dtype,
    )
    model = CoreAmplifierLM(
        spec,
        core_layers=int(model_cfg.get("core_layers", 5)),
        core_type=str(model_cfg.get("core_type", "mingru")),
        core_expansion=float(model_cfg.get("core_expansion", 2.0)),
        residual_core=_as_bool(model_cfg.get("residual_core", True)),
        residual_core_init=float(model_cfg.get("residual_core_init", -2.0)),
        branch_temporal_mode=str(model_cfg.get("branch_temporal_mode", "current")),
        branch_temporal_lag_scale=float(model_cfg.get("branch_temporal_lag_scale", 1.0)),
        residual_token_gate_mode=str(model_cfg.get("residual_token_gate_mode", "none")),
        branch_router_mode=str(model_cfg.get("branch_router_mode", "none")),
        base_bigram_delta=str(model_cfg.get("base_bigram_delta", "none")),
        trigram_memory=str(trigram_memory_config_value(model_cfg, "none")),
        trigram_log_scale_init=float(model_cfg.get("trigram_log_scale_init", 0.0)),
        residual_readout_delta_rank=int(model_cfg.get("residual_readout_delta_rank", 0)),
        residual_readout_delta_init_std=float(
            model_cfg.get("residual_readout_delta_init_std", 0.02)
        ),
        scan_backend=str(training_cfg.get("scan_backend", "auto")),
        gradient_checkpointing=False,
        amplifier_dtype=amplifier_dtype,
    ).to(device)
    model.prepare_runtime_buffers(device=device, amplifier_dtype=amplifier_dtype)

    checkpoint_path = _checkpoint_path(cfg, args.checkpoint)
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    load_trainable_state(model, state)
    model.eval()

    data_source = args.data or resolved.get("data", {}).get("source") or cfg.data.get("source")
    if not data_source:
        raise SystemExit("No data source found; pass --data")
    storage_dtype = str(cfg.data.get("storage_dtype", "uint16"))
    train_frac = float(cfg.data.get("train_frac", 0.98))
    val_np = _validation_tokens(
        data_source,
        storage_dtype=storage_dtype,
        train_frac=train_frac,
        allow_train_frac_val_split=bool(args.allow_train_frac_val_split),
        cache_dir=args.mmap_cache_dir,
    )
    val_tokens = torch.from_numpy(np.asarray(val_np))

    tokenizer = _find_tokenizer(cfg, resolved, args.tokenizer)
    if tokenizer is None and not args.allow_approx_bpb:
        raise SystemExit(
            "Tokenizer not found; exact bpb is unavailable. "
            "Pass --tokenizer or --allow-approx-bpb for loss-only diagnostics."
        )
    byte_count_lut = (
        build_byte_count_lut(tokenizer, spec.vocab_size, device) if tokenizer is not None else None
    )

    seq_len = int(args.seq_len or training_cfg.get("seq_len", 512))
    max_streams = max(1, (val_tokens.numel() - 1) // max(1, seq_len + 1))
    batch_size = min(int(args.batch_size), max_streams)
    batcher = SequentialStreamBatcher(
        val_tokens,
        seq_len=seq_len,
        batch_size=batch_size,
        output_device=device,
    )

    def autocast_context() -> Any:
        """Return a fresh inference autocast context.

        :return Any: Autocast context or ``nullcontext``.
        """
        if device_type == "cuda":
            return torch.autocast(device_type=device_type, dtype=default_amp_dtype)
        return nullcontext()

    base_loss_parts: list[torch.Tensor] = []
    full_loss_parts: list[torch.Tensor] = []
    no_delta_loss_parts: list[torch.Tensor] = []
    no_base_delta_loss_parts: list[torch.Tensor] = []
    no_trigram_loss_parts: list[torch.Tensor] = []
    base_loss_sum = 0.0
    full_loss_sum = 0.0
    no_delta_loss_sum = 0.0
    no_base_delta_loss_sum = 0.0
    no_trigram_loss_sum = 0.0
    total_tokens = 0
    total_bytes = 0
    state_obj: Optional[Any] = None
    checkpoint_step = int(checkpoint.get("step", -1)) if isinstance(checkpoint, dict) else -1

    with torch.no_grad():
        for _ in range(max(1, int(args.steps))):
            batch, reset_state = batcher.next_batch()
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            if state_obj is None or reset_state:
                state_in = model.initial_state(inputs.size(0), device=device)
            else:
                state_in = model.detach_state(state_obj)

            with autocast_context():
                base_logits = model.base_path_logits(inputs)
                full_logits, next_state = model(inputs, state=state_in, return_state=True)
                with _zero_readout_delta(model) as had_delta:
                    if had_delta:
                        no_delta_logits, _ = model(inputs, state=state_in, return_state=True)
                    else:
                        no_delta_logits = None
                with _zero_base_bigram_delta(model) as had_base_delta:
                    if had_base_delta:
                        no_base_delta_logits, _ = model(inputs, state=state_in, return_state=True)
                    else:
                        no_base_delta_logits = None
                with _disable_trigram_memory(model) as had_trigram:
                    if had_trigram:
                        no_trigram_logits, _ = model(inputs, state=state_in, return_state=True)
                    else:
                        no_trigram_logits = None
            base_loss = cross_entropy_per_token(base_logits, targets)
            full_loss = cross_entropy_per_token(full_logits, targets)
            base_loss_parts.append(base_loss.detach().cpu().flatten())
            full_loss_parts.append(full_loss.detach().cpu().flatten())
            base_loss_sum += float(base_loss.sum().item())
            full_loss_sum += float(full_loss.sum().item())
            if no_delta_logits is not None:
                no_delta_loss = cross_entropy_per_token(no_delta_logits, targets)
                no_delta_loss_parts.append(no_delta_loss.detach().cpu().flatten())
                no_delta_loss_sum += float(no_delta_loss.sum().item())
            if no_base_delta_logits is not None:
                no_base_delta_loss = cross_entropy_per_token(no_base_delta_logits, targets)
                no_base_delta_loss_parts.append(no_base_delta_loss.detach().cpu().flatten())
                no_base_delta_loss_sum += float(no_base_delta_loss.sum().item())
            if no_trigram_logits is not None:
                no_trigram_loss = cross_entropy_per_token(no_trigram_logits, targets)
                no_trigram_loss_parts.append(no_trigram_loss.detach().cpu().flatten())
                no_trigram_loss_sum += float(no_trigram_loss.sum().item())
            total_tokens += int(targets.numel())
            if byte_count_lut is not None:
                total_bytes += int(byte_count_lut[targets.long()].sum().item())
            state_obj = model.detach_state(next_state)

    base_losses = torch.cat(base_loss_parts)
    full_losses = torch.cat(full_loss_parts)
    no_delta_losses = torch.cat(no_delta_loss_parts) if no_delta_loss_parts else None
    no_base_delta_losses = torch.cat(no_base_delta_loss_parts) if no_base_delta_loss_parts else None
    no_trigram_losses = torch.cat(no_trigram_loss_parts) if no_trigram_loss_parts else None
    losses: dict[str, Any] = {
        "base_loss": base_loss_sum / max(1, total_tokens),
        "full_loss": full_loss_sum / max(1, total_tokens),
        "full_gain_nats": (base_loss_sum - full_loss_sum) / max(1, total_tokens),
    }
    if total_bytes > 0:
        losses["base_bpb"] = (base_loss_sum / math.log(2)) / total_bytes
        losses["full_bpb"] = (full_loss_sum / math.log(2)) / total_bytes
        losses["full_gain_bpb"] = losses["base_bpb"] - losses["full_bpb"]
    else:
        losses["base_bpb"] = None
        losses["full_bpb"] = None
        losses["full_gain_bpb"] = None
    if no_delta_losses is not None:
        losses["no_readout_delta_loss"] = no_delta_loss_sum / max(1, total_tokens)
        losses["readout_delta_gain_nats"] = (no_delta_loss_sum - full_loss_sum) / max(
            1, total_tokens
        )
        losses["readout_delta_gain_bpb"] = (
            ((no_delta_loss_sum - full_loss_sum) / math.log(2)) / total_bytes
            if total_bytes > 0
            else None
        )
    if no_base_delta_losses is not None:
        losses["no_base_bigram_delta_loss"] = no_base_delta_loss_sum / max(1, total_tokens)
        losses["base_bigram_delta_gain_nats"] = (no_base_delta_loss_sum - full_loss_sum) / max(
            1, total_tokens
        )
        losses["base_bigram_delta_gain_bpb"] = (
            ((no_base_delta_loss_sum - full_loss_sum) / math.log(2)) / total_bytes
            if total_bytes > 0
            else None
        )
    if no_trigram_losses is not None:
        losses["no_trigram_memory_loss"] = no_trigram_loss_sum / max(1, total_tokens)
        losses["trigram_memory_gain_nats"] = (no_trigram_loss_sum - full_loss_sum) / max(
            1, total_tokens
        )
        losses["trigram_memory_gain_bpb"] = (
            ((no_trigram_loss_sum - full_loss_sum) / math.log(2)) / total_bytes
            if total_bytes > 0
            else None
        )

    payload = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "model": {
            "core_layers": int(model_cfg.get("core_layers", 5)),
            "core_expansion": float(model_cfg.get("core_expansion", 2.0)),
            "branch_temporal_mode": str(model_cfg.get("branch_temporal_mode", "current")),
            "residual_token_gate_mode": str(model_cfg.get("residual_token_gate_mode", "none")),
            "branch_router_mode": str(model_cfg.get("branch_router_mode", "none")),
            "base_bigram_delta": str(model_cfg.get("base_bigram_delta", "none")),
            "trigram_memory": str(trigram_memory_config_value(model_cfg, "none")),
            "residual_readout_delta_rank": int(model_cfg.get("residual_readout_delta_rank", 0)),
            "trainable_parameters": int(model.trainable_parameters),
        },
        "validation": {
            "source": str(data_source),
            "seq_len": seq_len,
            "batch_size": batch_size,
            "steps": int(args.steps),
            "tokens": total_tokens,
            "bytes": total_bytes if total_bytes > 0 else None,
            "coverage_frac": min(1.0, float(total_tokens) / max(1, int(val_tokens.numel()) - 1)),
            "exact_bpb": byte_count_lut is not None,
        },
        "losses": losses,
        "base_loss_buckets": _bucket_rows(base_losses, full_losses, no_delta_losses),
    }

    out_path = Path(args.out).expanduser().resolve() if args.out else run_dir / "diagnostics.json"
    md_path = (
        Path(args.md_out).expanduser().resolve() if args.md_out else run_dir / "diagnostics.md"
    )
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    print(f"Wrote diagnostics JSON: {out_path}")
    print(f"Wrote diagnostics Markdown: {md_path}")
    print(
        "loss: "
        f"base={losses['base_loss']:.6f} full={losses['full_loss']:.6f} "
        f"gain={losses['full_gain_nats']:.6f}"
    )


if __name__ == "__main__":
    main()
