"""Experiment utilities for local core/amplifier studies.

These helpers keep the local 5090 workflow deterministic and summary-friendly:
- per-run JSON/JSONL artifacts
- hardware/runtime metadata capture
- rebuildable sweep summaries
"""

from __future__ import annotations

import gzip
import io
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from .config import trigram_memory_config_value

SUMMARY_FIELDS = [
    "run_name",
    "status",
    "phase",
    "git_commit",
    "seed",
    "core_dim",
    "core_layers",
    "core_inner_dim",
    "recurrent_cells",
    "core_expansion",
    "residual_core",
    "residual_core_init",
    "branch_temporal_mode",
    "branch_temporal_lag_scale",
    "residual_token_gate_mode",
    "branch_router_mode",
    "base_bigram_delta",
    "trigram_memory",
    "trigram_top_k",
    "residual_readout_delta_rank",
    "residual_readout_delta_init_std",
    "carry_chunks",
    "bptt_chunks",
    "branch_lags",
    "num_blocks",
    "readout_rank",
    "seq_len",
    "batch_size",
    "grad_accum",
    "local_step_tokens",
    "effective_step_tokens",
    "learning_rate",
    "min_lr",
    "warmup_steps",
    "lr_hold_steps",
    "weight_decay",
    "planned_steps",
    "planned_train_tokens",
    "processed_tokens",
    "best_step",
    "best_val_loss",
    "best_val_bpb",
    "last_eval_step",
    "last_val_loss",
    "last_val_bpb",
    "last_eval_tokens",
    "last_eval_bytes",
    "last_eval_coverage_denominator_tokens",
    "last_eval_coverage_frac",
    "last_eval_full_coverage",
    "elapsed_sec",
    "steady_state_tokens_per_sec",
    "peak_mem_alloc_mib",
    "peak_mem_reserved_mib",
    "compile_enabled",
    "compile_after",
    "compile_mode",
    "compile_duration_sec",
    "scan_backend",
    "gradient_checkpointing",
    "torch_version",
    "cuda_version",
    "gpu_name",
    "driver_version",
    "gpu_total_memory_mib",
    "tf32_matmul",
    "tf32_cudnn",
    "matmul_precision",
    "blas_prefer_cublaslt",
    "repo_code_bytes",
    "spec_bytes",
    "gzip_spec_bytes",
    "trainable_int8_zlib_bytes",
    "artifact_estimate_bytes",
    "artifact_headroom_bytes",
    "artifact_status",
    "exact_val_bpb",
    "data_path",
    "tokenizer_path",
    "run_dir",
]

ARTIFACT_LIMIT_BYTES = 16_000_000

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CORE_AMP_CONTROL_TENSOR_NAME_PATTERNS",
        (
            "_h0_raw,resid_logit,block_gain,branch_scale,branch_bias,"
            "readout_branch_scale,residual_log_scale,base_bigram_delta_log_scale,"
            "trigram_log_scale,residual_readout_delta_log_scale,logit_bias,norm.scale"
        ),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CORE_AMP_INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = int(os.environ.get("CORE_AMP_INT8_KEEP_FLOAT_MAX_NUMEL", "65536"))
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = float(os.environ.get("CORE_AMP_INT8_CLIP_PERCENTILE", "99.99984"))
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def _json_default(obj: Any) -> Any:
    """Serialize paths and dtypes for JSON output.

    Returns:
        Any: JSON-serializable fallback value.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    raise TypeError(f"not JSON serializable: {type(obj)}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting.

    :param str | Path path: Output file path.
    :param dict[str, Any] payload: JSON-serializable payload.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file.

    :param str | Path path: Output file path.
    :param dict[str, Any] payload: JSON-serializable payload.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read JSON, returning an empty dict when missing.

    :param str | Path path: Input file path.
    :return dict[str, Any]: Parsed JSON content or an empty dict.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL rows, skipping malformed lines.

    :param str | Path path: Input file path.
    :return list[dict[str, Any]]: Parsed rows, excluding malformed lines.
    """
    p = Path(path)
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def shell_join(argv: Iterable[str]) -> str:
    """Quote a command line for reproducible logs.

    :param Iterable[str] argv: Command arguments.
    :return str: Shell-quoted command string.
    """
    return shlex.join([str(x) for x in argv])


def git_commit(repo_root: str | Path) -> Optional[str]:
    """Return HEAD commit for the repo, if available.

    :param str | Path repo_root: Repository root directory.
    :return str | None: HEAD commit SHA or None if unavailable.
    """
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    out = res.stdout.strip()
    return out or None


def nvidia_smi_metadata(device_index: Optional[int]) -> dict[str, Any]:
    """Read GPU name and driver version from nvidia-smi.

    :param Optional[int] device_index: CUDA device index to query.
    :return dict[str, Any]: GPU metadata when available.
    """
    if device_index is None:
        return {}
    cmd = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        res = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return {}
    line = res.stdout.strip()
    if not line:
        return {}
    parts = [x.strip() for x in line.split(",", maxsplit=1)]
    payload: dict[str, Any] = {}
    if parts:
        payload["gpu_name"] = parts[0]
    if len(parts) > 1:
        payload["driver_version"] = parts[1]
    return payload


def spec_size_bytes(spec_path: str | Path) -> tuple[Optional[int], Optional[int]]:
    """Return raw and gzip-compressed spec.pt sizes.

    :param str | Path spec_path: Path to spec.pt.
    :return tuple[Optional[int], Optional[int]]: Raw and gzip-compressed sizes.
    """
    p = Path(spec_path)
    if not p.exists():
        return None, None
    raw = p.stat().st_size
    gz = len(gzip.compress(p.read_bytes(), compresslevel=9))
    return raw, gz


def estimate_repo_code_bytes(repo_root: str | Path) -> int:
    """Approximate code bytes for the maintained root code path.

    :param str | Path repo_root: Repository root directory.
    :return int: Total bytes of tracked Python source under the root path.
    """
    root = Path(repo_root)
    total = 0
    for path in root.rglob("*.py"):
        if any(
            part in {"records", "experiments", ".git", ".ruff_cache", "__pycache__"}
            for part in path.parts
        ):
            continue
        total += path.stat().st_size
    return total


def tensor_nbytes(t: torch.Tensor) -> int:
    """Return the raw storage bytes for a tensor.

    :param torch.Tensor t: Tensor to measure.
    :return int: Tensor storage size in bytes.
    """
    return int(t.numel()) * int(t.element_size())


def _keep_float_tensor(
    name: str,
    tensor: torch.Tensor,
    passthrough_orig_dtypes: dict[str, str],
) -> torch.Tensor:
    """Keep a tensor in floating-point form for the int8 export path.

    :param str name: Parameter name being exported.
    :param torch.Tensor tensor: Tensor to preserve or downcast.
    :param dict[str, str] passthrough_orig_dtypes: Original dtypes for passthrough values.
    Returns:
        torch.Tensor: Tensor prepared for the int8 export payload.
    """
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return tensor.float().contiguous()
    if tensor.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(tensor.dtype).removeprefix("torch.")
        return tensor.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return tensor


def _quantize_float_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating-point tensor with int8 payload and scale metadata.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scale tensor.
    """
    tensor_f32 = tensor.float()
    if tensor_f32.ndim == 2:
        clip_abs = (
            torch.quantile(tensor_f32.abs(), INT8_CLIP_Q, dim=1)
            if tensor_f32.numel()
            else torch.empty((tensor_f32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(
            torch.minimum(tensor_f32, clip_abs[:, None]),
            -clip_abs[:, None],
        )
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        quantized = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
        return quantized.contiguous(), scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = (
        float(torch.quantile(tensor_f32.abs().flatten(), INT8_CLIP_Q).item())
        if tensor_f32.numel()
        else 0.0
    )
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    quantized = torch.clamp(
        torch.round(torch.clamp(tensor_f32, -clip_abs, clip_abs) / scale),
        -127,
        127,
    ).to(torch.int8)
    return quantized.contiguous(), scale


def quantize_state_dict_int8(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, object], dict[str, int]]:
    """Mirror the record-side clean int8 export format for trainable weights.

    :param dict[str, torch.Tensor] state_dict: Trainable state dict to export.
    :return tuple[dict[str, object], dict[str, int]]: Quantized payload and export stats.
    """
    quantized: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, torch.Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = {
        "param_count": 0,
        "num_tensors": 0,
        "num_float_tensors": 0,
        "num_nonfloat_tensors": 0,
        "baseline_tensor_bytes": 0,
        "int8_payload_bytes": 0,
    }

    for name, tensor in state_dict.items():
        detached = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(detached.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(detached)

        if not detached.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = detached
            stats["int8_payload_bytes"] += tensor_nbytes(detached)
            continue

        if detached.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = _keep_float_tensor(name, detached, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        quantized_tensor, scale_tensor = _quantize_float_tensor(detached)
        if scale_tensor.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = quantized_tensor
        scales[name] = scale_tensor
        dtypes[name] = str(detached.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(quantized_tensor) + tensor_nbytes(scale_tensor)

    payload: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        payload["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        payload["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return payload, stats


def serialize_trainable_int8_zlib(
    state_dict: dict[str, torch.Tensor],
) -> tuple[bytes, dict[str, int]]:
    """Serialize a trainable state dict using the record-side int8+zlib convention.

    :param dict[str, torch.Tensor] state_dict: Trainable state dict to export.
    :return tuple[bytes, dict[str, int]]: Compressed payload and serialization stats.
    """
    quantized, stats = quantize_state_dict_int8(state_dict)
    buf = io.BytesIO()
    torch.save(quantized, buf)
    raw = buf.getvalue()
    blob = zlib.compress(raw, level=9)
    out_stats = dict(stats)
    out_stats["int8_serialized_bytes"] = len(raw)
    out_stats["int8_zlib_bytes"] = len(blob)
    return blob, out_stats


def export_trainable_int8_zlib(
    path: str | Path,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Write the compressed trainable artifact blob and return serialization stats.

    :param str | Path path: Output file path.
    :param dict[str, torch.Tensor] state_dict: Trainable state dict to export.
    :return dict[str, int]: Serialization statistics.
    """
    blob, stats = serialize_trainable_int8_zlib(state_dict)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    return stats


def trainable_int8_zlib_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    """Return the compressed trainable artifact bytes using the record-side export path.

    :param dict[str, torch.Tensor] state_dict: Trainable state dict to export.
    :return int: Compressed artifact size in bytes.
    """
    blob, _stats = serialize_trainable_int8_zlib(state_dict)
    return len(blob)


def artifact_estimate_bytes(
    *,
    repo_root: str | Path,
    spec_path: str | Path,
    trainable_payload_bytes: Optional[int] = None,
    repo_code_bytes: Optional[int] = None,
) -> Optional[int]:
    """Estimate artifact size as code bytes plus frozen spec plus controller payload.

    :param str | Path repo_root: Repository root directory.
    :param str | Path spec_path: Path to spec.pt.
    :param Optional[int] trainable_payload_bytes: Optional trainable artifact bytes.
    :param Optional[int] repo_code_bytes: Optional precomputed code byte count.
    :return Optional[int]: Estimated total artifact bytes, or None if unavailable.
    """
    _, gz = spec_size_bytes(spec_path)
    if gz is None:
        return None
    code_bytes = (
        estimate_repo_code_bytes(repo_root) if repo_code_bytes is None else int(repo_code_bytes)
    )
    total = code_bytes + gz
    if trainable_payload_bytes is not None:
        total += int(trainable_payload_bytes)
    return total


def artifact_headroom_bytes(artifact_bytes: Optional[int]) -> Optional[int]:
    """Return remaining artifact budget headroom against the 16 MB cap.

    :param Optional[int] artifact_bytes: Estimated artifact bytes.
    :return Optional[int]: Remaining bytes under the cap, or None if unknown.
    """
    if artifact_bytes is None:
        return None
    return int(ARTIFACT_LIMIT_BYTES) - int(artifact_bytes)


def artifact_status(artifact_bytes: Optional[int]) -> str:
    """Classify artifact budget usage for summary/reporting.

    :param Optional[int] artifact_bytes: Estimated artifact bytes.
    :return str: Budget classification string.
    """
    if artifact_bytes is None:
        return "UNKNOWN"
    if int(artifact_bytes) > ARTIFACT_LIMIT_BYTES:
        return "OVER_LIMIT"
    if int(artifact_bytes) == ARTIFACT_LIMIT_BYTES:
        return "EXACT_LIMIT"
    return "LEFT_ON_TABLE"


def compute_steady_state_tokens_per_sec(
    rows: Iterable[dict[str, Any]],
    *,
    compile_trigger_step: Optional[int] = None,
) -> Optional[float]:
    """Estimate steady-state throughput from recorded train rows.

    :param Iterable[dict[str, Any]] rows: Metric rows to inspect.
    :param Optional[int] compile_trigger_step: Optional compile warmup cutoff.
    :return Optional[float]: Median steady-state tokens per second, or None.
    """
    train_rows = [row for row in rows if row.get("kind") == "train"]
    if not train_rows:
        return None

    def _value(row: dict[str, Any]) -> Optional[float]:
        """Parse a finite positive tokens/sec value from a row.

        Returns:
            Optional[float]: Parsed tokens/sec value or None.
        """
        val = row.get("tokens_per_sec")
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) and f > 0 else None

    if compile_trigger_step is not None:
        post_compile = [
            row
            for row in train_rows
            if int(row.get("step", -1)) > int(compile_trigger_step) and _value(row) is not None
        ]
        if len(post_compile) >= 3:
            train_rows = post_compile

    usable = [row for row in train_rows if _value(row) is not None]
    if not usable:
        return None
    tail = usable[len(usable) // 2 :] if len(usable) >= 4 else usable
    values = [_value(row) for row in tail]
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(statistics.median(clean))


def best_and_last_eval(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return best and last eval rows from a metrics stream.

    :param Iterable[dict[str, Any]] rows: Metric rows to inspect.
    :return tuple[dict[str, Any], dict[str, Any]]: Best and last eval rows.
    """
    eval_rows = [row for row in rows if row.get("kind") == "eval"]
    if not eval_rows:
        return {}, {}

    def _loss(row: dict[str, Any]) -> float:
        """Parse a comparable validation loss from a row.

        Returns:
            float: Comparable validation loss.
        """
        try:
            return float(row.get("val_loss", math.inf))
        except (TypeError, ValueError):
            return math.inf

    best = min(eval_rows, key=_loss)
    last = eval_rows[-1]
    return best, last


def _stringify(value: Any) -> str:
    """Convert a value into a flat summary string.

    Returns:
        str: Stringified value or an empty string for ``None``.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_env_bool(value: Any) -> Optional[bool]:
    """Parse a boolean-like environment value.

    Returns:
        Optional[bool]: Parsed boolean or None if the value is unknown.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _resolved_core_inner_dim(model: dict[str, Any]) -> Optional[int]:
    """Compute the resolved recurrent inner dimension from model config.

    :param dict[str, Any] model: Resolved model config.
    :return Optional[int]: Explicit ``core_inner_dim`` or ``core_dim * core_expansion``.
    """
    raw_inner = model.get("core_inner_dim")
    if raw_inner is not None:
        try:
            return int(raw_inner)
        except (TypeError, ValueError):
            return None
    try:
        core_dim = int(model["core_dim"])
        core_expansion = float(model["core_expansion"])
    except (KeyError, TypeError, ValueError):
        return None
    return int(core_dim * core_expansion)


def _resolved_recurrent_cells(
    model: dict[str, Any], core_inner_dim: Optional[int]
) -> Optional[int]:
    """Compute the total stacked recurrent cell count from model config.

    :param dict[str, Any] model: Resolved model config.
    :param Optional[int] core_inner_dim: Resolved recurrent inner dimension.
    :return Optional[int]: ``core_layers * core_inner_dim`` when both are known.
    """
    if core_inner_dim is None:
        return None
    try:
        core_layers = int(model["core_layers"])
    except (KeyError, TypeError, ValueError):
        return None
    return core_layers * core_inner_dim


def summarize_run_dir(run_dir: str | Path) -> dict[str, str]:
    """Build one flat summary row from per-run artifacts.

    :param str | Path run_dir: Run directory to summarize.
    :return dict[str, str]: Flat summary row.
    """
    root = Path(run_dir)
    resolved = read_json(root / "resolved_config.json")
    results = read_json(root / "run_results.json")
    metadata = read_json(root / "run_metadata.json")
    metrics = read_jsonl(root / "metrics.jsonl")
    best, last = best_and_last_eval(metrics)

    training = resolved.get("training", {})
    model = resolved.get("model", {})
    runtime = resolved.get("runtime", {})
    data = resolved.get("data", {})
    spec = resolved.get("spec", {})
    system = metadata.get("system", {})
    env = metadata.get("env", {})
    compile_cfg = runtime.get("compile", {})
    core_inner_dim = _resolved_core_inner_dim(model)
    recurrent_cells = _resolved_recurrent_cells(model, core_inner_dim)

    status = "partial"
    if results.get("completed"):
        status = "completed"
    elif (root / "spec.pt").exists():
        status = "spec_only"
    if not (root / "config.json").exists():
        status = "missing"

    row: dict[str, str] = {
        "run_name": _stringify(resolved.get("run_name") or metadata.get("run_name") or root.name),
        "status": status,
        "phase": _stringify(resolved.get("phase")),
        "git_commit": _stringify(metadata.get("git_commit")),
        "seed": _stringify(resolved.get("seed")),
        "core_dim": _stringify(model.get("core_dim")),
        "core_layers": _stringify(model.get("core_layers")),
        "core_inner_dim": _stringify(core_inner_dim),
        "recurrent_cells": _stringify(recurrent_cells),
        "core_expansion": _stringify(model.get("core_expansion")),
        "residual_core": _stringify(model.get("residual_core")),
        "residual_core_init": _stringify(model.get("residual_core_init")),
        "branch_temporal_mode": _stringify(model.get("branch_temporal_mode")),
        "branch_temporal_lag_scale": _stringify(model.get("branch_temporal_lag_scale")),
        "residual_token_gate_mode": _stringify(model.get("residual_token_gate_mode")),
        "branch_router_mode": _stringify(model.get("branch_router_mode")),
        "base_bigram_delta": _stringify(model.get("base_bigram_delta")),
        "trigram_memory": _stringify(trigram_memory_config_value(model)),
        "trigram_top_k": _stringify(model.get("trigram_top_k")),
        "residual_readout_delta_rank": _stringify(model.get("residual_readout_delta_rank")),
        "residual_readout_delta_init_std": _stringify(model.get("residual_readout_delta_init_std")),
        "carry_chunks": _stringify(training.get("carry_chunks")),
        "bptt_chunks": _stringify(training.get("bptt_chunks")),
        "branch_lags": ",".join(str(x) for x in model.get("branch_lags", [])),
        "num_blocks": _stringify(model.get("num_blocks")),
        "readout_rank": _stringify(model.get("readout_rank")),
        "seq_len": _stringify(training.get("seq_len")),
        "batch_size": _stringify(training.get("batch_size")),
        "grad_accum": _stringify(training.get("grad_accum")),
        "local_step_tokens": _stringify(training.get("local_step_tokens")),
        "effective_step_tokens": _stringify(training.get("effective_step_tokens")),
        "learning_rate": _stringify(training.get("learning_rate")),
        "min_lr": _stringify(training.get("min_lr")),
        "warmup_steps": _stringify(training.get("warmup_steps")),
        "lr_hold_steps": _stringify(training.get("lr_hold_steps")),
        "weight_decay": _stringify(training.get("weight_decay")),
        "planned_steps": _stringify(training.get("num_steps")),
        "planned_train_tokens": _stringify(training.get("planned_train_tokens")),
        "processed_tokens": _stringify(results.get("seen_train_tokens")),
        "best_step": _stringify(best.get("step") or results.get("best_step")),
        "best_val_loss": _stringify(best.get("val_loss") or results.get("best_val_loss")),
        "best_val_bpb": _stringify(best.get("val_bpb") or results.get("best_val_bpb")),
        "last_eval_step": _stringify(last.get("step") or results.get("last_eval_step")),
        "last_val_loss": _stringify(last.get("val_loss") or results.get("last_val_loss")),
        "last_val_bpb": _stringify(last.get("val_bpb") or results.get("last_val_bpb")),
        "last_eval_tokens": _stringify(last.get("eval_tokens")),
        "last_eval_bytes": _stringify(last.get("eval_bytes")),
        "last_eval_coverage_denominator_tokens": _stringify(
            last.get("eval_coverage_denominator_tokens")
            or results.get("last_eval_coverage_denominator_tokens")
        ),
        "last_eval_coverage_frac": _stringify(last.get("eval_coverage_frac")),
        "last_eval_full_coverage": _stringify(last.get("eval_full_coverage")),
        "elapsed_sec": _stringify(results.get("elapsed_sec")),
        "steady_state_tokens_per_sec": _stringify(results.get("steady_state_tokens_per_sec")),
        "peak_mem_alloc_mib": _stringify(results.get("peak_mem_alloc_mib")),
        "peak_mem_reserved_mib": _stringify(results.get("peak_mem_reserved_mib")),
        "compile_enabled": _stringify(compile_cfg.get("enabled")),
        "compile_after": _stringify(compile_cfg.get("compile_after")),
        "compile_mode": _stringify(compile_cfg.get("compile_mode")),
        "compile_duration_sec": _stringify(results.get("compile_duration_sec")),
        "scan_backend": _stringify(runtime.get("scan_backend_active")),
        "gradient_checkpointing": _stringify(
            runtime.get("gradient_checkpointing", training.get("gradient_checkpointing"))
        ),
        "torch_version": _stringify(system.get("torch_version")),
        "cuda_version": _stringify(system.get("cuda_version")),
        "gpu_name": _stringify(system.get("gpu_name")),
        "driver_version": _stringify(system.get("driver_version")),
        "gpu_total_memory_mib": _stringify(system.get("gpu_total_memory_mib")),
        "tf32_matmul": _stringify(system.get("tf32_matmul")),
        "tf32_cudnn": _stringify(system.get("tf32_cudnn")),
        "matmul_precision": _stringify(system.get("float32_matmul_precision")),
        "blas_prefer_cublaslt": _stringify(_parse_env_bool(env.get("TORCH_BLAS_PREFER_CUBLASLT"))),
        "repo_code_bytes": _stringify(
            results.get("repo_code_bytes") or spec.get("repo_code_bytes")
        ),
        "spec_bytes": _stringify(results.get("spec_bytes") or spec.get("spec_bytes")),
        "gzip_spec_bytes": _stringify(
            results.get("gzip_spec_bytes") or spec.get("gzip_spec_bytes")
        ),
        "trainable_int8_zlib_bytes": _stringify(results.get("trainable_int8_zlib_bytes")),
        "artifact_estimate_bytes": _stringify(results.get("artifact_estimate_bytes")),
        "artifact_headroom_bytes": _stringify(results.get("artifact_headroom_bytes")),
        "artifact_status": _stringify(results.get("artifact_status")),
        "exact_val_bpb": _stringify(runtime.get("exact_val_bpb")),
        "data_path": _stringify(data.get("source")),
        "tokenizer_path": _stringify(resolved.get("tokenizer_path")),
        "run_dir": str(root),
    }
    return row


def collect_summary_rows(root: str | Path) -> list[dict[str, str]]:
    """Collect flat summary rows for all run directories under a root.

    :param str | Path root: Directory containing run subdirectories.
    :return list[dict[str, str]]: Summary rows for discovered runs.
    """
    base = Path(root)
    rows: list[dict[str, str]] = []
    if not base.exists():
        return rows
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if run_dir.name.startswith("_") or run_dir.name == "data":
            continue
        if not (run_dir / "config.json").exists():
            continue
        rows.append(summarize_run_dir(run_dir))
    return rows


def write_summary_tsv(rows: list[dict[str, str]], out_path: str | Path) -> None:
    """Write a deterministic summary TSV.

    :param list[dict[str, str]] rows: Summary rows to write.
    :param str | Path out_path: Output file path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("\t".join(SUMMARY_FIELDS) + "\n")
        for row in rows:
            f.write("\t".join(row.get(field, "") for field in SUMMARY_FIELDS) + "\n")


def write_summary_markdown(
    rows: list[dict[str, str]],
    out_path: str | Path,
    *,
    title: str = "Sweep Summary",
) -> None:
    """Render a compact Markdown summary for human inspection.

    :param list[dict[str, str]] rows: Summary rows to render.
    :param str | Path out_path: Output file path.
    :param str title: Markdown title line.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    completed = [row for row in rows if row.get("status") == "completed"]

    def _score(row: dict[str, str]) -> float:
        """Parse a sortable validation score from a summary row.

        Returns:
            float: Comparable score for sorting.
        """
        try:
            return float(row.get("best_val_bpb", "") or math.inf)
        except ValueError:
            return math.inf

    ranked = sorted(completed, key=_score)
    lines = [
        f"# {title}",
        "",
        f"Runs discovered: {len(rows)}",
        f"Completed runs: {len(completed)}",
        "",
    ]
    if ranked:
        lines.extend(
            [
                "| run | best val_bpb | best val_loss | steady tok/s | spec gzip bytes | notes |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in ranked[:12]:
            notes = []
            if row.get("num_blocks"):
                notes.append(f"blocks={row['num_blocks']}")
            if row.get("readout_rank"):
                notes.append(f"rank={row['readout_rank']}")
            if row.get("core_layers"):
                notes.append(f"layers={row['core_layers']}")
            if row.get("core_expansion"):
                notes.append(f"exp={row['core_expansion']}")
            if row.get("residual_token_gate_mode") and row["residual_token_gate_mode"] != "none":
                notes.append(f"gate={row['residual_token_gate_mode']}")
            if row.get("branch_router_mode") and row["branch_router_mode"] != "none":
                notes.append(f"router={row['branch_router_mode']}")
            if row.get("scan_backend"):
                notes.append(f"scan={row['scan_backend']}")
            lines.append(
                "| {run} | {bpb} | {loss} | {tok} | {spec} | {notes} |".format(
                    run=row.get("run_name", ""),
                    bpb=row.get("best_val_bpb", ""),
                    loss=row.get("best_val_loss", ""),
                    tok=row.get("steady_state_tokens_per_sec", ""),
                    spec=row.get("gzip_spec_bytes", ""),
                    notes=", ".join(notes),
                )
            )
        lines.append("")
    else:
        lines.append("No completed runs yet.")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def runtime_device_index(device: torch.device) -> Optional[int]:
    """Return the CUDA device index when applicable.

    :param torch.device device: Device to inspect.
    :return Optional[int]: CUDA device index or None for non-CUDA devices.
    """
    if device.type != "cuda":
        return None
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def current_peak_memory_mib(device: torch.device) -> tuple[Optional[float], Optional[float]]:
    """Return peak allocated and reserved memory in MiB.

    :param torch.device device: Device to inspect.
    :return tuple[Optional[float], Optional[float]]: Peak allocated and reserved MiB.
    """
    if device.type != "cuda":
        return None, None
    idx = runtime_device_index(device)
    assert idx is not None
    alloc = torch.cuda.max_memory_allocated(idx) / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved(idx) / (1024 * 1024)
    return float(alloc), float(reserved)


def reset_peak_memory(device: torch.device) -> None:
    """Reset CUDA peak memory stats when available."""
    if device.type != "cuda":
        return
    idx = runtime_device_index(device)
    assert idx is not None
    torch.cuda.reset_peak_memory_stats(idx)


def command_context(script_path: str | Path, argv: list[str]) -> dict[str, Any]:
    """Build reproducible command metadata for a Python entrypoint.

    :param str | Path script_path: Entry point script path.
    :param list[str] argv: Command-line arguments.
    :return dict[str, Any]: Reproducible command metadata.
    """
    full_argv = [sys.executable, str(script_path), *argv]
    return {
        "argv": full_argv,
        "shell": shell_join(full_argv),
        "cwd": os.getcwd(),
    }
