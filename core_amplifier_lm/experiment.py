"""Experiment utilities for local core/amplifier studies.

These helpers keep the local 5090 workflow deterministic and summary-friendly:
- per-run JSON/JSONL artifacts
- hardware/runtime metadata capture
- rebuildable sweep summaries
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

SUMMARY_FIELDS = [
    "run_name",
    "status",
    "phase",
    "git_commit",
    "seed",
    "core_layers",
    "core_expansion",
    "residual_core",
    "residual_core_init",
    "carry_chunks",
    "bptt_chunks",
    "branch_lags",
    "num_blocks",
    "readout_rank",
    "seq_len",
    "batch_size",
    "grad_accum",
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
    "elapsed_sec",
    "steady_state_tokens_per_sec",
    "peak_mem_alloc_mib",
    "peak_mem_reserved_mib",
    "compile_enabled",
    "compile_after",
    "compile_mode",
    "compile_duration_sec",
    "tf32_matmul",
    "tf32_cudnn",
    "matmul_precision",
    "spec_bytes",
    "gzip_spec_bytes",
    "artifact_estimate_bytes",
    "artifact_headroom_bytes",
    "artifact_status",
    "exact_val_bpb",
    "data_path",
    "tokenizer_path",
    "run_dir",
]

ARTIFACT_LIMIT_BYTES = 16_000_000


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    raise TypeError(f"not JSON serializable: {type(obj)}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read JSON, returning an empty dict when missing."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL rows, skipping malformed lines."""
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
    """Quote a command line for reproducible logs."""
    return shlex.join([str(x) for x in argv])


def git_commit(repo_root: str | Path) -> Optional[str]:
    """Return HEAD commit for the repo, if available."""
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
    """Read GPU name/driver from nvidia-smi when available."""
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
    """Return raw and gzip-compressed spec.pt sizes."""
    p = Path(spec_path)
    if not p.exists():
        return None, None
    raw = p.stat().st_size
    gz = len(gzip.compress(p.read_bytes(), compresslevel=9))
    return raw, gz


def estimate_repo_code_bytes(repo_root: str | Path) -> int:
    """Approximate code bytes for the maintained root code path."""
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


def artifact_estimate_bytes(
    *,
    repo_root: str | Path,
    spec_path: str | Path,
) -> Optional[int]:
    """Estimate artifact size as maintained repo Python files plus gzip(spec.pt)."""
    _, gz = spec_size_bytes(spec_path)
    if gz is None:
        return None
    return estimate_repo_code_bytes(repo_root) + gz


def artifact_headroom_bytes(artifact_bytes: Optional[int]) -> Optional[int]:
    """Return remaining artifact budget headroom against the 16 MB cap."""
    if artifact_bytes is None:
        return None
    return int(ARTIFACT_LIMIT_BYTES) - int(artifact_bytes)


def artifact_status(artifact_bytes: Optional[int]) -> str:
    """Classify artifact budget usage for summary/reporting."""
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
    """Estimate steady-state throughput from recorded train rows."""
    train_rows = [row for row in rows if row.get("kind") == "train"]
    if not train_rows:
        return None

    def _value(row: dict[str, Any]) -> Optional[float]:
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
    """Return best and last eval rows from a metrics stream."""
    eval_rows = [row for row in rows if row.get("kind") == "eval"]
    if not eval_rows:
        return {}, {}

    def _loss(row: dict[str, Any]) -> float:
        try:
            return float(row.get("val_loss", math.inf))
        except (TypeError, ValueError):
            return math.inf

    best = min(eval_rows, key=_loss)
    last = eval_rows[-1]
    return best, last


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def summarize_run_dir(run_dir: str | Path) -> dict[str, str]:
    """Build one flat summary row from per-run artifacts."""
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
    compile_cfg = runtime.get("compile", {})

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
        "core_layers": _stringify(model.get("core_layers")),
        "core_expansion": _stringify(model.get("core_expansion")),
        "residual_core": _stringify(model.get("residual_core")),
        "residual_core_init": _stringify(model.get("residual_core_init")),
        "carry_chunks": _stringify(training.get("carry_chunks")),
        "bptt_chunks": _stringify(training.get("bptt_chunks")),
        "branch_lags": ",".join(str(x) for x in model.get("branch_lags", [])),
        "num_blocks": _stringify(model.get("num_blocks")),
        "readout_rank": _stringify(model.get("readout_rank")),
        "seq_len": _stringify(training.get("seq_len")),
        "batch_size": _stringify(training.get("batch_size")),
        "grad_accum": _stringify(training.get("grad_accum")),
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
        "elapsed_sec": _stringify(results.get("elapsed_sec")),
        "steady_state_tokens_per_sec": _stringify(results.get("steady_state_tokens_per_sec")),
        "peak_mem_alloc_mib": _stringify(results.get("peak_mem_alloc_mib")),
        "peak_mem_reserved_mib": _stringify(results.get("peak_mem_reserved_mib")),
        "compile_enabled": _stringify(compile_cfg.get("enabled")),
        "compile_after": _stringify(compile_cfg.get("compile_after")),
        "compile_mode": _stringify(compile_cfg.get("compile_mode")),
        "compile_duration_sec": _stringify(results.get("compile_duration_sec")),
        "tf32_matmul": _stringify(system.get("tf32_matmul")),
        "tf32_cudnn": _stringify(system.get("tf32_cudnn")),
        "matmul_precision": _stringify(system.get("float32_matmul_precision")),
        "spec_bytes": _stringify(results.get("spec_bytes") or spec.get("spec_bytes")),
        "gzip_spec_bytes": _stringify(
            results.get("gzip_spec_bytes") or spec.get("gzip_spec_bytes")
        ),
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
    """Collect flat summary rows for all run directories under a root."""
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
    """Write a deterministic summary TSV."""
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
    """Render a compact Markdown summary for human inspection."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    completed = [row for row in rows if row.get("status") == "completed"]

    def _score(row: dict[str, str]) -> float:
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
    """Return the CUDA device index when applicable."""
    if device.type != "cuda":
        return None
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def current_peak_memory_mib(device: torch.device) -> tuple[Optional[float], Optional[float]]:
    """Return peak allocated/reserved memory in MiB."""
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
    """Build reproducible command metadata for a Python entrypoint."""
    full_argv = [sys.executable, str(script_path), *argv]
    return {
        "argv": full_argv,
        "shell": shell_join(full_argv),
        "cwd": os.getcwd(),
    }
