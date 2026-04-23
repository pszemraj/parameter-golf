#!/usr/bin/env python3
"""Screen artifact-size proxy metrics for HGDN architecture candidates.

This helper is intentionally CPU-only. It mirrors the trainer's dtype restore
and quantized artifact audit so candidate families can be shortlisted before
launching real training runs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from _repo_bootstrap import ensure_repo_root_on_sys_path

REPO_ROOT = ensure_repo_root_on_sys_path()

from hgdn_runtime_utils import (  # noqa: E402
    restore_low_dim_params_to_fp32,
    serialize_quantized_state_dict_int8,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "hgdn" / "current_winner_retune.toml"
TRAINER_PATH = REPO_ROOT / "train_gpt_hybrid.py"
PROFILE_ROOT = REPO_ROOT / "profiles" / "arch_size"
DEFAULT_ARTIFACT_LIMIT_BYTES = 16_000_000

DEFAULT_MODEL_KWARGS: dict[str, Any] = {
    "vocab_size": 1024,
    "num_layers": 16,
    "d_model": 384,
    "attn_heads": 8,
    "attn_kv_heads": 4,
    "gdn_n_heads": 8,
    "gdn_head_k_dim": 48,
    "gdn_expand_v": 1.0,
    "gdn_allow_neg_eigval": True,
    "gdn_conv_size": 4,
    "gdn_use_q_conv": True,
    "gdn_use_k_conv": True,
    "gdn_use_v_conv": True,
    "gdn_use_packed_qkv_conv": True,
    "gdn_use_packed_qkv_proj": True,
    "gdn_conv_output_contiguous": True,
    "gdn_q_conv_output_contiguous": True,
    "gdn_k_conv_output_contiguous": True,
    "gdn_v_conv_output_contiguous": True,
    "gdn_gates_fp32": True,
    "gdn_output_norm_fp32": True,
    "mlp_mult": 3.25,
    "leaky_slope": 0.5,
    "gdn_ratio": 1,
    "block_pattern": None,
    "rope_base": 10000.0,
    "qk_gain_init": 1.5,
    "logit_softcap": 30.0,
    "tie_embeddings": True,
    "tied_embed_init_std": 0.005,
    "norm_style": "pre",
    "residual_alpha": 1.0,
}

ALLOWED_CONFIG_KEYS = set(DEFAULT_MODEL_KWARGS) | {
    "artifact_limit_bytes",
    "gdn_control_proj_fp32",
    "gdn_w_g_optimizer",
}


@lru_cache(maxsize=1)
def load_project_bindings() -> tuple[Any, Any, Any]:
    """Load repo-local modules after bootstrapping repo imports.

    :return tuple[Any, Any, Any]: `HybridGPT`,
        `restore_low_dim_params_to_fp32`, and
        `serialize_quantized_state_dict_int8`.
    """
    from model import HybridGPT

    return (
        HybridGPT,
        restore_low_dim_params_to_fp32,
        serialize_quantized_state_dict_int8,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="TOML config with [base] and [[candidates]] tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Defaults to profiles/arch_size/<config-stem>.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=20,
        help="Maximum number of candidates to print to stdout, defaults to 20.",
    )
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the base model config and candidate overrides from TOML.

    :param Path path: Config path.
    :raises KeyError: If required TOML sections are missing.
    :raises ValueError: If a config key is unsupported.
    :return tuple[dict[str, Any], list[dict[str, Any]]]: Base config and candidate list.
    """

    with path.open("rb") as fh:
        data = tomllib.load(fh)
    if "base" not in data:
        raise KeyError(f"Missing [base] table in {path}")
    if "candidates" not in data:
        raise KeyError(f"Missing [[candidates]] tables in {path}")
    base = dict(data["base"])
    candidates = list(data["candidates"])
    unknown = (set(base) | {k for row in candidates for k in row}) - (
        ALLOWED_CONFIG_KEYS | {"name", "notes"}
    )
    if unknown:
        raise ValueError(f"Unsupported config keys in {path}: {sorted(unknown)}")
    return base, candidates


def build_model_kwargs(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Merge default, base, and candidate-specific model kwargs.

    :param dict[str, Any] base: Base config values.
    :param dict[str, Any] override: Candidate-specific overrides.
    :return dict[str, Any]: Model kwargs ready for `HybridGPT`.
    """

    kwargs = dict(DEFAULT_MODEL_KWARGS)
    for source in (base, override):
        for key, value in source.items():
            if key in DEFAULT_MODEL_KWARGS:
                kwargs[key] = value
    return kwargs


def format_int(value: int) -> str:
    """Format an integer with thousands separators.

    :param int value: Integer to format.
    :return str: Formatted integer string.
    """

    return f"{value:,}"


def format_delta(value: float) -> str:
    """Format a percentage delta.

    :param float value: Percent delta.
    :return str: Signed percentage string.
    """

    return f"{value:+.2f}%"


def candidate_metrics(
    name: str,
    base_cfg: dict[str, Any],
    override: dict[str, Any],
    *,
    code_bytes: int,
) -> dict[str, Any]:
    """Compute size metrics for a single architecture candidate.

    :param str name: Candidate label.
    :param dict[str, Any] base_cfg: Base config table.
    :param dict[str, Any] override: Candidate overrides.
    :param int code_bytes: Submission code bytes from `train_gpt_hybrid.py`.
    :return dict[str, Any]: Metric row for the candidate.
    """

    (
        HybridGPT,
        restore_low_dim_params_to_fp32,
        serialize_quantized_state_dict_int8,
    ) = load_project_bindings()
    model_kwargs = build_model_kwargs(base_cfg, override)
    gdn_control_proj_fp32 = bool(
        override.get(
            "gdn_control_proj_fp32",
            base_cfg.get("gdn_control_proj_fp32", True),
        )
    )
    gdn_w_g_optimizer = str(
        override.get(
            "gdn_w_g_optimizer",
            base_cfg.get("gdn_w_g_optimizer", "scalar"),
        )
    )
    artifact_limit_bytes = int(
        override.get(
            "artifact_limit_bytes",
            base_cfg.get("artifact_limit_bytes", DEFAULT_ARTIFACT_LIMIT_BYTES),
        )
    )

    model = HybridGPT(**model_kwargs).bfloat16()
    restore_low_dim_params_to_fp32(
        model,
        gdn_control_proj_fp32=gdn_control_proj_fp32,
        gdn_w_g_optimizer=gdn_w_g_optimizer,
    )

    state_dict = model.state_dict()
    raw_buf = io.BytesIO()
    torch.save(state_dict, raw_buf)
    raw_model_bytes = len(raw_buf.getvalue())
    _, quant_blob, audit = serialize_quantized_state_dict_int8(
        state_dict, gdn_w_g_optimizer=gdn_w_g_optimizer
    )
    int8_zlib_bytes = len(quant_blob)
    total_init_bytes = int8_zlib_bytes + code_bytes
    headroom_bytes = artifact_limit_bytes - total_init_bytes
    artifact_status = (
        "OVER_LIMIT"
        if headroom_bytes < 0
        else "UNDER_LIMIT"
        if headroom_bytes > 0
        else "ON_BUDGET"
    )
    ng = sum(1 for t in model.block_types if t == "gdn")
    na = sum(1 for t in model.block_types if t == "attn")
    params = sum(p.numel() for p in model.parameters())
    return {
        "name": name,
        "notes": override.get("notes", ""),
        "num_layers": model_kwargs["num_layers"],
        "d_model": model_kwargs["d_model"],
        "mlp_mult": model_kwargs["mlp_mult"],
        "gdn_ratio": model_kwargs["gdn_ratio"],
        "blocks": f"{ng}G+{na}A",
        "params": params,
        "raw_model_bytes": raw_model_bytes,
        "baseline_tensor_bytes": int(audit["baseline_tensor_bytes"]),
        "int8_payload_bytes": int(audit["int8_payload_bytes"]),
        "quant_raw_torch_bytes": int(audit["quant_raw_torch_bytes"]),
        "int8_zlib_init_bytes": int8_zlib_bytes,
        "code_bytes": code_bytes,
        "total_init_bytes": total_init_bytes,
        "headroom_bytes": headroom_bytes,
        "artifact_status": artifact_status,
    }


def add_reference_deltas(rows: list[dict[str, Any]]) -> None:
    """Add percentage deltas relative to the first row in-place.

    :param list[dict[str, Any]] rows: Metric rows. The first row is the reference.
    :return None: Mutates rows in place.
    """

    ref = rows[0]
    for row in rows:
        row["params_delta_pct_vs_ref"] = (
            100.0 * (row["params"] - ref["params"]) / max(ref["params"], 1)
        )
        row["total_init_delta_pct_vs_ref"] = (
            100.0
            * (row["total_init_bytes"] - ref["total_init_bytes"])
            / max(ref["total_init_bytes"], 1)
        )
        row["raw_model_delta_pct_vs_ref"] = (
            100.0
            * (row["raw_model_bytes"] - ref["raw_model_bytes"])
            / max(ref["raw_model_bytes"], 1)
        )


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSON, CSV, and markdown summaries for the candidate screen.

    :param Path output_dir: Output directory to create.
    :param list[dict[str, Any]] rows: Candidate metric rows.
    :return None: Writes files under `output_dir`.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    csv_fields = [
        "name",
        "num_layers",
        "d_model",
        "mlp_mult",
        "gdn_ratio",
        "blocks",
        "params",
        "params_delta_pct_vs_ref",
        "raw_model_bytes",
        "raw_model_delta_pct_vs_ref",
        "int8_zlib_init_bytes",
        "total_init_bytes",
        "total_init_delta_pct_vs_ref",
        "headroom_bytes",
        "artifact_status",
        "notes",
    ]
    with (output_dir / "rows.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# HGDN Architecture Size Screen",
        "",
        "Note: `total_init_bytes` is an initialization-time proxy built from the",
        "trainer's real bf16/fp32 dtype restore and int8+zlib audit. Final trained",
        "artifact bytes must still be verified with the trainer.",
        "",
        "| Candidate | Shape | Blocks | Params | Raw bytes | Init int8 zlib | Init total | Delta vs ref | Headroom |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        shape = (
            f"{row['num_layers']}L×{row['d_model']}d "
            f"mlp{row['mlp_mult']} r{row['gdn_ratio']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    shape,
                    row["blocks"],
                    format_int(row["params"]),
                    format_int(row["raw_model_bytes"]),
                    format_int(row["int8_zlib_init_bytes"]),
                    format_int(row["total_init_bytes"]),
                    format_delta(row["total_init_delta_pct_vs_ref"]),
                    format_int(row["headroom_bytes"]),
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def print_stdout_table(rows: list[dict[str, Any]], *, row_limit: int) -> None:
    """Print a compact markdown table to stdout.

    :param list[dict[str, Any]] rows: Candidate rows.
    :param int row_limit: Maximum number of rows to print.
    :return None: Prints to stdout.
    """

    print("# HGDN Architecture Size Screen")
    print()
    print(
        "Initialization-time proxy only; final trained artifact bytes must still be"
        " checked with the trainer."
    )
    print()
    print(
        "| Candidate | Shape | Params | Init total | Delta vs ref | Headroom | Status |"
    )
    print("|---|---|---:|---:|---:|---:|---|")
    for row in rows[:row_limit]:
        shape = (
            f"{row['num_layers']}L×{row['d_model']}d "
            f"mlp{row['mlp_mult']} r{row['gdn_ratio']}"
        )
        print(
            f"| {row['name']} | {shape} | {format_int(row['params'])} | "
            f"{format_int(row['total_init_bytes'])} | "
            f"{format_delta(row['total_init_delta_pct_vs_ref'])} | "
            f"{format_int(row['headroom_bytes'])} | {row['artifact_status']} |"
        )


def main() -> None:
    """Run the architecture size screen."""

    args = parse_args()
    base_cfg, candidates = load_config(args.config)
    code_bytes = len(TRAINER_PATH.read_text(encoding="utf-8").encode("utf-8"))
    rows = [
        candidate_metrics(row["name"], base_cfg, row, code_bytes=code_bytes)
        for row in candidates
    ]
    add_reference_deltas(rows)
    output_dir = args.output_dir or PROFILE_ROOT / args.config.stem
    write_outputs(output_dir, rows)
    print_stdout_table(rows, row_limit=args.row_limit)
    print()
    print(f"output_dir:{output_dir}")


if __name__ == "__main__":
    main()
