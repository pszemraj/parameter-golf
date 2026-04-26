#!/usr/bin/env python3
"""Probe the installed Flash Linear Attention stack."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import inspect
import json
from typing import Any


def package_version(name: str) -> dict[str, Any]:
    """Return package metadata for one distribution.

    :param str name: Distribution name.
    :return dict[str, Any]: Version or error payload.
    """
    try:
        return {"version": metadata.version(name)}
    except Exception as exc:  # pragma: no cover - environment probe
        return {"error": str(exc)}


def import_target(name: str) -> dict[str, Any]:
    """Import one target and report basic metadata.

    :param str name: Import target.
    :return dict[str, Any]: Import status payload.
    """
    try:
        module_name, _, attr_name = name.partition(":")
        module = importlib.import_module(module_name)
        obj = getattr(module, attr_name) if attr_name else module
        payload: dict[str, Any] = {
            "ok": True,
            "module_file": getattr(module, "__file__", None),
            "object": repr(obj),
        }
        try:
            payload["signature"] = str(inspect.signature(obj))
        except Exception:
            pass
        return payload
    except Exception as exc:  # pragma: no cover - environment probe
        return {"ok": False, "error": repr(exc)}


def build_report() -> dict[str, Any]:
    """Build the FLA stack report.

    :return dict[str, Any]: Probe report.
    """
    import torch

    try:
        import triton
    except Exception:  # pragma: no cover - optional dependency metadata
        triton = None
    return {
        "packages": {
            "flash-linear-attention": package_version("flash-linear-attention"),
            "fla-core": package_version("fla-core"),
            "fla": package_version("fla"),
            "torch": package_version("torch"),
            "triton": package_version("triton"),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "devices": [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": torch.cuda.get_device_capability(idx),
                }
                for idx in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        },
        "triton": {
            "version": getattr(triton, "__version__", None)
            if triton is not None
            else None,
            "imported": triton is not None,
        },
        "imports": {
            "fla": import_target("fla"),
            "fla.layers.GatedDeltaNet": import_target("fla.layers:GatedDeltaNet"),
            "fla.ops.gated_delta_rule.chunk_gated_delta_rule": import_target(
                "fla.ops.gated_delta_rule:chunk_gated_delta_rule"
            ),
            "fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule": import_target(
                "fla.ops.gated_delta_rule:fused_recurrent_gated_delta_rule"
            ),
            "fla.ops.gated_delta_rule.chunk": import_target(
                "fla.ops.gated_delta_rule.chunk"
            ),
            "fla.ops.gated_delta_rule.wy_fast": import_target(
                "fla.ops.gated_delta_rule.wy_fast"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print raw JSON only.")
    return parser.parse_args()


def main() -> int:
    """Run the probe.

    :return int: Exit code.
    """
    args = parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(json.dumps(report, indent=2, sort_keys=True))
    high_level = report["imports"]["fla.layers.GatedDeltaNet"]["ok"]
    low_level = report["imports"]["fla.ops.gated_delta_rule.chunk_gated_delta_rule"][
        "ok"
    ]
    print(f"fla_high_level_gated_deltanet:{int(high_level)}")
    print(f"fla_low_level_chunk_gated_delta_rule:{int(low_level)}")
    return 0 if high_level and low_level else 1


if __name__ == "__main__":
    raise SystemExit(main())
