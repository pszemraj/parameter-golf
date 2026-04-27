#!/usr/bin/env python3
"""Estimate artifact size for a built Core/Amplifier model."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from core_amplifier_lm import AmplifierSpec
from core_amplifier_lm.experiment import (
    artifact_estimate_bytes,
    artifact_headroom_bytes,
    artifact_status,
    estimate_repo_code_bytes,
    spec_size_bytes,
    trainable_int8_zlib_bytes,
)


def _buffer_bytes(spec: AmplifierSpec) -> dict[str, int]:
    """Return raw fixed-buffer bytes by spec field.

    :param AmplifierSpec spec: Loaded frozen spec.
    :return dict[str, int]: Buffer byte counts keyed by field name.
    """
    out: dict[str, int] = {}
    for name in (
        "token_embed",
        "base_bigram_logits",
        "lag_ops",
        "amp_w1",
        "amp_w2",
        "readout_weight",
        "readout_in_proj",
        "readout_out_proj",
        "trigram_top_tokens",
        "trigram_residual_values",
        "trigram_context_confidence",
    ):
        buf = getattr(spec, name, None)
        if buf is None:
            continue
        out[name] = int(buf.numel() * buf.element_size())
    return out


def _code_files(root: Path, mode: str) -> list[Path]:
    """List code files included in the artifact estimate.

    :param Path root: Repository or record root.
    :param str mode: Code accounting mode.
    :return list[Path]: Files included in code-byte accounting.
    """
    if mode == "train_gpt":
        return [root / "train_gpt.py"]
    return [
        p
        for p in sorted(root.rglob("*.py"))
        if p.is_file()
        and "experiments" not in p.parts
        and "records" not in p.parts
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
    ]


def _trainable_int8_payload_bytes(model_dir: Path) -> int | None:
    """Return existing or estimated compressed trainable payload bytes.

    :param Path model_dir: Model directory containing final artifacts.
    :return int | None: Payload size, or ``None`` when no final state exists.
    """
    exported_path = model_dir / "final_trainable.int8.ptz"
    if exported_path.exists():
        return int(exported_path.stat().st_size)

    final_path = model_dir / "final.pt"
    if not final_path.exists():
        return None
    obj = torch.load(final_path, map_location="cpu")
    state = obj.get("model")
    if not isinstance(state, dict):
        return None
    return trainable_int8_zlib_bytes(state)


def main() -> None:
    """Estimate artifact bytes for a built Core/Amplifier model directory."""
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Path to a built model dir containing spec.pt")
    ap.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help="Optional record folder for code-byte accounting",
    )
    ap.add_argument("--code-mode", choices=["all-py", "train_gpt"], default="all-py")
    ap.add_argument(
        "--assume-trainable-payload-bytes",
        type=int,
        default=None,
        help="Use this trainable payload estimate when final.pt/export is absent.",
    )
    ap.add_argument(
        "--fail-over-limit",
        action="store_true",
        help="Exit nonzero if the estimated artifact exceeds the 16 MB limit.",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a durable JSON artifact estimate.",
    )
    ap.add_argument("--label", default=None, help="Optional geometry/model label for JSON output.")
    ap.add_argument("--run-version", default=None, help="Optional run-version for JSON output.")
    ap.add_argument("--trigram-top-k", type=int, default=None, help="Optional trigram top-K.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    spec_path = model_dir / "spec.pt"
    if not spec_path.exists():
        raise SystemExit(f"spec.pt not found under {model_dir}")

    spec = AmplifierSpec.load(spec_path)
    spec_bytes, spec_gzip_bytes = spec_size_bytes(spec_path)
    buf_bytes = _buffer_bytes(spec)
    trainable_payload_bytes = _trainable_int8_payload_bytes(model_dir)

    print(f"spec: {spec.summary()}")
    print(f"raw fixed buffers: {spec.fixed_nbytes:,}")
    print(f"spec.pt bytes: {spec_bytes:,}")
    print(f"gzip(spec.pt): {spec_gzip_bytes:,}")
    if trainable_payload_bytes is not None:
        print(f"int8 zlib(trainable payload): {trainable_payload_bytes:,}")
    elif args.assume_trainable_payload_bytes is not None:
        trainable_payload_bytes = int(args.assume_trainable_payload_bytes)
        print(f"assumed int8 zlib(trainable payload): {trainable_payload_bytes:,}")
    print()
    print("Fixed-buffer breakdown:")
    for name, nbytes in sorted(buf_bytes.items(), key=lambda kv: kv[1], reverse=True):
        pct = nbytes / max(1, spec.fixed_nbytes)
        print(f"  {name:20s} {nbytes:>12,} bytes  ({pct:6.2%})")

    code_root = Path(args.record_dir).resolve() if args.record_dir else REPO_ROOT
    if args.code_mode == "all-py":
        code_total = estimate_repo_code_bytes(code_root)
        code_files = [p for p in _code_files(code_root, args.code_mode) if p.exists()]
    else:
        code_files = [p for p in _code_files(code_root, args.code_mode) if p.exists()]
        code_total = sum(p.stat().st_size for p in code_files)
    print()
    print(f"Code bytes ({args.code_mode}): {code_total:,}")
    for p in sorted(code_files, key=lambda q: q.stat().st_size, reverse=True)[:10]:
        rel = p.relative_to(code_root)
        print(f"  {str(rel):50s} {p.stat().st_size:>10,}")
    print()
    estimate = artifact_estimate_bytes(
        repo_root=code_root,
        spec_path=spec_path,
        trainable_payload_bytes=trainable_payload_bytes,
        repo_code_bytes=code_total,
    )
    headroom = artifact_headroom_bytes(estimate)
    status = artifact_status(estimate)
    print(f"artifact estimate = code + gzip(spec.pt) + int8 trainable: {estimate:,}")
    print(f"artifact headroom: {headroom:,}")
    print(f"artifact status: {status}")
    if args.json_out is not None:
        payload = {
            "label": args.label,
            "run_version": args.run_version,
            "trigram_top_k": args.trigram_top_k,
            "shared_spec_dir": str(model_dir),
            "record_dir": str(code_root),
            "code_mode": args.code_mode,
            "spec_summary": spec.summary(),
            "raw_fixed_buffer_bytes": int(spec.fixed_nbytes),
            "fixed_buffer_bytes": buf_bytes,
            "spec_bytes": int(spec_bytes),
            "gzip_spec_bytes": int(spec_gzip_bytes),
            "repo_code_bytes": int(code_total),
            "preflight_trainable_payload_bytes": (
                None if trainable_payload_bytes is None else int(trainable_payload_bytes)
            ),
            "artifact_estimate_bytes": None if estimate is None else int(estimate),
            "artifact_headroom_bytes": None if headroom is None else int(headroom),
            "artifact_status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"artifact JSON: {args.json_out}")
    if args.fail_over_limit and status == "OVER_LIMIT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
