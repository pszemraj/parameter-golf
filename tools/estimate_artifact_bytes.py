#!/usr/bin/env python3
"""Estimate artifact size for a built Core/Amplifier model."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from core_amplifier_lm import AmplifierSpec
from core_amplifier_lm.experiment import (
    artifact_estimate_bytes,
    estimate_repo_code_bytes,
    spec_size_bytes,
    trainable_int8_zlib_bytes,
)


def _buffer_bytes(spec: AmplifierSpec) -> dict[str, int]:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Path to a built model dir containing spec.pt")
    ap.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help="Optional record folder for code-byte accounting",
    )
    ap.add_argument("--code-mode", choices=["all-py", "train_gpt"], default="all-py")
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
    print(f"artifact estimate = code + gzip(spec.pt) + int8 trainable: {estimate:,}")


if __name__ == "__main__":
    main()
