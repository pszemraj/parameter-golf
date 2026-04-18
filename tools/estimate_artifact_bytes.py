#!/usr/bin/env python3
"""Estimate artifact size for a built Core/Amplifier model."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from core_amplifier_lm import AmplifierSpec


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
    ):
        buf = getattr(spec, name, None)
        if buf is None:
            continue
        out[name] = int(buf.numel() * buf.element_size())
    return out


def _code_files(record_dir: Path, mode: str) -> list[Path]:
    if mode == "train_gpt":
        return [record_dir / "train_gpt.py"]
    return [
        p for p in sorted(record_dir.rglob("*.py")) if p.is_file() and "experiments" not in p.parts
    ]


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
    spec_bytes = spec_path.stat().st_size
    spec_gzip_bytes = len(gzip.compress(spec_path.read_bytes(), compresslevel=9))
    buf_bytes = _buffer_bytes(spec)

    print(f"spec: {spec.summary()}")
    print(f"raw fixed buffers: {spec.fixed_nbytes:,}")
    print(f"spec.pt bytes: {spec_bytes:,}")
    print(f"gzip(spec.pt): {spec_gzip_bytes:,}")
    print()
    print("Fixed-buffer breakdown:")
    for name, nbytes in sorted(buf_bytes.items(), key=lambda kv: kv[1], reverse=True):
        pct = nbytes / max(1, spec.fixed_nbytes)
        print(f"  {name:20s} {nbytes:>12,} bytes  ({pct:6.2%})")

    if args.record_dir:
        record_dir = Path(args.record_dir).resolve()
        code_files = [p for p in _code_files(record_dir, args.code_mode) if p.exists()]
        code_total = sum(p.stat().st_size for p in code_files)
        print()
        print(f"Code bytes ({args.code_mode}): {code_total:,}")
        for p in sorted(code_files, key=lambda q: q.stat().st_size, reverse=True)[:10]:
            rel = p.relative_to(record_dir)
            print(f"  {str(rel):50s} {p.stat().st_size:>10,}")
        print()
        print(f"artifact estimate = code + gzip(spec.pt): {code_total + spec_gzip_bytes:,}")


if __name__ == "__main__":
    main()
