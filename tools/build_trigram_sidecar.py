#!/usr/bin/env python3
"""Build a dense SP1024 trigram top-K sidecar spec."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_amplifier_lm import AmplifierSpec, add_trigram_sidecar_to_spec


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_spec_dir", help="Existing model/spec directory with spec.pt")
    ap.add_argument("out_spec_dir", help="Destination model/spec directory")
    ap.add_argument("--data", required=True, help="Training data path")
    ap.add_argument(
        "--storage-dtype",
        default="uint16",
        choices=["uint8", "uint16", "int32", "int64"],
    )
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--smoothing", type=float, default=0.25)
    ap.add_argument("--residual-clip", type=float, default=8.0)
    ap.add_argument("--confidence-count-cap", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=50_000_000)
    ap.add_argument("--force", action="store_true", help="Overwrite an existing output spec")
    return ap.parse_args()


def _copy_model_dir_metadata(source: Path, dest: Path) -> None:
    """Copy lightweight model-dir metadata needed by launchers.

    :param Path source: Source spec directory.
    :param Path dest: Destination spec directory.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.model", "fineweb_1024_bpe.model"):
        src = source / name
        if src.exists():
            shutil.copy2(src, dest / name)


def main() -> None:
    """Build and write the trigram sidecar spec."""
    args = parse_args()
    source_dir = Path(args.source_spec_dir).expanduser().resolve()
    out_dir = Path(args.out_spec_dir).expanduser().resolve()
    source_spec = source_dir / "spec.pt"
    out_spec = out_dir / "spec.pt"

    if not source_spec.exists():
        raise SystemExit(f"missing source spec: {source_spec}")
    if out_spec.exists() and not args.force:
        existing = AmplifierSpec.load(out_spec)
        if existing.trigram_top_tokens is not None:
            print(f"Existing trigram sidecar spec found: {out_spec}")
            print(existing.summary())
            return
        raise SystemExit(f"{out_spec} exists but has no trigram sidecar; pass --force")

    base_spec = AmplifierSpec.load(source_spec)
    sidecar_spec = add_trigram_sidecar_to_spec(
        base_spec,
        args.data,
        storage_dtype=args.storage_dtype,
        top_k=args.top_k,
        smoothing=args.smoothing,
        residual_clip=args.residual_clip,
        confidence_count_cap=args.confidence_count_cap,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
        verbose=True,
    )
    _copy_model_dir_metadata(source_dir, out_dir)
    sidecar_spec.save(out_spec)
    print(f"Wrote trigram sidecar spec: {out_spec}")
    print(sidecar_spec.summary())


if __name__ == "__main__":
    main()
