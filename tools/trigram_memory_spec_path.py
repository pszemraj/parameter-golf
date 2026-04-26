#!/usr/bin/env python3
"""Resolve the deterministic cache directory for a trigram memory spec."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "experiments" / "param-golf-coreamp"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_spec_dir", help="Existing model/spec directory with spec.pt")
    ap.add_argument("--data", required=True, help="Training data path")
    ap.add_argument("--family", default="", help="Readable family label, e.g. blocks0")
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--storage-dtype", default="uint16")
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--smoothing", type=float, default=0.25)
    ap.add_argument("--residual-clip", type=float, default=8.0)
    ap.add_argument("--confidence-count-cap", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--mkdir", action="store_true", help="Create the cache directory")
    return ap.parse_args()


def _slug(value: str) -> str:
    """Return a path-safe slug.

    :param str value: Raw label.
    :return str: Sanitized label.
    """
    out = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return out.strip("-") or "spec"


def _file_sha256(path: Path) -> str:
    """Hash one file.

    :param Path path: File path.
    :return str: Hex SHA256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(payload: dict[str, Any]) -> str:
    """Hash a canonical JSON cache-key payload.

    :param dict[str, Any] payload: Cache-key fields.
    :return str: Short digest.
    """
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def main() -> None:
    """Print the cache directory path."""
    args = parse_args()
    source_dir = Path(args.source_spec_dir).expanduser().resolve()
    spec_path = source_dir / "spec.pt"
    if not spec_path.exists():
        raise SystemExit(f"missing source spec: {spec_path}")
    data_path = Path(args.data).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    source_digest = _file_sha256(spec_path)
    payload = {
        "source_spec_sha256": source_digest,
        "data_path": str(data_path),
        "storage_dtype": str(args.storage_dtype),
        "top_k": int(args.top_k),
        "smoothing": float(args.smoothing),
        "residual_clip": float(args.residual_clip),
        "confidence_count_cap": int(args.confidence_count_cap),
        "max_tokens": args.max_tokens,
    }
    readable = _slug(args.family or source_dir.name)
    token_scope = "full" if args.max_tokens is None else f"max{int(args.max_tokens)}"
    dirname = (
        f"{readable}_k{int(args.top_k)}_smooth{float(args.smoothing):g}_"
        f"clip{float(args.residual_clip):g}_cap{int(args.confidence_count_cap)}_"
        f"{token_scope}_{source_digest[:12]}_{_cache_key(payload)}"
    )
    out = cache_root / "trigram_memory_specs" / dirname
    if args.mkdir:
        out.mkdir(parents=True, exist_ok=True)
    print(out)


if __name__ == "__main__":
    main()
