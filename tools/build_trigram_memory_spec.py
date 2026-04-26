#!/usr/bin/env python3
"""Build a dense SP1024 trigram top-K memory spec."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core_amplifier_lm import AmplifierSpec, add_trigram_memory_to_spec


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
    ap.add_argument(
        "--table-cache-root",
        default=os.environ.get("TRIGRAM_MEMORY_TABLE_CACHE_ROOT"),
        help="Optional cache root for reusable trigram memory tensors",
    )
    ap.add_argument(
        "--rebuild-table-cache",
        action="store_true",
        help="Recount and overwrite a cached trigram tensor table",
    )
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


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a CPU tensor by dtype, shape, and raw bytes.

    :param torch.Tensor tensor: Tensor to hash.
    :return str: Hex SHA256 digest.
    """
    cpu = tensor.detach().cpu().contiguous()
    payload = {"dtype": str(cpu.dtype), "shape": list(cpu.shape)}
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    h.update(cpu.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _cache_key(payload: dict[str, Any]) -> str:
    """Return a deterministic short cache key.

    :param dict[str, Any] payload: Cache-key fields.
    :return str: Short digest.
    """
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def _table_cache_path(
    *,
    cache_root: str | Path,
    base_spec: AmplifierSpec,
    data: str | Path,
    storage_dtype: str,
    top_k: int,
    smoothing: float,
    residual_clip: float,
    confidence_count_cap: int,
    max_tokens: int | None,
) -> Path:
    """Resolve the reusable trigram tensor cache path.

    :param str | Path cache_root: Cache root directory.
    :param AmplifierSpec base_spec: Source spec containing base bigram logits.
    :param str | Path data: Training-token source.
    :param str storage_dtype: Token storage dtype.
    :param int top_k: Number of top tokens per context.
    :param float smoothing: Trigram smoothing.
    :param float residual_clip: Residual clip.
    :param int confidence_count_cap: Confidence count cap.
    :param int | None max_tokens: Optional smoke cap.
    :return Path: Cache path.
    """
    payload = {
        "version": 1,
        "vocab_size": int(base_spec.vocab_size),
        "base_bigram_sha256": _tensor_sha256(base_spec.base_bigram_logits),
        "data_path": str(Path(data).expanduser().resolve()),
        "storage_dtype": str(storage_dtype),
        "top_k": int(top_k),
        "smoothing": float(smoothing),
        "residual_clip": float(residual_clip),
        "confidence_count_cap": int(confidence_count_cap),
        "max_tokens": max_tokens,
    }
    scope = "full" if max_tokens is None else f"max{int(max_tokens)}"
    name = f"trigram_memory_table_k{int(top_k)}_{scope}_{_cache_key(payload)}.pt"
    return Path(cache_root).expanduser().resolve() / name


def _spec_with_trigram_table(spec: AmplifierSpec, table: dict[str, Any]) -> AmplifierSpec:
    """Attach cached trigram tensors to a spec.

    :param AmplifierSpec spec: Base spec.
    :param dict[str, Any] table: Cached trigram table payload.
    :return AmplifierSpec: Spec with memory tensors attached.
    """
    metadata = dict(spec.metadata)
    metadata.update(dict(table["metadata"]))
    return AmplifierSpec(
        vocab_size=spec.vocab_size,
        core_dim=spec.core_dim,
        branch_lags=spec.branch_lags,
        num_blocks=spec.num_blocks,
        token_embed=spec.token_embed,
        base_bigram_logits=spec.base_bigram_logits,
        lag_ops=spec.lag_ops,
        amp_w1=spec.amp_w1,
        amp_w2=spec.amp_w2,
        readout_weight=spec.readout_weight,
        readout_in_proj=spec.readout_in_proj,
        readout_out_proj=spec.readout_out_proj,
        trigram_top_tokens=table["trigram_top_tokens"],
        trigram_residual_values=table["trigram_residual_values"],
        trigram_context_confidence=table["trigram_context_confidence"],
        metadata=metadata,
    )


def _table_from_spec(spec: AmplifierSpec) -> dict[str, Any]:
    """Extract a reusable trigram table payload from a memory spec.

    :param AmplifierSpec spec: Spec containing trigram tensors.
    :return dict[str, Any]: Cache payload.
    """
    if (
        spec.trigram_top_tokens is None
        or spec.trigram_residual_values is None
        or spec.trigram_context_confidence is None
    ):
        raise ValueError("spec does not contain trigram memory tensors")
    metadata = {
        key: value for key, value in spec.metadata.items() if str(key).startswith("trigram_")
    }
    return {
        "metadata": metadata,
        "trigram_top_tokens": spec.trigram_top_tokens.cpu(),
        "trigram_residual_values": spec.trigram_residual_values.cpu(),
        "trigram_context_confidence": spec.trigram_context_confidence.cpu(),
    }


def main() -> None:
    """Build and write the trigram memory spec."""
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
            print(f"Existing trigram memory spec found: {out_spec}")
            print(existing.summary())
            return
        raise SystemExit(f"{out_spec} exists but has no trigram memory; pass --force")

    base_spec = AmplifierSpec.load(source_spec)
    table_cache_path = None
    table_payload = None
    if args.table_cache_root:
        table_cache_path = _table_cache_path(
            cache_root=args.table_cache_root,
            base_spec=base_spec,
            data=args.data,
            storage_dtype=args.storage_dtype,
            top_k=args.top_k,
            smoothing=args.smoothing,
            residual_clip=args.residual_clip,
            confidence_count_cap=args.confidence_count_cap,
            max_tokens=args.max_tokens,
        )
        if table_cache_path.exists() and not args.rebuild_table_cache:
            table_payload = torch.load(table_cache_path, map_location="cpu", weights_only=False)
            print(f"Loaded cached trigram memory table: {table_cache_path}")
        elif args.rebuild_table_cache:
            print(f"Rebuilding trigram memory table cache: {table_cache_path}")
        else:
            print(f"Trigram memory table cache miss: {table_cache_path}")
            print("The counted table will be cached there after the full pass completes.")

    if table_payload is None:
        memory_spec = add_trigram_memory_to_spec(
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
        table_payload = _table_from_spec(memory_spec)
        if table_cache_path is not None:
            table_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(table_payload, table_cache_path)
            print(f"Cached trigram memory table: {table_cache_path}")
    else:
        memory_spec = _spec_with_trigram_table(base_spec, table_payload)

    _copy_model_dir_metadata(source_dir, out_dir)
    memory_spec.save(out_spec)
    print(f"Wrote trigram memory spec: {out_spec}")
    print(memory_spec.summary())


if __name__ == "__main__":
    main()
