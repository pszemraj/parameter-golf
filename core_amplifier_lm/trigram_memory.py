"""Shared helpers for frozen dense trigram memory specs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .model import AmplifierSpec


def tensor_sha256(tensor: torch.Tensor) -> str:
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


def trigram_memory_expected_metadata(
    *,
    top_k: int,
    smoothing: float,
    residual_clip: float,
    confidence_count_cap: int,
    max_tokens: int | None,
    data_fingerprint: dict[str, object],
) -> dict[str, Any]:
    """Return metadata values that identify a requested trigram memory table.

    :param int top_k: Number of top next-token entries per context.
    :param float smoothing: Additive trigram smoothing.
    :param float residual_clip: Residual clipping bound.
    :param int confidence_count_cap: Count at which confidence saturates.
    :param int | None max_tokens: Optional smoke/debug token cap.
    :param dict[str, object] data_fingerprint: Training-shard fingerprint.
    :return dict[str, Any]: Expected metadata fields.
    """
    return {
        "trigram_top_k": int(top_k),
        "trigram_smoothing": float(smoothing),
        "trigram_residual_clip": float(residual_clip),
        "trigram_confidence_count_cap": int(confidence_count_cap),
        "trigram_max_tokens": None if max_tokens is None else int(max_tokens),
        "trigram_train_file_count": data_fingerprint["train_file_count"],
        "trigram_train_total_bytes": data_fingerprint["train_total_bytes"],
        "trigram_train_fingerprint": data_fingerprint["digest"],
    }


def trigram_memory_metadata_mismatches(
    metadata: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    """Return cache-safety metadata mismatches.

    :param dict[str, Any] metadata: Existing metadata.
    :param dict[str, Any] expected: Expected metadata fields.
    :return list[str]: Human-readable mismatch strings.
    """
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: existing={actual_value!r} expected={expected_value!r}")
    return mismatches


def trigram_memory_table_cache_path(
    *,
    cache_root: str | Path,
    base_spec: AmplifierSpec,
    data: str | Path,
    data_fingerprint: dict[str, object],
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
    :param dict[str, object] data_fingerprint: Training-shard fingerprint.
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
        "base_bigram_sha256": tensor_sha256(base_spec.base_bigram_logits),
        "data_path": str(Path(data).expanduser().resolve()),
        "data_fingerprint": data_fingerprint,
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


def validate_trigram_memory_table(
    table: dict[str, Any],
    *,
    base_spec: AmplifierSpec,
    top_k: int,
    expected_metadata: dict[str, Any] | None = None,
) -> None:
    """Validate a cached trigram tensor table before attaching it.

    :param dict[str, Any] table: Cached trigram table payload.
    :param AmplifierSpec base_spec: Base spec that will receive the table.
    :param int top_k: Expected number of top tokens per context.
    :param dict[str, Any] | None expected_metadata: Optional exact metadata
        fields expected for this cache request.
    :raises ValueError: If required keys, tensor shapes, or dtypes are invalid.
    """
    if not isinstance(table, dict):
        raise ValueError("trigram table cache payload must be a dict")
    required = {
        "metadata",
        "trigram_top_tokens",
        "trigram_residual_values",
        "trigram_context_confidence",
    }
    missing = sorted(required.difference(table))
    if missing:
        raise ValueError(f"trigram table cache is missing required keys: {missing}")
    if not isinstance(table["metadata"], dict):
        raise ValueError("trigram table cache metadata must be a dict")
    if expected_metadata is not None:
        mismatches = []
        for key, expected_value in expected_metadata.items():
            actual_value = table["metadata"].get(key)
            if actual_value != expected_value:
                mismatches.append(f"{key}: cached={actual_value!r} expected={expected_value!r}")
        if mismatches:
            joined = "\n  - ".join(mismatches)
            raise ValueError(f"trigram table cache metadata mismatch:\n  - {joined}")

    vocab_size = int(base_spec.vocab_size)
    expected_top_shape = (vocab_size * vocab_size, int(top_k))
    expected_confidence_shape = (vocab_size * vocab_size,)
    checks = (
        ("trigram_top_tokens", expected_top_shape, torch.int16),
        ("trigram_residual_values", expected_top_shape, torch.int8),
        ("trigram_context_confidence", expected_confidence_shape, torch.uint8),
    )
    for key, expected_shape, expected_dtype in checks:
        value = table[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"trigram table cache {key} must be a torch.Tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"trigram table cache {key} has shape {tuple(value.shape)}; "
                f"expected {expected_shape}"
            )
        if value.dtype != expected_dtype:
            raise ValueError(
                f"trigram table cache {key} has dtype {value.dtype}; expected {expected_dtype}"
            )


def spec_with_trigram_memory_table(
    spec: AmplifierSpec,
    table: dict[str, Any],
    *,
    top_k: int,
) -> AmplifierSpec:
    """Attach cached trigram tensors to a spec.

    :param AmplifierSpec spec: Base spec.
    :param dict[str, Any] table: Cached trigram table payload.
    :param int top_k: Expected number of top tokens per context.
    :return AmplifierSpec: Spec with memory tensors attached.
    """
    validate_trigram_memory_table(table, base_spec=spec, top_k=top_k)
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


def trigram_memory_table_from_spec(spec: AmplifierSpec) -> dict[str, Any]:
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
    table = {
        "metadata": metadata,
        "trigram_top_tokens": spec.trigram_top_tokens.cpu(),
        "trigram_residual_values": spec.trigram_residual_values.cpu(),
        "trigram_context_confidence": spec.trigram_context_confidence.cpu(),
    }
    validate_trigram_memory_table(
        table,
        base_spec=spec,
        top_k=int(spec.trigram_top_tokens.shape[1]),
    )
    return table


def record_trigram_memory_data_fingerprint(
    spec: AmplifierSpec,
    data_fingerprint: dict[str, object],
) -> None:
    """Record training-shard fingerprint metadata on a memory spec.

    :param AmplifierSpec spec: Spec to annotate.
    :param dict[str, object] data_fingerprint: Training-shard fingerprint.
    """
    spec.metadata.update(
        {
            "trigram_train_file_count": data_fingerprint["train_file_count"],
            "trigram_train_total_bytes": data_fingerprint["train_total_bytes"],
            "trigram_train_fingerprint": data_fingerprint["digest"],
        }
    )
