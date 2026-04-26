"""Tests for shared frozen trigram memory helpers."""

from __future__ import annotations

import torch

from core_amplifier_lm import (
    AmplifierSpec,
    spec_with_trigram_memory_table,
    validate_trigram_memory_table,
)


def _minimal_spec(vocab_size: int = 4) -> AmplifierSpec:
    """Build a small base spec for table-cache validation tests.

    :param int vocab_size: Vocabulary size.
    :return AmplifierSpec: Minimal valid spec.
    """
    return AmplifierSpec(
        vocab_size=vocab_size,
        core_dim=2,
        branch_lags=(1,),
        num_blocks=1,
        token_embed=torch.zeros((vocab_size, 2), dtype=torch.float32),
        base_bigram_logits=torch.zeros((vocab_size, vocab_size), dtype=torch.float32),
        lag_ops=torch.zeros((1, 2, 2, 2), dtype=torch.float32),
        amp_w1=torch.zeros((1, 2, 2), dtype=torch.float32),
        amp_w2=torch.zeros((1, 2, 2), dtype=torch.float32),
    )


def _valid_table(vocab_size: int = 4, top_k: int = 2) -> dict[str, object]:
    """Build a valid cached trigram tensor table.

    :param int vocab_size: Vocabulary size.
    :param int top_k: Number of top tokens per context.
    :return dict[str, object]: Cache payload.
    """
    contexts = vocab_size * vocab_size
    return {
        "metadata": {"trigram_top_k": top_k, "trigram_residual_scale": 1.0},
        "trigram_top_tokens": torch.zeros((contexts, top_k), dtype=torch.int16),
        "trigram_residual_values": torch.zeros((contexts, top_k), dtype=torch.int8),
        "trigram_context_confidence": torch.zeros((contexts,), dtype=torch.uint8),
    }


def test_spec_with_trigram_table_rejects_missing_tensor() -> None:
    """Cached table reuse should fail before attaching malformed tensors."""
    spec = _minimal_spec(vocab_size=4)
    table = _valid_table(vocab_size=4, top_k=2)
    table.pop("trigram_residual_values")

    try:
        spec_with_trigram_memory_table(spec, table, top_k=2)
    except ValueError as exc:
        assert "missing required keys" in str(exc)
    else:
        raise AssertionError("expected malformed trigram table rejection")


def test_spec_with_trigram_table_rejects_bad_shape() -> None:
    """Cached table reuse should reject stale shape metadata."""
    spec = _minimal_spec(vocab_size=4)
    table = _valid_table(vocab_size=4, top_k=2)
    table["trigram_top_tokens"] = torch.zeros((15, 2), dtype=torch.int16)

    try:
        spec_with_trigram_memory_table(spec, table, top_k=2)
    except ValueError as exc:
        assert "trigram_top_tokens has shape" in str(exc)
    else:
        raise AssertionError("expected malformed trigram table rejection")


def test_validate_trigram_table_rejects_metadata_mismatch() -> None:
    """Cached table reuse should fail on stale request metadata."""
    spec = _minimal_spec(vocab_size=4)
    table = _valid_table(vocab_size=4, top_k=2)

    try:
        validate_trigram_memory_table(
            table,
            base_spec=spec,
            top_k=2,
            expected_metadata={"trigram_top_k": 4},
        )
    except ValueError as exc:
        assert "metadata mismatch" in str(exc)
    else:
        raise AssertionError("expected stale trigram table rejection")
