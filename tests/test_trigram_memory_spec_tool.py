"""Tests for the trigram memory spec build tool."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core_amplifier_lm import AmplifierSpec
from tools.build_trigram_memory_spec import (
    _copy_model_dir_metadata,
    _spec_with_trigram_table,
    _validate_trigram_table,
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


def test_copy_model_dir_metadata_copies_config_models_and_vocabs(tmp_path: Path) -> None:
    """Metadata copy should include config plus every model/vocab sidecar."""
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "tokenizer.model").write_text("model-a", encoding="utf-8")
    (source / "fineweb_1024_bpe.model").write_text("model-b", encoding="utf-8")
    (source / "tokenizer.vocab").write_text("vocab-a", encoding="utf-8")
    (source / "notes.txt").write_text("skip", encoding="utf-8")

    _copy_model_dir_metadata(source, dest)

    assert sorted(path.name for path in dest.iterdir()) == [
        "config.json",
        "fineweb_1024_bpe.model",
        "tokenizer.model",
        "tokenizer.vocab",
    ]


def test_copy_model_dir_metadata_clean_dest_removes_stale_sidecars(tmp_path: Path) -> None:
    """Forced rebuild metadata copy should not leave stale known sidecars."""
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "config.json").write_text('{"current": true}', encoding="utf-8")
    (source / "current.model").write_text("model", encoding="utf-8")
    (dest / "old.model").write_text("stale", encoding="utf-8")
    (dest / "old.vocab").write_text("stale", encoding="utf-8")
    (dest / "keep.txt").write_text("unowned", encoding="utf-8")

    _copy_model_dir_metadata(source, dest, clean_dest=True)

    assert sorted(path.name for path in dest.iterdir()) == [
        "config.json",
        "current.model",
        "keep.txt",
    ]
    assert (dest / "config.json").read_text(encoding="utf-8") == '{"current": true}'


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda table: table.pop("trigram_residual_values"), "missing required keys"),
        (
            lambda table: table.__setitem__(
                "trigram_top_tokens",
                torch.zeros((15, 2), dtype=torch.int16),
            ),
            "trigram_top_tokens has shape",
        ),
        (
            lambda table: table.__setitem__(
                "trigram_context_confidence",
                torch.zeros((16,), dtype=torch.int16),
            ),
            "trigram_context_confidence has dtype",
        ),
    ],
)
def test_spec_with_trigram_table_rejects_bad_cache_payloads(mutate, match: str) -> None:
    """Cached table reuse should fail before attaching malformed tensors."""
    spec = _minimal_spec(vocab_size=4)
    table = _valid_table(vocab_size=4, top_k=2)
    mutate(table)

    with pytest.raises(ValueError, match=match):
        _spec_with_trigram_table(spec, table, top_k=2)


def test_validate_trigram_table_rejects_metadata_mismatch() -> None:
    """Cached table reuse should fail on stale request metadata."""
    spec = _minimal_spec(vocab_size=4)
    table = _valid_table(vocab_size=4, top_k=2)

    with pytest.raises(ValueError, match="metadata mismatch"):
        _validate_trigram_table(
            table,
            base_spec=spec,
            top_k=2,
            expected_metadata={"trigram_top_k": 4},
        )
