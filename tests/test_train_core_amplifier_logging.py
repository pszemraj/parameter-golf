"""Trainer logging helper tests."""

from __future__ import annotations

import torch

from train_core_amplifier import (
    EvalResult,
    SequentialStreamBatcher,
    eval_payload_fields,
    file_sha256,
    format_eval_coverage,
    full_validation_steps,
    validate_exact_bpb_targets,
)
from train_core_amplifier import _format_debug_vector


def test_format_debug_vector_keeps_all_values():
    values = [float(i) for i in range(12)]
    formatted = _format_debug_vector(values)
    assert formatted.count(",") == 11
    assert formatted.startswith("0.000,1.000,2.000")
    assert formatted.endswith("9.000,10.000,11.000")


def test_format_debug_vector_handles_empty_values():
    assert _format_debug_vector([]) == "none"


def test_full_validation_steps_covers_stream_once():
    tokens = torch.arange(10_001)
    steps = full_validation_steps(tokens, seq_len=100, batch_size=8)
    assert steps > 0
    covered = steps * 8 * 100
    assert covered >= tokens.numel() - 1


def test_full_validation_batcher_covers_each_target_once():
    tokens = torch.arange(10_001)
    seq_len = 100
    batch_size = 8
    batcher = SequentialStreamBatcher(
        tokens,
        seq_len=seq_len,
        batch_size=batch_size,
        output_device=torch.device("cpu"),
        allow_tail=True,
        cover_remainder=True,
    )
    steps = full_validation_steps(tokens, seq_len=seq_len, batch_size=batch_size)

    seen = []
    for _ in range(steps):
        batch, _reset = batcher.next_batch()
        seen.extend(batch[:, 1:].flatten().tolist())

    assert sorted(seen) == list(range(1, tokens.numel()))


def test_exact_bpb_target_validation_rejects_zero_byte_targets():
    """Exact BPB should fail if validation targets include zero-byte token ids."""
    byte_count_lut = torch.tensor([0, 1, 2], dtype=torch.int32)
    val_tokens = torch.tensor([1, 2, 0, 1], dtype=torch.long)

    try:
        validate_exact_bpb_targets(byte_count_lut, val_tokens)
    except ValueError as exc:
        assert "zero-byte" in str(exc)
    else:
        raise AssertionError("expected zero-byte validation target failure")


def test_file_sha256_hashes_file_contents(tmp_path):
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"abc")

    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_eval_payload_fields_are_explicit():
    result = EvalResult(
        loss=1.0,
        bpb=2.0,
        tokens=100,
        bytes=50,
        usable_tokens=400,
        coverage_frac=0.25,
        full_coverage=False,
        steps=3,
        batch_size=4,
        seq_len=8,
    )
    fields = eval_payload_fields(result)
    assert fields["eval_tokens"] == 100
    assert fields["eval_bytes"] == 50
    assert fields["eval_coverage_denominator_tokens"] == 400
    assert fields["eval_coverage_frac"] == 0.25
    assert fields["eval_full_coverage"] is False
    assert fields["eval_steps"] == 3


def test_format_eval_coverage_names_validation_source():
    result = EvalResult(
        loss=1.0,
        bpb=2.0,
        tokens=100,
        bytes=50,
        usable_tokens=400,
        coverage_frac=0.25,
        full_coverage=False,
        steps=3,
        batch_size=4,
        seq_len=8,
    )
    text = format_eval_coverage(result, validation_source="explicit_val_shard")
    assert text.startswith("val_coverage 25.000%")
    assert "100/400 target tokens" in text
    assert "source=explicit_val_shard" in text
