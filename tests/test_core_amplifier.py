"""Tests for core_amplifier_lm package."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from core_amplifier_lm import (
    AmplifierSpec,
    DEFAULTS,
    build_amplifier_spec,
    build_spec_optimized,
    load_tokens_int32,
    load_train_val_int32,
)
from core_amplifier_lm.model import (
    _count_bigrams,
    _count_lag_pairs,
    _count_unigrams,
)
from core_amplifier_lm.scan_backend import (
    apply_affine_scan,
    resolve_scan_backend,
    sequential_affine_scan,
)
from core_amplifier_lm.spec_builder import count_all
from train_core_amplifier import mmap_train_val

VOCAB = 256
LAGS = (1, 2, 4, 8)
SEED = 42
TOTAL = 50_000

PKG_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_shard_dir(tmpdir: Path) -> tuple[np.ndarray, Path]:
    """Create a fineweb-style shard directory."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=TOTAL, dtype=np.int32)
    shard_dir = tmpdir / "shards"
    shard_dir.mkdir()
    train_split = int(TOTAL * 0.85)
    offset, idx = 0, 0
    while offset < train_split:
        end = min(offset + 12_000, train_split)
        tokens[offset:end].astype(np.uint16).tofile(shard_dir / f"fineweb_train_{idx:06d}.bin")
        offset = end
        idx += 1
    tokens[train_split:].astype(np.uint16).tofile(shard_dir / "fineweb_val_000000.bin")
    return tokens, shard_dir


def _make_train_only_shard_dir(tmpdir: Path) -> tuple[np.ndarray, Path]:
    """Create a fineweb-style shard directory without validation shards."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=TOTAL, dtype=np.int32)
    shard_dir = tmpdir / "train_only_shards"
    shard_dir.mkdir()
    offset, idx = 0, 0
    while offset < TOTAL:
        end = min(offset + 12_000, TOTAL)
        tokens[offset:end].astype(np.uint16).tofile(shard_dir / f"fineweb_train_{idx:06d}.bin")
        offset = end
        idx += 1
    return tokens, shard_dir


def _make_gz(tmpdir: Path) -> tuple[np.ndarray, Path]:
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=TOTAL, dtype=np.int32)
    gz_path = tmpdir / "tokens.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(tokens.astype(np.uint8).tobytes())
    return tokens, gz_path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_single_pass_counts_match():
    """count_all must match the original per-lag counting functions."""
    rng = np.random.default_rng(SEED)
    tokens_i32 = rng.integers(0, VOCAB, size=TOTAL, dtype=np.int32)
    tokens_i64 = tokens_i32.astype(np.int64)

    uni, bigram, lag_counts = count_all(
        tokens_i32, vocab_size=VOCAB, branch_lags=LAGS, verbose=False
    )

    assert np.abs(uni - _count_unigrams(tokens_i64, VOCAB)).max() == 0
    assert np.abs(bigram - _count_bigrams(tokens_i64, VOCAB)).max() == 0
    for lag in LAGS:
        mono = _count_lag_pairs(tokens_i64, lag=lag, vocab_size=VOCAB)
        assert np.abs(lag_counts[lag] - mono).max() == 0, f"lag {lag} mismatch"


def test_spec_match_single_file():
    """Optimized spec from .gz must match original monolithic builder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tokens, gz_path = _make_gz(tmpdir)

        mono = build_amplifier_spec(
            tokens.astype(np.int64),
            vocab_size=VOCAB,
            core_dim=8,
            branch_lags=LAGS,
            num_blocks=1,
            fixed_dtype=torch.float16,
        )
        opt = build_spec_optimized(
            gz_path,
            vocab_size=VOCAB,
            core_dim=8,
            branch_lags=LAGS,
            num_blocks=1,
            fixed_dtype=torch.float16,
            storage_dtype="uint8",
            verbose=False,
        )
        _assert_specs_equal(mono, opt)


def test_spec_match_shard_dir():
    """Optimized spec from shard directory must produce a valid, usable spec."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, shard_dir = _make_shard_dir(tmpdir)

        opt = build_spec_optimized(
            shard_dir,
            vocab_size=VOCAB,
            core_dim=8,
            branch_lags=LAGS,
            num_blocks=1,
            fixed_dtype=torch.float16,
            storage_dtype="uint16",
            verbose=False,
        )

        # Verify structural correctness
        assert opt.vocab_size == VOCAB
        assert opt.core_dim == 8
        assert opt.branch_lags == LAGS
        assert opt.num_blocks == 1
        assert opt.token_embed.shape == (VOCAB, 8)
        assert opt.base_bigram_logits.shape == (VOCAB, VOCAB)

        # Verify the spec produces a working model
        from core_amplifier_lm import CoreAmplifierLM

        model = CoreAmplifierLM(opt, core_layers=1, amplifier_dtype=torch.float32)
        model.prepare_runtime_buffers(amplifier_dtype=torch.float32)
        x = torch.randint(0, VOCAB, (2, 17))
        with torch.no_grad():
            logits = model(x[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, VOCAB), x[:, 1:].reshape(-1)
            )
        # Loss should be in a reasonable range (not NaN/inf, near ln(V))
        assert not torch.isnan(loss) and not torch.isinf(loss)
        assert loss.item() < 2 * np.log(VOCAB), f"loss {loss.item():.2f} too high"

        # Step-vs-forward consistency
        with torch.no_grad():
            full = model(x[:, :-1])
            state = model.initial_state(2)
            pieces = []
            for t in range(x.size(1) - 1):
                sl, state = model.step(x[:, t], state)
                pieces.append(sl[:, None, :])
            step_all = torch.cat(pieces, dim=1)
        diff = (full - step_all).abs().max().item()
        assert diff < 1e-4, f"step vs forward diff={diff}"


def test_residual_mingru_controller_step_consistency():
    """Residualized minGRU keeps the same scan/step semantics."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    spec = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
    )
    from core_amplifier_lm import CoreAmplifierLM

    model = CoreAmplifierLM(
        spec,
        core_layers=4,
        core_expansion=2.0,
        residual_core=True,
        residual_core_init=-2.0,
        amplifier_dtype=torch.float32,
    )
    gates = model.residual_gate_values()
    assert len(gates) == 4
    assert all(0.0 < g < 0.25 for g in gates)

    x = torch.randint(0, VOCAB, (2, 19))
    with torch.no_grad():
        full = model(x[:, :-1])
        state = model.initial_state(2)
        parts = []
        for t in range(x.size(1) - 1):
            logits, state = model.step(x[:, t], state)
            parts.append(logits[:, None, :])
        stepwise = torch.cat(parts, dim=1)
    assert (full - stepwise).abs().max().item() < 1e-4


def test_lagged_temporal_mode_step_consistency():
    """Explicit lagged branch taps should preserve forward/step equivalence."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    spec = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
    )
    from core_amplifier_lm import CoreAmplifierLM

    model = CoreAmplifierLM(
        spec,
        core_layers=3,
        core_expansion=2.0,
        residual_core=True,
        residual_core_init=-2.0,
        branch_temporal_mode="lagged",
        amplifier_dtype=torch.float32,
    )

    x = torch.randint(0, VOCAB, (2, 19))
    with torch.no_grad():
        full = model(x[:, :-1])
        state = model.initial_state(2)
        parts = []
        for t in range(x.size(1) - 1):
            logits, state = model.step(x[:, t], state)
            parts.append(logits[:, None, :])
        stepwise = torch.cat(parts, dim=1)
    assert (full - stepwise).abs().max().item() < 1e-4


def test_lagged_temporal_mode_chunk_carry_matches_full_forward():
    """Lagged branch history should carry exactly across chunk boundaries."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    spec = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
    )
    from core_amplifier_lm import CoreAmplifierLM

    model = CoreAmplifierLM(
        spec,
        core_layers=3,
        core_expansion=2.0,
        residual_core=True,
        residual_core_init=-2.0,
        branch_temporal_mode="lagged",
        amplifier_dtype=torch.float32,
    )

    x = torch.randint(0, VOCAB, (2, 17))
    with torch.no_grad():
        full = model(x)
        first, state = model(x[:, :7], return_state=True)
        second, _ = model(x[:, 7:], state=state, return_state=True)
        stitched = torch.cat([first, second], dim=1)
    assert (full - stitched).abs().max().item() < 1e-4


def test_hybrid_temporal_mode_chunk_carry_matches_full_forward():
    """Hybrid current+lagged taps should preserve forward/chunk consistency."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    spec = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
    )
    from core_amplifier_lm import CoreAmplifierLM

    model = CoreAmplifierLM(
        spec,
        core_layers=3,
        core_expansion=2.0,
        residual_core=True,
        residual_core_init=-2.0,
        branch_temporal_mode="hybrid",
        branch_temporal_lag_scale=1.0,
        amplifier_dtype=torch.float32,
    )

    x = torch.randint(0, VOCAB, (2, 17))
    with torch.no_grad():
        full = model(x)
        first, state = model(x[:, :7], return_state=True)
        second, _ = model(x[:, 7:], state=state, return_state=True)
        stitched = torch.cat([first, second], dim=1)
    assert (full - stitched).abs().max().item() < 1e-4


def test_ema_temporal_mode_chunk_carry_matches_full_forward():
    """EMA temporal taps should preserve forward/chunk consistency."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    spec = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
    )
    from core_amplifier_lm import CoreAmplifierLM

    model = CoreAmplifierLM(
        spec,
        core_layers=3,
        core_expansion=2.0,
        residual_core=True,
        residual_core_init=-2.0,
        branch_temporal_mode="ema",
        scan_backend="assoc",
        amplifier_dtype=torch.float32,
    )

    x = torch.randint(0, VOCAB, (2, 17))
    with torch.no_grad():
        full = model(x)
        first, state = model(x[:, :7], return_state=True)
        second, _ = model(x[:, 7:], state=state, return_state=True)
        stitched = torch.cat([first, second], dim=1)
    assert (full - stitched).abs().max().item() < 1e-4


def test_mingru_assoc_scan_matches_heinsen_reference():
    """Assoc-scan should match the Heinsen minGRU reference path on CPU."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    spec = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
    )
    from core_amplifier_lm import CoreAmplifierLM

    reference = CoreAmplifierLM(
        spec,
        core_layers=3,
        core_expansion=2.0,
        residual_core=True,
        residual_core_init=-2.0,
        branch_temporal_mode="current",
        scan_backend="heinsen",
        amplifier_dtype=torch.float32,
    )
    candidate = CoreAmplifierLM(
        spec,
        core_layers=3,
        core_expansion=2.0,
        residual_core=True,
        residual_core_init=-2.0,
        branch_temporal_mode="current",
        scan_backend="assoc",
        amplifier_dtype=torch.float32,
    )
    candidate.load_state_dict(reference.state_dict())

    x = torch.randint(0, VOCAB, (2, 17))
    with torch.no_grad():
        ref_logits, ref_state = reference(x, return_state=True)
        out_logits, out_state = candidate(x, return_state=True)
    assert reference.active_scan_backend_name() == "heinsen"
    assert candidate.active_scan_backend_name() == "assoc"
    assert torch.allclose(ref_logits, out_logits, atol=1e-4, rtol=1e-4)
    assert torch.allclose(ref_state, out_state, atol=1e-4, rtol=1e-4)


def test_auto_scan_backend_requires_accelerated_cuda_stack(monkeypatch):
    """CUDA auto backend should fail loudly when accelerated scan deps are missing."""
    monkeypatch.setattr("core_amplifier_lm.scan_backend.assoc_scan_installed", lambda: True)
    monkeypatch.setattr("core_amplifier_lm.scan_backend.accelerated_scan_installed", lambda: False)
    with pytest.raises(RuntimeError, match="requires both assoc-scan and accelerated-scan"):
        resolve_scan_backend("auto", device=torch.device("cuda"), allow_heinsen=True)


def test_min_gru_active_scan_backend_resolves_before_first_forward(monkeypatch):
    """The reported active backend should resolve before the first training step."""
    monkeypatch.setattr("core_amplifier_lm.scan_backend.assoc_scan_installed", lambda: True)
    monkeypatch.setattr("core_amplifier_lm.scan_backend.accelerated_scan_installed", lambda: False)
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    spec = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
    )
    from core_amplifier_lm import CoreAmplifierLM

    model = CoreAmplifierLM(spec, core_layers=2, scan_backend="auto", amplifier_dtype=torch.float32)
    assert model.active_scan_backend_name() == "assoc"


def test_assoc_accel_direct_scalar_path_matches_sequential_with_prev(monkeypatch):
    """The direct accelerated scalar path should preserve scan semantics."""

    def fake_scalar_scan(gates_bct: torch.Tensor, inputs_bct: torch.Tensor) -> torch.Tensor:
        gates_btd = gates_bct.permute(0, 2, 1).contiguous()
        inputs_btd = inputs_bct.permute(0, 2, 1).contiguous()
        outputs_btd = sequential_affine_scan(gates_btd, inputs_btd)
        return outputs_btd.permute(0, 2, 1).contiguous()

    monkeypatch.setattr(
        "core_amplifier_lm.scan_backend._get_accelerated_scalar_scan",
        lambda: fake_scalar_scan,
    )

    gates = torch.sigmoid(torch.randn(2, 5, 3))
    inputs = torch.randn(2, 5, 3)
    prev = torch.randn(2, 3)

    out, backend = apply_affine_scan(gates, inputs, prev=prev, backend="assoc_accel")
    ref = sequential_affine_scan(gates, inputs, prev=prev)

    assert backend == "assoc_accel"
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_ema_hybrid_temporal_mode_chunk_carry_matches_full_forward():
    """EMA-hybrid temporal taps should preserve forward/chunk consistency."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    spec = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
    )
    from core_amplifier_lm import CoreAmplifierLM

    model = CoreAmplifierLM(
        spec,
        core_layers=3,
        core_expansion=2.0,
        residual_core=True,
        residual_core_init=-2.0,
        branch_temporal_mode="ema_hybrid",
        branch_temporal_lag_scale=1.0,
        scan_backend="assoc",
        amplifier_dtype=torch.float32,
    )

    x = torch.randint(0, VOCAB, (2, 17))
    with torch.no_grad():
        full = model(x)
        first, state = model(x[:, :7], return_state=True)
        second, _ = model(x[:, 7:], state=state, return_state=True)
        stitched = torch.cat([first, second], dim=1)
    assert (full - stitched).abs().max().item() < 1e-4


def test_softmax_branch_router_is_initially_neutral():
    """Zero-init softmax routing should preserve the baseline forward path."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    spec = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
    )
    from core_amplifier_lm import CoreAmplifierLM

    baseline = CoreAmplifierLM(spec, core_layers=2, amplifier_dtype=torch.float32)
    routed = CoreAmplifierLM(
        spec,
        core_layers=2,
        branch_router_mode="softmax",
        amplifier_dtype=torch.float32,
    )
    routed.load_state_dict(baseline.state_dict(), strict=False)

    x = torch.randint(0, VOCAB, (2, 17))
    with torch.no_grad():
        ref = baseline(x[:, :-1])
        out = routed(x[:, :-1])
    assert (ref - out).abs().max().item() < 1e-5


def test_hold_then_cosine_lr_schedule():
    """The controller LR should warm up, hold, then decay."""
    from train_core_amplifier import get_lr

    kwargs = dict(max_lr=3e-3, min_lr=3e-4, warmup_steps=10, hold_steps=20, total_steps=100)
    assert abs(get_lr(0, **kwargs) - 3e-4) < 1e-12
    assert abs(get_lr(9, **kwargs) - 3e-3) < 1e-12
    assert abs(get_lr(10, **kwargs) - 3e-3) < 1e-12
    assert abs(get_lr(29, **kwargs) - 3e-3) < 1e-12
    assert get_lr(60, **kwargs) < 3e-3
    assert abs(get_lr(100, **kwargs) - 3e-4) < 1e-12


def test_record_defaults_match_recommended_run():
    """Fresh init defaults should reflect the recommended controller settings."""
    model = DEFAULTS["model"]
    train = DEFAULTS["training"]
    assert model["core_layers"] == 5
    assert model["core_expansion"] == 2.0
    assert model["residual_core"] is True
    assert model["branch_temporal_mode"] == "current"
    assert model["branch_temporal_lag_scale"] == 1.0
    assert model["residual_token_gate_mode"] == "none"
    assert model["branch_router_mode"] == "none"
    assert train["batch_size"] == 256
    assert train["carry_chunks"] == 16
    assert train["bptt_chunks"] == 2
    assert train["learning_rate"] == 3e-3
    assert train["warmup_steps"] == 100
    assert train["lr_hold_steps"] == 1500
    assert train["min_lr"] == 3e-4
    assert train["scan_backend"] == "auto"


def test_zero_block_spec_and_model():
    """num_blocks=0 should build a valid zero-block frozen spec and usable model."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    spec = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=0,
        fixed_dtype=torch.float16,
    )
    assert spec.num_blocks == 0
    assert spec.amp_w1.shape == (0, spec.amp_dim, spec.amp_dim)
    assert spec.amp_w2.shape == (0, spec.amp_dim, spec.amp_dim)

    from core_amplifier_lm import CoreAmplifierLM

    model = CoreAmplifierLM(spec, core_layers=2, amplifier_dtype=torch.float32)
    x = torch.randint(0, VOCAB, (2, 17))
    with torch.no_grad():
        logits = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, VOCAB), x[:, 1:].reshape(-1)
        )
    assert not torch.isnan(loss) and not torch.isinf(loss)


def test_low_rank_readout_spec_and_model():
    """Low-rank readout should reduce frozen bytes while keeping the model usable."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.int64)
    full = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
        readout_rank=None,
    )
    low_rank = build_amplifier_spec(
        tokens,
        vocab_size=VOCAB,
        core_dim=8,
        branch_lags=(1, 2, 4),
        num_blocks=1,
        fixed_dtype=torch.float16,
        readout_rank=4,
    )

    assert low_rank.readout_weight is None
    assert low_rank.readout_in_proj is not None
    assert low_rank.readout_out_proj is not None
    assert low_rank.readout_rank == 4
    assert low_rank.fixed_nbytes < full.fixed_nbytes

    from core_amplifier_lm import CoreAmplifierLM

    model = CoreAmplifierLM(low_rank, core_layers=2, amplifier_dtype=torch.float32)
    x = torch.randint(0, VOCAB, (2, 17))
    with torch.no_grad():
        logits = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, VOCAB), x[:, 1:].reshape(-1)
        )
    assert not torch.isnan(loss) and not torch.isinf(loss)


def test_train_val_split():
    """load_train_val_int32 returns correct splits from shard dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, shard_dir = _make_shard_dir(tmpdir)
        train, val = load_train_val_int32(shard_dir, storage_dtype="uint16")

        assert train.dtype == np.int32
        assert val.dtype == np.int32
        assert train.size > 0
        assert val.size > 0

        val_direct = np.fromfile(shard_dir / "fineweb_val_000000.bin", dtype=np.uint16).astype(
            np.int32
        )
        assert np.array_equal(val, val_direct)


def test_train_val_split_requires_explicit_val_shard():
    """Directory fineweb shards should fail without the provided validation shard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, shard_dir = _make_train_only_shard_dir(tmpdir)
        try:
            load_train_val_int32(shard_dir, storage_dtype="uint16")
        except FileNotFoundError as exc:
            assert "fineweb_val_*" in str(exc)
        else:
            raise AssertionError("expected explicit validation shard requirement")


def test_train_val_split_can_fallback_when_explicitly_allowed():
    """Train-split fallback should require an explicit opt-in."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tokens, shard_dir = _make_train_only_shard_dir(tmpdir)
        train, val = load_train_val_int32(
            shard_dir,
            storage_dtype="uint16",
            allow_train_frac_val_split=True,
        )
        split = int(TOTAL * 0.98)
        split = max(1, min(TOTAL - 1, split))
        assert np.array_equal(train, tokens[:split].astype(np.int32))
        assert np.array_equal(val, tokens[split:].astype(np.int32))


def test_mmap_train_val_requires_explicit_val_shard():
    """Mmap loader should also reject train-only shard directories by default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, shard_dir = _make_train_only_shard_dir(tmpdir)
        try:
            mmap_train_val(shard_dir, storage_dtype="uint16", verbose=False)
        except FileNotFoundError as exc:
            assert "fineweb_val_*" in str(exc)
        else:
            raise AssertionError("expected explicit validation shard requirement")


def test_spec_save_load_roundtrip():
    """Spec survives save → load cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, gz_path = _make_gz(tmpdir)
        spec = build_spec_optimized(
            gz_path,
            vocab_size=VOCAB,
            core_dim=8,
            branch_lags=LAGS,
            num_blocks=1,
            fixed_dtype=torch.float16,
            storage_dtype="uint8",
            verbose=False,
        )
        save_path = tmpdir / "spec.pt"
        spec.save(save_path)
        loaded = AmplifierSpec.load(save_path)

        assert loaded.core_dim == spec.core_dim
        assert loaded.vocab_size == spec.vocab_size
        assert loaded.branch_lags == spec.branch_lags
        _assert_specs_equal(spec, loaded)


# ---------------------------------------------------------------------------
# Integration test: run the training script
# ---------------------------------------------------------------------------


def test_training_script_shard_dir():
    """Full training run with sharded data directory via model-dir workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, shard_dir = _make_shard_dir(tmpdir)
        model_dir = tmpdir / "model"

        # init model dir
        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(shard_dir),
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--num-blocks",
                "1",
                "--spec-strategy",
                "stream",
            ]
        )
        assert (model_dir / "config.json").exists()
        assert (model_dir / "spec.pt").exists()

        # train
        _run_train(
            [
                str(model_dir),
                "--num-steps",
                "3",
                "--seq-len",
                "32",
                "--batch-size",
                "4",
                "--carry-chunks",
                "2",
                "--val-every",
                "1",
                "--val-steps",
                "1",
                "--log-every",
                "1",
                "--learning-rate",
                "1e-3",
                "--hard-loss-gamma",
                "1.0",
                "--lr-schedule",
                "none",
                "--force-device",
                "cpu",
                "--no-mmap",
            ],
            expect_in_stdout="Training complete",
        )


def test_inspect_init_max_tokens_alias():
    """inspect_model --max-tokens should populate spec.max_tokens and truncate the build."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, gz_path = _make_gz(tmpdir)
        model_dir = tmpdir / "model_max_tokens"

        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(gz_path),
                "--storage-dtype",
                "uint8",
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--num-blocks",
                "1",
                "--max-tokens",
                "12345",
                "--spec-strategy",
                "stream",
            ]
        )

        spec = AmplifierSpec.load(model_dir / "spec.pt")
        assert int(spec.metadata.get("total_tokens", 0)) == 12345
        cfg = (model_dir / "config.json").read_text()
        assert "12345" in cfg


def test_training_script_shard_dir_streaming_loader():
    """Shard directories should train without building a flat mmap cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, shard_dir = _make_shard_dir(tmpdir)
        model_dir = tmpdir / "model_stream"

        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(shard_dir),
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--num-blocks",
                "1",
                "--spec-strategy",
                "stream",
            ]
        )

        stdout = _run_train(
            [
                str(model_dir),
                "--num-steps",
                "2",
                "--seq-len",
                "32",
                "--batch-size",
                "4",
                "--carry-chunks",
                "2",
                "--val-every",
                "1",
                "--val-steps",
                "1",
                "--log-every",
                "1",
                "--learning-rate",
                "1e-3",
                "--lr-schedule",
                "none",
                "--force-device",
                "cpu",
            ],
            expect_in_stdout="Training complete",
        )
        assert "Direct shard streaming" in stdout
        assert not (shard_dir / ".mmap_cache").exists()


def test_training_script_shard_dir_data_max_tokens():
    """data_max_tokens should truncate direct shard streaming instead of silently reading all shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, shard_dir = _make_shard_dir(tmpdir)
        model_dir = tmpdir / "model_stream_limited"

        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(shard_dir),
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--num-blocks",
                "1",
                "--spec-strategy",
                "stream",
            ]
        )

        stdout = _run_train(
            [
                str(model_dir),
                "--num-steps",
                "2",
                "--seq-len",
                "32",
                "--batch-size",
                "4",
                "--carry-chunks",
                "2",
                "--val-every",
                "1",
                "--val-steps",
                "1",
                "--log-every",
                "1",
                "--learning-rate",
                "1e-3",
                "--lr-schedule",
                "none",
                "--force-device",
                "cpu",
                "--data-max-tokens",
                "2000",
            ],
            expect_in_stdout="Training complete",
        )
        assert "data_max_tokens=2,000" in stdout
        assert "train=2,000" in stdout


def test_training_fails_on_spec_config_mismatch():
    """Training should fail fast when config.json no longer matches spec.pt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, gz_path = _make_gz(tmpdir)
        model_dir = tmpdir / "model_mismatch"

        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(gz_path),
                "--storage-dtype",
                "uint8",
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--num-blocks",
                "1",
                "--spec-strategy",
                "stream",
            ]
        )

        cfg_path = model_dir / "config.json"
        cfg = cfg_path.read_text()
        cfg = cfg.replace('"core_dim": 8', '"core_dim": 12')
        cfg_path.write_text(cfg)

        cmd = [
            sys.executable,
            str(PKG_ROOT / "train_core_amplifier.py"),
            str(model_dir),
            "--force-device",
            "cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PKG_ROOT))
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "config.json does not match spec.pt" in combined


def test_training_script_single_file():
    """Full training run with single .gz file via model-dir workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, gz_path = _make_gz(tmpdir)
        model_dir = tmpdir / "model"

        # init model dir
        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(gz_path),
                "--storage-dtype",
                "uint8",
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--num-blocks",
                "1",
                "--spec-strategy",
                "stream",
            ]
        )

        # train
        _run_train(
            [
                str(model_dir),
                "--num-steps",
                "2",
                "--seq-len",
                "32",
                "--batch-size",
                "4",
                "--carry-chunks",
                "2",
                "--val-every",
                "1",
                "--val-steps",
                "1",
                "--log-every",
                "1",
                "--learning-rate",
                "1e-3",
                "--lr-schedule",
                "none",
                "--force-device",
                "cpu",
                "--no-mmap",
            ],
            expect_in_stdout="Training complete",
        )


def test_training_script_lagged_branch_mode():
    """Training should support lagged branch taps and record them in resolved config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, gz_path = _make_gz(tmpdir)
        model_dir = tmpdir / "model_lagged"

        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(gz_path),
                "--storage-dtype",
                "uint8",
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--branch-temporal-mode",
                "lagged",
                "--num-blocks",
                "1",
                "--spec-strategy",
                "stream",
            ]
        )

        _run_train(
            [
                str(model_dir),
                "--num-steps",
                "2",
                "--seq-len",
                "32",
                "--batch-size",
                "4",
                "--carry-chunks",
                "2",
                "--val-every",
                "1",
                "--val-steps",
                "1",
                "--log-every",
                "1",
                "--learning-rate",
                "1e-3",
                "--lr-schedule",
                "none",
                "--force-device",
                "cpu",
                "--no-mmap",
            ],
            expect_in_stdout="Training complete",
        )

        resolved = (model_dir / "resolved_config.json").read_text()
        assert '"branch_temporal_mode": "lagged"' in resolved


def test_training_script_hybrid_branch_mode():
    """Training should support hybrid branch taps and preserve lag scale in config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, gz_path = _make_gz(tmpdir)
        model_dir = tmpdir / "model_hybrid"

        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(gz_path),
                "--storage-dtype",
                "uint8",
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--branch-temporal-mode",
                "hybrid",
                "--branch-temporal-lag-scale",
                "0.75",
                "--num-blocks",
                "1",
                "--spec-strategy",
                "stream",
            ]
        )

        _run_train(
            [
                str(model_dir),
                "--num-steps",
                "2",
                "--seq-len",
                "32",
                "--batch-size",
                "4",
                "--carry-chunks",
                "2",
                "--val-every",
                "1",
                "--val-steps",
                "1",
                "--log-every",
                "1",
                "--learning-rate",
                "1e-3",
                "--lr-schedule",
                "none",
                "--force-device",
                "cpu",
                "--no-mmap",
            ],
            expect_in_stdout="Training complete",
        )

        resolved = (model_dir / "resolved_config.json").read_text()
        assert '"branch_temporal_mode": "hybrid"' in resolved
        assert '"branch_temporal_lag_scale": 0.75' in resolved


def test_training_script_gradient_checkpointing_exports_quantized_payload():
    """Training should support checkpointed minGRU runs and write the final int8 payload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, gz_path = _make_gz(tmpdir)
        model_dir = tmpdir / "model_ckpt"

        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(gz_path),
                "--storage-dtype",
                "uint8",
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--num-blocks",
                "1",
                "--spec-strategy",
                "stream",
            ]
        )

        _run_train(
            [
                str(model_dir),
                "--num-steps",
                "2",
                "--seq-len",
                "32",
                "--batch-size",
                "4",
                "--carry-chunks",
                "2",
                "--val-every",
                "1",
                "--val-steps",
                "1",
                "--log-every",
                "1",
                "--learning-rate",
                "1e-3",
                "--lr-schedule",
                "none",
                "--gradient-checkpointing",
                "--force-device",
                "cpu",
                "--no-mmap",
            ],
            expect_in_stdout="Training complete",
        )

        resolved = (model_dir / "resolved_config.json").read_text()
        assert '"gradient_checkpointing": true' in resolved
        payload_path = model_dir / "final_trainable.int8.ptz"
        assert payload_path.exists()
        assert payload_path.stat().st_size > 0


def test_training_script_architecture_modes_roundtrip():
    """Training should round-trip new gating, routing, temporal, and scan modes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, gz_path = _make_gz(tmpdir)
        model_dir = tmpdir / "model_arch_modes"

        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(gz_path),
                "--storage-dtype",
                "uint8",
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--branch-temporal-mode",
                "ema_hybrid",
                "--residual-token-gate-mode",
                "core_base",
                "--branch-router-mode",
                "softmax",
                "--scan-backend",
                "assoc",
                "--num-blocks",
                "1",
                "--spec-strategy",
                "stream",
            ]
        )

        _run_train(
            [
                str(model_dir),
                "--num-steps",
                "2",
                "--seq-len",
                "32",
                "--batch-size",
                "4",
                "--carry-chunks",
                "2",
                "--val-every",
                "1",
                "--val-steps",
                "1",
                "--log-every",
                "1",
                "--learning-rate",
                "1e-3",
                "--lr-schedule",
                "none",
                "--force-device",
                "cpu",
                "--no-mmap",
            ],
            expect_in_stdout="Training complete",
        )

        resolved = json.loads((model_dir / "resolved_config.json").read_text())
        assert resolved["model"]["branch_temporal_mode"] == "ema_hybrid"
        assert resolved["model"]["residual_token_gate_mode"] == "core_base"
        assert resolved["model"]["branch_router_mode"] == "softmax"
        assert resolved["runtime"]["scan_backend_requested"] == "assoc"
        assert resolved["runtime"]["scan_backend_active"] == "assoc"


def test_training_script_records_step_token_contract():
    """Resolved config should record local and effective step-token counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, gz_path = _make_gz(tmpdir)
        model_dir = tmpdir / "model_step_tokens"

        _run_inspect(
            [
                "init",
                str(model_dir),
                "--data",
                str(gz_path),
                "--storage-dtype",
                "uint8",
                "--vocab-size",
                "256",
                "--core-dim",
                "8",
                "--branch-lags",
                "1,2,4",
                "--num-blocks",
                "1",
                "--spec-strategy",
                "stream",
            ]
        )

        _run_train(
            [
                str(model_dir),
                "--num-steps",
                "2",
                "--seq-len",
                "32",
                "--batch-size",
                "4",
                "--grad-accum",
                "2",
                "--bptt-chunks",
                "3",
                "--carry-chunks",
                "2",
                "--val-every",
                "1",
                "--val-steps",
                "1",
                "--log-every",
                "1",
                "--learning-rate",
                "1e-3",
                "--lr-schedule",
                "none",
                "--force-device",
                "cpu",
                "--no-mmap",
            ],
            expect_in_stdout="Training complete",
        )

        resolved = json.loads((model_dir / "resolved_config.json").read_text())
        training = resolved["training"]
        assert training["local_step_tokens"] == 4 * 32 * 3
        assert training["effective_step_tokens"] == 4 * 32 * 3 * 2
        assert training["planned_train_tokens"] == 4 * 32 * 3 * 2 * 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_specs_equal(a: AmplifierSpec, b: AmplifierSpec, tol: float = 1e-4) -> None:
    for name in (
        "token_embed",
        "base_bigram_logits",
        "lag_ops",
        "amp_w1",
        "amp_w2",
        "readout_weight",
    ):
        ta = getattr(a, name).float()
        tb = getattr(b, name).float()
        diff = (ta - tb).abs().max().item()
        assert diff < tol, f"{name} max_diff={diff:.6e}"


def _run_inspect(args: list[str], expect_in_stdout: str = "") -> str:
    cmd = [sys.executable, str(PKG_ROOT / "inspect_model.py")] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PKG_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f"inspect failed (rc={result.returncode})\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        )
    if expect_in_stdout:
        assert expect_in_stdout in result.stdout, f"expected '{expect_in_stdout}' in stdout"
    return result.stdout


def _run_train(args: list[str], expect_in_stdout: str = "") -> str:
    cmd = [sys.executable, str(PKG_ROOT / "train_core_amplifier.py")] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PKG_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f"train failed (rc={result.returncode})\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        )
    if expect_in_stdout:
        assert expect_in_stdout in result.stdout, f"expected '{expect_in_stdout}' in stdout"
    return result.stdout


def _write_competition_shard(path: Path, tokens: np.ndarray) -> None:
    """Write a .bin shard with the competition header format."""
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520  # magic
    header[1] = 1  # version
    header[2] = len(tokens)
    toks_u16 = tokens.astype("<u2", copy=False)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks_u16.tobytes())


def _make_header_shard_dir(tmpdir: Path) -> tuple[np.ndarray, Path]:
    """Create shards WITH the competition header format."""
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, VOCAB, size=TOTAL, dtype=np.int32)
    shard_dir = tmpdir / "header_shards"
    shard_dir.mkdir()
    train_split = int(TOTAL * 0.85)
    offset, idx = 0, 0
    while offset < train_split:
        end = min(offset + 12_000, train_split)
        _write_competition_shard(shard_dir / f"fineweb_train_{idx:06d}.bin", tokens[offset:end])
        offset = end
        idx += 1
    _write_competition_shard(shard_dir / "fineweb_val_000000.bin", tokens[train_split:])
    return tokens, shard_dir


def test_header_shard_loading():
    """Shards with competition header must load correctly, skipping header bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tokens, shard_dir = _make_header_shard_dir(tmpdir)
        loaded = load_tokens_int32(shard_dir, storage_dtype="uint16")

        # Should only get train shards (load_tokens_int32 prefers train if present)
        train_split = int(TOTAL * 0.85)
        expected = tokens[:train_split].astype(np.int32)
        assert loaded.size == expected.size, f"size mismatch: {loaded.size} vs {expected.size}"
        assert np.array_equal(loaded, expected), "header shard content mismatch"
        assert loaded.max() < VOCAB, (
            f"max token {loaded.max()} >= vocab {VOCAB} — header not skipped?"
        )
        print(f"  loaded {loaded.size} tokens, max={loaded.max()}")


def test_header_shard_train_val():
    """Train/val split from header-format shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tokens, shard_dir = _make_header_shard_dir(tmpdir)
        train, val = load_train_val_int32(shard_dir, storage_dtype="uint16")

        train_split = int(TOTAL * 0.85)
        expected_train = tokens[:train_split].astype(np.int32)
        expected_val = tokens[train_split:].astype(np.int32)
        assert np.array_equal(train, expected_train), "train mismatch"
        assert np.array_equal(val, expected_val), "val mismatch"
        assert train.max() < VOCAB
        assert val.max() < VOCAB


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_single_pass_counts_match,
        test_spec_match_single_file,
        test_spec_match_shard_dir,
        test_train_val_split,
        test_spec_save_load_roundtrip,
        test_header_shard_loading,
        test_header_shard_train_val,
        test_training_script_shard_dir,
        test_inspect_init_max_tokens_alias,
        test_training_script_shard_dir_streaming_loader,
        test_training_script_single_file,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
