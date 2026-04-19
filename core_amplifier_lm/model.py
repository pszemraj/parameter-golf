"""Core/amplifier language model in PyTorch.

This module implements a concrete version of the core/amplifier idea:

- a tiny trainable recurrent core
- a large frozen amplifier built from corpus statistics
- a cheap fixed bigram path to absorb easy tokens
- a residual path that lets the learned core correct the fixed local model

The fixed buffers are stored in compact dtypes inside the spec/checkpoint. At
runtime the model materializes one cached copy of the frozen amplifier in the
chosen compute dtype. This keeps the CPU path stable and avoids repeated casts
inside the inner loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

ArrayLike = np.ndarray | torch.Tensor
BranchTemporalState: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _to_numpy_1d(tokens: ArrayLike) -> np.ndarray:
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy()
    arr = np.asarray(tokens)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D token array, got shape {arr.shape}")
    return arr.astype(np.int64, copy=False)


def _validate_tokens(tokens: np.ndarray, vocab_size: int) -> None:
    if tokens.size == 0:
        raise ValueError("token array is empty")
    token_min = int(tokens.min())
    token_max = int(tokens.max())
    if token_min < 0 or token_max >= vocab_size:
        raise ValueError(
            f"token ids must lie in [0, {vocab_size - 1}], got min={token_min}, max={token_max}"
        )


def _normalize_branch_lags(branch_lags: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(lag) for lag in branch_lags)
    if not out:
        raise ValueError("branch_lags must be non-empty")
    if len(set(out)) != len(out):
        raise ValueError(f"branch_lags must be unique, got {out}")
    if any(lag <= 0 for lag in out):
        raise ValueError(f"all lags must be positive, got {out}")
    return out


def _normalize_branch_temporal_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"current", "lagged", "hybrid"}:
        raise ValueError(
            f"unknown branch_temporal_mode {mode!r}; expected 'current', 'lagged', or 'hybrid'"
        )
    return normalized


def _count_unigrams(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    return np.bincount(tokens, minlength=vocab_size).astype(np.float64, copy=False)


def _count_pairs(left: np.ndarray, right: np.ndarray, vocab_size: int) -> np.ndarray:
    flat = left * vocab_size + right
    counts = np.bincount(flat, minlength=vocab_size * vocab_size)
    return counts.reshape(vocab_size, vocab_size).astype(np.float64, copy=False)


def _count_bigrams(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    return _count_pairs(tokens[:-1], tokens[1:], vocab_size=vocab_size)


def _count_lag_pairs(tokens: np.ndarray, lag: int, vocab_size: int) -> np.ndarray:
    if lag <= 0:
        raise ValueError(f"lag must be positive, got {lag}")
    if lag >= tokens.size:
        raise ValueError(f"lag {lag} exceeds token length {tokens.size}")
    return _count_pairs(tokens[:-lag], tokens[lag:], vocab_size=vocab_size)


def _smoothed_log_conditional(counts: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    counts = counts + alpha
    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probs)


def _spectral_scale(matrix: np.ndarray, target_radius: float = 0.95) -> np.ndarray:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    top = float(singular_values[0]) if singular_values.size > 0 else 1.0
    if top < 1e-8:
        return matrix.astype(np.float32, copy=False)
    return (matrix * (target_radius / top)).astype(np.float32, copy=False)


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _effective_rank(singular_values: np.ndarray) -> float:
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > 1e-12]
    if s.size == 0:
        return 1.0
    probs = s / s.sum()
    entropy = -(probs * np.log(probs + 1e-12)).sum()
    return float(np.exp(entropy))


def _topk_row_sparsify(matrix: np.ndarray, k: int) -> np.ndarray:
    """Keep the top-k off-diagonal entries per row, zeroing the rest."""
    n = matrix.shape[0]
    if k <= 0 or k >= n - 1:
        out = matrix.copy()
        np.fill_diagonal(out, 0.0)
        return out
    idx = np.argpartition(matrix, kth=n - k, axis=1)[:, -k:]
    out = np.zeros_like(matrix)
    rows = np.arange(n)[:, None]
    out[rows, idx] = matrix[rows, idx]
    np.fill_diagonal(out, 0.0)
    return out


def _umap_style_spectral_basis(
    log_bigram: np.ndarray,
    *,
    dim: int,
    neighbors: int = 64,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Approximate UMAP-style spectral initialization from the bigram graph."""
    n = log_bigram.shape[0]
    dim = max(1, min(int(dim), n - 1))
    neighbors = max(1, min(int(neighbors), n - 1))

    cond = np.exp(log_bigram).astype(np.float64, copy=False)
    np.fill_diagonal(cond, 0.0)
    directed = _topk_row_sparsify(cond, neighbors)
    graph = directed + directed.T - directed * directed.T
    graph = np.maximum(graph, 0.0)

    degree = graph.sum(axis=1)
    if np.any(degree <= 1e-12):
        graph = graph + np.eye(n, dtype=np.float64) * 1e-6
        degree = graph.sum(axis=1)

    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = (graph * inv_sqrt[:, None]) * inv_sqrt[None, :]
    laplacian = np.eye(n, dtype=np.float64) - normalized

    evals, evecs = np.linalg.eigh(laplacian)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    non_trivial = np.flatnonzero(evals > 1e-10)
    if non_trivial.size >= dim:
        take = non_trivial[:dim]
    else:
        start = 1 if n > 1 else 0
        take = np.arange(start, min(start + dim, n))
    basis = evecs[:, take].astype(np.float64, copy=False)
    return basis, evals[take].astype(np.float64, copy=False), neighbors


def _factorize_bigram_residual(
    residual: np.ndarray,
    log_bigram: np.ndarray,
    *,
    core_dim: int,
    embedding_init: str = "spectral",
    spectral_neighbors: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build token_embed / token_out from the bigram residual matrix."""
    embedding_init = str(embedding_init).lower()
    effective_core_dim = max(1, min(int(core_dim), residual.shape[0], residual.shape[1]))

    def _svd_factorization() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        u, s, vt = np.linalg.svd(residual, full_matrices=False)
        s = np.maximum(s[:effective_core_dim], 1e-8)
        token_embed = (u[:, :effective_core_dim] * np.sqrt(s)).astype(np.float32)
        token_out = (vt[:effective_core_dim].T * np.sqrt(s)).astype(np.float32)
        return (
            token_embed,
            token_out,
            {
                "embedding_init": "svd",
                "spectral_neighbors": 0,
                "embedding_singular_values": s.tolist(),
            },
        )

    if embedding_init == "svd":
        return _svd_factorization()
    if embedding_init != "spectral":
        raise ValueError(f"unknown embedding_init {embedding_init!r}; expected 'spectral' or 'svd'")

    try:
        basis, evals, neighbors_used = _umap_style_spectral_basis(
            log_bigram,
            dim=effective_core_dim,
            neighbors=spectral_neighbors,
        )
        projected = basis.T @ residual @ basis
        ub, sb, vtb = np.linalg.svd(projected, full_matrices=False)
        sb = np.maximum(sb[:effective_core_dim], 1e-8)
        token_embed = (basis @ (ub[:, :effective_core_dim] * np.sqrt(sb))).astype(np.float32)
        token_out = (basis @ (vtb[:effective_core_dim].T * np.sqrt(sb))).astype(np.float32)
        return (
            token_embed,
            token_out,
            {
                "embedding_init": "spectral",
                "spectral_neighbors": int(neighbors_used),
                "spectral_eigenvalues": evals[:effective_core_dim].tolist(),
                "embedding_singular_values": sb.tolist(),
            },
        )
    except np.linalg.LinAlgError:
        token_embed, token_out, meta = _svd_factorization()
        meta["embedding_init"] = "svd_fallback"
        return token_embed, token_out, meta


def _build_lag_operator(
    projected: np.ndarray,
    *,
    identity_base: float = 0.15,
    min_identity_blend: float = 0.01,
    target_radius: float = 0.95,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build one lag operator without tanh-squashing the spectrum."""
    u2, s2, vt2 = np.linalg.svd(projected, full_matrices=False)
    if s2.size == 0:
        dim = projected.shape[0]
        eye = np.eye(dim, dtype=np.float32)
        return _spectral_scale(eye, target_radius=target_radius), {
            "effective_rank": 1.0,
            "identity_blend": float(identity_base),
        }

    top = max(float(s2[0]), 1e-8)
    s_rel = (s2 / top).astype(np.float64, copy=False)
    op = (u2 * s_rel) @ vt2
    erank = _effective_rank(s2)
    dim = max(1, projected.shape[0])
    blend = max(float(min_identity_blend), float(identity_base) * erank / float(dim))
    eye = np.eye(dim, dtype=np.float32)
    op = (1.0 - blend) * op + blend * eye
    return _spectral_scale(op, target_radius=target_radius), {
        "effective_rank": float(erank),
        "identity_blend": float(blend),
        "top_singular": float(top),
    }


def _factorize_readout_matrix(
    weight: np.ndarray,
    *,
    rank: Optional[int],
) -> tuple[
    Optional[np.ndarray], Optional[tuple[np.ndarray, np.ndarray]], dict[str, float | int | str]
]:
    """Optionally factorize a [vocab, amp_dim] readout into two smaller matrices."""
    if rank is None:
        return weight.astype(np.float32, copy=False), None, {"readout_type": "full"}

    vocab_size, amp_dim = weight.shape
    max_rank = min(vocab_size, amp_dim)
    if rank <= 0 or rank >= max_rank:
        return weight.astype(np.float32, copy=False), None, {"readout_type": "full"}

    u, s, vt = np.linalg.svd(weight.astype(np.float64, copy=False), full_matrices=False)
    kept = int(rank)
    s_kept = s[:kept]
    sqrt_s = np.sqrt(np.clip(s_kept, a_min=1e-12, a_max=None))
    out_proj = (u[:, :kept] * sqrt_s[None, :]).astype(np.float32)
    in_proj = (sqrt_s[:, None] * vt[:kept, :]).astype(np.float32)
    energy = float((s_kept * s_kept).sum() / np.clip((s * s).sum(), a_min=1e-12, a_max=None))
    return (
        None,
        (in_proj, out_proj),
        {
            "readout_type": "low_rank",
            "readout_rank": kept,
            "readout_energy": energy,
        },
    )


@dataclass
class AmplifierSpec:
    vocab_size: int
    core_dim: int
    branch_lags: tuple[int, ...]
    num_blocks: int
    token_embed: torch.Tensor
    base_bigram_logits: torch.Tensor
    lag_ops: torch.Tensor
    amp_w1: torch.Tensor
    amp_w2: torch.Tensor
    readout_weight: Optional[torch.Tensor] = None
    readout_in_proj: Optional[torch.Tensor] = None
    readout_out_proj: Optional[torch.Tensor] = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def num_branches(self) -> int:
        return len(self.branch_lags)

    @property
    def amp_dim(self) -> int:
        return self.num_branches * self.core_dim

    @property
    def fixed_nbytes(self) -> int:
        total = (
            self.token_embed.element_size() * self.token_embed.numel()
            + self.base_bigram_logits.element_size() * self.base_bigram_logits.numel()
            + self.lag_ops.element_size() * self.lag_ops.numel()
            + self.amp_w1.element_size() * self.amp_w1.numel()
            + self.amp_w2.element_size() * self.amp_w2.numel()
        )
        if self.readout_weight is not None:
            total += self.readout_weight.element_size() * self.readout_weight.numel()
        if self.readout_in_proj is not None:
            total += self.readout_in_proj.element_size() * self.readout_in_proj.numel()
        if self.readout_out_proj is not None:
            total += self.readout_out_proj.element_size() * self.readout_out_proj.numel()
        return total

    @property
    def readout_rank(self) -> Optional[int]:
        if self.readout_in_proj is None:
            return None
        return int(self.readout_in_proj.shape[0])

    def summary(self) -> str:
        mb = self.fixed_nbytes / 1_000_000
        mib = self.fixed_nbytes / (1024 * 1024)
        requested_core_dim = self.metadata.get("requested_core_dim", self.core_dim)
        suffix = ""
        if int(requested_core_dim) != self.core_dim:
            suffix = f", requested_core_dim={requested_core_dim}"
        readout = "full" if self.readout_rank is None else f"rank{self.readout_rank}"
        return (
            f"AmplifierSpec(vocab={self.vocab_size}, core_dim={self.core_dim}, "
            f"branches={self.num_branches}, amp_dim={self.amp_dim}, blocks={self.num_blocks}, "
            f"readout={readout}, fixed={mb:.2f} MB / {mib:.2f} MiB{suffix})"
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "core_dim": self.core_dim,
            "branch_lags": self.branch_lags,
            "num_blocks": self.num_blocks,
            "token_embed": self.token_embed.cpu(),
            "base_bigram_logits": self.base_bigram_logits.cpu(),
            "lag_ops": self.lag_ops.cpu(),
            "amp_w1": self.amp_w1.cpu(),
            "amp_w2": self.amp_w2.cpu(),
            "readout_weight": None if self.readout_weight is None else self.readout_weight.cpu(),
            "readout_in_proj": None if self.readout_in_proj is None else self.readout_in_proj.cpu(),
            "readout_out_proj": None
            if self.readout_out_proj is None
            else self.readout_out_proj.cpu(),
            "metadata": self.metadata,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "AmplifierSpec":
        try:
            payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location=map_location)
        return cls(
            vocab_size=int(payload["vocab_size"]),
            core_dim=int(payload["core_dim"]),
            branch_lags=tuple(int(x) for x in payload["branch_lags"]),
            num_blocks=int(payload["num_blocks"]),
            token_embed=payload["token_embed"],
            base_bigram_logits=payload["base_bigram_logits"],
            lag_ops=payload["lag_ops"],
            amp_w1=payload["amp_w1"],
            amp_w2=payload["amp_w2"],
            readout_weight=payload.get("readout_weight"),
            readout_in_proj=payload.get("readout_in_proj"),
            readout_out_proj=payload.get("readout_out_proj"),
            metadata=dict(payload.get("metadata", {})),
        )


def _build_spec_from_counts(
    *,
    unigram: np.ndarray,
    bigram: np.ndarray,
    lag_pair_counts: dict[int, np.ndarray],
    vocab_size: int,
    core_dim: int,
    branch_lags: Sequence[int],
    num_blocks: int,
    smoothing: float,
    fixed_dtype: torch.dtype,
    embedding_init: str = "spectral",
    spectral_neighbors: int = 64,
    lag_identity_base: float = 0.15,
    readout_rank: Optional[int] = None,
    metadata: Optional[dict[str, object]] = None,
) -> AmplifierSpec:
    branch_lags = _normalize_branch_lags(branch_lags)
    log_bigram = _smoothed_log_conditional(bigram, alpha=smoothing)
    unigram_probs = (unigram + smoothing) / (unigram.sum() + smoothing * vocab_size)
    log_unigram = np.log(unigram_probs)
    residual = log_bigram - log_unigram[None, :]

    token_embed, token_out, basis_meta = _factorize_bigram_residual(
        residual,
        log_bigram,
        core_dim=core_dim,
        embedding_init=embedding_init,
        spectral_neighbors=spectral_neighbors,
    )
    effective_core_dim = token_embed.shape[1]

    lag_ops_list: list[np.ndarray] = []
    lag_meta: dict[str, dict[str, float]] = {}
    for lag in branch_lags:
        lag_counts = lag_pair_counts[lag]
        log_lag = _smoothed_log_conditional(lag_counts, alpha=smoothing)
        lag_residual = log_lag - log_unigram[None, :]
        projected = token_embed.T @ lag_residual @ token_out
        op, op_meta = _build_lag_operator(
            projected,
            identity_base=lag_identity_base,
            min_identity_blend=0.01,
            target_radius=0.95,
        )
        lag_ops_list.append(op)
        lag_meta[str(lag)] = op_meta
    lag_ops = np.stack(lag_ops_list, axis=0).astype(np.float32)

    num_branches = len(branch_lags)
    amp_dim = num_branches * effective_core_dim

    def make_block(offset: int, transpose: bool) -> np.ndarray:
        out = np.zeros((amp_dim, amp_dim), dtype=np.float32)
        for row in range(num_branches):
            for col in range(num_branches):
                if transpose:
                    idx = (2 * row + col + offset) % num_branches
                    block = lag_ops[idx].T
                else:
                    idx = (row + 2 * col + offset) % num_branches
                    block = lag_ops[idx]
                sign = 1.0 if ((row + col + offset) % 2 == 0) else -1.0
                rs = slice(row * effective_core_dim, (row + 1) * effective_core_dim)
                cs = slice(col * effective_core_dim, (col + 1) * effective_core_dim)
                out[rs, cs] = sign * block
        out /= np.sqrt(float(num_branches))
        return _spectral_scale(out, target_radius=0.92)

    if num_blocks > 0:
        amp_w1 = np.stack([make_block(i, transpose=False) for i in range(num_blocks)], axis=0)
        amp_w2 = np.stack(
            [make_block(i + num_branches // 2, transpose=True) for i in range(num_blocks)],
            axis=0,
        )
    else:
        amp_w1 = np.zeros((0, amp_dim, amp_dim), dtype=np.float32)
        amp_w2 = np.zeros((0, amp_dim, amp_dim), dtype=np.float32)

    branch_readouts = []
    token_out_t = token_out.T
    for op in lag_ops_list:
        branch_readouts.append((op @ token_out_t).astype(np.float32))
    readout = np.concatenate(branch_readouts, axis=0).astype(np.float32)
    readout /= np.sqrt(float(num_branches))
    readout_weight_full = readout.T.astype(np.float32, copy=False)
    readout_weight, readout_factors, readout_meta = _factorize_readout_matrix(
        readout_weight_full,
        rank=readout_rank,
    )

    out_metadata: dict[str, object] = dict(metadata or {})
    out_metadata.update(
        {
            "description": "count-derived core/amplifier basis",
            "requested_core_dim": int(core_dim),
            "effective_core_dim": int(effective_core_dim),
            "fixed_dtype": str(fixed_dtype).replace("torch.", ""),
            "lag_identity_base": float(lag_identity_base),
            "lag_operator_scale": "relative_to_top_singular_value",
            "lag_operator_metadata": lag_meta,
            "smoothing": float(smoothing),
        }
    )
    out_metadata.update(readout_meta)
    out_metadata.update(basis_meta)

    readout_in_proj_t: Optional[torch.Tensor] = None
    readout_out_proj_t: Optional[torch.Tensor] = None
    if readout_factors is not None:
        in_proj_np, out_proj_np = readout_factors
        readout_in_proj_t = torch.tensor(in_proj_np, dtype=fixed_dtype)
        readout_out_proj_t = torch.tensor(out_proj_np, dtype=fixed_dtype)

    return AmplifierSpec(
        vocab_size=vocab_size,
        core_dim=effective_core_dim,
        branch_lags=branch_lags,
        num_blocks=num_blocks,
        token_embed=torch.tensor(token_embed, dtype=fixed_dtype),
        base_bigram_logits=torch.tensor(log_bigram, dtype=fixed_dtype),
        lag_ops=torch.tensor(lag_ops, dtype=fixed_dtype),
        amp_w1=torch.tensor(amp_w1, dtype=fixed_dtype),
        amp_w2=torch.tensor(amp_w2, dtype=fixed_dtype),
        readout_weight=None
        if readout_weight is None
        else torch.tensor(readout_weight, dtype=fixed_dtype),
        readout_in_proj=readout_in_proj_t,
        readout_out_proj=readout_out_proj_t,
        metadata=out_metadata,
    )


def build_amplifier_spec(
    tokens: ArrayLike,
    *,
    vocab_size: int = 1024,
    core_dim: int = 48,
    branch_lags: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64),
    num_blocks: int = 9,
    smoothing: float = 0.25,
    fixed_dtype: torch.dtype = torch.bfloat16,
    max_tokens: Optional[int] = None,
    embedding_init: str = "spectral",
    spectral_neighbors: int = 64,
    lag_identity_base: float = 0.15,
    readout_rank: Optional[int] = None,
) -> AmplifierSpec:
    branch_lags = _normalize_branch_lags(branch_lags)
    tokens_np = _to_numpy_1d(tokens)
    if max_tokens is not None:
        tokens_np = tokens_np[:max_tokens]
    if tokens_np.size < max(branch_lags) + 2:
        raise ValueError(f"need at least {max(branch_lags) + 2} tokens, got {tokens_np.size}")
    _validate_tokens(tokens_np, vocab_size=vocab_size)

    unigram = _count_unigrams(tokens_np, vocab_size)
    bigram = _count_bigrams(tokens_np, vocab_size)
    lag_pair_counts = {
        lag: _count_lag_pairs(tokens_np, lag=lag, vocab_size=vocab_size) for lag in branch_lags
    }

    return _build_spec_from_counts(
        unigram=unigram,
        bigram=bigram,
        lag_pair_counts=lag_pair_counts,
        vocab_size=vocab_size,
        core_dim=core_dim,
        branch_lags=branch_lags,
        num_blocks=num_blocks,
        smoothing=smoothing,
        fixed_dtype=fixed_dtype,
        embedding_init=embedding_init,
        spectral_neighbors=spectral_neighbors,
        lag_identity_base=lag_identity_base,
        readout_rank=readout_rank,
        metadata={
            "max_tokens": int(tokens_np.size),
            "bigram_tokens": int(tokens_np.size - 1),
        },
    )


# ---------------------------------------------------------------------------
# Log-space associative scan (Heinsen, 2023) — numerically stable for long seqs
# ---------------------------------------------------------------------------


def _heinsen_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
    """Associative scan in log space. Numerically stable for arbitrarily long sequences."""
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    return (a_star + log_h0_plus_b_star).exp()


def _g(x: torch.Tensor) -> torch.Tensor:
    """Activation that maps to positive reals. Eq B.3 from minGRU paper."""
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def _log_g(x: torch.Tensor) -> torch.Tensor:
    """Log of _g, numerically stable."""
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


class MinGRULayer(nn.Module):
    """Single minGRU layer (Feng et al. 2024, "Were RNNs All We Needed?").

    Log-space parallel scan for training, simple recurrence for generation.
    Hidden states are positive (required for log-space). Output projection
    maps back to unconstrained reals if expansion_factor != 1.
    """

    def __init__(self, dim: int, expansion_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        dim_inner = int(dim * expansion_factor)
        self.dim_inner = dim_inner
        self.proj_out = dim_inner != dim

        self.to_hidden_and_gate = nn.Linear(dim, dim_inner * 2, bias=False)
        self.out_proj = nn.Linear(dim_inner, dim, bias=False) if self.proj_out else nn.Identity()

    def forward(
        self, x: torch.Tensor, prev_hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, dim]
            prev_hidden: [B, dim_inner] or None (positive values)
        Returns:
            out: [B, T, dim]
            next_hidden: [B, dim_inner] (positive values, for carry)
        """
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if seq_len == 1:
            # Sequential mode (generation)
            hidden = _g(hidden)
            gate = gate.sigmoid()
            if prev_hidden is not None:
                out = torch.lerp(prev_hidden.unsqueeze(1), hidden, gate)
            else:
                out = hidden * gate
        else:
            # Parallel mode — log-space scan. MUST use float32 for log/exp stability.
            orig_dtype = hidden.dtype
            hidden = hidden.float()
            gate = gate.float()

            log_coeffs = -F.softplus(gate)
            log_z = -F.softplus(-gate)
            log_tilde_h = _log_g(hidden)
            log_values = log_z + log_tilde_h

            if prev_hidden is not None:
                prev_hidden_log = (
                    prev_hidden.float().clamp_min(1e-8).unsqueeze(1).log().clamp(min=-30)
                )
                log_values = torch.cat((prev_hidden_log, log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = _heinsen_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]
            out = out.to(orig_dtype)

        next_hidden = out[:, -1]  # [B, dim_inner], positive
        out = self.out_proj(out)
        return out, next_hidden


class LearnedRMSNorm(nn.Module):
    """Tiny RMSNorm with a learned scale.

    This is deliberately minimal: no bias, no mean subtraction, and only one
    vector parameter. It makes deeper recurrent stacks easier to optimize while
    keeping the controller small and parallelizable.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.scale


class ResidualMinGRUBlock(nn.Module):
    """Pre-norm residual wrapper around a parallel minGRU layer.

    The recurrence itself is unchanged: for T>1 the minGRU still uses the
    log-space Heinsen scan. The residual gate starts near identity so that
    5-6 layer controllers do not have to learn through a randomly deep stack
    on the first few hundred steps.
    """

    def __init__(
        self,
        dim: int,
        *,
        expansion_factor: float = 2.0,
        residual_init: float = -2.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = LearnedRMSNorm(dim, eps=norm_eps)
        self.rnn = MinGRULayer(dim, expansion_factor=expansion_factor)
        self.resid_logit = nn.Parameter(torch.tensor(float(residual_init)))

    @property
    def dim_inner(self) -> int:
        return self.rnn.dim_inner

    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, next_h = self.rnn(self.norm(x), prev_hidden=prev_hidden)
        alpha = torch.sigmoid(self.resid_logit).to(dtype=y.dtype)
        return x + alpha * y, next_h


class MinGRUCore(nn.Module):
    """Multi-layer minGRU with a GRU-compatible interface.

    State: [num_layers, B, dim_inner]. The hidden state lives in the positive
    log-space minGRU domain, while the exposed output remains unconstrained.

    residual_blocks=True is now the default because it makes controller-depth
    sweeps much less brittle without changing the O(T) parallel scan path.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        expansion_factor: float = 2.0,
        batch_first: bool = True,
        residual_blocks: bool = True,
        residual_init: float = -2.0,
        norm_eps: float = 1e-5,
        gradient_checkpointing: bool = False,
        **_kwargs,
    ):
        super().__init__()
        assert batch_first
        if input_size != hidden_size:
            raise ValueError(
                "Residualized MinGRUCore assumes input_size == hidden_size; "
                f"got input_size={input_size}, hidden_size={hidden_size}"
            )
        self.hidden_size = hidden_size
        self.num_layers = int(num_layers)
        self.dim_inner = int(hidden_size * expansion_factor)
        self.residual_blocks = bool(residual_blocks)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        if self.residual_blocks:
            self.blocks = nn.ModuleList(
                [
                    ResidualMinGRUBlock(
                        hidden_size,
                        expansion_factor=expansion_factor,
                        residual_init=residual_init,
                        norm_eps=norm_eps,
                    )
                    for _ in range(self.num_layers)
                ]
            )
            self.layers = self.blocks  # backwards-friendly alias
        else:
            self.layers = nn.ModuleList(
                [
                    MinGRULayer(hidden_size, expansion_factor=expansion_factor)
                    for _ in range(self.num_layers)
                ]
            )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        if state is None:
            state = x.new_full((self.num_layers, B, self.dim_inner), 0.5)

        new_states = []
        h = x
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                h, next_h = activation_checkpoint(
                    layer,
                    h,
                    state[i],
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                h, next_h = layer(h, prev_hidden=state[i])
            new_states.append(next_h)

        return h, torch.stack(new_states, dim=0)

    @torch.no_grad()
    def residual_gate_values(self) -> list[float]:
        if not self.residual_blocks:
            return []
        return [float(torch.sigmoid(block.resid_logit).item()) for block in self.blocks]


# Keep simple ScanCore as a lightweight option
class ScanCore(nn.Module):
    """Simple diagonal linear recurrence. Fastest, fewest params, no log-space stability."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        **_kwargs,
    ):
        super().__init__()
        assert batch_first
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dim_in = input_size if i == 0 else hidden_size
            self.layers.append(nn.Linear(dim_in, hidden_size * 2, bias=True))

        for layer in self.layers:
            with torch.no_grad():
                layer.bias[:hidden_size].fill_(2.0)

        self._scan = None

    def _get_scan(self):
        if self._scan is None:
            from assoc_scan import AssocScan

            self._scan = AssocScan(use_accelerated=True)
        return self._scan

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if state is None:
            state = x.new_zeros(self.num_layers, B, self.hidden_size)

        new_states = []
        h = x
        for i, layer in enumerate(self.layers):
            proj = layer(h)
            gate_logits, values = proj.chunk(2, dim=-1)
            gates = torch.sigmoid(gate_logits)
            inputs = (1 - gates) * values

            if T == 1:
                h_prev = state[i]
                h_new = gates[:, 0] * h_prev + inputs[:, 0]
                h = h_new.unsqueeze(1)
                new_states.append(h_new)
            else:
                scan = self._get_scan()
                prev = state[i].unsqueeze(1)
                gates_with_prev = torch.cat([gates.new_ones(B, 1, self.hidden_size), gates], dim=1)
                inputs_with_prev = torch.cat([prev, inputs], dim=1)
                h_all = scan(gates_with_prev, inputs_with_prev)
                h = h_all[:, 1:]
                new_states.append(h[:, -1])

        return h, torch.stack(new_states, dim=0)


class CoreAmplifierLM(nn.Module):
    def __init__(
        self,
        spec: AmplifierSpec,
        *,
        core_layers: int = 3,
        core_type: str = "mingru",
        dropout: float = 0.0,
        learn_input_adapter: bool = True,
        amplifier_dtype: Optional[torch.dtype] = None,
        core_expansion: float = 2.0,
        residual_core: bool = True,
        residual_core_init: float = -2.0,
        branch_temporal_mode: str = "current",
        branch_temporal_lag_scale: float = 1.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = spec.vocab_size
        self.core_dim = spec.core_dim
        self.branch_lags = spec.branch_lags
        self.branch_temporal_mode = _normalize_branch_temporal_mode(branch_temporal_mode)
        self.branch_temporal_lag_scale = float(branch_temporal_lag_scale)
        self.num_branches = len(spec.branch_lags)
        self.max_branch_lag = max(self.branch_lags)
        self.amp_dim = self.core_dim * self.num_branches
        self.num_blocks = spec.num_blocks
        self.preferred_amplifier_dtype = amplifier_dtype
        self.core_type = core_type
        self.residual_core = bool(residual_core)
        self.residual_core_init = float(residual_core_init)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.register_buffer("token_embed", spec.token_embed.clone(), persistent=True)
        self.register_buffer("base_bigram_logits", spec.base_bigram_logits.clone(), persistent=True)
        self.register_buffer("lag_ops", spec.lag_ops.clone(), persistent=True)
        self.register_buffer("amp_w1", spec.amp_w1.clone(), persistent=True)
        self.register_buffer("amp_w2", spec.amp_w2.clone(), persistent=True)
        self.register_buffer(
            "readout_weight",
            None if spec.readout_weight is None else spec.readout_weight.clone(),
            persistent=True,
        )
        self.register_buffer(
            "readout_in_proj",
            None if spec.readout_in_proj is None else spec.readout_in_proj.clone(),
            persistent=True,
        )
        self.register_buffer(
            "readout_out_proj",
            None if spec.readout_out_proj is None else spec.readout_out_proj.clone(),
            persistent=True,
        )
        lag_mix = torch.tensor(
            [self.branch_temporal_lag_scale / math.sqrt(float(lag)) for lag in self.branch_lags],
            dtype=torch.float32,
        )
        self.register_buffer("branch_temporal_lag_mix", lag_mix, persistent=True)

        if learn_input_adapter:
            self.input_adapter = nn.Linear(self.core_dim, self.core_dim, bias=False)
            with torch.no_grad():
                self.input_adapter.weight.copy_(torch.eye(self.core_dim))
        else:
            self.input_adapter = nn.Identity()

        if core_type == "mingru":
            self.core = MinGRUCore(
                input_size=self.core_dim,
                hidden_size=self.core_dim,
                num_layers=core_layers,
                expansion_factor=core_expansion,
                batch_first=True,
                residual_blocks=self.residual_core,
                residual_init=self.residual_core_init,
                gradient_checkpointing=self.gradient_checkpointing,
            )
        elif core_type == "scan":
            self.core = ScanCore(
                input_size=self.core_dim,
                hidden_size=self.core_dim,
                num_layers=core_layers,
                batch_first=True,
            )
        elif core_type == "gru":
            self.core = nn.GRU(
                input_size=self.core_dim,
                hidden_size=self.core_dim,
                num_layers=core_layers,
                batch_first=True,
                dropout=dropout if core_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"unknown core_type: {core_type!r}")

        # h0: learnable initial hidden state. Shape depends on core type.
        if core_type == "mingru":
            dim_inner = int(getattr(self.core, "dim_inner", int(self.core_dim * core_expansion)))
            # Initialize positive (required for log-space). Use softplus to keep positive during training.
            self._h0_raw = nn.Parameter(torch.zeros(core_layers, 1, dim_inner))
        else:
            self._h0_raw = nn.Parameter(torch.zeros(core_layers, 1, self.core_dim))
        self.block_gain = nn.Parameter(torch.full((self.num_blocks,), -1.5))
        self.branch_scale = nn.Parameter(torch.ones(self.num_blocks, self.num_branches))
        self.branch_bias = nn.Parameter(torch.zeros(self.num_blocks, self.num_branches))
        self.readout_branch_scale = nn.Parameter(torch.ones(self.num_branches))
        self.residual_log_scale = nn.Parameter(torch.tensor(-0.5))
        self.logit_bias = nn.Parameter(torch.zeros(self.vocab_size))

        self._runtime_device_str: Optional[str] = None
        self._runtime_amp_dtype: Optional[torch.dtype] = None
        self._lag_ops_runtime: Optional[torch.Tensor] = None
        self._amp_w1_runtime: Optional[torch.Tensor] = None
        self._amp_w2_runtime: Optional[torch.Tensor] = None
        self._readout_weight_runtime: Optional[torch.Tensor] = None
        self._readout_in_proj_runtime: Optional[torch.Tensor] = None
        self._readout_out_proj_runtime: Optional[torch.Tensor] = None
        self._base_bigram_logits_runtime: Optional[torch.Tensor] = None
        self._branch_temporal_lag_mix_runtime: Optional[torch.Tensor] = None

    def _use_gradient_checkpointing(self) -> bool:
        return bool(self.gradient_checkpointing and self.training and torch.is_grad_enabled())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def fixed_nbytes(self) -> int:
        total = (
            self.token_embed.numel() * self.token_embed.element_size()
            + self.base_bigram_logits.numel() * self.base_bigram_logits.element_size()
            + self.lag_ops.numel() * self.lag_ops.element_size()
            + self.amp_w1.numel() * self.amp_w1.element_size()
            + self.amp_w2.numel() * self.amp_w2.element_size()
        )
        if self.readout_weight is not None:
            total += self.readout_weight.numel() * self.readout_weight.element_size()
        if self.readout_in_proj is not None:
            total += self.readout_in_proj.numel() * self.readout_in_proj.element_size()
        if self.readout_out_proj is not None:
            total += self.readout_out_proj.numel() * self.readout_out_proj.element_size()
        return total

    def _resolve_amplifier_dtype(self, device: torch.device) -> torch.dtype:
        dtype = self.preferred_amplifier_dtype or self.amp_w1.dtype
        if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype

    def prepare_runtime_buffers(
        self,
        *,
        device: Optional[torch.device] = None,
        amplifier_dtype: Optional[torch.dtype] = None,
    ) -> None:
        device = device or self.token_embed.device
        dtype = amplifier_dtype or self._resolve_amplifier_dtype(device)
        if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        self._lag_ops_runtime = self.lag_ops.to(device=device, dtype=dtype)
        self._amp_w1_runtime = self.amp_w1.to(device=device, dtype=dtype)
        self._amp_w2_runtime = self.amp_w2.to(device=device, dtype=dtype)
        self._readout_weight_runtime = (
            None
            if self.readout_weight is None
            else self.readout_weight.to(device=device, dtype=dtype)
        )
        self._readout_in_proj_runtime = (
            None
            if self.readout_in_proj is None
            else self.readout_in_proj.to(device=device, dtype=dtype)
        )
        self._readout_out_proj_runtime = (
            None
            if self.readout_out_proj is None
            else self.readout_out_proj.to(device=device, dtype=dtype)
        )
        self._base_bigram_logits_runtime = self.base_bigram_logits.to(device=device, dtype=dtype)
        self._branch_temporal_lag_mix_runtime = self.branch_temporal_lag_mix.to(
            device=device, dtype=dtype
        )
        self._runtime_device_str = str(device)
        self._runtime_amp_dtype = dtype

    def _ensure_runtime_buffers(self) -> None:
        device = self.token_embed.device
        desired = self._resolve_amplifier_dtype(device)
        if (
            self._runtime_device_str != str(device)
            or self._runtime_amp_dtype != desired
            or self._lag_ops_runtime is None
            or self._amp_w1_runtime is None
            or self._amp_w2_runtime is None
            or self._base_bigram_logits_runtime is None
            or self._branch_temporal_lag_mix_runtime is None
            or (self.readout_weight is not None and self._readout_weight_runtime is None)
            or (self.readout_in_proj is not None and self._readout_in_proj_runtime is None)
            or (self.readout_out_proj is not None and self._readout_out_proj_runtime is None)
        ):
            self.prepare_runtime_buffers(device=device, amplifier_dtype=desired)

    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> BranchTemporalState:
        if self.core_type == "mingru":
            # MinGRU needs positive hidden states for log-space stability
            core_state = (
                F.softplus(self._h0_raw).clamp_min(1e-8).expand(-1, batch_size, -1).contiguous()
            )
        else:
            core_state = self._h0_raw.expand(-1, batch_size, -1).contiguous()
        if device is not None:
            core_state = core_state.to(device)
        if self.branch_temporal_mode == "current":
            return core_state
        history = core_state.new_zeros(batch_size, self.max_branch_lag, self.core_dim)
        return core_state, history

    def detach_state(self, state: BranchTemporalState) -> BranchTemporalState:
        if isinstance(state, tuple):
            return tuple(part.detach() for part in state)  # type: ignore[return-value]
        return state.detach()

    def state_norms(self, state: BranchTemporalState) -> list[float]:
        core_state = state[0] if isinstance(state, tuple) else state
        return core_state.float().norm(dim=-1).mean(dim=1).detach().cpu().tolist()

    def _split_state(
        self,
        state: BranchTemporalState,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.branch_temporal_mode == "current":
            if isinstance(state, tuple):
                return state[0].to(device=device, dtype=dtype), None
            return state.to(device=device, dtype=dtype), None

        if isinstance(state, tuple):
            core_state, history = state
        else:
            core_state = state
            history = state.new_zeros(batch_size, self.max_branch_lag, self.core_dim)
        core_state = core_state.to(device=device, dtype=dtype)
        history = history.to(device=device, dtype=dtype)
        expected_shape = (batch_size, self.max_branch_lag, self.core_dim)
        if tuple(history.shape) != expected_shape:
            raise ValueError(
                "branch history shape mismatch: "
                f"expected {expected_shape}, got {tuple(history.shape)}"
            )
        return core_state, history

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = F.embedding(input_ids, self.token_embed)
        if isinstance(self.input_adapter, nn.Linear):
            emb = emb.to(dtype=self.input_adapter.weight.dtype)
        else:
            # Get dtype from any core parameter
            core_param = next(self.core.parameters())
            emb = emb.to(dtype=core_param.dtype)
        return self.input_adapter(emb)

    def _expand_branches(
        self,
        core_out: torch.Tensor,
        history: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._ensure_runtime_buffers()
        assert self._lag_ops_runtime is not None
        core_out = core_out.to(dtype=self._lag_ops_runtime.dtype)
        if self.branch_temporal_mode == "current":
            branches = torch.einsum("btc,ncd->btnd", core_out, self._lag_ops_runtime)
            return branches, None

        assert history is not None
        assert self._branch_temporal_lag_mix_runtime is not None
        full = torch.cat([history, core_out], dim=1)
        lagged_inputs = torch.stack(
            [
                full[
                    :, self.max_branch_lag - lag : self.max_branch_lag - lag + core_out.shape[1], :
                ]
                for lag in self.branch_lags
            ],
            dim=2,
        )
        next_history = full[:, -self.max_branch_lag :, :].contiguous()
        if self.branch_temporal_mode == "lagged":
            branch_inputs = lagged_inputs
        else:
            lag_mix = self._branch_temporal_lag_mix_runtime.view(1, 1, self.num_branches, 1)
            branch_inputs = (core_out[:, :, None, :] + lag_mix * lagged_inputs) / torch.sqrt(
                1.0 + lag_mix.square()
            )
        branches = torch.einsum("btnc,ncd->btnd", branch_inputs, self._lag_ops_runtime)
        return branches, next_history

    def _amplify(self, branches: torch.Tensor) -> torch.Tensor:
        self._ensure_runtime_buffers()
        assert self._amp_w1_runtime is not None and self._amp_w2_runtime is not None
        b, t, n, c = branches.shape
        x = branches.reshape(b, t, n * c)
        block_gain = torch.sigmoid(self.block_gain).to(device=branches.device, dtype=branches.dtype)
        branch_scale = self.branch_scale.to(device=branches.device, dtype=branches.dtype)
        branch_bias = self.branch_bias.to(device=branches.device, dtype=branches.dtype)
        for i in range(self.num_blocks):
            if self._use_gradient_checkpointing():
                branches, x = activation_checkpoint(
                    lambda cur_branches, cur_x, block_idx=i: self._amplify_block(
                        cur_branches,
                        cur_x,
                        block_idx=block_idx,
                        branch_scale=branch_scale,
                        branch_bias=branch_bias,
                        block_gain=block_gain,
                        branch_count=n,
                        core_dim=c,
                    ),
                    branches,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                branches, x = self._amplify_block(
                    branches,
                    x,
                    block_idx=i,
                    branch_scale=branch_scale,
                    branch_bias=branch_bias,
                    block_gain=block_gain,
                    branch_count=n,
                    core_dim=c,
                )
        return branches

    def _amplify_block(
        self,
        branches: torch.Tensor,
        x: torch.Tensor,
        *,
        block_idx: int,
        branch_scale: torch.Tensor,
        branch_bias: torch.Tensor,
        block_gain: torch.Tensor,
        branch_count: int,
        core_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = branches * branch_scale[block_idx].view(1, 1, branch_count, 1)
        y = y + branch_bias[block_idx].view(1, 1, branch_count, 1)
        y = y.reshape(branches.shape[0], branches.shape[1], branch_count * core_dim)
        y = _rms_norm(y)
        assert self._amp_w1_runtime is not None and self._amp_w2_runtime is not None
        y = F.linear(y, self._amp_w1_runtime[block_idx])
        y = F.silu(y)
        y = F.linear(y, self._amp_w2_runtime[block_idx])
        x = x + block_gain[block_idx] * y
        return x.reshape(branches.shape[0], branches.shape[1], branch_count, core_dim), x

    def _residual_logits(self, branches: torch.Tensor) -> torch.Tensor:
        self._ensure_runtime_buffers()
        n = self.num_branches
        c = self.core_dim
        scales = self.readout_branch_scale.to(device=branches.device, dtype=branches.dtype)
        x = (branches * scales.view(1, 1, n, 1)).reshape(
            branches.shape[0], branches.shape[1], n * c
        )
        if self._readout_weight_runtime is not None:
            return F.linear(x, self._readout_weight_runtime)
        assert (
            self._readout_in_proj_runtime is not None and self._readout_out_proj_runtime is not None
        )
        hidden = F.linear(x, self._readout_in_proj_runtime)
        return F.linear(hidden, self._readout_out_proj_runtime)

    def base_path_logits(
        self, input_ids: torch.Tensor, *, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        self._ensure_runtime_buffers()
        assert self._base_bigram_logits_runtime is not None
        out = F.embedding(input_ids.long(), self._base_bigram_logits_runtime)
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype=dtype)
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[BranchTemporalState] = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, BranchTemporalState]:
        if input_ids.dtype not in (torch.int32, torch.int64):
            input_ids = input_ids.long()
        batch_size = input_ids.shape[0]
        if state is None:
            state = self.initial_state(batch_size, device=input_ids.device)
        x = self._embed(input_ids)
        core_state, history = self._split_state(
            state,
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        core_out, next_core_state = self.core(x, core_state)
        branches, next_history = self._expand_branches(core_out, history=history)
        branches = self._amplify(branches)
        residual_logits = self._residual_logits(branches)
        base_logits = self.base_path_logits(input_ids, dtype=residual_logits.dtype)
        logits = (
            base_logits
            + self.logit_bias.to(device=residual_logits.device, dtype=residual_logits.dtype)
            + torch.exp(self.residual_log_scale).to(
                device=residual_logits.device, dtype=residual_logits.dtype
            )
            * residual_logits
        )
        next_state: BranchTemporalState
        if next_history is None:
            next_state = next_core_state
        else:
            next_state = (next_core_state, next_history)
        if return_state:
            return logits, next_state
        return logits

    @torch.no_grad()
    def residual_gate_values(self) -> list[float]:
        if hasattr(self.core, "residual_gate_values"):
            return self.core.residual_gate_values()
        return []

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        state: Optional[BranchTemporalState] = None,
    ) -> tuple[torch.Tensor, BranchTemporalState]:
        if input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        logits, next_state = self.forward(input_ids, state=state, return_state=True)
        return logits[:, -1], next_state

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        if prompt.ndim == 1:
            prompt = prompt[None, :]
        device = next(self.parameters()).device
        prompt = prompt.to(device)
        batch_size = prompt.shape[0]
        state = self.initial_state(batch_size, device=device)
        if prompt.shape[1] > 1:
            _, state = self.forward(prompt[:, :-1], state=state, return_state=True)
        cur = prompt[:, -1]
        generated = [prompt]
        for _ in range(max_new_tokens):
            logits, state = self.step(cur, state=state)
            logits = logits.float() / max(temperature, 1e-5)
            if top_k is not None and top_k < logits.size(-1):
                values, _ = torch.topk(logits, k=top_k, dim=-1)
                cutoff = values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            cur = torch.multinomial(probs, num_samples=1).squeeze(1)
            generated.append(cur[:, None])
        return torch.cat(generated, dim=1)


def estimate_storage_bytes(
    *,
    vocab_size: int,
    core_dim: int,
    branch_count: int,
    num_blocks: int,
    readout_rank: Optional[int] = None,
    fixed_dtype: torch.dtype = torch.bfloat16,
) -> int:
    bytes_per = torch.tensor([], dtype=fixed_dtype).element_size()
    amp_dim = core_dim * branch_count
    total_numel = 0
    total_numel += vocab_size * core_dim
    total_numel += vocab_size * vocab_size
    total_numel += branch_count * core_dim * core_dim
    total_numel += num_blocks * amp_dim * amp_dim
    total_numel += num_blocks * amp_dim * amp_dim
    if readout_rank is None or readout_rank <= 0 or readout_rank >= min(vocab_size, amp_dim):
        total_numel += vocab_size * amp_dim
    else:
        total_numel += readout_rank * amp_dim
        total_numel += vocab_size * readout_rank
    return total_numel * bytes_per


def default_16mb_recipe() -> dict[str, object]:
    branch_lags = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    bytes_ = estimate_storage_bytes(
        vocab_size=1024,
        core_dim=48,
        branch_count=len(branch_lags),
        num_blocks=9,
        readout_rank=None,
        fixed_dtype=torch.float16,
    )
    return {
        "vocab_size": 1024,
        "core_dim": 48,
        "branch_lags": branch_lags,
        "num_blocks": 9,
        "readout_rank": None,
        "fixed_dtype": torch.float16,
        "estimated_fixed_bytes": bytes_,
        "estimated_fixed_mb": bytes_ / 1_000_000,
        "estimated_fixed_mib": bytes_ / (1024 * 1024),
        "comment": "~15.35 MB fixed amplifier + ~110k trainable parameters with the 5-layer residual minGRU controller",
    }


__all__ = [
    "AmplifierSpec",
    "CoreAmplifierLM",
    "build_amplifier_spec",
    "estimate_storage_bytes",
    "default_16mb_recipe",
]
