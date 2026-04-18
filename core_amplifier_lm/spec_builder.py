"""Optimized corpus-statistics builder for the core/amplifier spec.

Key improvements over the original build_amplifier_spec path:
- Loads sharded .bin directories via glob (competition layout)
- Keeps tokens as int32 (not int64): 16GB on disk → 32GB in RAM, not 64GB
- Accumulates unigram + bigram + all lag-pair counts in a single pass
- No spec_max_tokens cap: uses the full dataset by default
- Parallel shard counting via multiprocessing (--spec-workers)

Drop this next to core_amplifier_lm.py and use build_spec_optimized().
"""

from __future__ import annotations

import gzip
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .model import AmplifierSpec, _build_spec_from_counts, _normalize_branch_lags

NUMPY_DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int32": np.int32,
    "int64": np.int64,
}


def _get_rss_gb() -> float:
    """Current process RSS in GB (main process only, excludes child workers)."""
    # resource.getrusage returns maxrss in KB on Linux, bytes on macOS
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return maxrss / 1e9
    return maxrss / 1e6  # Linux: KB → GB


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Competition shard header: 256 x int32 (1024 bytes)
# header[0] = magic (20240520), header[1] = version (1), header[2] = num_tokens
_SHARD_MAGIC = 20240520
_SHARD_VERSION = 1
_HEADER_INTS = 256
_HEADER_BYTES = _HEADER_INTS * np.dtype("<i4").itemsize  # 1024


def _load_shard_with_header(path: Path) -> np.ndarray:
    """Load a competition .bin shard, skipping the 1024-byte header.

    Returns tokens as int32.
    """
    header = np.fromfile(path, dtype="<i4", count=_HEADER_INTS)
    if (
        header.size == _HEADER_INTS
        and int(header[0]) == _SHARD_MAGIC
        and int(header[1]) == _SHARD_VERSION
    ):
        num_tokens = int(header[2])
        tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=_HEADER_BYTES)
        return tokens.astype(np.int32, copy=False)
    # No header / unknown format: fall back to raw uint16
    return np.fromfile(path, dtype=np.uint16).astype(np.int32, copy=False)


def _detect_has_header(path: Path) -> bool:
    """Check if a .bin file has the competition shard header."""
    header = np.fromfile(path, dtype="<i4", count=2)
    return header.size >= 2 and int(header[0]) == _SHARD_MAGIC and int(header[1]) == _SHARD_VERSION


def load_tokens_int32(
    source: str | Path,
    *,
    storage_dtype: str = "uint16",
    max_tokens: Optional[int] = None,
) -> np.ndarray:
    """Load tokens from a single file or a directory of shards as int32.

    Automatically detects and skips the competition shard header if present.
    """
    p = Path(source)
    np_dtype = NUMPY_DTYPE_MAP[storage_dtype]

    if p.is_dir():
        train = sorted(p.glob("fineweb_train_*.bin"))
        val = sorted(p.glob("fineweb_val_*.bin"))
        all_shards = train + val if not train else train
        if not all_shards:
            all_shards = sorted(p.glob("*.bin"))
        if not all_shards:
            raise FileNotFoundError(f"no .bin files in {p}")
        # Auto-detect header from first shard
        has_header = _detect_has_header(all_shards[0])
        if has_header:
            chunks = [_load_shard_with_header(s) for s in all_shards]
        else:
            chunks = [
                np.fromfile(s, dtype=np_dtype).astype(np.int32, copy=False) for s in all_shards
            ]
        arr = np.concatenate(chunks)
    elif p.suffix.lower() == ".gz":
        with gzip.open(p, "rb") as f:
            raw = f.read()
        itemsize = np.dtype(np_dtype).itemsize
        usable = (len(raw) // itemsize) * itemsize
        arr = np.frombuffer(raw[:usable], dtype=np_dtype).copy()
    elif p.suffix.lower() == ".npy":
        arr = np.load(p).ravel()
    else:
        arr = np.fromfile(p, dtype=np_dtype)

    arr = arr.astype(np.int32, copy=False).ravel()
    if max_tokens is not None:
        arr = arr[:max_tokens]
    return arr


def load_train_val_int32(
    source: str | Path,
    *,
    storage_dtype: str = "uint16",
    train_frac: float = 0.98,
    max_tokens: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load train/val splits, respecting shard layout if present."""
    p = Path(source)
    np_dtype = NUMPY_DTYPE_MAP[storage_dtype]

    if p.is_dir():
        train_shards = sorted(p.glob("fineweb_train_*.bin"))
        val_shards = sorted(p.glob("fineweb_val_*.bin"))
        if train_shards:
            has_header = _detect_has_header(train_shards[0])
            if has_header:
                train = np.concatenate([_load_shard_with_header(s) for s in train_shards])
                if val_shards:
                    val = np.concatenate([_load_shard_with_header(s) for s in val_shards])
                else:
                    split = int(train.size * train_frac)
                    split = max(1, min(train.size - 1, split))
                    train, val = train[:split], train[split:]
            else:
                train = np.concatenate(
                    [
                        np.fromfile(s, dtype=np_dtype).astype(np.int32, copy=False)
                        for s in train_shards
                    ]
                )
                if val_shards:
                    val = np.concatenate(
                        [
                            np.fromfile(s, dtype=np_dtype).astype(np.int32, copy=False)
                            for s in val_shards
                        ]
                    )
                else:
                    split = int(train.size * train_frac)
                    split = max(1, min(train.size - 1, split))
                    train, val = train[:split], train[split:]
            if max_tokens is not None:
                train = train[:max_tokens]
            return train, val

    # Single file: load and split
    tokens = load_tokens_int32(source, storage_dtype=storage_dtype, max_tokens=max_tokens)
    split = int(tokens.size * train_frac)
    split = max(1, min(tokens.size - 1, split))
    return tokens[:split], tokens[split:]


# ---------------------------------------------------------------------------
# Single-pass multi-lag counting
# ---------------------------------------------------------------------------


def count_all(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    branch_lags: tuple[int, ...],
    chunk_size: int = 200_000_000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Count unigram, bigram, and all lag-pair matrices.

    Processes in chunks to avoid allocating a flat-index buffer the size of
    the full token array.  Peak RAM = token array + chunk_size*4 bytes + accumulators.
    For 8B tokens with chunk_size=200M: ~32 GB tokens + 0.8 GB buffer = ~33 GB peak
    (vs 64 GB without chunking).

    Returns (unigram, bigram, {lag: counts}).
    """
    n = tokens.shape[0]
    all_lags = sorted(set(branch_lags) | {1})
    v32 = np.int32(vocab_size)
    vsq = vocab_size * vocab_size

    t0 = time.monotonic()

    # Unigram — bincount handles any size cheaply
    unigram = np.bincount(tokens, minlength=vocab_size).astype(np.float64)

    # Accumulate lag-pair counts in chunks to avoid a full-length flat buffer.
    # Cross-chunk boundary pairs (≤max_lag per boundary) are dropped — negligible
    # error: max_lag × num_lags × num_boundaries / total_pairs < 1e-6 for 200M chunks.
    counts = {lag: np.zeros((vocab_size, vocab_size), dtype=np.float64) for lag in all_lags}
    buf = np.empty(min(chunk_size, n - 1), dtype=np.int32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = tokens[start:end]
        chunk_len = chunk.shape[0]

        for lag in all_lags:
            if lag >= chunk_len:
                continue
            length = chunk_len - lag
            flat = buf[:length]
            np.multiply(chunk[:length], v32, out=flat)
            np.add(flat, chunk[lag : lag + length], out=flat)
            counts[lag] += np.bincount(flat, minlength=vsq).reshape(vocab_size, vocab_size)

    if verbose:
        elapsed = time.monotonic() - t0
        print(
            f"  counted {len(all_lags)} lag matrices over {n / 1e9:.2f}B tokens "
            f"in {elapsed:.1f}s ({n / elapsed / 1e6:.0f}M tok/s)",
            flush=True,
        )

    bigram = counts[1]
    lag_pair_counts = {lag: counts[lag] for lag in branch_lags}
    return unigram, bigram, lag_pair_counts


# ---------------------------------------------------------------------------
# Counts → AmplifierSpec
# ---------------------------------------------------------------------------


def _count_one_shard(
    path: Path, vocab_size: int, all_lags: list[int]
) -> tuple[np.ndarray, dict[int, np.ndarray], int]:
    """Count stats for a single shard. Standalone function for pickling."""
    v32 = np.int32(vocab_size)
    vsq = vocab_size * vocab_size
    shard = (
        _load_shard_with_header(path)
        if _detect_has_header(path)
        else np.fromfile(path, dtype=np.uint16).astype(np.int32, copy=False)
    )
    n = shard.shape[0]
    if n == 0:
        return (
            np.zeros(vocab_size, dtype=np.float64),
            {lag: np.zeros((vocab_size, vocab_size), dtype=np.float64) for lag in all_lags},
            0,
        )
    uni = np.bincount(shard, minlength=vocab_size).astype(np.float64)
    counts: dict[int, np.ndarray] = {}
    buf = np.empty(n - 1, dtype=np.int32)
    for lag in all_lags:
        if lag >= n:
            counts[lag] = np.zeros((vocab_size, vocab_size), dtype=np.float64)
            continue
        length = n - lag
        flat = buf[:length]
        np.multiply(shard[:length], v32, out=flat)
        np.add(flat, shard[lag : lag + length], out=flat)
        counts[lag] = (
            np.bincount(flat, minlength=vsq).reshape(vocab_size, vocab_size).astype(np.float64)
        )
    return uni, counts, n


def _count_shard_batch(args: tuple) -> tuple[np.ndarray, dict[int, np.ndarray], int]:
    """Process a batch of shards in one worker. Accumulates locally to minimize IPC.

    Each worker holds: one shard at a time (~400 MB) + one set of accumulators (~104 MB).
    IPC transfer on completion: one accumulated result (~104 MB) per worker.
    """
    shard_paths, vocab_size, all_lags = args
    uni_acc = np.zeros(vocab_size, dtype=np.float64)
    counts_acc = {lag: np.zeros((vocab_size, vocab_size), dtype=np.float64) for lag in all_lags}
    total = 0
    for path in shard_paths:
        uni, counts, n = _count_one_shard(path, vocab_size, all_lags)
        uni_acc += uni
        for lag in all_lags:
            counts_acc[lag] += counts[lag]
        total += n
    return uni_acc, counts_acc, total


def _default_num_workers() -> int:
    """Sensible default: half of CPU cores, clamped to [1, 8]."""
    cpus = os.cpu_count() or 1
    return max(1, min(cpus // 2, 8))


# ---------------------------------------------------------------------------
# Strategy: preload — load all tokens into RAM, then parallel count on slices
# ---------------------------------------------------------------------------

# Module-level ref for fork-shared memory (workers inherit via COW)
_shared_tokens: Optional[np.ndarray] = None
_shared_vocab_size: int = 0


def _init_shared_tokens(tokens: np.ndarray, vocab_size: int) -> None:
    global _shared_tokens, _shared_vocab_size
    _shared_tokens = tokens
    _shared_vocab_size = vocab_size


def _count_lags_worker(lags: list[int]) -> dict[int, np.ndarray]:
    """Count lag-pair matrices for a subset of lags on the shared token array.

    Pre-allocates two int32 buffers and reuses them across all lags and chunks.
    Per-worker peak: 2 × chunk_size × 4 bytes + accumulators.
    With chunk=200M: 2 × 0.8 GB = 1.6 GB per worker.
    """
    assert _shared_tokens is not None
    tokens = _shared_tokens
    vocab_size = _shared_vocab_size
    n = tokens.shape[0]
    v32 = np.int32(vocab_size)
    vsq = vocab_size * vocab_size
    chunk_size = 200_000_000

    # Two reusable buffers — allocated once, reused for all lags and chunks
    alloc = min(chunk_size, n)
    buf_a = np.empty(alloc, dtype=np.int32)
    buf_b = np.empty(alloc, dtype=np.int32)

    results: dict[int, np.ndarray] = {}
    for lag in lags:
        if lag >= n:
            results[lag] = np.zeros((vocab_size, vocab_size), dtype=np.float64)
            continue
        acc = np.zeros(vsq, dtype=np.float64)
        for start in range(0, n - lag, chunk_size):
            end = min(start + chunk_size, n - lag)
            length = end - start
            a = buf_a[:length]
            b = buf_b[:length]
            np.copyto(a, tokens[start:end])  # uint16 → int32, no temp alloc
            np.copyto(b, tokens[start + lag : end + lag])
            np.multiply(a, v32, out=a)  # a *= 1024 in-place
            np.add(a, b, out=a)  # a += b in-place, a is now flat index
            acc += np.bincount(a, minlength=vsq)
        results[lag] = acc.reshape(vocab_size, vocab_size)
    return results


def _load_all_shards_uint16(
    shard_paths: list[Path], *, max_tokens: Optional[int] = None, verbose: bool = True
) -> np.ndarray:
    """Load all shards into a single contiguous uint16 array."""
    chunks = []
    total = 0
    for si, sp in enumerate(shard_paths):
        if max_tokens is not None and total >= max_tokens:
            break
        if _detect_has_header(sp):
            header = np.fromfile(sp, dtype="<i4", count=_HEADER_INTS)
            num_tokens = int(header[2])
            shard = np.fromfile(sp, dtype="<u2", count=num_tokens, offset=_HEADER_BYTES)
        else:
            shard = np.fromfile(sp, dtype=np.uint16)
        if max_tokens is not None:
            shard = shard[: max_tokens - total]
        chunks.append(shard)
        total += shard.shape[0]
        if verbose and (si + 1) % 20 == 0:
            print(
                f"  loaded {si + 1}/{len(shard_paths)} shards ({total / 1e9:.2f}B tokens, "
                f"{total * 2 / 1e9:.1f} GB)",
                flush=True,
            )
    arr = np.concatenate(chunks)
    if verbose:
        print(
            f"  all shards loaded: {arr.shape[0] / 1e9:.2f}B tokens ({arr.nbytes / 1e9:.1f} GB RAM)",
            flush=True,
        )
    return arr


def _count_preloaded(
    shard_paths: list[Path],
    *,
    vocab_size: int,
    all_lags: list[int],
    num_workers: int,
    max_tokens: Optional[int] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[int, np.ndarray], int]:
    """Load all tokens into RAM, then parallel-count with one worker per lag batch.

    Parallelizes across LAGS (not array slices), so there are no chunk-boundary
    misses. Uses fork-based COW sharing: the token array exists once in physical
    RAM, all workers read the same pages.

    Peak RAM ≈ token_array + N × 2 × chunk_size × 4 bytes.
    For 8B uint16 tokens, 12 workers, 200M chunk: 16 GB + 12 × 1.6 GB ≈ 35 GB.
    """
    t0 = time.monotonic()

    if verbose:
        print("  preload strategy: loading all tokens as uint16 ...", flush=True)
    tokens = _load_all_shards_uint16(shard_paths, max_tokens=max_tokens, verbose=verbose)
    load_time = time.monotonic() - t0

    n = tokens.shape[0]
    effective_workers = max(1, min(num_workers, len(all_lags)))

    # Distribute lags across workers round-robin
    lag_batches: list[list[int]] = [[] for _ in range(effective_workers)]
    for i, lag in enumerate(all_lags):
        lag_batches[i % effective_workers].append(lag)

    if verbose:
        print(
            f"  loaded in {load_time:.1f}s | {effective_workers} workers × "
            f"~{len(all_lags) // effective_workers + 1} lags each on "
            f"{n / 1e9:.2f}B tokens",
            flush=True,
        )

    # Unigram: cheap, do in parent
    unigram = np.bincount(tokens.astype(np.int32, copy=False), minlength=vocab_size).astype(
        np.float64
    )

    # Parallel lag counting via fork (COW shared tokens)
    ctx = mp.get_context("fork")
    with ctx.Pool(
        effective_workers, initializer=_init_shared_tokens, initargs=(tokens, vocab_size)
    ) as pool:
        results = pool.map(_count_lags_worker, lag_batches)

    # Merge
    counts: dict[int, np.ndarray] = {}
    for result_dict in results:
        counts.update(result_dict)

    if verbose:
        elapsed = time.monotonic() - t0
        rss = _get_rss_gb()
        print(
            f"  counted {len(all_lags)} lag matrices over {n / 1e9:.2f}B tokens "
            f"in {elapsed:.1f}s ({n / max(elapsed, 0.01) / 1e6:.0f}M tok/s) | "
            f"peak RSS {rss:.1f} GB",
            flush=True,
        )

    return unigram, counts, n


# ---------------------------------------------------------------------------
# Strategy: gpu — cupy GPU counting (if available)
# ---------------------------------------------------------------------------


def _count_gpu(
    shard_paths: list[Path],
    *,
    vocab_size: int,
    all_lags: list[int],
    max_tokens: Optional[int] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[int, np.ndarray], int]:
    """GPU-accelerated counting using cupy. Requires cupy and a CUDA GPU.

    Loads tokens as uint16 on GPU, processes each lag in chunks to avoid
    allocating full-length int32 arrays. Peak VRAM ≈ tokens (uint16) + 2 × chunk (int32).
    For 8B tokens with 500M chunk: ~16 GB + 2 × 2 GB = ~20 GB VRAM.
    """
    import cupy as cp

    t0 = time.monotonic()
    if verbose:
        print("  gpu strategy: loading all tokens as uint16 ...", flush=True)
    tokens_cpu = _load_all_shards_uint16(shard_paths, max_tokens=max_tokens, verbose=verbose)

    if verbose:
        print(f"  transferring {tokens_cpu.nbytes / 1e9:.1f} GB to GPU ...", flush=True)
    tokens_gpu = cp.asarray(tokens_cpu)
    del tokens_cpu

    n = int(tokens_gpu.shape[0])
    v32 = cp.int32(vocab_size)
    vsq = vocab_size * vocab_size
    chunk_size = 500_000_000  # 500M tokens per chunk

    unigram = cp.bincount(tokens_gpu.astype(cp.int32), minlength=vocab_size)

    counts: dict[int, np.ndarray] = {}
    for lag in all_lags:
        if lag >= n:
            counts[lag] = np.zeros((vocab_size, vocab_size), dtype=np.float64)
            continue
        acc = np.zeros(vsq, dtype=np.float64)
        for start in range(0, n - lag, chunk_size):
            end = min(start + chunk_size, n - lag)
            a = tokens_gpu[start:end].astype(cp.int32)
            b = tokens_gpu[start + lag : end + lag].astype(cp.int32)
            flat = a * v32 + b
            acc += cp.asnumpy(cp.bincount(flat, minlength=vsq)).astype(np.float64)
            del a, b, flat
        counts[lag] = acc.reshape(vocab_size, vocab_size)

    unigram_np = cp.asnumpy(unigram).astype(np.float64)
    del tokens_gpu, unigram
    cp.get_default_memory_pool().free_all_blocks()

    if verbose:
        elapsed = time.monotonic() - t0
        print(
            f"  counted {len(all_lags)} lag matrices over {n / 1e9:.2f}B tokens "
            f"in {elapsed:.1f}s ({n / max(elapsed, 0.01) / 1e6:.0f}M tok/s) [GPU]",
            flush=True,
        )

    return unigram_np, counts, n


def _count_shards_streamed(
    shard_paths: list[Path],
    *,
    vocab_size: int,
    all_lags: list[int],
    max_tokens: Optional[int] = None,
    num_workers: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[int, np.ndarray], int]:
    """Count statistics from shard files.

    Args:
        num_workers: 0 = serial streaming (~500 MB peak RAM, slowest).
                     N>0 = parallel with N processes. Each worker accumulates
                     over its shard batch locally, so IPC is one set of
                     accumulators per worker (~104 MB for V=1024, 13 lags).
                     Peak RAM ≈ N × (shard_size + 104 MB).
                     -1 = auto (half of CPU count, clamped to [1, 8]).

    Note: max_tokens forces serial mode (needs running total to stop early).
    """
    if max_tokens is not None:
        num_workers = 0
    if num_workers < 0:
        num_workers = _default_num_workers()

    t0 = time.monotonic()

    if num_workers > 0 and len(shard_paths) > 1:
        effective_workers = min(num_workers, len(shard_paths))
        # Round-robin assignment so each worker gets a mix of early/late shards
        batches: list[list[Path]] = [[] for _ in range(effective_workers)]
        for i, sp in enumerate(shard_paths):
            batches[i % effective_workers].append(sp)

        if verbose:
            per_worker = [len(b) for b in batches]
            print(
                f"  parallel counting: {effective_workers} workers × "
                f"{per_worker[0]} shards, {len(all_lags)} lags",
                flush=True,
            )

        args_list = [(batch, vocab_size, all_lags) for batch in batches]
        with mp.Pool(effective_workers) as pool:
            results = pool.map(_count_shard_batch, args_list)

        # Merge worker results
        unigram = np.zeros(vocab_size, dtype=np.float64)
        counts = {lag: np.zeros((vocab_size, vocab_size), dtype=np.float64) for lag in all_lags}
        total = 0
        for uni, cts, n in results:
            unigram += uni
            for lag in all_lags:
                counts[lag] += cts[lag]
            total += n
    else:
        # Serial streaming — original low-RAM path
        v32 = np.int32(vocab_size)
        vsq = vocab_size * vocab_size
        unigram = np.zeros(vocab_size, dtype=np.float64)
        counts = {lag: np.zeros((vocab_size, vocab_size), dtype=np.float64) for lag in all_lags}
        total = 0

        for si, path in enumerate(shard_paths):
            if max_tokens is not None and total >= max_tokens:
                break
            shard = (
                _load_shard_with_header(path)
                if _detect_has_header(path)
                else np.fromfile(path, dtype=np.uint16).astype(np.int32, copy=False)
            )
            if max_tokens is not None:
                shard = shard[: max_tokens - total]
            n = shard.shape[0]
            if n == 0:
                continue

            unigram += np.bincount(shard, minlength=vocab_size)
            total += n

            buf = np.empty(n - 1, dtype=np.int32)
            for lag in all_lags:
                if lag >= n:
                    continue
                length = n - lag
                flat = buf[:length]
                np.multiply(shard[:length], v32, out=flat)
                np.add(flat, shard[lag : lag + length], out=flat)
                counts[lag] += np.bincount(flat, minlength=vsq).reshape(vocab_size, vocab_size)

            if verbose and (si + 1) % 10 == 0:
                elapsed = time.monotonic() - t0
                print(
                    f"  shard {si + 1}/{len(shard_paths)} | {total / 1e9:.2f}B tokens "
                    f"| {total / elapsed / 1e6:.0f}M tok/s",
                    flush=True,
                )

    if verbose:
        elapsed = time.monotonic() - t0
        rss = _get_rss_gb()
        print(
            f"  counted {len(all_lags)} lag matrices over {total / 1e9:.2f}B tokens "
            f"in {elapsed:.1f}s ({total / max(elapsed, 0.01) / 1e6:.0f}M tok/s) | "
            f"peak RSS {rss:.1f} GB",
            flush=True,
        )

    return unigram, counts, total


SPEC_STRATEGIES = ("auto", "stream", "parallel", "preload", "gpu")


def build_spec_optimized(
    source: str | Path,
    *,
    vocab_size: int = 1024,
    core_dim: int = 48,
    branch_lags: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64),
    num_blocks: int = 9,
    smoothing: float = 0.25,
    fixed_dtype: torch.dtype = torch.bfloat16,
    storage_dtype: str = "uint16",
    max_tokens: Optional[int] = None,
    num_workers: int = -1,
    strategy: str = "auto",
    embedding_init: str = "spectral",
    spectral_neighbors: int = 64,
    lag_identity_base: float = 0.15,
    readout_rank: Optional[int] = None,
    verbose: bool = True,
) -> AmplifierSpec:
    """Build an AmplifierSpec from a file or shard directory.

    Args:
        num_workers: Worker count for parallel/preload strategies.
            -1 = auto (cpu_count // 2, clamped [1, 8]).
        strategy: Counting strategy for shard directories.
            'auto': tries gpu → parallel (safe default that avoids preload RAM blowups).
            'stream': serial, one shard at a time (~500 MB RAM, slowest).
            'parallel': parallel per-shard counting (moderate RAM, I/O bound).
            'preload': load all tokens into RAM first, then parallel count
                       on array slices (~16 GB + N×104 MB, fast).
            'gpu': cupy GPU counting (~16 GB VRAM, fastest). Requires cupy.
    """
    if strategy not in SPEC_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, must be one of {SPEC_STRATEGIES}")

    branch_lags = _normalize_branch_lags(branch_lags)
    all_lags = sorted(set(branch_lags) | {1})
    p = Path(source)

    if num_workers < 0:
        num_workers = _default_num_workers()

    # --- counting phase ---
    if p.is_dir():
        train = sorted(p.glob("fineweb_train_*.bin"))
        all_shards = train if train else sorted(p.glob("*.bin"))
        if not all_shards:
            raise FileNotFoundError(f"no .bin files in {p}")

        # Resolve 'auto' strategy
        resolved = strategy
        if resolved == "auto":
            # Try gpu first
            try:
                import cupy as cp

                if cp.cuda.runtime.getDeviceCount() > 0:
                    resolved = "gpu"
            except Exception:
                pass
            if resolved == "auto":
                resolved = "parallel"  # safer CPU fallback than preload

        if verbose:
            print(
                f"Spec build from {len(all_shards)} shards in {p} (strategy={resolved}, "
                f"workers={num_workers}) ...",
                flush=True,
            )

        if resolved == "gpu":
            unigram, counts, total_tokens = _count_gpu(
                all_shards,
                vocab_size=vocab_size,
                all_lags=all_lags,
                max_tokens=max_tokens,
                verbose=verbose,
            )
        elif resolved == "preload":
            unigram, counts, total_tokens = _count_preloaded(
                all_shards,
                vocab_size=vocab_size,
                all_lags=all_lags,
                num_workers=num_workers,
                max_tokens=max_tokens,
                verbose=verbose,
            )
        elif resolved == "parallel":
            unigram, counts, total_tokens = _count_shards_streamed(
                all_shards,
                vocab_size=vocab_size,
                all_lags=all_lags,
                max_tokens=max_tokens,
                num_workers=num_workers,
                verbose=verbose,
            )
        else:  # stream
            unigram, counts, total_tokens = _count_shards_streamed(
                all_shards,
                vocab_size=vocab_size,
                all_lags=all_lags,
                max_tokens=max_tokens,
                num_workers=0,
                verbose=verbose,
            )

        bigram = counts[1]
        lag_pair_counts = {lag: counts[lag] for lag in branch_lags}
    else:
        # Single file: load and count in chunks
        if verbose:
            print(f"Loading tokens from {source} as int32 ...", flush=True)
        tokens = load_tokens_int32(source, storage_dtype=storage_dtype)
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        if verbose:
            print(
                f"  {tokens.size / 1e9:.2f}B tokens loaded ({tokens.nbytes / 1e9:.1f} GB RAM)",
                flush=True,
            )
        if tokens.size < max(branch_lags) + 2:
            raise ValueError(f"need at least {max(branch_lags) + 2} tokens, got {tokens.size}")
        token_min, token_max = int(tokens.min()), int(tokens.max())
        if token_min < 0 or token_max >= vocab_size:
            raise ValueError(
                f"token ids must be in [0, {vocab_size - 1}], got min={token_min}, max={token_max}"
            )
        unigram, bigram, lag_pair_counts = count_all(
            tokens,
            vocab_size=vocab_size,
            branch_lags=branch_lags,
            verbose=verbose,
        )
        total_tokens = tokens.size
        del tokens

    # --- from here on it's the same math as the original ---
    if verbose:
        print("  building amplifier matrices from counted statistics ...", flush=True)

    spec = _build_spec_from_counts(
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
            "total_tokens": int(total_tokens),
            "bigram_tokens": int(total_tokens - 1),
        },
    )

    if verbose:
        print(spec.summary())

    return spec


__all__ = [
    "SPEC_STRATEGIES",
    "build_spec_optimized",
    "count_all",
    "load_tokens_int32",
    "load_train_val_int32",
]
