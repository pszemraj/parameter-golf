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
import math
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
    """Return the current process RSS in GB.

    :return float: Main-process RSS in GB, excluding child workers.
    """
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
    """Load a competition shard and skip the 1024-byte header if present.

    :param Path path: Shard path to read.
    :return np.ndarray: Token array as int32.
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
    """Check whether a shard uses the competition header format.

    :param Path path: Shard path to inspect.
    :return bool: ``True`` when the expected header is present.
    """
    header = np.fromfile(path, dtype="<i4", count=2)
    return header.size >= 2 and int(header[0]) == _SHARD_MAGIC and int(header[1]) == _SHARD_VERSION


def load_tokens_int32(
    source: str | Path,
    *,
    storage_dtype: str = "uint16",
    max_tokens: Optional[int] = None,
) -> np.ndarray:
    """Load tokens from a file or shard directory as int32.

    :param str | Path source: File or directory to load.
    :param str storage_dtype: On-disk dtype to use for non-header shards.
    :param Optional[int] max_tokens: Optional cap on returned tokens.
    :return np.ndarray: Flattened token array as int32.
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
    allow_train_frac_val_split: bool = False,
    max_tokens: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load train and validation token splits.

    :param str | Path source: File or directory to load.
    :param str storage_dtype: On-disk dtype to use for non-header shards.
    :param float train_frac: Fraction used for a single-file train split.
    :param bool allow_train_frac_val_split: Whether directory data may fall back
        to splitting train tokens when explicit validation shards are missing.
    :param Optional[int] max_tokens: Optional cap on loaded train tokens.
    :return tuple[np.ndarray, np.ndarray]: Train and validation token arrays.
    """
    p = Path(source)
    np_dtype = NUMPY_DTYPE_MAP[storage_dtype]

    if p.is_dir():
        train_shards = sorted(p.glob("fineweb_train_*.bin"))
        val_shards = sorted(p.glob("fineweb_val_*.bin"))
        if train_shards:
            if not val_shards and not allow_train_frac_val_split:
                raise FileNotFoundError(
                    f"{p} has fineweb_train_* shards but no fineweb_val_* shards. "
                    "Official-style runs must use the provided validation shard. "
                    "For a deliberate local smoke fallback, set "
                    "allow_train_frac_val_split=True."
                )
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
    """Count unigram, bigram, and lag-pair matrices.

    :param np.ndarray tokens: Token sequence to count.
    :param int vocab_size: Vocabulary size.
    :param tuple[int, ...] branch_lags: Lag offsets to count.
    :param int chunk_size: Chunk size used to bound temporary buffers.
    :param bool verbose: Whether to print progress.
    :return tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]: Unigram,
        bigram, and lag-pair counts.
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
    """Count unigram and lag-pair statistics for one shard.

    :param Path path: Shard path to count.
    :param int vocab_size: Vocabulary size.
    :param list[int] all_lags: Lag offsets to count.
    :return tuple[np.ndarray, dict[int, np.ndarray], int]: Unigram counts,
        lag-pair counts, and token count.
    """
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
    """Process a shard batch in one worker.

    :param tuple args: ``(shard_paths, vocab_size, all_lags)`` batch payload.
    :return tuple[np.ndarray, dict[int, np.ndarray], int]: Accumulated unigram
        counts, lag-pair counts, and token count.
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
    """Return the default worker count.

    :return int: Half the CPU cores, clamped to ``[1, 8]``.
    """
    cpus = os.cpu_count() or 1
    return max(1, min(cpus // 2, 8))


# ---------------------------------------------------------------------------
# Strategy: preload — load all tokens into RAM, then parallel count on slices
# ---------------------------------------------------------------------------

# Module-level ref for fork-shared memory (workers inherit via COW)
_shared_tokens: Optional[np.ndarray] = None
_shared_vocab_size: int = 0


def _init_shared_tokens(tokens: np.ndarray, vocab_size: int) -> None:
    """Initialize the fork-shared token buffer for worker processes.

    :param np.ndarray tokens: Shared token array.
    :param int vocab_size: Vocabulary size for the shared buffer.
    """
    global _shared_tokens, _shared_vocab_size
    _shared_tokens = tokens
    _shared_vocab_size = vocab_size


def _count_lags_worker(lags: list[int]) -> dict[int, np.ndarray]:
    """Count a subset of lags against the shared token array.

    :param list[int] lags: Lag offsets assigned to this worker.
    :return dict[int, np.ndarray]: Lag-pair matrices keyed by lag.
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
    """Load all shards into one contiguous uint16 array.

    :param list[Path] shard_paths: Shard paths to load.
    :param Optional[int] max_tokens: Optional token cap.
    :param bool verbose: Whether to print progress.
    :return np.ndarray: Concatenated token array.
    """
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
    """Preload tokens into RAM and count lags in parallel.

    :param list[Path] shard_paths: Shard paths to load.
    :param int vocab_size: Vocabulary size.
    :param list[int] all_lags: Lag offsets to count.
    :param int num_workers: Worker count to use.
    :param Optional[int] max_tokens: Optional token cap.
    :param bool verbose: Whether to print progress.
    :return tuple[np.ndarray, dict[int, np.ndarray], int]: Unigram counts,
        lag-pair counts, and token count.
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
    """Count statistics on GPU with CuPy.

    :param list[Path] shard_paths: Shard paths to load.
    :param int vocab_size: Vocabulary size.
    :param list[int] all_lags: Lag offsets to count.
    :param Optional[int] max_tokens: Optional token cap.
    :param bool verbose: Whether to print progress.
    :return tuple[np.ndarray, dict[int, np.ndarray], int]: Unigram counts,
        lag-pair counts, and token count.
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
    """Count shard statistics with optional streaming or multiprocessing.

    :param list[Path] shard_paths: Shard paths to count.
    :param int vocab_size: Vocabulary size.
    :param list[int] all_lags: Lag offsets to count.
    :param Optional[int] max_tokens: Optional token cap.
    :param int num_workers: Worker count; ``0`` forces serial mode.
    :param bool verbose: Whether to print progress.
    :return tuple[np.ndarray, dict[int, np.ndarray], int]: Unigram counts,
        lag-pair counts, and token count.
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
    """Build an ``AmplifierSpec`` from a file or shard directory.

    :param str | Path source: File or directory to load.
    :param int vocab_size: Vocabulary size.
    :param int core_dim: Core width for the factorized spec.
    :param Sequence[int] branch_lags: Lag offsets to include.
    :param int num_blocks: Number of amplifier blocks.
    :param float smoothing: Additive smoothing for count-derived logits.
    :param torch.dtype fixed_dtype: Storage dtype for fixed spec tensors.
    :param str storage_dtype: On-disk dtype for non-header shards.
    :param Optional[int] max_tokens: Optional token cap.
    :param int num_workers: Worker count for parallel/preload strategies.
    :param str strategy: Counting strategy for shard directories.
    :param str embedding_init: Initialization method for the token basis.
    :param int spectral_neighbors: Neighbor count for spectral basis build.
    :param float lag_identity_base: Base identity blend for lag operators.
    :param Optional[int] readout_rank: Optional low-rank readout factorization.
    :param bool verbose: Whether to print progress.
    :return AmplifierSpec: Built spec.
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


def _mmap_raw_token_file(path: Path, *, storage_dtype: str) -> np.ndarray:
    """Memory-map a raw or competition-header token file.

    :param Path path: Token shard path.
    :param str storage_dtype: On-disk dtype for non-header files.
    :return np.ndarray: Read-only token array view.
    """
    if _detect_has_header(path):
        header = np.fromfile(path, dtype="<i4", count=_HEADER_INTS)
        num_tokens = int(header[2])
        return np.memmap(
            path,
            dtype="<u2",
            mode="r",
            offset=_HEADER_BYTES,
            shape=(num_tokens,),
        )
    return np.memmap(path, dtype=NUMPY_DTYPE_MAP[storage_dtype], mode="r")


def _training_token_files(source: str | Path) -> list[Path]:
    """List training token files without validation shards.

    :param str | Path source: Token source.
    :return list[Path]: Ordered training files.
    """
    p = Path(source)
    if p.is_dir():
        train = sorted(p.glob("fineweb_train_*.bin"))
        if train:
            return train
        files = sorted(p.glob("*.bin"))
        valish = [sp for sp in files if "val" in sp.name]
        if valish and len(valish) == len(files):
            raise FileNotFoundError(f"no training shards found in {p}")
        return [sp for sp in files if "val" not in sp.name]
    return [p]


def add_trigram_memory_to_spec(
    spec: AmplifierSpec,
    source: str | Path,
    *,
    storage_dtype: str = "uint16",
    top_k: int = 2,
    smoothing: float = 0.25,
    residual_clip: float = 8.0,
    confidence_count_cap: int = 4096,
    max_tokens: Optional[int] = None,
    chunk_size: int = 50_000_000,
    verbose: bool = True,
) -> AmplifierSpec:
    """Attach exact dense trigram top-K memory tensors to a spec.

    :param AmplifierSpec spec: Base amplifier spec.
    :param str | Path source: Training-token source.
    :param str storage_dtype: On-disk dtype for non-header shards.
    :param int top_k: Number of next-token residuals to store per context.
    :param float smoothing: Additive smoothing for trigram log probabilities.
    :param float residual_clip: Absolute residual-logit clipping bound.
    :param int confidence_count_cap: Count at which confidence saturates.
    :param Optional[int] max_tokens: Optional cap for local smoke builds.
    :param int chunk_size: Counting chunk size in trigram positions.
    :param bool verbose: Whether to print progress.
    :return AmplifierSpec: Copy of ``spec`` with trigram memory tensors.
    """
    vocab_size = int(spec.vocab_size)
    if vocab_size > np.iinfo(np.int16).max:
        raise ValueError(f"trigram_top_tokens use int16 storage; got vocab_size={vocab_size}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if top_k >= vocab_size:
        raise ValueError(f"top_k must be less than vocab_size={vocab_size}, got {top_k}")
    if residual_clip <= 0:
        raise ValueError(f"residual_clip must be positive, got {residual_clip}")
    if confidence_count_cap <= 0:
        raise ValueError(f"confidence_count_cap must be positive, got {confidence_count_cap}")

    num_contexts = vocab_size * vocab_size
    counts = np.zeros((num_contexts, vocab_size), dtype=np.uint32)
    flat_counts = counts.reshape(-1)
    files = _training_token_files(source)
    carry = np.empty(0, dtype=np.int32)
    tokens_seen = 0
    triples_seen = 0
    started = time.monotonic()

    if verbose:
        print(
            f"Building dense trigram top-{top_k} memory from {len(files)} training files "
            f"({counts.nbytes / 1e9:.2f} GB count table) ...",
            flush=True,
        )

    for file_idx, path in enumerate(files):
        if max_tokens is not None and tokens_seen >= max_tokens:
            break
        view = _mmap_raw_token_file(path, storage_dtype=storage_dtype)
        take = int(view.shape[0])
        if max_tokens is not None:
            take = min(take, int(max_tokens) - tokens_seen)
        if take <= 0:
            break
        tokens = np.asarray(view[:take], dtype=np.int32)
        tokens_seen += int(take)
        if carry.size:
            tokens = np.concatenate([carry, tokens])
        if tokens.size >= 3:
            for start in range(0, tokens.size - 2, chunk_size):
                end = min(tokens.size, start + chunk_size + 2)
                chunk = tokens[start:end]
                if chunk.size < 3:
                    continue
                prev = chunk[:-2].astype(np.int64, copy=False)
                cur = chunk[1:-1].astype(np.int64, copy=False)
                nxt = chunk[2:].astype(np.int64, copy=False)
                flat = ((prev * vocab_size + cur) * vocab_size + nxt).astype(np.int64)
                unique, counts_in_chunk = np.unique(flat, return_counts=True)
                flat_counts[unique] += counts_in_chunk.astype(np.uint32, copy=False)
                triples_seen += int(flat.shape[0])
        carry = tokens[-2:].astype(np.int32, copy=True)
        if verbose and ((file_idx + 1) % 10 == 0 or file_idx + 1 == len(files)):
            elapsed = time.monotonic() - started
            print(
                f"  trigram memory counted {file_idx + 1}/{len(files)} files | "
                f"tokens={tokens_seen / 1e9:.2f}B triples={triples_seen / 1e9:.2f}B "
                f"| {triples_seen / max(elapsed, 1e-6) / 1e6:.1f}M triples/s",
                flush=True,
            )

    context_counts = counts.sum(axis=1, dtype=np.uint64)
    top_idx = np.argpartition(counts, kth=vocab_size - top_k, axis=1)[:, -top_k:]
    top_counts = np.take_along_axis(counts, top_idx, axis=1)
    order = np.argsort(-top_counts, axis=1)
    top_tokens = np.take_along_axis(top_idx, order, axis=1).astype(np.int16)
    top_counts = np.take_along_axis(top_counts, order, axis=1).astype(np.float64)

    cur_tokens = (np.arange(num_contexts, dtype=np.int64) % vocab_size)[:, None]
    base_logits = spec.base_bigram_logits.float().cpu().numpy()
    denom = context_counts[:, None].astype(np.float64) + float(smoothing * vocab_size)
    log_trigram = np.log((top_counts + float(smoothing)) / np.maximum(denom, 1e-12))
    residual = log_trigram - base_logits[cur_tokens, top_tokens.astype(np.int64)]
    residual[context_counts == 0] = 0.0
    residual = np.clip(residual, -float(residual_clip), float(residual_clip))
    residual_scale = float(residual_clip) / 127.0
    residual_q = np.rint(residual / residual_scale).clip(-127, 127).astype(np.int8)
    confidence = 255.0 * np.minimum(
        1.0,
        np.log1p(context_counts.astype(np.float64)) / math.log1p(confidence_count_cap),
    )
    confidence_q = np.rint(confidence).clip(0, 255).astype(np.uint8)

    metadata = dict(spec.metadata)
    metadata.update(
        {
            "trigram_memory": "dense_topk_residual",
            "trigram_top_k": int(top_k),
            "trigram_tokens_seen": int(tokens_seen),
            "trigram_triples_seen": int(triples_seen),
            "trigram_smoothing": float(smoothing),
            "trigram_residual_clip": float(residual_clip),
            "trigram_residual_scale": residual_scale,
            "trigram_confidence_count_cap": int(confidence_count_cap),
        }
    )
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
        trigram_top_tokens=torch.from_numpy(top_tokens),
        trigram_residual_values=torch.from_numpy(residual_q),
        trigram_context_confidence=torch.from_numpy(confidence_q),
        metadata=metadata,
    )


__all__ = [
    "SPEC_STRATEGIES",
    "add_trigram_memory_to_spec",
    "build_spec_optimized",
    "count_all",
    "load_tokens_int32",
    "load_train_val_int32",
]
