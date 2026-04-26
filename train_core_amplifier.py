"""Training script for the core/amplifier language model.

Fixes in this version:
- spec compatibility checks with optional rebuilds
- optional token truncation (`--data-max-tokens`)
- vectorized batch sampling without DataLoader overhead
- streamed recurrent training with hidden-state carry (`--carry-chunks`)
- optional hard-token-focused training loss (`--hard-loss-gamma`)
- optimizer split into decay / no-decay parameter groups
- explicit runtime amplifier dtype control
- delayed/partial torch.compile control for warmup-sensitive runs
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from core_amplifier_lm import (
    AmplifierSpec,
    CoreAmplifierLM,
    build_amplifier_spec,
    build_spec_optimized,
    load_train_val_int32,
)
from core_amplifier_lm.experiment import (
    ARTIFACT_LIMIT_BYTES,
    append_jsonl,
    artifact_estimate_bytes,
    artifact_headroom_bytes,
    artifact_status,
    best_and_last_eval,
    command_context,
    compute_steady_state_tokens_per_sec,
    current_peak_memory_mib,
    estimate_repo_code_bytes,
    export_trainable_int8_zlib,
    git_commit,
    nvidia_smi_metadata,
    read_jsonl,
    reset_peak_memory,
    runtime_device_index,
    spec_size_bytes,
    write_json,
)

DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

NUMPY_DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int32": np.int32,
    "int64": np.int64,
}

# Competition shard header: 256 x int32 (1024 bytes)
_SHARD_MAGIC = 20240520
_SHARD_VERSION = 1
_HEADER_INTS = 256
_HEADER_BYTES = _HEADER_INTS * np.dtype("<i4").itemsize
T = TypeVar("T")


@dataclass(frozen=True)
class EvalResult:
    """Validation metrics plus coverage metadata."""

    loss: float
    bpb: float
    tokens: int
    bytes: int
    usable_tokens: int
    coverage_frac: float
    full_coverage: bool
    steps: int
    batch_size: int
    seq_len: int


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs.

    :param int seed: Seed value to apply.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a compiled model if needed.

    :param torch.nn.Module model: Model to unwrap.
    :return torch.nn.Module: The original module when wrapped by ``torch.compile``.
    """
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def parse_branch_lags(text: str) -> tuple[int, ...]:
    """Parse a comma-separated branch-lag list.

    :param str text: Comma-separated lag values.
    :return tuple[int, ...]: Parsed lag tuple.
    """
    values = tuple(int(x) for x in text.split(",") if x)
    if not values:
        raise ValueError("branch_lags must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"branch_lags must be unique, got {values}")
    if any(v <= 0 for v in values):
        raise ValueError(f"branch_lags must all be positive, got {values}")
    return values


def load_tokens(
    path: str | Path,
    *,
    storage_dtype: str = "uint16",
    max_tokens: Optional[int] = None,
) -> np.ndarray:
    """Load token ids from a file into a flat int32 array.

    :param str | Path path: Token file path.
    :param str storage_dtype: On-disk integer dtype for raw files.
    :param Optional[int] max_tokens: Optional token cap.
    :return np.ndarray: Flat int32 token array.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    np_dtype = NUMPY_DTYPE_MAP[storage_dtype]

    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        arr = data["tokens"] if "tokens" in data else data[sorted(data.files)[0]]
    elif suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            arr = obj.cpu().numpy()
        elif isinstance(obj, dict):
            value = obj["tokens"] if "tokens" in obj else obj[next(iter(obj))]
            arr = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        else:
            arr = np.asarray(obj)
    elif suffix == ".gz":
        with gzip.open(path, "rb") as f:
            if max_tokens is None:
                raw = f.read()
            else:
                byte_count = int(max_tokens) * np.dtype(np_dtype).itemsize
                raw = f.read(byte_count)
        itemsize = np.dtype(np_dtype).itemsize
        usable = (len(raw) // itemsize) * itemsize
        arr = np.frombuffer(raw[:usable], dtype=np_dtype).copy()
    else:
        arr = np.fromfile(path, dtype=np_dtype)

    if arr.ndim != 1:
        arr = arr.reshape(-1)
    arr = arr.astype(np.int32, copy=False)
    if max_tokens is not None:
        arr = arr[:max_tokens]
    return arr


def _detect_has_header(path: Path) -> bool:
    """Detect the competition shard header.

    :param Path path: Candidate shard path.
    :return bool: ``True`` when the magic header is present.
    """
    header = np.fromfile(path, dtype="<i4", count=2)
    return header.size >= 2 and int(header[0]) == _SHARD_MAGIC and int(header[1]) == _SHARD_VERSION


def _list_train_val_shards(source: str | Path) -> tuple[list[Path], list[Path]]:
    """List train and validation shard files under a directory.

    :param str | Path source: Data source path.
    :return tuple[list[Path], list[Path]]: Train and validation shard lists.
    """
    p = Path(source)
    if not p.is_dir():
        return [], []
    train_shards = sorted(p.glob("fineweb_train_*.bin"))
    val_shards = sorted(p.glob("fineweb_val_*.bin"))
    if train_shards:
        return train_shards, val_shards
    generic = sorted(p.glob("*.bin"))
    return generic, []


def _memmap_token_file(path: str | Path, *, storage_dtype: str = "uint16") -> np.ndarray:
    """Open a token file as a memory map when supported.

    :param str | Path path: Token file path.
    :param str storage_dtype: On-disk integer dtype for raw files.
    :return np.ndarray: Memory-mapped token array.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p, mmap_mode="r")
        return arr.reshape(-1) if arr.ndim != 1 else arr
    if suffix in (".bin", ".raw"):
        if _detect_has_header(p):
            header = np.fromfile(p, dtype="<i4", count=_HEADER_INTS)
            num_tokens = int(header[2])
            return np.memmap(
                p, dtype=np.uint16, mode="c", offset=_HEADER_BYTES, shape=(num_tokens,)
            )
        np_dtype = NUMPY_DTYPE_MAP.get(storage_dtype, np.uint16)
        return np.memmap(p, dtype=np_dtype, mode="c")
    raise ValueError(f"cannot memmap {p}; expected .bin/.raw/.npy")


def _sample_directory_tokens(
    arrays: list[np.ndarray], *, max_tokens: int = 1_000_000
) -> np.ndarray:
    """Sample a bounded token prefix from a shard list.

    :param list[np.ndarray] arrays: Token arrays to sample from.
    :param int max_tokens: Maximum number of tokens to collect.
    :return np.ndarray: Concatenated sampled tokens.
    """
    chunks = []
    remaining = int(max_tokens)
    for arr in arrays:
        if remaining <= 0:
            break
        take = min(int(arr.shape[0]), remaining)
        if take > 0:
            chunks.append(np.asarray(arr[:take]))
            remaining -= take
    if not chunks:
        raise ValueError("sampled zero tokens from directory")
    return np.concatenate(chunks, axis=0)


def _truncate_array_views(arrays: list[np.ndarray], max_tokens: Optional[int]) -> list[np.ndarray]:
    """Truncate a list of arrays without copying more than needed.

    :param list[np.ndarray] arrays: Token arrays to truncate.
    :param Optional[int] max_tokens: Token cap or ``None``.
    :return list[np.ndarray]: Truncated array views.
    """
    if max_tokens is None:
        return list(arrays)
    remaining = int(max_tokens)
    out: list[np.ndarray] = []
    for arr in arrays:
        if remaining <= 0:
            break
        take = min(int(arr.shape[0]), remaining)
        if take > 0:
            out.append(arr[:take])
            remaining -= take
    if not out:
        raise ValueError(f"data_max_tokens={max_tokens} leaves zero tokens")
    return out


def validate_token_range(tokens: np.ndarray, vocab_size: int) -> None:
    """Validate token ids against the vocabulary size.

    :param np.ndarray tokens: Token array to validate.
    :param int vocab_size: Vocabulary upper bound.
    """
    if tokens.size == 0:
        raise ValueError("loaded zero tokens")
    token_min = int(tokens.min())
    token_max = int(tokens.max())
    if token_min < 0 or token_max >= vocab_size:
        raise ValueError(
            f"token ids must lie in [0, {vocab_size - 1}], got min={token_min}, max={token_max}"
        )


def maybe_split_tokens(tokens: np.ndarray, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Split a flat token array into train and validation views.

    :param np.ndarray tokens: Token array to split.
    :param float train_frac: Fraction assigned to training.
    :return tuple[np.ndarray, np.ndarray]: Train and validation views.
    """
    split = int(tokens.size * train_frac)
    split = max(1, min(tokens.size - 1, split))
    return tokens[:split], tokens[split:]


def fingerprint_tokens(tokens: np.ndarray, *, max_tokens: int = 1_000_000) -> str:
    """Fingerprint token data for spec validation.

    :param np.ndarray tokens: Token array to fingerprint.
    :param int max_tokens: Maximum prefix size included in the digest.
    :return str: Stable hexadecimal fingerprint.
    """
    sample = np.asarray(tokens[:max_tokens], dtype=np.int64)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray([tokens.size], dtype=np.int64).tobytes())
    digest.update(np.asarray([int(sample.min()), int(sample.max())], dtype=np.int64).tobytes())
    digest.update(sample.view(np.uint8))
    return digest.hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file by SHA256.

    :param str | Path path: File to hash.
    :param int chunk_size: Read chunk size.
    :return str: Hex SHA256 digest.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_device(force: Optional[str] = None) -> tuple[torch.device, str, torch.dtype]:
    """Select the runtime device and default AMP dtype.

    :param Optional[str] force: Optional explicit device string.
    :return tuple[torch.device, str, torch.dtype]: Device, device type, and default AMP dtype.
    """
    force = force or os.environ.get("FORCE_DEVICE")
    if force:
        device = torch.device(force)
        device_type = device.type
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
    else:
        device = torch.device("cpu")
        device_type = "cpu"
    amp_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
    return device, device_type, amp_dtype


class RandomStreamBatcher:
    """Stream random training chunks from a flat token tensor."""

    def __init__(
        self,
        tokens: torch.Tensor,
        *,
        seq_len: int,
        batch_size: int,
        output_device: torch.device,
        carry_chunks: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initialize a random streamed batcher.

        :param torch.Tensor tokens: Flat token tensor.
        :param int seq_len: Chunk length.
        :param int batch_size: Batch size.
        :param torch.device output_device: Target device for emitted batches.
        :param int carry_chunks: Number of consecutive chunks per stream.
        :param Optional[torch.Generator] generator: Optional RNG.
        """
        self.tokens = tokens  # keep original dtype (int16/int32), do NOT .long()
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.output_device = output_device
        self.carry_chunks = max(1, int(carry_chunks))
        self.generator = generator
        self.index_device = self.tokens.device
        self.numpy_tokens = (
            self.tokens.detach().cpu().numpy()
            if self.tokens.device.type == "cpu"
            and self.tokens.dtype not in (torch.int32, torch.int64)
            else None
        )
        self.offsets = torch.arange(self.seq_len + 1, device=self.index_device)
        self._pinned_batch_cpu: Optional[torch.Tensor] = (
            torch.empty(
                (self.batch_size, self.seq_len + 1),
                dtype=torch.long,
                pin_memory=True,
            )
            if self.output_device.type == "cuda" and self.numpy_tokens is not None
            else None
        )
        self.starts: Optional[torch.Tensor] = None
        self.remaining = 0
        self.max_start = self.tokens.numel() - self.carry_chunks * self.seq_len - 1
        if self.max_start < 0:
            raise ValueError(
                "not enough tokens for streamed training: "
                f"tokens={self.tokens.numel()}, seq_len={self.seq_len}, carry_chunks={self.carry_chunks}"
            )

    def _start_new_stream(self) -> None:
        """Sample fresh stream starts for the next carry window."""
        self.starts = torch.randint(
            0,
            self.max_start + 1,
            (self.batch_size,),
            device=self.index_device,
            generator=self.generator,
        )
        self.remaining = self.carry_chunks

    def next_batch(self) -> tuple[torch.Tensor, bool]:
        """Return the next training batch and reset flag.

        :return tuple[torch.Tensor, bool]: Batch tensor and whether the stream reset.
        """
        reset_state = False
        if self.starts is None or self.remaining <= 0:
            self._start_new_stream()
            reset_state = True
        assert self.starts is not None
        idx = self.starts[:, None] + self.offsets[None, :]
        if self.numpy_tokens is not None:
            gathered = np.asarray(self.numpy_tokens[idx.cpu().numpy()], dtype=np.int64)
            gathered_t = torch.from_numpy(gathered)
            if self._pinned_batch_cpu is not None:
                self._pinned_batch_cpu.copy_(gathered_t)
                batch = self._pinned_batch_cpu
            else:
                batch = gathered_t
        else:
            batch = self.tokens[idx].long()  # .long() only on the small gathered batch
        self.starts = self.starts + self.seq_len
        self.remaining -= 1
        if batch.device != self.output_device:
            batch = batch.to(
                self.output_device,
                non_blocking=(self.output_device.type == "cuda" and self._pinned_batch_cpu is None),
            )
        return batch, reset_state


class DirectoryRandomStreamBatcher:
    """Random streamed training batches directly from per-shard memmaps."""

    def __init__(
        self,
        shard_arrays: list[np.ndarray],
        *,
        seq_len: int,
        batch_size: int,
        output_device: torch.device,
        carry_chunks: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initialize a random streamed batcher over shard arrays.

        :param list[np.ndarray] shard_arrays: Memmapped shard arrays.
        :param int seq_len: Chunk length.
        :param int batch_size: Batch size.
        :param torch.device output_device: Target device for emitted batches.
        :param int carry_chunks: Number of consecutive chunks per stream.
        :param Optional[torch.Generator] generator: Optional RNG.
        """
        self.shard_arrays = list(shard_arrays)
        if not self.shard_arrays:
            raise ValueError("need at least one training shard")
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.output_device = output_device
        self.carry_chunks = max(1, int(carry_chunks))
        self.generator = generator
        self.offsets = torch.arange(self.seq_len + 1, dtype=torch.long)

        valid_ids = []
        max_starts = []
        start_weights = []
        needed = self.carry_chunks * self.seq_len + 1
        for shard_id, shard in enumerate(self.shard_arrays):
            max_start = int(shard.shape[0]) - needed
            if max_start >= 0:
                valid_ids.append(shard_id)
                max_starts.append(int(max_start))
                start_weights.append(float(max_start + 1))
        if not valid_ids:
            raise ValueError(
                "not enough tokens for streamed training in any shard: "
                f"seq_len={self.seq_len}, carry_chunks={self.carry_chunks}"
            )

        self.valid_shard_ids = torch.tensor(valid_ids, dtype=torch.long)
        self.valid_max_starts = torch.tensor(max_starts, dtype=torch.long)
        self.valid_start_weights = torch.tensor(start_weights, dtype=torch.float64)
        self._batch_cpu: Optional[torch.Tensor] = (
            torch.empty(
                (self.batch_size, self.seq_len + 1),
                dtype=torch.long,
                pin_memory=True,
            )
            if self.output_device.type == "cuda"
            else None
        )
        self.shard_ids: Optional[torch.Tensor] = None
        self.starts: Optional[torch.Tensor] = None
        self.remaining = 0

    def _start_new_stream(self) -> None:
        """Sample fresh shard/start pairs for the next carry window."""
        choice_idx = torch.multinomial(
            self.valid_start_weights,
            num_samples=self.batch_size,
            replacement=True,
            generator=self.generator,
        )
        self.shard_ids = self.valid_shard_ids[choice_idx]
        starts = torch.empty(self.batch_size, dtype=torch.long)
        for local_idx in range(self.valid_shard_ids.numel()):
            rows = (choice_idx == local_idx).nonzero(as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            max_start = int(self.valid_max_starts[local_idx].item())
            starts[rows] = torch.randint(
                0, max_start + 1, (rows.numel(),), generator=self.generator
            )
        self.starts = starts
        self.remaining = self.carry_chunks

    def next_batch(self) -> tuple[torch.Tensor, bool]:
        """Return the next training batch and reset flag.

        :return tuple[torch.Tensor, bool]: Batch tensor and whether the stream reset.
        """
        reset_state = False
        if self.shard_ids is None or self.starts is None or self.remaining <= 0:
            self._start_new_stream()
            reset_state = True

        assert self.shard_ids is not None and self.starts is not None
        batch_cpu = (
            self._batch_cpu
            if self._batch_cpu is not None
            else torch.empty((self.batch_size, self.seq_len + 1), dtype=torch.long)
        )

        for shard_id in torch.unique(self.shard_ids).tolist():
            rows = (self.shard_ids == shard_id).nonzero(as_tuple=False).flatten()
            idx = self.starts[rows][:, None] + self.offsets[None, :]
            idx_np = idx.cpu().numpy()
            gathered = np.asarray(self.shard_arrays[shard_id][idx_np], dtype=np.int64)
            batch_cpu[rows] = torch.from_numpy(gathered)

        self.starts = self.starts + self.seq_len
        self.remaining -= 1

        if self.output_device.type != "cpu":
            batch = batch_cpu.to(self.output_device, non_blocking=False)
        else:
            batch = batch_cpu
        return batch, reset_state


class SequentialStreamBatcher:
    """Stream validation chunks in deterministic order."""

    def __init__(
        self,
        tokens: torch.Tensor,
        *,
        seq_len: int,
        batch_size: int,
        output_device: torch.device,
        allow_tail: bool = False,
        cover_remainder: bool = False,
    ) -> None:
        """Initialize a sequential batcher over one token tensor.

        :param torch.Tensor tokens: Flat token tensor.
        :param int seq_len: Chunk length.
        :param int batch_size: Batch size.
        :param torch.device output_device: Target device for emitted batches.
        :param bool allow_tail: Whether to emit a shorter final chunk instead of
            wrapping to the stream prefix.
        :param bool cover_remainder: Whether to emit leftover validation targets
            after the equal-length streams.
        """
        self.tokens = tokens  # keep original dtype, do NOT .long()
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.output_device = output_device
        self.allow_tail = bool(allow_tail)
        self.cover_remainder = bool(cover_remainder)
        self.index_device = self.tokens.device
        self.numpy_tokens = (
            self.tokens.detach().cpu().numpy()
            if self.tokens.device.type == "cpu"
            and self.tokens.dtype not in (torch.int32, torch.int64)
            else None
        )
        self.offsets = torch.arange(self.seq_len + 1, device=self.index_device)
        self._pinned_batch_cpu: Optional[torch.Tensor] = (
            torch.empty(
                (self.batch_size, self.seq_len + 1),
                dtype=torch.long,
                pin_memory=True,
            )
            if self.output_device.type == "cuda" and self.numpy_tokens is not None
            else None
        )

        usable = self.tokens.numel() - 1
        if usable < self.batch_size * (self.seq_len + 1):
            raise ValueError(
                "validation set too short for the requested batch_size / seq_len combination"
            )
        self.stream_len = usable // self.batch_size
        self.remainder = usable - self.stream_len * self.batch_size
        self.remainder_start = self.stream_len * self.batch_size
        self.remainder_emitted = False
        if self.stream_len <= self.seq_len:
            raise ValueError("validation streams are shorter than one chunk")
        self.base_starts = torch.arange(self.batch_size, device=self.index_device) * self.stream_len
        self.position = 0

    def _gather(self, starts: torch.Tensor, length: int) -> torch.Tensor:
        """Gather one validation batch.

        :param torch.Tensor starts: Start offsets for each row.
        :param int length: Target-token count to gather.
        :return torch.Tensor: Batch with ``length + 1`` tokens per row.
        """
        offsets = (
            self.offsets
            if int(length) == self.seq_len
            else torch.arange(int(length) + 1, device=self.index_device)
        )
        idx = starts[:, None] + offsets[None, :]
        if self.numpy_tokens is not None:
            gathered = np.asarray(self.numpy_tokens[idx.cpu().numpy()], dtype=np.int64)
            gathered_t = torch.from_numpy(gathered)
            if (
                self._pinned_batch_cpu is not None
                and gathered_t.shape == self._pinned_batch_cpu.shape
            ):
                self._pinned_batch_cpu.copy_(gathered_t)
                batch = self._pinned_batch_cpu
            else:
                batch = gathered_t
        else:
            batch = self.tokens[idx].long()  # .long() only on the small gathered batch
        if batch.device != self.output_device:
            batch = batch.to(
                self.output_device,
                non_blocking=(self.output_device.type == "cuda" and self._pinned_batch_cpu is None),
            )
        return batch

    def next_batch(self) -> tuple[torch.Tensor, bool]:
        """Return the next validation batch and reset flag.

        :return tuple[torch.Tensor, bool]: Batch tensor and whether the stream reset.
        """
        reset_state = False
        if self.allow_tail and self.position >= self.stream_len:
            if self.cover_remainder and self.remainder > 0 and not self.remainder_emitted:
                self.remainder_emitted = True
                starts = torch.tensor([self.remainder_start], device=self.index_device)
                return self._gather(starts, self.remainder), True
            self.position = 0
            self.remainder_emitted = False
            reset_state = True

        if not self.allow_tail and self.position + self.seq_len + 1 > self.stream_len:
            self.position = 0
            reset_state = True
        length = self.seq_len
        if self.allow_tail:
            length = min(self.seq_len, self.stream_len - self.position)
        starts = self.base_starts + self.position
        batch = self._gather(starts, length)
        self.position += length
        return batch, reset_state


class BasePathLookup(torch.nn.Module):
    """Tiny wrapper so the frozen bigram path can be compiled independently."""

    def __init__(self, core_model: CoreAmplifierLM) -> None:
        """Wrap the core model's base path for separate compilation.

        :param CoreAmplifierLM core_model: Core model to wrap.
        """
        super().__init__()
        self.core_model = core_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute base-path logits for the provided token ids.

        :param torch.Tensor input_ids: Input token ids.
        :return torch.Tensor: Base-path logits.
        """
        return self.core_model.base_path_logits(input_ids)


def maybe_move_tokens(tokens: torch.Tensor, device: torch.device, enabled: bool) -> torch.Tensor:
    """Move tokens to a device when the transfer is enabled.

    :param torch.Tensor tokens: Token tensor.
    :param torch.device device: Destination device.
    :param bool enabled: Whether device transfer is allowed.
    :return torch.Tensor: Original or moved tensor.
    """
    if not enabled or device.type == "cpu":
        return tokens
    return tokens.to(device=device, non_blocking=(device.type == "cuda"))


# ---------------------------------------------------------------------------
# Memory-mapped token loading
# ---------------------------------------------------------------------------


def _concat_shards_to_flat(
    shard_paths: list[Path],
    out_path: Path,
    *,
    max_tokens: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """Stream-concatenate shard files into a single flat int32 file.

    Writes one shard at a time — peak RAM is one shard (~400 MB).
    Returns total number of tokens written.
    :param list[Path] shard_paths: Shards to concatenate.
    :param Path out_path: Output file path.
    :param Optional[int] max_tokens: Optional token cap.
    :param bool verbose: Whether to print progress.
    :return int: Total tokens written.
    """
    from core_amplifier_lm.spec_builder import _detect_has_header, _load_shard_with_header

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "wb") as f:
        for si, sp in enumerate(shard_paths):
            if max_tokens is not None and total >= max_tokens:
                break
            shard = (
                _load_shard_with_header(sp)
                if _detect_has_header(sp)
                else np.fromfile(sp, dtype=np.uint16).astype(np.int32, copy=False)
            )
            if max_tokens is not None:
                shard = shard[: max_tokens - total]
            shard.astype(np.int32).tofile(f)
            total += shard.shape[0]
            if verbose and (si + 1) % 20 == 0:
                print(
                    f"  concatenated {si + 1}/{len(shard_paths)} shards ({total / 1e9:.2f}B tokens)",
                    flush=True,
                )
    if verbose:
        print(
            f"  wrote {out_path} ({total:,} tokens, {out_path.stat().st_size / 1e9:.1f} GB)",
            flush=True,
        )
    return total


def mmap_train_val(
    source: str | Path,
    *,
    storage_dtype: str = "uint16",
    train_frac: float = 0.98,
    allow_train_frac_val_split: bool = False,
    max_tokens: Optional[int] = None,
    cache_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and validation token views with minimal RAM use.

    :param str | Path source: Data source path.
    :param str storage_dtype: On-disk integer dtype for raw files.
    :param float train_frac: Train split fraction for single-file fallback.
    :param bool allow_train_frac_val_split: Whether directory data may fall back
        to splitting train tokens when explicit validation shards are missing.
    :param Optional[int] max_tokens: Optional token cap.
    :param Optional[str | Path] cache_dir: Optional mmap cache directory.
    :param bool verbose: Whether to print loading progress.
    :return tuple[np.ndarray, np.ndarray]: Train and validation arrays.
    """
    p = Path(source)

    if p.is_dir():
        train_shards = sorted(p.glob("fineweb_train_*.bin"))
        val_shards = sorted(p.glob("fineweb_val_*.bin"))
        if not train_shards:
            train_shards = sorted(p.glob("*.bin"))
            val_shards = []
        elif not val_shards and not allow_train_frac_val_split:
            raise FileNotFoundError(
                f"{p} has fineweb_train_* shards but no fineweb_val_* shards. "
                "Official-style runs must use the provided validation shard. "
                "For a deliberate local smoke fallback, pass "
                "--allow-train-frac-val-split."
            )

        if not train_shards:
            raise FileNotFoundError(f"no .bin files in {p}")

        cdir = Path(cache_dir) if cache_dir else p / ".mmap_cache"
        train_flat = cdir / "train_int32.bin"
        val_flat = cdir / "val_int32.bin"

        # Build caches independently
        if not train_flat.exists():
            if verbose:
                print(f"Building train mmap cache ({len(train_shards)} shards) ...", flush=True)
            cdir.mkdir(parents=True, exist_ok=True)
            _concat_shards_to_flat(train_shards, train_flat, max_tokens=max_tokens, verbose=verbose)

        if val_shards and not val_flat.exists():
            if verbose:
                print(f"Building val mmap cache ({len(val_shards)} shards) ...", flush=True)
            cdir.mkdir(parents=True, exist_ok=True)
            _concat_shards_to_flat(val_shards, val_flat, verbose=verbose)

        if verbose and train_flat.exists():
            print(f"Using mmap cache: {cdir}", flush=True)

        train_mmap = np.memmap(train_flat, dtype=np.int32, mode="c")
        if max_tokens is not None:
            train_mmap = train_mmap[:max_tokens]

        if val_flat.exists():
            val_mmap = np.memmap(val_flat, dtype=np.int32, mode="c")
            if verbose:
                print(
                    f"  train={train_mmap.shape[0]:,} | val={val_mmap.shape[0]:,} (explicit val shard)",
                    flush=True,
                )
        else:
            n = train_mmap.shape[0]
            split = max(1, min(n - 1, int(n * train_frac)))
            val_mmap = train_mmap[split:]
            train_mmap = train_mmap[:split]
            if verbose:
                print(
                    f"  train={train_mmap.shape[0]:,} | val={val_mmap.shape[0]:,} (split from train, no val shard found)",
                    flush=True,
                )

        return train_mmap, val_mmap

    # Single file — mmap directly when possible (including competition headers)
    if p.suffix.lower() in (".bin", ".raw", ".npy"):
        try:
            mmap = _memmap_token_file(p, storage_dtype=storage_dtype)
        except ValueError:
            mmap = None
        if mmap is not None:
            if max_tokens is not None:
                mmap = mmap[:max_tokens]
            n = mmap.shape[0]
            split = max(1, min(n - 1, int(n * train_frac)))
            return mmap[:split], mmap[split:]

    # Fallback: load into RAM (compressed files, .npy, .pt, etc.)
    if verbose:
        print(f"WARNING: cannot mmap {p.suffix} files, loading into RAM", flush=True)
    tokens = load_tokens(str(p), storage_dtype=storage_dtype, max_tokens=max_tokens)
    train, val = maybe_split_tokens(tokens, train_frac)
    return train, val


def assert_model_config_matches_spec(cfg: Any, spec: AmplifierSpec) -> None:
    """Validate the saved config against the loaded spec.

    :param Any cfg: Loaded model config object.
    :param AmplifierSpec spec: Loaded amplifier spec.
    """
    mismatches = []
    model_cfg = cfg.model

    cfg_vocab = int(model_cfg.get("vocab_size", spec.vocab_size))
    if cfg_vocab != int(spec.vocab_size):
        mismatches.append(f"vocab_size mismatch: config={cfg_vocab} spec={spec.vocab_size}")

    cfg_core_dim = int(
        model_cfg.get("core_dim", spec.metadata.get("requested_core_dim", spec.core_dim))
    )
    spec_requested_core_dim = int(spec.metadata.get("requested_core_dim", spec.core_dim))
    if cfg_core_dim != spec_requested_core_dim:
        mismatches.append(
            f"core_dim mismatch: config={cfg_core_dim} spec_requested={spec_requested_core_dim}"
        )

    cfg_branch_lags = tuple(int(x) for x in cfg.branch_lags_tuple)
    if cfg_branch_lags != tuple(spec.branch_lags):
        mismatches.append(
            f"branch_lags mismatch: config={cfg_branch_lags} spec={tuple(spec.branch_lags)}"
        )

    cfg_num_blocks = int(model_cfg.get("num_blocks", spec.num_blocks))
    if cfg_num_blocks != int(spec.num_blocks):
        mismatches.append(f"num_blocks mismatch: config={cfg_num_blocks} spec={spec.num_blocks}")

    cfg_readout_rank = model_cfg.get("readout_rank")
    cfg_readout_rank = None if cfg_readout_rank in (None, 0) else int(cfg_readout_rank)
    spec_readout_rank = spec.metadata.get("readout_rank")
    spec_readout_rank = None if spec_readout_rank in (None, 0) else int(spec_readout_rank)
    if cfg_readout_rank != spec_readout_rank:
        mismatches.append(
            f"readout_rank mismatch: config={cfg_readout_rank} spec={spec_readout_rank}"
        )

    for key in ("smoothing", "embedding_init", "spectral_neighbors", "lag_identity_base"):
        if key not in spec.metadata:
            continue
        cfg_val = model_cfg.get(key)
        if cfg_val is None:
            continue
        spec_val = spec.metadata.get(key)
        if key == "embedding_init":
            if str(cfg_val) == str(spec_val):
                continue
        if key == "spectral_neighbors":
            if str(model_cfg.get("embedding_init", "spectral")) == "svd":
                continue
            cfg_effective = max(1, min(int(cfg_val), int(spec.vocab_size) - 1))
            if cfg_effective != int(spec_val):
                mismatches.append(
                    f"{key} mismatch: config={cfg_val} effective={cfg_effective} spec={spec_val}"
                )
            continue
        if isinstance(spec_val, float):
            if abs(float(cfg_val) - float(spec_val)) > 1e-8:
                mismatches.append(f"{key} mismatch: config={cfg_val} spec={spec_val}")
        else:
            if cfg_val != spec_val:
                mismatches.append(f"{key} mismatch: config={cfg_val} spec={spec_val}")

    if mismatches:
        joined = "\n  - ".join(mismatches)
        raise ValueError(
            "config.json does not match spec.pt:\n"
            f"  - {joined}\n"
            "Rebuild the model dir with inspect_model.py init ... or restore matching files."
        )


def resolve_runtime_amplifier_dtype(
    name: str,
    *,
    device_type: str,
    default_amp_dtype: torch.dtype,
) -> torch.dtype:
    """Resolve the runtime amplifier dtype.

    :param str name: User-facing dtype name.
    :param str device_type: Runtime device type.
    :param torch.dtype default_amp_dtype: Device default AMP dtype.
    :return torch.dtype: Resolved amplifier dtype.
    """
    if name == "auto":
        return torch.float32 if device_type == "cpu" else default_amp_dtype
    return DTYPE_MAP[name]


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Collect the trainable parameters on CPU.

    :param torch.nn.Module model: Model to snapshot.
    :return dict[str, torch.Tensor]: Trainable parameter tensors.
    """
    core_model = unwrap_model(model)
    return {name: p.detach().cpu() for name, p in core_model.named_parameters()}


def load_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a trainable-parameter snapshot into a model.

    :param torch.nn.Module model: Target model.
    :param dict[str, torch.Tensor] state: Parameter snapshot.
    """
    core_model = unwrap_model(model)
    param_map = dict(core_model.named_parameters())
    missing = []
    for name, tensor in state.items():
        if name not in param_map:
            missing.append(name)
            continue
        param_map[name].data.copy_(
            tensor.to(device=param_map[name].device, dtype=param_map[name].dtype)
        )
    if missing:
        print(f"Warning: skipped unknown trainable parameters: {missing}")


def build_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    device_type: str,
) -> AdamW:
    """Build the AdamW optimizer with decay and no-decay groups.

    :param torch.nn.Module model: Model to optimize.
    :param float learning_rate: Base learning rate.
    :param float weight_decay: Weight decay for decay parameters.
    :param str device_type: Runtime device type.
    :return AdamW: Configured optimizer.
    """
    core_model = unwrap_model(model)
    decay_params = []
    no_decay_params = []
    for name, param in core_model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or name.endswith("bias")
            or name
            in {
                "_h0_raw",
                "block_gain",
                "branch_scale",
                "branch_bias",
                "readout_branch_scale",
                "residual_log_scale",
                "base_bigram_delta_log_scale",
                "trigram_log_scale",
                "residual_readout_delta_log_scale",
                "logit_bias",
            }
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    print(
        "Optimizer groups | "
        f"decay={sum(p.numel() for p in decay_params):,} params | "
        f"no_decay={sum(p.numel() for p in no_decay_params):,} params"
    )
    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        fused=(device_type == "cuda"),
    )


def get_lr(
    step: int,
    *,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    hold_steps: int,
    total_steps: int,
) -> float:
    """Compute the warmup-hold-cosine learning rate.

    :param int step: Current step.
    :param float max_lr: Peak learning rate.
    :param float min_lr: Minimum learning rate.
    :param int warmup_steps: Warmup duration.
    :param int hold_steps: Flat hold duration after warmup.
    :param int total_steps: Total step budget.
    :return float: Learning rate for the current step.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step < warmup_steps + hold_steps:
        return max_lr
    if step >= total_steps:
        return min_lr
    decay_steps = max(1, total_steps - warmup_steps - hold_steps)
    progress = min(1.0, (step - warmup_steps - hold_steps) / decay_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: AdamW, lr: float) -> None:
    """Set the learning rate on every optimizer param group.

    :param AdamW optimizer: Optimizer to update.
    :param float lr: Learning rate to apply.
    """
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def cross_entropy_per_token(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute token-wise cross entropy.

    :param torch.Tensor logits: Model logits.
    :param torch.Tensor targets: Target token ids.
    :return torch.Tensor: Per-token loss tensor.
    """
    if logits.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        loss_logits = logits
    else:
        loss_logits = logits.float()
    per_token = F.cross_entropy(
        loss_logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    return per_token.view_as(targets)


def build_byte_count_lut(
    tokenizer_path: str | Path, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Build the competition byte-count lookup table.

    :param str | Path tokenizer_path: SentencePiece tokenizer path.
    :param int vocab_size: Vocabulary size.
    :param torch.device device: Target device.
    :return torch.Tensor: Per-token byte counts.
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    sp_vocab = sp.vocab_size()
    if sp_vocab != int(vocab_size):
        raise ValueError(
            f"tokenizer vocab size mismatch: tokenizer={sp_vocab} spec={int(vocab_size)}"
        )
    table_size = int(vocab_size)
    byte_counts = np.zeros(table_size, dtype=np.int16)
    has_leading_space = np.zeros(table_size, dtype=np.bool_)

    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        if sp.is_byte(tid):
            byte_counts[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space[tid] = True
            piece = piece[1:]
        byte_counts[tid] = len(piece.encode("utf-8"))

    byte_counts_t = torch.tensor(byte_counts[:vocab_size], dtype=torch.int16, device=device)
    has_space_t = torch.tensor(has_leading_space[:vocab_size], dtype=torch.bool, device=device)
    # Total bytes per token = base_bytes + has_leading_space (the space is 1 extra byte)
    total_bytes = byte_counts_t.to(torch.int32) + has_space_t.to(torch.int32)
    return total_bytes


def validate_exact_bpb_targets(
    byte_count_lut: torch.Tensor,
    val_tokens: torch.Tensor,
    *,
    chunk_size: int = 8_000_000,
) -> None:
    """Verify validation target ids all have positive byte counts.

    :param torch.Tensor byte_count_lut: Per-token byte-count lookup.
    :param torch.Tensor val_tokens: Flat validation token tensor.
    :param int chunk_size: Validation target chunk size.
    :raises ValueError: If any validation target id maps to zero bytes.
    """
    if val_tokens.numel() <= 1:
        raise ValueError("validation token stream is empty; exact BPB is undefined")
    lut_cpu = byte_count_lut.detach().cpu()
    if int(lut_cpu.min().item()) <= 0:
        zero_ids = (lut_cpu <= 0).nonzero(as_tuple=False).flatten()[:16].tolist()
        for start in range(1, int(val_tokens.numel()), int(chunk_size)):
            end = min(int(val_tokens.numel()), start + int(chunk_size))
            targets = val_tokens[start:end]
            target_ids = targets.detach().cpu().long()
            target_byte_counts = lut_cpu[target_ids]
            if int(target_byte_counts.min().item()) <= 0:
                bad = target_ids[target_byte_counts <= 0][:16].tolist()
                raise ValueError(
                    "validation targets include zero-byte tokenizer ids; "
                    f"bad_target_ids={bad} zero_byte_ids_sample={zero_ids}"
                )


def compute_training_objective(
    base_path_model: torch.nn.Module,
    logits: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    hard_loss_gamma: float,
    hard_loss_cap: float = 0.0,
    base_nll_model: Optional[CoreAmplifierLM] = None,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute weighted and raw training losses.

    :param torch.nn.Module base_path_model: Base-path model for hard-loss weighting.
    :param torch.Tensor logits: Current model logits.
    :param torch.Tensor inputs: Input token ids.
    :param torch.Tensor targets: Target token ids.
    :param float hard_loss_gamma: Hard-loss exponent.
    :param float hard_loss_cap: Optional weight cap.
    :param Optional[CoreAmplifierLM] base_nll_model: Optional model with a
        precomputed frozen-base NLL table.
    :param bool collect_stats: Whether to synchronize scalar diagnostics.
    :return tuple[torch.Tensor, torch.Tensor, dict[str, float]]: Weighted sum, raw sum, and stats.
    """
    per_token = cross_entropy_per_token(logits, targets)
    raw_sum = per_token.sum()
    stats = {"base_loss": float("nan"), "weight_max": 1.0}
    if hard_loss_gamma <= 0:
        return raw_sum, raw_sum, stats
    with torch.no_grad():
        if base_nll_model is not None:
            base_per_token = base_nll_model.base_nll_for_targets(inputs, targets)
        else:
            base_logits = base_path_model(inputs)
            base_per_token = cross_entropy_per_token(base_logits, targets)
        weights = base_per_token.clamp_min(1e-3).pow(hard_loss_gamma)
        if hard_loss_cap > 0:
            weights = weights.clamp_max(hard_loss_cap)
        weights = weights / weights.mean().clamp_min(1e-6)
        if collect_stats:
            stats = {
                "base_loss": float(base_per_token.mean().item()),
                "weight_max": float(weights.max().item()),
            }
    weighted_sum = (per_token * weights).sum()
    return weighted_sum, raw_sum, stats


def evaluate(
    model: torch.nn.Module,
    val_tokens: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    device_type: str,
    amp_dtype: torch.dtype,
    steps: int,
    use_autocast: bool,
    byte_count_lut: Optional[torch.Tensor] = None,
    cover_once: bool = False,
) -> EvalResult:
    """Evaluate loss and bits-per-byte on validation data.

    :param torch.nn.Module model: Model to evaluate.
    :param torch.Tensor val_tokens: Validation token tensor.
    :param int seq_len: Chunk length.
    :param int batch_size: Validation batch size.
    :param torch.device device: Runtime device.
    :param str device_type: Runtime device type.
    :param torch.dtype amp_dtype: Autocast dtype.
    :param int steps: Validation steps to run.
    :param bool use_autocast: Whether to use autocast.
    :param Optional[torch.Tensor] byte_count_lut: Optional exact byte-count lookup.
    :param bool cover_once: Whether to emit tail/remainder chunks exactly once
        instead of wrapping to the validation prefix.
    :return EvalResult: Validation loss, bpb, and coverage metadata.
    """
    usable_tokens = max(0, int(val_tokens.numel()) - 1)
    if steps <= 0:
        return EvalResult(
            loss=float("nan"),
            bpb=float("nan"),
            tokens=0,
            bytes=0,
            usable_tokens=usable_tokens,
            coverage_frac=0.0,
            full_coverage=False,
            steps=int(steps),
            batch_size=0,
            seq_len=int(seq_len),
        )
    core_model = unwrap_model(model)
    model.eval()
    total_loss_t = torch.zeros((), dtype=torch.float32, device=device)
    total_tokens = 0
    total_bits_t = torch.zeros((), dtype=torch.float32, device=device)
    total_bytes_t = torch.zeros((), dtype=torch.int64, device=device)

    max_streams = max(1, usable_tokens // max(1, seq_len + 1))
    eval_batch_size = min(batch_size, max_streams)
    batcher = SequentialStreamBatcher(
        val_tokens,
        seq_len=seq_len,
        batch_size=eval_batch_size,
        output_device=device,
        allow_tail=cover_once,
        cover_remainder=cover_once,
    )

    def autocast_context() -> Any:
        """Return the validation autocast context manager.

        :return Any: Validation autocast context manager or ``nullcontext``.
        """
        if use_autocast and device_type != "cpu":
            return torch.autocast(device_type=device_type, dtype=amp_dtype)
        return nullcontext()

    state: Optional[Any] = None
    with torch.no_grad():
        for _ in range(steps):
            batch, reset_state = batcher.next_batch()
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            if state is None or reset_state:
                state = core_model.initial_state(inputs.size(0), device=device)
            else:
                state = core_model.detach_state(state)
            with autocast_context():
                logits, state = model(inputs, state=state, return_state=True)
            per_token = cross_entropy_per_token(logits, targets)
            total_loss_t += per_token.sum()
            total_tokens += targets.numel()

            if byte_count_lut is not None:
                bits = per_token / math.log(2)  # nats → bits, per token
                total_bits_t += bits.sum()
                total_bytes_t += byte_count_lut[targets.long()].sum()
            state = core_model.detach_state(state)

    total_loss = float(total_loss_t.item())
    total_bits = float(total_bits_t.item())
    total_bytes = int(total_bytes_t.item())
    val_loss = total_loss / max(1, total_tokens)
    val_bpb = (total_bits / max(1, total_bytes)) if total_bytes > 0 else float("nan")
    coverage_frac = min(1.0, float(total_tokens) / max(1, usable_tokens))
    return EvalResult(
        loss=val_loss,
        bpb=val_bpb,
        tokens=int(total_tokens),
        bytes=int(total_bytes),
        usable_tokens=int(usable_tokens),
        coverage_frac=coverage_frac,
        full_coverage=coverage_frac >= 0.999,
        steps=int(steps),
        batch_size=int(eval_batch_size),
        seq_len=int(seq_len),
    )


def full_validation_steps(
    val_tokens: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
) -> int:
    """Compute deterministic sequential steps needed to cover validation once.

    :param torch.Tensor val_tokens: Flat validation token tensor.
    :param int seq_len: Chunk length.
    :param int batch_size: Requested validation batch size.
    :return int: Step count covering the validation streams once.
    """
    usable_tokens = max(0, int(val_tokens.numel()) - 1)
    if usable_tokens <= 0:
        return 0
    max_streams = max(1, usable_tokens // max(1, seq_len + 1))
    eval_batch_size = min(batch_size, max_streams)
    stream_len = usable_tokens // max(1, eval_batch_size)
    remainder = usable_tokens - stream_len * eval_batch_size
    return max(1, math.ceil(stream_len / max(1, seq_len)) + (1 if remainder > 0 else 0))


def eval_payload_fields(result: EvalResult) -> dict[str, int | float | bool]:
    """Flatten validation coverage metadata for metrics rows.

    :param EvalResult result: Evaluation result to flatten.
    :return dict[str, int | float | bool]: Coverage fields.
    """
    return {
        "eval_tokens": int(result.tokens),
        "eval_bytes": int(result.bytes),
        "eval_coverage_denominator_tokens": int(result.usable_tokens),
        "eval_coverage_frac": float(result.coverage_frac),
        "eval_full_coverage": bool(result.full_coverage),
        "eval_steps": int(result.steps),
        "eval_batch_size": int(result.batch_size),
        "eval_seq_len": int(result.seq_len),
    }


def format_eval_coverage(result: EvalResult, *, validation_source: str) -> str:
    """Format validation coverage so sampled evals cannot be confused with train coverage.

    :param EvalResult result: Evaluation result to describe.
    :param str validation_source: Validation-token source label.
    :return str: Human-readable validation coverage string.
    """
    if result.usable_tokens > 0:
        return (
            f"val_coverage {result.coverage_frac:.3%} "
            f"({result.tokens:,}/{result.usable_tokens:,} target tokens, "
            f"source={validation_source})"
        )
    return f"val_coverage {result.coverage_frac:.3%} (source={validation_source})"


def build_expected_spec_metadata(
    args: argparse.Namespace,
    train_tokens: np.ndarray,
    branch_lags: tuple[int, ...],
) -> dict[str, object]:
    """Build the spec metadata expected for a run.

    :param argparse.Namespace args: Parsed CLI arguments.
    :param np.ndarray train_tokens: Training token array.
    :param tuple[int, ...] branch_lags: Branch lag tuple.
    :return dict[str, object]: Expected spec metadata.
    """
    return {
        "spec_version": 2,
        "requested_core_dim": int(args.core_dim),
        "vocab_size": int(args.vocab_size),
        "branch_lags": branch_lags,
        "num_blocks": int(args.num_blocks),
        "readout_rank": None
        if getattr(args, "readout_rank", None) in (None, 0)
        else int(args.readout_rank),
        "smoothing": float(args.smoothing),
        "train_token_count": int(train_tokens.size),
        "train_token_fingerprint": fingerprint_tokens(train_tokens),
        "storage_dtype": str(args.storage_dtype),
        "data_max_tokens": None if args.data_max_tokens is None else int(args.data_max_tokens),
    }


def assert_spec_matches(spec: AmplifierSpec, *, expected: dict[str, object]) -> None:
    """Verify that a loaded spec matches the expected metadata.

    :param AmplifierSpec spec: Loaded spec.
    :param dict[str, object] expected: Expected metadata.
    """
    errors = []
    if spec.vocab_size != int(expected["vocab_size"]):
        errors.append(
            f"vocab_size mismatch: spec={spec.vocab_size} requested={expected['vocab_size']}"
        )
    requested_core_dim = int(spec.metadata.get("requested_core_dim", spec.core_dim))
    if requested_core_dim != int(expected["requested_core_dim"]):
        errors.append(
            f"core_dim mismatch: spec_requested={requested_core_dim} requested={expected['requested_core_dim']}"
        )
    if tuple(spec.branch_lags) != tuple(expected["branch_lags"]):
        errors.append(
            f"branch_lags mismatch: spec={spec.branch_lags} requested={expected['branch_lags']}"
        )
    if spec.num_blocks != int(expected["num_blocks"]):
        errors.append(
            f"num_blocks mismatch: spec={spec.num_blocks} requested={expected['num_blocks']}"
        )
    expected_readout_rank = expected.get("readout_rank")
    spec_readout_rank = spec.metadata.get("readout_rank")
    if expected_readout_rank not in (None, 0):
        expected_readout_rank = int(expected_readout_rank)
    else:
        expected_readout_rank = None
    if spec_readout_rank not in (None, 0):
        spec_readout_rank = int(spec_readout_rank)
    else:
        spec_readout_rank = None
    if spec_readout_rank != expected_readout_rank:
        errors.append(
            f"readout_rank mismatch: spec={spec_readout_rank} requested={expected_readout_rank}"
        )
    stored_smoothing = spec.metadata.get("smoothing")
    if (
        stored_smoothing is not None
        and abs(float(stored_smoothing) - float(expected["smoothing"])) > 1e-8
    ):
        errors.append(
            f"smoothing mismatch: spec={stored_smoothing} requested={expected['smoothing']}"
        )
    stored_fp = spec.metadata.get("train_token_fingerprint")
    if stored_fp is not None and stored_fp != expected["train_token_fingerprint"]:
        errors.append("training data fingerprint mismatch")
    stored_count = spec.metadata.get("train_token_count")
    if stored_count is not None and int(stored_count) != int(expected["train_token_count"]):
        errors.append(
            f"train_token_count mismatch: spec={stored_count} requested={expected['train_token_count']}"
        )
    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(
            f"Existing spec does not match the requested run:\n  - {joined}\nUse --force-rebuild-spec to overwrite it."
        )


def build_spec_if_needed(
    args: argparse.Namespace,
    train_tokens: np.ndarray,
    *,
    data_source: Optional[str | Path] = None,
) -> AmplifierSpec:
    """Load or build the fixed amplifier spec for a run.

    :param argparse.Namespace args: Parsed CLI arguments.
    :param np.ndarray train_tokens: Training token array.
    :param Optional[str | Path] data_source: Optional source path for optimized spec building.
    :return AmplifierSpec: Loaded or newly built spec.
    """
    fixed_dtype = DTYPE_MAP[args.fixed_dtype]
    branch_lags = parse_branch_lags(args.branch_lags)
    expected = build_expected_spec_metadata(args, train_tokens, branch_lags)
    spec_path = Path(args.spec_path) if args.spec_path else None
    if spec_path is not None and spec_path.exists() and not args.force_rebuild_spec:
        spec = AmplifierSpec.load(spec_path)
        assert_spec_matches(spec, expected=expected)
        print(f"Loaded fixed amplifier from {spec_path}")
        print(spec.summary())
        return spec

    # Use the optimized builder when we have a data source path (handles
    # both shard directories and single files, uses int32, single-pass counting).
    # Fall back to the original monolithic builder if no source path is given.
    if data_source is not None:
        spec = build_spec_optimized(
            data_source,
            vocab_size=args.vocab_size,
            core_dim=args.core_dim,
            branch_lags=branch_lags,
            num_blocks=args.num_blocks,
            smoothing=args.smoothing,
            fixed_dtype=fixed_dtype,
            storage_dtype=args.storage_dtype,
            max_tokens=args.spec_max_tokens,
            num_workers=args.spec_workers,
            strategy=args.spec_strategy,
            readout_rank=None
            if getattr(args, "readout_rank", None) in (None, 0)
            else int(args.readout_rank),
        )
    else:
        spec = build_amplifier_spec(
            train_tokens,
            vocab_size=args.vocab_size,
            core_dim=args.core_dim,
            branch_lags=branch_lags,
            num_blocks=args.num_blocks,
            smoothing=args.smoothing,
            fixed_dtype=fixed_dtype,
            max_tokens=args.spec_max_tokens,
            readout_rank=None
            if getattr(args, "readout_rank", None) in (None, 0)
            else int(args.readout_rank),
        )
    spec.metadata.update(expected)
    print(spec.summary())
    if spec_path is not None:
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec.save(spec_path)
        print(f"Saved fixed amplifier to {spec_path}")
    return spec


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train the core/amplifier language model")
    parser.add_argument("model_dir", help="Model directory (must contain config.json + spec.pt)")

    # Training params — all None, resolved from config.json. CLI overrides config.
    parser.add_argument("--data", type=str, default=None, help="Override data source")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--carry-chunks", type=int, default=None)
    parser.add_argument("--bptt-chunks", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-schedule", type=str, default=None, choices=["cosine", "none"])
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--lr-hold-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--hard-loss-gamma", type=float, default=None)
    parser.add_argument("--hard-loss-cap", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--amplifier-dtype", type=str, default=None)
    parser.add_argument("--log-state-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument(
        "--full-val-final",
        dest="full_val_final",
        action="store_true",
        default=None,
        help="Run the final validation over the full validation stream once.",
    )
    parser.add_argument(
        "--no-full-val-final",
        dest="full_val_final",
        action="store_false",
    )
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--allow-train-frac-val-split", action="store_true")
    parser.add_argument("--allow-approx-bpb", action="store_true")
    parser.add_argument(
        "--storage-dtype", type=str, default=None, choices=["uint8", "uint16", "int32", "int64"]
    )
    parser.add_argument("--data-max-tokens", type=int, default=None)

    # Optional architecture overrides. These do not alter the frozen spec unless
    # core_dim / branch_lags / num_blocks change, so they are useful for cheap
    # controller-capacity sweeps.
    parser.add_argument("--core-type", type=str, default=None, choices=["mingru", "scan", "gru"])
    parser.add_argument("--core-layers", type=int, default=None)
    parser.add_argument("--core-expansion", type=float, default=None)
    parser.add_argument("--residual-core", type=int, default=None, choices=[0, 1])
    parser.add_argument("--residual-core-init", type=float, default=None)
    parser.add_argument(
        "--branch-temporal-mode",
        type=str,
        default=None,
        choices=["current", "lagged", "hybrid", "ema", "ema_hybrid"],
    )
    parser.add_argument("--branch-temporal-lag-scale", type=float, default=None)
    parser.add_argument(
        "--residual-token-gate-mode",
        type=str,
        default=None,
        choices=["none", "base", "core_base"],
    )
    parser.add_argument(
        "--branch-router-mode",
        type=str,
        default=None,
        choices=["none", "softmax"],
    )
    parser.add_argument("--base-bigram-delta", type=str, default=None, choices=["none", "full"])
    parser.add_argument("--trigram-memory", type=str, default=None, choices=["none", "frozen"])
    parser.add_argument("--trigram-log-scale-init", type=float, default=None)
    parser.add_argument("--residual-readout-delta-rank", type=int, default=None)
    parser.add_argument("--residual-readout-delta-init-std", type=float, default=None)
    parser.add_argument("--readout-rank", type=int, default=None)
    parser.add_argument(
        "--scan-backend",
        type=str,
        default=None,
        choices=["auto", "heinsen", "assoc", "assoc_accel", "sequential"],
    )

    # Non-config args
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-after", type=int, default=0)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--compile-base-path", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--force-device", type=str, default=None)
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--mmap-cache-dir", type=str, default=None)
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--tokens-on-device", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument(
        "--wandb-watch",
        type=str,
        default=None,
        choices=["off", "gradients", "all"],
    )
    parser.add_argument("--wandb-watch-log-freq", type=int, default=None)

    return parser.parse_args()


def _resolve(cli_val: T | None, config_val: T | None, default: T) -> T:
    """Resolve a value with CLI precedence.

    :param T | None cli_val: CLI override.
    :param T | None config_val: Config value.
    :param T default: Fallback default.
    :return T: Resolved value.
    """
    if cli_val is not None:
        return cli_val
    if config_val is not None:
        return config_val
    return default


def _format_debug_vector(values: list[float], *, precision: int = 3) -> str:
    """Format per-layer debug values.

    :param list[float] values: Values to format.
    :param int precision: Decimal precision.
    :return str: Comma-separated formatted values or ``none``.
    """
    if not values:
        return "none"
    return ",".join(f"{float(x):.{precision}f}" for x in values)


def dtype_name(dtype: Optional[torch.dtype]) -> Optional[str]:
    """Render a torch dtype as a short string.

    :param Optional[torch.dtype] dtype: Torch dtype or ``None``.
    :return Optional[str]: Short dtype name.
    """
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def parse_wandb_tags(raw: Optional[str]) -> list[str]:
    """Parse comma-separated W&B tags.

    :param Optional[str] raw: Raw tag string.
    :return list[str]: Stable, deduplicated tags.
    """
    if not raw:
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        tag = part.strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def build_wandb_config(
    *,
    resolved_config: dict[str, Any],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the static W&B config payload.

    :param dict[str, Any] resolved_config: Resolved run configuration.
    :param dict[str, Any] run_metadata: Static run metadata.
    :return dict[str, Any]: W&B config payload.
    """
    system = run_metadata.get("system", {})
    env = run_metadata.get("env", {})
    runtime = resolved_config.get("runtime", {})
    return {
        "trainer_family": "core_amplifier",
        "run_name": resolved_config.get("run_name"),
        "phase": resolved_config.get("phase"),
        "seed": resolved_config.get("seed"),
        "git_commit": run_metadata.get("git_commit"),
        "model": resolved_config.get("model", {}),
        "training": resolved_config.get("training", {}),
        "data": resolved_config.get("data", {}),
        "runtime": {
            "device": runtime.get("device"),
            "device_type": runtime.get("device_type"),
            "default_amp_dtype": runtime.get("default_amp_dtype"),
            "amplifier_dtype": runtime.get("amplifier_dtype"),
            "autocast": runtime.get("autocast"),
            "gradient_checkpointing": runtime.get("gradient_checkpointing"),
            "exact_val_bpb": runtime.get("exact_val_bpb"),
            "scan_backend_requested": runtime.get("scan_backend_requested"),
            "scan_backend_active": runtime.get("scan_backend_active"),
            "compile": runtime.get("compile", {}),
            "wandb": runtime.get("wandb", {}),
            "float32_matmul_precision": system.get("float32_matmul_precision"),
            "tf32_matmul": system.get("tf32_matmul"),
            "tf32_cudnn": system.get("tf32_cudnn"),
            "torch_blas_prefer_cublaslt": env.get("TORCH_BLAS_PREFER_CUBLASLT"),
        },
        "system": {
            "torch_version": system.get("torch_version"),
            "cuda_version": system.get("cuda_version"),
            "cudnn_version": system.get("cudnn_version"),
            "gpu_name": system.get("gpu_name"),
            "gpu_capability": system.get("gpu_capability"),
            "gpu_total_memory_mib": system.get("gpu_total_memory_mib"),
            "driver_version": system.get("driver_version"),
        },
        "spec": {
            "repo_code_bytes": resolved_config.get("spec", {}).get("repo_code_bytes"),
            "spec_bytes": resolved_config.get("spec", {}).get("spec_bytes"),
            "gzip_spec_bytes": resolved_config.get("spec", {}).get("gzip_spec_bytes"),
            "summary": resolved_config.get("spec", {}).get("summary"),
        },
        "tokenizer_path": resolved_config.get("tokenizer_path"),
        "artifact_limit_bytes": ARTIFACT_LIMIT_BYTES,
    }


def normalize_wandb_run_path(run: Any) -> Optional[str]:
    """Normalize a W&B run path for local metadata.

    :param Any run: W&B run object.
    :return Optional[str]: Normalized run path.
    """
    raw = getattr(run, "path", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    return "/".join(str(part) for part in raw)


def build_resolved_config_payload(
    *,
    cfg: Any,
    model_dir: Path,
    args: argparse.Namespace,
    data_source: str | Path,
    train_tokens_count: int,
    val_tokens_count: int,
    storage_dtype: str,
    train_frac: float,
    allow_train_frac_val_split: bool,
    allow_approx_bpb: bool,
    validation_source: str,
    data_max_tokens: Optional[int],
    seq_len: int,
    batch_size: int,
    grad_accum: int,
    carry_chunks: int,
    bptt_chunks: int,
    num_steps: int,
    learning_rate: float,
    lr_schedule: str,
    min_lr: float,
    warmup_steps: int,
    lr_hold_steps: int,
    weight_decay: float,
    hard_loss_gamma: float,
    hard_loss_cap: float,
    grad_clip: float,
    dropout: float,
    val_every: int,
    val_steps: int,
    full_val_final: bool,
    log_every: int,
    log_state_every: int,
    save_every: int,
    device: torch.device,
    device_type: str,
    default_amp_dtype: torch.dtype,
    runtime_amp_dtype: torch.dtype,
    use_autocast: bool,
    spec: AmplifierSpec,
    core_type: str,
    core_layers: int,
    core_expansion: float,
    residual_core: bool,
    residual_core_init: float,
    branch_temporal_mode: str,
    branch_temporal_lag_scale: float,
    residual_token_gate_mode: str,
    branch_router_mode: str,
    base_bigram_delta: str,
    trigram_memory: str,
    trigram_log_scale_init: float,
    residual_readout_delta_rank: int,
    residual_readout_delta_init_std: float,
    trainable_parameters: int,
    fixed_buffer_bytes: int,
    compile_requested: bool,
    compile_after: int,
    compile_mode: Optional[str],
    compile_base_path: bool,
    scan_backend_requested: str,
    scan_backend_active: str,
    gradient_checkpointing: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    wandb_group: Optional[str],
    wandb_run_name: str,
    wandb_tags: list[str],
    wandb_watch: str,
    wandb_watch_log_freq: int,
    wandb_mode: Optional[str],
    tok_path: Optional[Path],
    tokenizer_sha256: Optional[str],
    byte_count_lut: Optional[torch.Tensor],
) -> dict[str, Any]:
    """Build the resolved run snapshot.

    :param Any cfg: Loaded config object.
    :param Path model_dir: Model directory.
    :param argparse.Namespace args: Parsed CLI arguments.
    :param str | Path data_source: Resolved data source.
    :param int train_tokens_count: Training token count.
    :param int val_tokens_count: Validation token count.
    :param str storage_dtype: Storage dtype.
    :param float train_frac: Train split fraction.
    :param bool allow_train_frac_val_split: Whether directory data may fall back
        to splitting train tokens for validation.
    :param bool allow_approx_bpb: Whether approximate bpb is explicitly allowed.
    :param str validation_source: Source of validation tokens.
    :param Optional[int] data_max_tokens: Optional token cap.
    :param int seq_len: Sequence length.
    :param int batch_size: Batch size.
    :param int grad_accum: Gradient accumulation steps.
    :param int carry_chunks: Carry chunks.
    :param int bptt_chunks: BPTT chunks.
    :param int num_steps: Training steps.
    :param float learning_rate: Learning rate.
    :param str lr_schedule: LR schedule name.
    :param float min_lr: Minimum learning rate.
    :param int warmup_steps: Warmup steps.
    :param int lr_hold_steps: Hold steps.
    :param float weight_decay: Weight decay.
    :param float hard_loss_gamma: Hard-loss gamma.
    :param float hard_loss_cap: Hard-loss cap.
    :param float grad_clip: Gradient clip value.
    :param float dropout: Dropout rate.
    :param int val_every: Validation cadence.
    :param int val_steps: Validation step count.
    :param bool full_val_final: Whether final eval covers validation once.
    :param int log_every: Log cadence.
    :param int log_state_every: State log cadence.
    :param int save_every: Save cadence.
    :param torch.device device: Runtime device.
    :param str device_type: Runtime device type.
    :param torch.dtype default_amp_dtype: Default AMP dtype.
    :param torch.dtype runtime_amp_dtype: Runtime AMP dtype.
    :param bool use_autocast: Whether autocast is enabled.
    :param AmplifierSpec spec: Loaded spec.
    :param str core_type: Controller core type.
    :param int core_layers: Controller layer count.
    :param float core_expansion: Controller expansion.
    :param bool residual_core: Whether residual core is enabled.
    :param float residual_core_init: Residual core init value.
    :param str branch_temporal_mode: Branch temporal mode.
    :param float branch_temporal_lag_scale: Branch temporal lag scale.
    :param str residual_token_gate_mode: Tokenwise residual gating mode.
    :param str branch_router_mode: Per-token branch router mode.
    :param str base_bigram_delta: Trainable base-bigram delta mode.
    :param str trigram_memory: Frozen trigram memory mode.
    :param float trigram_log_scale_init: Initial trigram boost log scale.
    :param int residual_readout_delta_rank: Low-rank trainable residual
        readout correction rank.
    :param float residual_readout_delta_init_std: Init std for the correction
        input projection.
    :param int trainable_parameters: Trainable parameter count.
    :param int fixed_buffer_bytes: Fixed buffer bytes.
    :param bool compile_requested: Whether compile was requested.
    :param int compile_after: Step to start compilation.
    :param Optional[str] compile_mode: torch.compile mode.
    :param bool compile_base_path: Whether to compile the base path.
    :param str scan_backend_requested: Requested scan backend.
    :param str scan_backend_active: Active scan backend.
    :param bool gradient_checkpointing: Whether gradient checkpointing is enabled.
    :param bool wandb_enabled: Whether W&B logging is enabled.
    :param str wandb_project: W&B project name.
    :param Optional[str] wandb_entity: W&B entity.
    :param Optional[str] wandb_group: W&B group.
    :param str wandb_run_name: W&B run name.
    :param list[str] wandb_tags: W&B tags.
    :param str wandb_watch: W&B watch mode.
    :param int wandb_watch_log_freq: W&B watch log frequency.
    :param Optional[str] wandb_mode: W&B mode.
    :param Optional[Path] tok_path: Tokenizer path.
    :param Optional[str] tokenizer_sha256: Tokenizer file SHA256.
    :param Optional[torch.Tensor] byte_count_lut: Byte-count LUT.
    :return dict[str, Any]: Resolved configuration payload.
    """
    spec_bytes, gzip_spec_bytes = spec_size_bytes(cfg.spec_path)
    local_step_tokens = int(batch_size * seq_len * bptt_chunks)
    effective_step_tokens = int(local_step_tokens * grad_accum)
    planned_train_tokens = int(effective_step_tokens * num_steps)
    run_name = cfg.meta.get("run_name", model_dir.name)
    phase = cfg.meta.get("phase")
    readout_rank = spec.metadata.get("readout_rank")
    if readout_rank in (0, None):
        readout_rank = None

    metadata = {
        "run_name": run_name,
        "phase": phase,
        "seed": int(args.seed),
        "model_dir": str(model_dir),
        "model": {
            "vocab_size": int(spec.vocab_size),
            "core_dim": int(spec.metadata.get("requested_core_dim", spec.core_dim)),
            "branch_lags": list(spec.branch_lags),
            "num_blocks": int(spec.num_blocks),
            "readout_rank": readout_rank,
            "core_type": core_type,
            "core_layers": int(core_layers),
            "core_expansion": float(core_expansion),
            "residual_core": bool(residual_core),
            "residual_core_init": float(residual_core_init),
            "branch_temporal_mode": branch_temporal_mode,
            "branch_temporal_lag_scale": float(branch_temporal_lag_scale),
            "residual_token_gate_mode": residual_token_gate_mode,
            "branch_router_mode": branch_router_mode,
            "base_bigram_delta": base_bigram_delta,
            "trigram_memory": trigram_memory,
            "trigram_log_scale_init": float(trigram_log_scale_init),
            "trigram_top_k": spec.metadata.get("trigram_top_k"),
            "trigram_residual_scale": spec.metadata.get("trigram_residual_scale"),
            "residual_readout_delta_rank": int(residual_readout_delta_rank),
            "residual_readout_delta_init_std": float(residual_readout_delta_init_std),
            "trainable_parameters": int(trainable_parameters),
            "fixed_buffer_bytes": int(fixed_buffer_bytes),
            "smoothing": spec.metadata.get("smoothing"),
            "embedding_init": spec.metadata.get("embedding_init"),
            "spectral_neighbors": spec.metadata.get("spectral_neighbors"),
            "lag_identity_base": spec.metadata.get("lag_identity_base"),
            "fixed_dtype": spec.metadata.get("fixed_dtype"),
        },
        "training": {
            "seq_len": int(seq_len),
            "batch_size": int(batch_size),
            "grad_accum": int(grad_accum),
            "local_step_tokens": local_step_tokens,
            "effective_step_tokens": effective_step_tokens,
            "carry_chunks": int(carry_chunks),
            "bptt_chunks": int(bptt_chunks),
            "num_steps": int(num_steps),
            "planned_train_tokens": planned_train_tokens,
            "learning_rate": float(learning_rate),
            "lr_schedule": lr_schedule,
            "min_lr": float(min_lr),
            "warmup_steps": int(warmup_steps),
            "lr_hold_steps": int(lr_hold_steps),
            "weight_decay": float(weight_decay),
            "hard_loss_gamma": float(hard_loss_gamma),
            "hard_loss_cap": float(hard_loss_cap),
            "grad_clip": float(grad_clip),
            "dropout": float(dropout),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "val_every": int(val_every),
            "val_steps": int(val_steps),
            "full_val_final": bool(full_val_final),
            "log_every": int(log_every),
            "log_state_every": int(log_state_every),
            "save_every": int(save_every),
        },
        "data": {
            "source": str(data_source),
            "storage_dtype": storage_dtype,
            "allow_train_frac_val_split": bool(allow_train_frac_val_split),
            "allow_approx_bpb": bool(allow_approx_bpb),
            "validation_source": validation_source,
            "validation_policy": (
                "explicit_val_shard_required"
                if not allow_train_frac_val_split
                else "train_split_fallback_allowed"
            ),
            "data_max_tokens": None if data_max_tokens is None else int(data_max_tokens),
            "train_tokens": int(train_tokens_count),
            "val_tokens": int(val_tokens_count),
            "train_token_fingerprint": spec.metadata.get("train_token_fingerprint"),
            "train_token_count": spec.metadata.get("train_token_count"),
        },
        "runtime": {
            "device": str(device),
            "device_type": device_type,
            "default_amp_dtype": dtype_name(default_amp_dtype),
            "amplifier_dtype": dtype_name(runtime_amp_dtype),
            "autocast": bool(use_autocast),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "exact_val_bpb": byte_count_lut is not None,
            "allow_approx_bpb": bool(allow_approx_bpb),
            "scan_backend_requested": scan_backend_requested,
            "scan_backend_active": scan_backend_active,
            "compile": {
                "enabled": bool(compile_requested),
                "compile_after": int(compile_after),
                "compile_mode": compile_mode or "default",
                "compile_base_path": bool(compile_base_path),
            },
            "wandb": {
                "enabled": bool(wandb_enabled),
                "project": wandb_project,
                "entity": wandb_entity,
                "group": wandb_group,
                "run_name": wandb_run_name,
                "tags": list(wandb_tags),
                "watch": wandb_watch,
                "watch_log_freq": int(wandb_watch_log_freq),
                "mode": wandb_mode,
            },
        },
        "tokenizer_path": None if tok_path is None else str(tok_path),
        "tokenizer_sha256": tokenizer_sha256,
        "spec": {
            "summary": spec.summary(),
            "spec_path": str(cfg.spec_path),
            "repo_code_bytes": int(estimate_repo_code_bytes(Path(__file__).resolve().parent)),
            "spec_bytes": spec_bytes,
            "gzip_spec_bytes": gzip_spec_bytes,
        },
        "config_source": {
            "config_path": str(cfg.config_path),
            "cli_args": vars(args),
        },
    }
    if allow_train_frac_val_split or validation_source != "explicit_val_shard":
        metadata["data"]["fallback_train_frac"] = float(train_frac)
    return metadata


def build_run_metadata(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    model_dir: Path,
    resolved_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Collect static metadata for audit and summary rebuilds.

    :param Path repo_root: Repository root.
    :param argparse.Namespace args: Parsed CLI arguments.
    :param Path model_dir: Model directory.
    :param dict[str, Any] resolved_config: Resolved run configuration.
    :param torch.device device: Runtime device.
    :return dict[str, Any]: Static run metadata.
    """
    device_idx = runtime_device_index(device)
    system: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))
        if torch.cuda.is_available()
        else None,
        "tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False))
        if torch.cuda.is_available()
        else None,
        "device_index": device_idx,
    }
    if device.type == "cuda" and device_idx is not None:
        props = torch.cuda.get_device_properties(device_idx)
        system.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_mib": int(props.total_memory / (1024 * 1024)),
                "gpu_capability": f"{props.major}.{props.minor}",
            }
        )
        system.update(nvidia_smi_metadata(device_idx))

    return {
        "run_name": resolved_config.get("run_name", model_dir.name),
        "phase": resolved_config.get("phase"),
        "git_commit": git_commit(repo_root),
        "command": command_context(Path(__file__), sys.argv[1:]),
        "system": system,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TORCH_BLAS_PREFER_CUBLASLT": os.environ.get("TORCH_BLAS_PREFER_CUBLASLT"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "WANDB_MODE": os.environ.get("WANDB_MODE"),
            "WANDB_PROJECT": os.environ.get("WANDB_PROJECT"),
        },
        "model_dir": str(model_dir),
        "started_at_epoch_sec": time.time(),
    }


def main() -> None:
    """Run core/amplifier training from the CLI."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    from core_amplifier_lm import AmplifierSpec, CoreAmplifierLM, ModelConfig
    from core_amplifier_lm.config import DEFAULTS, trigram_memory_config_value

    # --- Load config ---
    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        raise SystemExit(
            f"ERROR: {model_dir}/config.json not found. "
            f"Create with: inspect_model.py init {model_dir} --data <path>"
        )

    cfg = ModelConfig.load(model_dir)
    t = cfg.training  # training section from config
    td = DEFAULTS["training"]  # hardcoded defaults
    m = cfg.model
    md = DEFAULTS["model"]

    # --- Resolve all training params: CLI > config.json > defaults ---
    data_source = _resolve(args.data, cfg.data.get("source"), None)
    if not data_source:
        raise SystemExit("ERROR: no data source in config or --data")

    storage_dtype = _resolve(args.storage_dtype, cfg.data.get("storage_dtype"), "uint16")
    train_frac = _resolve(args.train_frac, cfg.data.get("train_frac"), 0.98)
    allow_train_frac_val_split = bool(
        _resolve(
            args.allow_train_frac_val_split,
            cfg.data.get("allow_train_frac_val_split"),
            False,
        )
    )
    allow_approx_bpb = bool(args.allow_approx_bpb)
    data_max_tokens = _resolve(args.data_max_tokens, cfg.data.get("max_tokens"), None)

    seq_len = _resolve(args.seq_len, t.get("seq_len"), td["seq_len"])
    batch_size = _resolve(args.batch_size, t.get("batch_size"), td["batch_size"])
    grad_accum = _resolve(args.grad_accum, t.get("grad_accum"), 1)
    carry_chunks = _resolve(args.carry_chunks, t.get("carry_chunks"), td["carry_chunks"])
    bptt_chunks = max(
        1, int(_resolve(args.bptt_chunks, t.get("bptt_chunks"), td.get("bptt_chunks", 1)))
    )
    num_steps = _resolve(args.num_steps, t.get("num_steps"), td["num_steps"])
    learning_rate = _resolve(args.learning_rate, t.get("learning_rate"), td["learning_rate"])
    lr_schedule = _resolve(args.lr_schedule, t.get("lr_schedule"), td["lr_schedule"])
    min_lr = _resolve(args.min_lr, t.get("min_lr"), td["min_lr"])
    warmup_steps = _resolve(args.warmup_steps, t.get("warmup_steps"), td["warmup_steps"])
    lr_hold_steps = int(
        _resolve(args.lr_hold_steps, t.get("lr_hold_steps"), td.get("lr_hold_steps", 0))
    )
    weight_decay = _resolve(args.weight_decay, t.get("weight_decay"), td["weight_decay"])
    hard_loss_gamma = _resolve(
        args.hard_loss_gamma, t.get("hard_loss_gamma"), td["hard_loss_gamma"]
    )
    hard_loss_cap = _resolve(args.hard_loss_cap, t.get("hard_loss_cap"), td["hard_loss_cap"])
    grad_clip = _resolve(args.grad_clip, t.get("grad_clip"), td["grad_clip"])
    dropout = _resolve(args.dropout, t.get("dropout"), td["dropout"])
    amplifier_dtype_str = _resolve(
        args.amplifier_dtype, t.get("amplifier_dtype"), td["amplifier_dtype"]
    )
    log_state_every = int(
        _resolve(args.log_state_every, t.get("log_state_every"), td.get("log_state_every", 0))
    )
    gradient_checkpointing = bool(
        _resolve(
            args.gradient_checkpointing,
            t.get("gradient_checkpointing"),
            td.get("gradient_checkpointing", False),
        )
    )
    val_every = _resolve(args.val_every, t.get("val_every"), 200)
    val_steps = _resolve(args.val_steps, t.get("val_steps"), 20)
    full_val_final = bool(
        _resolve(args.full_val_final, t.get("full_val_final"), td.get("full_val_final", False))
    )
    save_every = _resolve(args.save_every, t.get("save_every"), 1000)
    log_every = _resolve(args.log_every, t.get("log_every"), 20)

    # Controller architecture. core_dim/branch_lags/num_blocks are spec-bound;
    # these knobs only change the tiny trainable recurrent steering policy.
    core_type = _resolve(args.core_type, m.get("core_type"), md.get("core_type", "mingru"))
    core_layers = int(_resolve(args.core_layers, m.get("core_layers"), md.get("core_layers", 3)))
    core_expansion = float(
        _resolve(args.core_expansion, m.get("core_expansion"), md.get("core_expansion", 2.0))
    )
    residual_core = bool(
        _resolve(args.residual_core, m.get("residual_core"), md.get("residual_core", True))
    )
    residual_core_init = float(
        _resolve(
            args.residual_core_init, m.get("residual_core_init"), md.get("residual_core_init", -2.0)
        )
    )
    branch_temporal_mode = str(
        _resolve(
            args.branch_temporal_mode,
            m.get("branch_temporal_mode"),
            md.get("branch_temporal_mode", "current"),
        )
    )
    branch_temporal_lag_scale = float(
        _resolve(
            args.branch_temporal_lag_scale,
            m.get("branch_temporal_lag_scale"),
            md.get("branch_temporal_lag_scale", 1.0),
        )
    )
    residual_token_gate_mode = str(
        _resolve(
            args.residual_token_gate_mode,
            m.get("residual_token_gate_mode"),
            md.get("residual_token_gate_mode", "none"),
        )
    )
    branch_router_mode = str(
        _resolve(
            args.branch_router_mode,
            m.get("branch_router_mode"),
            md.get("branch_router_mode", "none"),
        )
    )
    base_bigram_delta = str(
        _resolve(
            args.base_bigram_delta,
            m.get("base_bigram_delta"),
            md.get("base_bigram_delta", "none"),
        )
    )
    trigram_memory = str(
        _resolve(
            args.trigram_memory,
            trigram_memory_config_value(m),
            trigram_memory_config_value(md, "none"),
        )
    )
    trigram_log_scale_init = float(
        _resolve(
            args.trigram_log_scale_init,
            m.get("trigram_log_scale_init"),
            md.get("trigram_log_scale_init", 0.0),
        )
    )
    residual_readout_delta_rank = int(
        _resolve(
            args.residual_readout_delta_rank,
            m.get("residual_readout_delta_rank"),
            md.get("residual_readout_delta_rank", 0),
        )
    )
    residual_readout_delta_init_std = float(
        _resolve(
            args.residual_readout_delta_init_std,
            m.get("residual_readout_delta_init_std"),
            md.get("residual_readout_delta_init_std", 0.02),
        )
    )
    scan_backend = str(
        _resolve(
            args.scan_backend,
            t.get("scan_backend"),
            td.get("scan_backend", "auto"),
        )
    )
    run_name = str(cfg.meta.get("run_name", model_dir.name))
    phase = cfg.meta.get("phase")
    wandb_enabled = bool(args.wandb)
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT") or "pg-core-amp"
    wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    wandb_group = args.wandb_group or phase
    wandb_run_name = args.wandb_run_name or run_name
    wandb_tags = parse_wandb_tags(args.wandb_tags or os.environ.get("WANDB_TAGS"))
    wandb_watch = args.wandb_watch or os.environ.get("WANDB_WATCH") or "gradients"
    wandb_watch_log_freq = int(
        _resolve(
            args.wandb_watch_log_freq,
            None,
            int(os.environ.get("WANDB_WATCH_LOG_FREQ", "25")),
        )
    )
    wandb_mode = os.environ.get("WANDB_MODE")

    print(f"Model dir: {model_dir}")
    print(f"  data: {data_source}")
    if data_max_tokens is not None:
        print(f"  data_max_tokens={int(data_max_tokens):,}")
    validation_policy = (
        "explicit_val_shard_required"
        if not allow_train_frac_val_split
        else "train_split_fallback_allowed"
    )
    print(f"  validation_policy={validation_policy}")
    if allow_train_frac_val_split:
        print(f"  fallback_train_frac={train_frac}")
    print(f"  allow_approx_bpb={allow_approx_bpb}")
    print(
        f"  batch_size={batch_size} seq_len={seq_len} steps={num_steps} lr={learning_rate} "
        f"warmup={warmup_steps} hold={lr_hold_steps} min_lr={min_lr}"
    )
    print(
        f"  controller: type={core_type} layers={core_layers} expansion={core_expansion} "
        f"residual_core={residual_core} branch_temporal_mode={branch_temporal_mode} "
        f"branch_temporal_lag_scale={branch_temporal_lag_scale} "
        f"residual_token_gate_mode={residual_token_gate_mode} "
        f"branch_router_mode={branch_router_mode} "
        f"base_bigram_delta={base_bigram_delta} "
        f"trigram_memory={trigram_memory} "
        f"residual_readout_delta_rank={residual_readout_delta_rank} "
        f"scan_backend={scan_backend} "
        f"bptt_chunks={bptt_chunks} carry_chunks={carry_chunks} "
        f"grad_ckpt={gradient_checkpointing}"
    )
    if wandb_enabled:
        print(
            f"  wandb: project={wandb_project} group={wandb_group or '-'} "
            f"run={wandb_run_name} watch={wandb_watch} mode={wandb_mode or 'online'}"
        )

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    device, device_type, default_amp_dtype = get_device(force=args.force_device)
    runtime_amp_dtype = resolve_runtime_amplifier_dtype(
        amplifier_dtype_str,
        device_type=device_type,
        default_amp_dtype=default_amp_dtype,
    )
    use_autocast = device_type == "cuda" and not args.no_autocast
    print(
        f"Device: {device} | autocast: {use_autocast} ({default_amp_dtype}) | amp dtype: {runtime_amp_dtype}"
    )

    # --- Load spec ---
    spec = AmplifierSpec.load(cfg.spec_path)
    assert_model_config_matches_spec(cfg, spec)
    print(spec.summary())

    # --- Load data ---
    train_tokens: Optional[torch.Tensor] = None
    val_tokens: torch.Tensor
    train_batcher: Optional[object] = None
    train_tokens_count = 0
    val_tokens_count = 0
    validation_source = "unknown"

    train_shards, val_shards = _list_train_val_shards(data_source)
    use_direct_shard_streaming = (
        not args.no_mmap
        and not args.tokens_on_device
        and bool(train_shards)
        and len(val_shards) == 1
    )

    if use_direct_shard_streaming:
        train_arrays = [_memmap_token_file(sp, storage_dtype=storage_dtype) for sp in train_shards]
        train_arrays = _truncate_array_views(train_arrays, data_max_tokens)
        validate_token_range(
            _sample_directory_tokens(train_arrays, max_tokens=1_000_000), spec.vocab_size
        )
        val_tokens_np = _memmap_token_file(val_shards[0], storage_dtype=storage_dtype)
        validate_token_range(
            np.asarray(val_tokens_np[: min(1_000_000, val_tokens_np.shape[0])]), spec.vocab_size
        )
        val_tokens = torch.from_numpy(np.asarray(val_tokens_np))
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(args.seed + 1)
        train_batcher = DirectoryRandomStreamBatcher(
            train_arrays,
            seq_len=seq_len,
            batch_size=batch_size,
            output_device=device,
            carry_chunks=carry_chunks,
            generator=cpu_generator,
        )
        total_train = sum(int(arr.shape[0]) for arr in train_arrays)
        train_tokens_count = total_train
        val_tokens_count = int(val_tokens_np.shape[0])
        validation_source = "explicit_val_shard"
        print(
            f"Direct shard streaming: {len(train_arrays)} train shard views | "
            f"train={total_train:,} | val={val_tokens_np.shape[0]:,}"
        )
    else:
        if args.no_mmap:
            train_tokens_np, val_tokens_np = load_train_val_int32(
                data_source,
                storage_dtype=storage_dtype,
                train_frac=train_frac,
                allow_train_frac_val_split=allow_train_frac_val_split,
                max_tokens=data_max_tokens,
            )
            _sample = np.array(train_tokens_np[: min(1_000_000, len(train_tokens_np))])
            validate_token_range(_sample, spec.vocab_size)
            print(f"Loaded into RAM: train={train_tokens_np.size:,} | val={val_tokens_np.size:,}")
        else:
            train_tokens_np, val_tokens_np = mmap_train_val(
                data_source,
                storage_dtype=storage_dtype,
                train_frac=train_frac,
                allow_train_frac_val_split=allow_train_frac_val_split,
                max_tokens=data_max_tokens,
                cache_dir=args.mmap_cache_dir,
                verbose=True,
            )
            _sample = np.array(train_tokens_np[: min(1_000_000, train_tokens_np.shape[0])])
            validate_token_range(_sample, spec.vocab_size)
            print(
                f"Memory-mapped: train={train_tokens_np.shape[0]:,} | val={val_tokens_np.shape[0]:,}"
            )

        train_tokens = torch.from_numpy(np.asarray(train_tokens_np))
        val_tokens = torch.from_numpy(np.asarray(val_tokens_np))
        train_tokens_count = int(train_tokens.shape[0])
        val_tokens_count = int(val_tokens.shape[0])
        if train_shards and val_shards:
            validation_source = "explicit_val_shard"
        elif train_shards and not val_shards:
            validation_source = "train_split_fallback"
        else:
            validation_source = "single_file_split"
        if args.tokens_on_device:
            if not args.no_mmap:
                print("WARNING: --tokens-on-device copies all tokens to GPU, defeating mmap")
            train_tokens = train_tokens.to(device=device, non_blocking=True)
            val_tokens = val_tokens.to(device=device, non_blocking=True)

    print(f"  validation_source={validation_source}")

    # --- Model ---
    model = CoreAmplifierLM(
        spec,
        core_layers=core_layers,
        core_type=core_type,
        dropout=dropout,
        amplifier_dtype=runtime_amp_dtype,
        core_expansion=core_expansion,
        residual_core=residual_core,
        residual_core_init=residual_core_init,
        branch_temporal_mode=branch_temporal_mode,
        branch_temporal_lag_scale=branch_temporal_lag_scale,
        residual_token_gate_mode=residual_token_gate_mode,
        branch_router_mode=branch_router_mode,
        base_bigram_delta=base_bigram_delta,
        trigram_memory=trigram_memory,
        trigram_log_scale_init=trigram_log_scale_init,
        residual_readout_delta_rank=residual_readout_delta_rank,
        residual_readout_delta_init_std=residual_readout_delta_init_std,
        scan_backend=scan_backend,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)
    model.prepare_runtime_buffers(device=device, amplifier_dtype=runtime_amp_dtype)
    core_model = model
    base_path_model: torch.nn.Module = BasePathLookup(core_model)

    optimizer = build_optimizer(
        model, learning_rate=learning_rate, weight_decay=weight_decay, device_type=device_type
    )

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        load_trainable_state(model, ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        print(f"Resumed from step {step}")

    metrics_path = model_dir / "metrics.jsonl"
    resolved_config_path = model_dir / "resolved_config.json"
    run_metadata_path = model_dir / "run_metadata.json"
    run_results_path = model_dir / "run_results.json"
    existing_rows = read_jsonl(metrics_path)
    compile_trigger_step: Optional[int] = None
    compile_duration_sec = 0.0
    compile_requested = bool(args.compile)
    compile_after = max(0, int(args.compile_after))
    compile_mode = None if args.compile_mode in (None, "default") else args.compile_mode
    compile_base_path = bool(args.compile_base_path or (compile_requested and hard_loss_gamma > 0))
    compiled_models = False

    def _compile_models(trigger_step: int) -> None:
        """Compile the model stack after the warmup trigger.

        :param int trigger_step: Step that triggered compilation.
        """
        nonlocal model, base_path_model, compiled_models, compile_duration_sec, compile_trigger_step
        if compiled_models:
            return
        msg = "Compiling model"
        if compile_mode is not None:
            msg += f" (mode={compile_mode})"
        if compile_base_path and hard_loss_gamma > 0:
            msg += " + base_path"
        print(msg + "...")
        started = time.time()
        compile_kwargs = {} if compile_mode is None else {"mode": compile_mode}
        model = torch.compile(model, **compile_kwargs)
        if compile_base_path and hard_loss_gamma > 0:
            base_path_model = torch.compile(base_path_model, **compile_kwargs)
        model.train()
        compiled_models = True
        compile_duration_sec = time.time() - started
        compile_trigger_step = int(trigger_step)

    if compile_requested and step >= compile_after:
        _compile_models(step)
    elif compile_requested:
        print(f"Compile delayed until step {compile_after}")

    print(
        f"Trainable params: {core_model.trainable_parameters:,} | "
        f"Fixed buffers: {core_model.fixed_nbytes / 1e6:.2f} MB"
    )
    print(f"Active scan backend: {core_model.active_scan_backend_name()}")

    # --- Byte count LUT for competition-exact bpb ---
    byte_count_lut: Optional[torch.Tensor] = None
    avg_bytes_per_token: Optional[float] = None
    tokenizer_sha256: Optional[str] = None
    tok_path = cfg.tokenizer_path
    if tok_path is not None:
        try:
            tokenizer_sha256 = file_sha256(tok_path)
            byte_count_lut = build_byte_count_lut(tok_path, spec.vocab_size, device)
            validate_exact_bpb_targets(byte_count_lut, val_tokens)
            avg_bytes_per_token = float(byte_count_lut.float().mean().item())
            print(
                f"Byte count LUT loaded from {tok_path.name} "
                f"(avg {avg_bytes_per_token:.2f} bytes/token, sha256={tokenizer_sha256[:12]})"
            )
        except Exception as e:
            if not allow_approx_bpb:
                raise SystemExit(
                    "exact val_bpb setup failed while building the byte-count LUT. "
                    "The maintained Core/Amplifier path does not silently fall back "
                    "to approximate bpb; fix the tokenizer path or pass "
                    "--allow-approx-bpb for a deliberate local smoke run."
                ) from e
            print(f"WARNING: could not build byte count LUT: {e} — bpb will be approximate")
    else:
        if not allow_approx_bpb:
            raise SystemExit(
                "no tokenizer found in the model directory, so exact val_bpb is unavailable. "
                "The maintained Core/Amplifier path does not silently fall back to approximate "
                "bpb; add the tokenizer or pass --allow-approx-bpb for a deliberate local smoke run."
            )
        print("WARNING: no tokenizer in model dir — bpb will be approximate")

    resolved_config = build_resolved_config_payload(
        cfg=cfg,
        model_dir=model_dir,
        args=args,
        data_source=data_source,
        train_tokens_count=train_tokens_count,
        val_tokens_count=val_tokens_count,
        storage_dtype=storage_dtype,
        train_frac=train_frac,
        allow_train_frac_val_split=allow_train_frac_val_split,
        allow_approx_bpb=allow_approx_bpb,
        validation_source=validation_source,
        data_max_tokens=data_max_tokens,
        seq_len=seq_len,
        batch_size=batch_size,
        grad_accum=grad_accum,
        carry_chunks=carry_chunks,
        bptt_chunks=bptt_chunks,
        num_steps=num_steps,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        lr_hold_steps=lr_hold_steps,
        weight_decay=weight_decay,
        hard_loss_gamma=hard_loss_gamma,
        hard_loss_cap=hard_loss_cap,
        grad_clip=grad_clip,
        dropout=dropout,
        val_every=val_every,
        val_steps=val_steps,
        full_val_final=full_val_final,
        log_every=log_every,
        log_state_every=log_state_every,
        save_every=save_every,
        device=device,
        device_type=device_type,
        default_amp_dtype=default_amp_dtype,
        runtime_amp_dtype=runtime_amp_dtype,
        use_autocast=use_autocast,
        spec=spec,
        core_type=core_type,
        core_layers=core_layers,
        core_expansion=core_expansion,
        residual_core=residual_core,
        residual_core_init=residual_core_init,
        branch_temporal_mode=branch_temporal_mode,
        branch_temporal_lag_scale=branch_temporal_lag_scale,
        residual_token_gate_mode=residual_token_gate_mode,
        branch_router_mode=branch_router_mode,
        base_bigram_delta=base_bigram_delta,
        trigram_memory=trigram_memory,
        trigram_log_scale_init=trigram_log_scale_init,
        residual_readout_delta_rank=residual_readout_delta_rank,
        residual_readout_delta_init_std=residual_readout_delta_init_std,
        trainable_parameters=core_model.trainable_parameters,
        fixed_buffer_bytes=core_model.fixed_nbytes,
        compile_requested=compile_requested,
        compile_after=compile_after,
        compile_mode=compile_mode,
        compile_base_path=compile_base_path,
        scan_backend_requested=scan_backend,
        scan_backend_active=core_model.active_scan_backend_name(),
        gradient_checkpointing=gradient_checkpointing,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_run_name=wandb_run_name,
        wandb_tags=wandb_tags,
        wandb_watch=wandb_watch,
        wandb_watch_log_freq=wandb_watch_log_freq,
        wandb_mode=wandb_mode,
        tok_path=tok_path,
        tokenizer_sha256=tokenizer_sha256,
        byte_count_lut=byte_count_lut,
    )
    write_json(resolved_config_path, resolved_config)
    run_metadata = build_run_metadata(
        repo_root=repo_root,
        args=args,
        model_dir=model_dir,
        resolved_config=resolved_config,
        device=device,
    )
    write_json(run_metadata_path, run_metadata)

    wandb_run = None
    if wandb_enabled:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(
                "W&B logging requested but wandb is not installed in the active environment"
            ) from exc

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            job_type=phase or "core_amplifier_train",
            name=wandb_run_name,
            tags=wandb_tags,
            dir=str(model_dir),
            config=build_wandb_config(
                resolved_config=resolved_config,
                run_metadata=run_metadata,
            ),
        )
    wandb_watch_attached = False

    def maybe_attach_wandb_watch() -> None:
        """Attach W&B watch once the model is ready."""
        nonlocal wandb_watch_attached
        if wandb_run is None or wandb_watch_attached or wandb_watch == "off":
            return
        if compile_requested and not compiled_models:
            return
        wandb_run.watch(core_model, log=wandb_watch, log_freq=wandb_watch_log_freq)
        wandb_watch_attached = True

    if train_batcher is None:
        assert train_tokens is not None
        gen_device = train_tokens.device if train_tokens.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(args.seed + 1)
        train_batcher = RandomStreamBatcher(
            train_tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            output_device=device,
            carry_chunks=carry_chunks,
            generator=generator,
        )
    maybe_attach_wandb_watch()

    def autocast_context() -> Any:
        """Return the training autocast context manager.

        :return Any: Training autocast context manager or ``nullcontext``.
        """
        if use_autocast and device_type != "cpu":
            return torch.autocast(device_type=device_type, dtype=default_amp_dtype)
        return nullcontext()

    if device.type == "cuda":
        reset_peak_memory(device)

    start_time = time.time()
    train_rows = [row for row in existing_rows if row.get("kind") == "train"]
    eval_rows = [row for row in existing_rows if row.get("kind") == "eval"]
    prior_processed_tokens = 0
    prior_elapsed_sec = 0.0
    if train_rows:
        prior_processed_tokens = max(int(row.get("processed_tokens", 0)) for row in train_rows)
        prior_elapsed_sec = max(float(row.get("elapsed_sec", 0.0)) for row in train_rows)
    seen_train_tokens = prior_processed_tokens
    train_state: Optional[Any] = None
    model.train()
    use_lr_schedule = lr_schedule == "cosine"

    while step < num_steps:
        if compile_requested and not compiled_models and step >= compile_after:
            _compile_models(step)
            maybe_attach_wandb_watch()

        if use_lr_schedule:
            lr = get_lr(
                step,
                max_lr=learning_rate,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                hold_steps=lr_hold_steps,
                total_steps=num_steps,
            )
            set_lr(optimizer, lr)

        step_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        log_due = step % max(1, log_every) == 0
        eval_due = val_every > 0 and step > 0 and step % val_every == 0
        collect_step_scalars = log_due or eval_due
        total_raw_loss_sum_t: Optional[torch.Tensor] = None
        total_raw_loss_sum = float("nan")
        total_tokens = 0
        loss_accum: Optional[torch.Tensor] = None
        last_stats = {"base_loss": float("nan"), "weight_max": 1.0}
        last_inputs: Optional[torch.Tensor] = None
        last_targets: Optional[torch.Tensor] = None

        for accum_idx in range(grad_accum):
            # Truncated BPTT window. The state is detached once at the start of
            # the window, then left connected across `bptt_chunks` consecutive
            # chunks. This preserves the parallel minGRU scan inside each chunk
            # but gives the controller a longer gradient horizon than one batch.
            state = train_state
            window_loss: Optional[torch.Tensor] = None

            for j in range(bptt_chunks):
                batch, reset_state = train_batcher.next_batch()
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                last_inputs = inputs
                last_targets = targets
                if state is None or reset_state:
                    state = core_model.initial_state(inputs.size(0), device=device)
                elif j == 0:
                    state = core_model.detach_state(state)

                objective_collect_stats = (
                    log_due and accum_idx == grad_accum - 1 and j == bptt_chunks - 1
                )
                with autocast_context():
                    logits, state = model(inputs, state=state, return_state=True)
                    weighted_sum, raw_sum, stats = compute_training_objective(
                        base_path_model,
                        logits,
                        inputs,
                        targets,
                        hard_loss_gamma=hard_loss_gamma,
                        hard_loss_cap=hard_loss_cap,
                        base_nll_model=core_model,
                        collect_stats=objective_collect_stats,
                    )

                if collect_step_scalars:
                    detached_raw_sum = raw_sum.detach()
                    total_raw_loss_sum_t = (
                        detached_raw_sum
                        if total_raw_loss_sum_t is None
                        else total_raw_loss_sum_t + detached_raw_sum
                    )
                total_tokens += targets.numel()
                seen_train_tokens += targets.numel()
                if objective_collect_stats:
                    last_stats = stats
                window_loss = weighted_sum if window_loss is None else window_loss + weighted_sum

            assert state is not None and window_loss is not None
            train_state = core_model.detach_state(state)
            loss_accum = window_loss if loss_accum is None else loss_accum + window_loss

        assert loss_accum is not None
        loss = loss_accum / max(1, total_tokens)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(core_model.parameters(), grad_clip)
        optimizer.step()
        step_elapsed = time.time() - step_started

        if collect_step_scalars:
            assert total_raw_loss_sum_t is not None
            total_raw_loss_sum = float(total_raw_loss_sum_t.item())

        if log_due:
            elapsed = prior_elapsed_sec + (time.time() - start_time)
            tok_per_sec = seen_train_tokens / max(elapsed, 1e-6)
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss = total_raw_loss_sum / max(1, total_tokens)
            # Approximate train bpb (exact would require per-token byte counts accumulated in loop)
            train_bpb_approx = train_loss / math.log(2)
            if avg_bytes_per_token is not None:
                train_bpb_approx /= avg_bytes_per_token
            peak_alloc_mib, peak_reserved_mib = current_peak_memory_mib(device)
            train_payload = {
                "kind": "train",
                "step": step,
                "elapsed_sec": elapsed,
                "step_wallclock_sec": step_elapsed,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_bpb_approx": train_bpb_approx,
                "base_loss": last_stats["base_loss"],
                "weight_max": last_stats["weight_max"],
                "tokens_per_sec": tok_per_sec,
                "processed_tokens": seen_train_tokens,
                "step_tokens": total_tokens,
                "peak_mem_alloc_mib": peak_alloc_mib,
                "peak_mem_reserved_mib": peak_reserved_mib,
            }
            append_jsonl(metrics_path, train_payload)
            train_rows.append(train_payload)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": train_loss,
                        "train/lr": current_lr,
                        "train/lr_scale": current_lr / max(float(learning_rate), 1e-12),
                        "train/processed_tokens": seen_train_tokens,
                        "train/tokens_per_s": tok_per_sec,
                    },
                    step=step,
                )
            print(
                f"step {step:6d} | train_bpb ~{train_bpb_approx:.4f} | train_loss {train_loss:.4f} | "
                f"base_loss {last_stats['base_loss']:.4f} | w_max {last_stats['weight_max']:.2f} | "
                f"lr {current_lr:.2e} | tok/s {tok_per_sec:,.0f}"
            )
            if log_state_every > 0 and train_state is not None and step % log_state_every == 0:
                watch_metrics = core_model.latest_watch_metrics()
                token_gate = core_model.latest_residual_token_gate()
                if token_gate is not None and last_inputs is not None and last_targets is not None:
                    with torch.no_grad():
                        base_per_token = core_model.base_nll_for_targets(last_inputs, last_targets)
                        hard_mask = base_per_token > base_per_token.mean()
                        easy_mask = ~hard_mask
                        if hard_mask.any() and easy_mask.any():
                            watch_metrics["residual_gate_hard_minus_easy"] = float(
                                token_gate[hard_mask].mean().item()
                                - token_gate[easy_mask].mean().item()
                            )
                with torch.no_grad():
                    state_norms = core_model.state_norms(train_state)
                    gate_values = core_model.residual_gate_values()
                state_s = _format_debug_vector(state_norms)
                gate_s = _format_debug_vector(gate_values)
                extra = ""
                if watch_metrics:
                    metric_s = " ".join(
                        f"{key}={float(value):.3f}" for key, value in sorted(watch_metrics.items())
                    )
                    extra = f" {metric_s}"
                print(f"           state_norms=[{state_s}] residual_gates=[{gate_s}]{extra}")
                if wandb_run is not None and watch_metrics:
                    wandb_run.log(
                        {f"watch/{key}": value for key, value in watch_metrics.items()},
                        step=step,
                    )

        if eval_due:
            eval_result = evaluate(
                model,
                val_tokens,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device,
                device_type=device_type,
                amp_dtype=default_amp_dtype,
                steps=val_steps,
                use_autocast=use_autocast,
                byte_count_lut=byte_count_lut,
                cover_once=False,
            )
            payload = {
                "kind": "eval",
                "step": step,
                "val_loss": eval_result.loss,
                "val_bpb": eval_result.bpb,
                "train_loss": total_raw_loss_sum / max(1, total_tokens),
                "elapsed_sec": prior_elapsed_sec + (time.time() - start_time),
                "lr": optimizer.param_groups[0]["lr"],
                "bptt_chunks": bptt_chunks,
                "carry_chunks": carry_chunks,
                "processed_tokens": seen_train_tokens,
            }
            payload.update(eval_payload_fields(eval_result))
            print(
                f"step {step:6d} | val_loss {eval_result.loss:.4f} | "
                f"val_bpb {eval_result.bpb:.4f} | "
                f"{format_eval_coverage(eval_result, validation_source=validation_source)}"
            )
            append_jsonl(metrics_path, payload)
            eval_rows.append(payload)
            if wandb_run is not None and step > 0:
                wandb_run.log(
                    {
                        "eval/loss": eval_result.loss,
                        "eval/bpb": eval_result.bpb,
                        "train/processed_tokens": seen_train_tokens,
                    },
                    step=step,
                )
            model.train()

        completed_steps = step + 1
        if save_every > 0 and completed_steps > 0 and completed_steps % save_every == 0:
            ckpt_path = model_dir / f"checkpoint_{completed_steps}.pt"
            torch.save(
                {
                    "step": completed_steps,
                    "model": trainable_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved {ckpt_path}")

        step += 1

    final_eval_step = max(0, step - 1)
    final_val_steps = (
        full_validation_steps(val_tokens, seq_len=seq_len, batch_size=batch_size)
        if full_val_final
        else val_steps
    )
    need_final_eval = final_val_steps > 0 and (
        not eval_rows
        or int(eval_rows[-1].get("step", -1)) != final_eval_step
        or (full_val_final and not bool(eval_rows[-1].get("eval_full_coverage", False)))
    )
    if need_final_eval:
        eval_result = evaluate(
            model,
            val_tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            device_type=device_type,
            amp_dtype=default_amp_dtype,
            steps=final_val_steps,
            use_autocast=use_autocast,
            byte_count_lut=byte_count_lut,
            cover_once=full_val_final,
        )
        payload = {
            "kind": "eval",
            "eval_name": "final",
            "step": final_eval_step,
            "val_loss": eval_result.loss,
            "val_bpb": eval_result.bpb,
            "elapsed_sec": prior_elapsed_sec + (time.time() - start_time),
            "lr": optimizer.param_groups[0]["lr"],
            "bptt_chunks": bptt_chunks,
            "carry_chunks": carry_chunks,
            "processed_tokens": seen_train_tokens,
        }
        payload.update(eval_payload_fields(eval_result))
        print(
            f"final eval @ step {final_eval_step:6d} | val_loss {eval_result.loss:.4f} | "
            f"val_bpb {eval_result.bpb:.4f} | "
            f"{format_eval_coverage(eval_result, validation_source=validation_source)}"
        )
        append_jsonl(metrics_path, payload)
        eval_rows.append(payload)
        if wandb_run is not None and final_eval_step > 0:
            wandb_run.log(
                {
                    "eval/loss": eval_result.loss,
                    "eval/bpb": eval_result.bpb,
                    "train/processed_tokens": seen_train_tokens,
                },
                step=final_eval_step,
            )
        model.train()

    final_path = model_dir / "final.pt"
    torch.save(
        {
            "step": step,
            "model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
        },
        final_path,
    )
    elapsed_sec = prior_elapsed_sec + (time.time() - start_time)
    peak_alloc_mib, peak_reserved_mib = current_peak_memory_mib(device)
    steady_state_tok_s = compute_steady_state_tokens_per_sec(
        train_rows, compile_trigger_step=compile_trigger_step
    )
    best_eval, last_eval = best_and_last_eval(eval_rows)
    spec_bytes, gzip_spec_bytes = spec_size_bytes(cfg.spec_path)
    repo_code_bytes = estimate_repo_code_bytes(repo_root)
    trainable_payload_path = model_dir / "final_trainable.int8.ptz"
    trainable_quant_stats = export_trainable_int8_zlib(
        trainable_payload_path, trainable_state_dict(model)
    )
    trainable_payload_bytes = int(trainable_quant_stats["int8_zlib_bytes"])
    artifact_bytes = artifact_estimate_bytes(
        repo_root=repo_root,
        spec_path=cfg.spec_path,
        trainable_payload_bytes=trainable_payload_bytes,
        repo_code_bytes=repo_code_bytes,
    )
    artifact_headroom = artifact_headroom_bytes(artifact_bytes)
    artifact_budget_status = artifact_status(artifact_bytes)
    run_results = {
        "completed": True,
        "final_step": int(step),
        "planned_steps": int(num_steps),
        "seen_train_tokens": int(seen_train_tokens),
        "elapsed_sec": float(elapsed_sec),
        "steady_state_tokens_per_sec": steady_state_tok_s,
        "compile_trigger_step": compile_trigger_step,
        "compile_duration_sec": float(compile_duration_sec),
        "peak_mem_alloc_mib": peak_alloc_mib,
        "peak_mem_reserved_mib": peak_reserved_mib,
        "best_step": best_eval.get("step"),
        "best_val_loss": best_eval.get("val_loss"),
        "best_val_bpb": best_eval.get("val_bpb"),
        "last_eval_step": last_eval.get("step"),
        "last_val_loss": last_eval.get("val_loss"),
        "last_val_bpb": last_eval.get("val_bpb"),
        "last_eval_tokens": last_eval.get("eval_tokens"),
        "last_eval_bytes": last_eval.get("eval_bytes"),
        "last_eval_coverage_denominator_tokens": last_eval.get("eval_coverage_denominator_tokens"),
        "last_eval_coverage_frac": last_eval.get("eval_coverage_frac"),
        "last_eval_full_coverage": last_eval.get("eval_full_coverage"),
        "last_eval_steps": last_eval.get("eval_steps"),
        "last_eval_batch_size": last_eval.get("eval_batch_size"),
        "last_eval_seq_len": last_eval.get("eval_seq_len"),
        "repo_code_bytes": int(repo_code_bytes),
        "spec_bytes": spec_bytes,
        "gzip_spec_bytes": gzip_spec_bytes,
        "trainable_baseline_tensor_bytes": int(trainable_quant_stats["baseline_tensor_bytes"]),
        "trainable_int8_payload_bytes": int(trainable_quant_stats["int8_payload_bytes"]),
        "trainable_int8_serialized_bytes": int(trainable_quant_stats["int8_serialized_bytes"]),
        "trainable_int8_zlib_bytes": trainable_payload_bytes,
        "artifact_estimate_bytes": artifact_bytes,
        "artifact_headroom_bytes": artifact_headroom,
        "artifact_status": artifact_budget_status,
        "trainable_payload_path": str(trainable_payload_path),
        "final_checkpoint": str(final_path),
        "wandb_project": wandb_project if wandb_run is not None else None,
        "wandb_group": wandb_group if wandb_run is not None else None,
        "wandb_run_name": wandb_run_name if wandb_run is not None else None,
        "wandb_run_path": normalize_wandb_run_path(wandb_run) if wandb_run is not None else None,
        "wandb_url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
    }
    write_json(run_results_path, run_results)
    if wandb_run is not None:
        summary = wandb_run.summary
        summary["system/peak_mem_alloc_mib"] = peak_alloc_mib
        summary["system/peak_mem_reserved_mib"] = peak_reserved_mib
        summary["runtime/elapsed_sec_final"] = float(elapsed_sec)
        summary["runtime/seen_train_tokens_final"] = int(seen_train_tokens)
        summary["throughput/steady_state_tokens_per_s_final"] = steady_state_tok_s
        summary["compile/trigger_step_final"] = compile_trigger_step
        summary["compile/duration_sec_final"] = float(compile_duration_sec)
        summary["eval/best_step"] = best_eval.get("step")
        summary["eval/best_loss"] = best_eval.get("val_loss")
        summary["eval/best_bpb"] = best_eval.get("val_bpb")
        summary["eval/final_step"] = last_eval.get("step")
        summary["eval/loss_final"] = last_eval.get("val_loss")
        summary["eval/bpb_final"] = last_eval.get("val_bpb")
        summary["eval/tokens_final"] = last_eval.get("eval_tokens")
        summary["eval/bytes_final"] = last_eval.get("eval_bytes")
        summary["eval/coverage_denominator_tokens_final"] = last_eval.get(
            "eval_coverage_denominator_tokens"
        )
        summary["eval/coverage_frac_final"] = last_eval.get("eval_coverage_frac")
        summary["eval/full_coverage_final"] = last_eval.get("eval_full_coverage")
        summary["artifact/code_bytes_final"] = int(repo_code_bytes)
        summary["artifact/spec_bytes_final"] = spec_bytes
        summary["artifact/gzip_spec_bytes_final"] = gzip_spec_bytes
        summary["artifact/trainable_int8_payload_bytes_final"] = int(
            trainable_quant_stats["int8_payload_bytes"]
        )
        summary["artifact/trainable_int8_serialized_bytes_final"] = int(
            trainable_quant_stats["int8_serialized_bytes"]
        )
        summary["artifact/trainable_int8_zlib_bytes_final"] = trainable_payload_bytes
        summary["artifact/estimate_bytes_final"] = artifact_bytes
        summary["artifact/headroom_bytes_final"] = artifact_headroom
        summary["artifact/status_final"] = artifact_budget_status
        summary["artifact/limit_bytes"] = ARTIFACT_LIMIT_BYTES
        wandb_run.finish()
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
