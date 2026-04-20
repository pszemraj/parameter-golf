"""Shared trainer utilities used by both baseline and hybrid entrypoints."""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Callable

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch import Tensor, nn


def resolve_glob_files(
    pattern: str, *, missing_message: str | None = None
) -> list[Path]:
    """Resolve one glob pattern into a sorted file list.

    :param str pattern: Glob pattern for token shards.
    :param str | None missing_message: Optional custom error when no files match.
    :raises FileNotFoundError: If the pattern matches no files.
    :return list[Path]: Sorted matching paths.
    """
    files = [Path(path) for path in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(missing_message or f"No files for: {pattern}")
    return files


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Build lookup tables used for tokenizer-aware byte accounting.

    :param spm.SentencePieceProcessor sp: SentencePiece tokenizer processor.
    :param int vocab_size: Model vocabulary size.
    :param torch.device device: Target device for the lookup tables.
    :return tuple[Tensor, Tensor, Tensor]: Byte, leading-space, and boundary LUTs.
    """
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens_from_files(
    files: list[Path],
    seq_len: int,
    *,
    load_data_shard: Callable[[Path], Tensor],
    missing_message: str,
) -> Tensor:
    """Load and trim validation tokens to one whole-number sequence span.

    :param list[Path] files: Ordered validation shard paths.
    :param int seq_len: Sequence length used to truncate the validation stream.
    :param Callable[[Path], Tensor] load_data_shard: Shard loader function.
    :param str missing_message: Error to raise when no files are provided.
    :raises FileNotFoundError: If `files` is empty.
    :raises ValueError: If the validation split is shorter than one sequence.
    :return Tensor: Validation tokens with one extra token for next-token targets.
    """
    if not files:
        raise FileNotFoundError(missing_message)
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def stage_eval_batch_views(
    x_cpu: Tensor,
    y_cpu: Tensor,
    device: torch.device,
    x_stage: Tensor | None,
    y_stage: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Stage one validation batch pair through reusable pinned host buffers.

    :param Tensor x_cpu: CPU input token ids shaped `(batch, seq)`.
    :param Tensor y_cpu: CPU target token ids shaped `(batch, seq)`.
    :param torch.device device: Target device for the staged batch.
    :param Tensor | None x_stage: Reusable pinned input staging buffer.
    :param Tensor | None y_stage: Reusable pinned target staging buffer.
    :return tuple[Tensor, Tensor, Tensor | None, Tensor | None]: Device batches plus updated staging buffers.
    """
    if device.type != "cuda":
        return (
            x_cpu.to(device=device, dtype=torch.int32),
            y_cpu.to(device=device, dtype=torch.int64),
            x_stage,
            y_stage,
        )
    if x_stage is None or tuple(x_stage.shape) != tuple(x_cpu.shape):
        x_stage = torch.empty_like(x_cpu, dtype=torch.int32, pin_memory=True)
        y_stage = torch.empty_like(y_cpu, dtype=torch.int64, pin_memory=True)
    x_stage.copy_(x_cpu)
    y_stage.copy_(y_cpu)
    return (
        x_stage.to(device=device, non_blocking=True),
        y_stage.to(device=device, non_blocking=True),
        x_stage,
        y_stage,
    )


def eval_val(
    *,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    train_seq_len: int,
    val_batch_size: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    use_inference_mode: bool,
) -> tuple[float, float]:
    """Run validation and return token loss plus bits per byte.

    :param nn.Module model: Model under evaluation.
    :param int rank: Distributed rank.
    :param int world_size: Distributed world size.
    :param torch.device device: CUDA device used for evaluation.
    :param int grad_accum_steps: Gradient accumulation steps.
    :param int train_seq_len: Sequence length used for validation reshaping.
    :param int val_batch_size: Global validation batch size in tokens.
    :param Tensor val_tokens: Validation token stream.
    :param Tensor base_bytes_lut: Per-token byte counts.
    :param Tensor has_leading_space_lut: Leading-space flags per token.
    :param Tensor is_boundary_token_lut: Boundary-token flags per token.
    :param bool use_inference_mode: Whether to use `torch.inference_mode`.
    :return tuple[float, float]: Validation loss and bits-per-byte.
    """
    local_batch_tokens = val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // train_seq_len
    total_seqs = (val_tokens.numel() - 1) // train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    grad_context = torch.inference_mode if use_inference_mode else torch.no_grad
    model.eval()
    x_stage: Tensor | None = None
    y_stage: Tensor | None = None
    with grad_context():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * train_seq_len
            raw_end = batch_seq_end * train_seq_len + 1
            local = val_tokens[raw_start:raw_end]
            x, y, x_stage, y_stage = stage_eval_batch_views(
                local[:-1].reshape(-1, train_seq_len),
                local[1:].reshape(-1, train_seq_len),
                device,
                x_stage,
                y_stage,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids]
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for tensor in (val_loss_sum, val_token_count, val_byte_count):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


class TokenStream:
    """Sequential shard reader that wraps around at the end of the file list."""

    def __init__(
        self, files: list[Path], *, load_data_shard: Callable[[Path], Tensor]
    ) -> None:
        """Initialize a streaming shard reader.

        :param list[Path] files: Ordered token shard paths.
        :param Callable[[Path], Tensor] load_data_shard: Shard loader function.
        """
        self.files = list(files)
        if not self.files:
            raise ValueError("TokenStream requires at least one shard file")
        self._load_data_shard = load_data_shard
        self.file_idx = 0
        self.tokens = self._load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        """Advance to the next shard, wrapping around when needed."""
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = self._load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def skip(self, n: int) -> None:
        """Advance the stream by ``n`` tokens without materializing them.

        :param int n: Number of tokens to discard.
        """
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            chunk = min(remaining, avail)
            self.pos += chunk
            remaining -= chunk

    def take(self, n: int) -> Tensor:
        """Take a contiguous token span across shard boundaries.

        :param int n: Number of tokens to read.
        :return Tensor: Requested token span.
        """
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            chunk = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + chunk])
            self.pos += chunk
            remaining -= chunk
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    """Shard loader that slices each global batch by distributed rank."""

    def __init__(
        self,
        files: list[Path],
        rank: int,
        world_size: int,
        device: torch.device,
        *,
        load_data_shard: Callable[[Path], Tensor],
    ) -> None:
        """Initialize the distributed token loader.

        :param list[Path] files: Ordered token shard paths.
        :param int rank: Distributed rank.
        :param int world_size: Distributed world size.
        :param torch.device device: Target device for batches.
        :param Callable[[Path], Tensor] load_data_shard: Shard loader function.
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(files, load_data_shard=load_data_shard)
        self._x_stage: Tensor | None = None
        self._y_stage: Tensor | None = None
        self._per_rank_span: int | None = None

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> tuple[Tensor, Tensor]:
        """Build the next local input/target batch pair.

        :param int global_tokens: Global tokens per optimizer step.
        :param int seq_len: Sequence length.
        :param int grad_accum_steps: Gradient accumulation factor.
        :return tuple[Tensor, Tensor]: Input and target batches on the target device.
        """
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        if self._per_rank_span is None:
            self.stream.skip(self.rank * per_rank_span)
            self._per_rank_span = per_rank_span
        elif self._per_rank_span != per_rank_span:
            raise RuntimeError(
                "DistributedTokenLoader requires a stable per-rank span within one run, "
                f"got {self._per_rank_span} then {per_rank_span}."
            )
        local = self.stream.take(per_rank_span)
        if self.world_size > 1:
            self.stream.skip((self.world_size - 1) * per_rank_span)
        x_cpu = local[:-1].reshape(-1, seq_len)
        y_cpu = local[1:].reshape(-1, seq_len)
        x, y, x_stage, y_stage = stage_eval_batch_views(
            x_cpu,
            y_cpu,
            self.device,
            self._x_stage,
            self._y_stage,
        )
        self._x_stage = x_stage
        self._y_stage = y_stage
        return x, y


def make_stream_sync_event(device: torch.device) -> torch.cuda.Event | None:
    """Create a reusable current-stream sync event for CUDA timing boundaries.

    :param torch.device device: Active execution device.
    :return torch.cuda.Event | None: Reusable CUDA event, or `None` for non-CUDA devices.
    """
    if device.type != "cuda":
        return None
    return torch.cuda.Event()


def wait_current_stream(event: torch.cuda.Event | None) -> None:
    """Wait for work already queued on the current CUDA stream.

    :param torch.cuda.Event | None event: Reusable CUDA event created by `make_stream_sync_event`.
    """
    if event is None:
        return
    event.record()
    event.synchronize()
