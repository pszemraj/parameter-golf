"""
ALlama Reborn: a decoder-only ALBERT-style trainer for OpenAI Parameter Golf.

Key features in this version:
- configurable cross-layer sharing (ALL, SOME, untied-ish)
- pre-norm or post-norm
- factorized embeddings with optional tying
- global RoPE cache shared across blocks
- ALBERT-style x0 / resid_mix shortcut at virtual depth
- GQA via SDPA enable_gqa=True when supported by the local PyTorch
- full fixed-validation scanning for Parameter Golf, plus sampled proxy eval
- optional q_delta / v_delta hooks and per-token loss return path for TTT-style eval work
- optional W&B logging directly from the trainer
- checkpoint save/load plus compact int8 payload export/load for local iteration
- concise size/parameter reporting for W&B and logs
"""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import math
import os
import random
import sys
import time
import uuid
import zlib
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import sentencepiece as spm  # type: ignore
except Exception:  # pragma: no cover
    spm = None

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def env_csv(name: str, default: str = "") -> list[str]:
    raw = os.environ.get(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def normalize_wandb_watch_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode in {"", "0", "false", "none", "off"}:
        return "off"
    if mode not in {"gradients", "parameters", "all"}:
        raise ValueError(
            f"Unsupported WANDB_WATCH={value!r}; expected off, gradients, parameters, or all."
        )
    return mode


def normalize_sdpa_backend(value: str) -> str:
    backend = value.strip().lower()
    aliases = {
        "default": "auto",
        "fa": "flash",
        "flash_attention": "flash",
        "mem_efficient": "efficient",
        "mem-efficient": "efficient",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"auto", "flash", "efficient", "math", "cudnn"}:
        raise ValueError(
            f"Unsupported SDPA_BACKEND={value!r}; expected auto, flash, efficient, math, or cudnn."
        )
    return backend


@dataclass(frozen=True)
class CliOverrides:
    compile_model: Optional[bool]


def parse_cli_overrides(argv: Sequence[str]) -> CliOverrides:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="Enable torch.compile() for single-process runs.",
    )
    compile_group.add_argument(
        "--no-compile",
        dest="compile_model",
        action="store_false",
        help="Disable torch.compile() even if COMPILE=1 is set.",
    )
    parser.set_defaults(compile_model=None)
    parsed = parser.parse_args(argv)
    return CliOverrides(compile_model=parsed.compile_model)


@dataclass
class Hyperparameters:
    # Data
    data_backend: str
    data_path: str
    train_files: str
    val_files: str
    enwik8_path: str
    tokenizer_path: str
    vocab_size: int
    train_split: float
    data_bytes_limit: int

    # Run
    run_id: str
    out_dir: str
    seed: int
    device: str
    dtype: str
    compile_model: bool
    load_path: str
    save_path: str
    export_int8_path: str
    eval_only: bool
    strict_load: bool

    # Training / eval
    num_epochs: int
    max_steps: int
    train_seq_len: int
    train_batch_tokens: int
    grad_accum_steps: int
    eval_mode: str
    eval_seq_len: int
    eval_batch_tokens: int
    val_batch_size: int
    val_batches: int
    val_loss_every: int
    train_log_every: int
    learning_rate: float
    min_lr: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    adam_eps: float
    grad_clip_norm: float
    max_wallclock_seconds: float
    sdpa_backend: str

    # Model
    num_layers: int
    num_shared_blocks: int
    share_pattern: str
    model_dim: int
    embed_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: float
    mlp_multiple_of: int
    norm_kind: str
    norm_layout: str
    norm_eps: float
    tie_embeddings: bool
    rope_base: float
    qk_norm: bool
    zero_init_residual: bool
    layer_scale_init: float
    attn_dropout: float
    resid_dropout: float
    use_bias: bool
    logit_softcap: float
    use_final_norm: bool
    use_x0_shortcut: bool
    resid_mix_init: float

    # Reporting / artifact estimation
    report_artifact: bool
    quant_keep_float_numel: int
    control_tensor_name_patterns: tuple[str, ...]

    # W&B
    wandb_enable: bool
    wandb_project: str
    wandb_entity: str
    wandb_group: str
    wandb_run_name: str
    wandb_tags: list[str]
    wandb_notes: str
    wandb_mode: str
    wandb_watch: str
    wandb_watch_log_freq: int

    # Local reporting
    print_model_summary: bool
    model_summary_max_depth: int
    model_summary_show_shapes: bool

    @classmethod
    def from_env(cls, cli: Optional[CliOverrides] = None) -> "Hyperparameters":
        data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
        wandb_project = os.environ.get("WANDB_PROJECT", "")
        wandb_enable = env_bool("WANDB", False) or bool(wandb_project)
        compile_model = env_bool("COMPILE", False)
        if cli is not None and cli.compile_model is not None:
            compile_model = cli.compile_model
        eval_seq_len_default = int(
            os.environ.get("EVAL_SEQ_LEN", os.environ.get("TRAIN_SEQ_LEN", "1024"))
        )
        return cls(
            data_backend=os.environ.get("DATA_BACKEND", "auto"),
            data_path=data_path,
            train_files=os.environ.get(
                "TRAIN_FILES", os.path.join(data_path, "fineweb_train_*.bin")
            ),
            val_files=os.environ.get(
                "VAL_FILES", os.path.join(data_path, "fineweb_val_*.bin")
            ),
            enwik8_path=os.environ.get("ENWIK8_PATH", "./data/enwik8.gz"),
            tokenizer_path=os.environ.get(
                "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
            ),
            vocab_size=int(os.environ.get("VOCAB_SIZE", "0")),
            train_split=float(os.environ.get("TRAIN_SPLIT", "0.9")),
            data_bytes_limit=int(float(os.environ.get("DATA_BYTES_LIMIT", "95000000"))),
            run_id=os.environ.get("RUN_ID", str(uuid.uuid4())),
            out_dir=os.environ.get("OUT_DIR", "./runs"),
            seed=int(os.environ.get("SEED", "1337")),
            device=os.environ.get("DEVICE", "auto"),
            dtype=os.environ.get("DTYPE", "auto"),
            compile_model=compile_model,
            load_path=os.environ.get("LOAD_PATH", ""),
            save_path=os.environ.get("SAVE_PATH", ""),
            export_int8_path=os.environ.get("EXPORT_INT8_PATH", ""),
            eval_only=env_bool("EVAL_ONLY", False),
            strict_load=env_bool("STRICT_LOAD", True),
            num_epochs=int(os.environ.get("NUM_EPOCHS", "1")),
            max_steps=int(
                os.environ.get("MAX_STEPS", os.environ.get("ITERATIONS", "0"))
            ),
            train_seq_len=int(os.environ.get("TRAIN_SEQ_LEN", "1024")),
            train_batch_tokens=int(os.environ.get("TRAIN_BATCH_TOKENS", "65536")),
            grad_accum_steps=int(os.environ.get("GRAD_ACCUM_STEPS", "4")),
            eval_mode=os.environ.get("EVAL_MODE", "auto"),
            eval_seq_len=eval_seq_len_default,
            eval_batch_tokens=int(os.environ.get("EVAL_BATCH_TOKENS", "0")),
            val_batch_size=int(os.environ.get("VAL_BATCH_SIZE", "0")),
            val_batches=int(os.environ.get("VAL_BATCHES", "8")),
            val_loss_every=int(os.environ.get("VAL_LOSS_EVERY", "500")),
            train_log_every=int(os.environ.get("TRAIN_LOG_EVERY", "25")),
            learning_rate=float(os.environ.get("LEARNING_RATE", "3e-4")),
            min_lr=float(os.environ.get("MIN_LR", "3e-5")),
            warmup_steps=int(os.environ.get("WARMUP_STEPS", "50")),
            weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.01")),
            beta1=float(os.environ.get("BETA1", "0.9")),
            beta2=float(os.environ.get("BETA2", "0.95")),
            adam_eps=float(os.environ.get("ADAM_EPS", "1e-8")),
            grad_clip_norm=float(os.environ.get("GRAD_CLIP_NORM", "1.0")),
            max_wallclock_seconds=float(os.environ.get("MAX_WALLCLOCK_SECONDS", "0.0")),
            sdpa_backend=normalize_sdpa_backend(os.environ.get("SDPA_BACKEND", "auto")),
            num_layers=int(os.environ.get("NUM_LAYERS", "24")),
            num_shared_blocks=int(os.environ.get("NUM_SHARED_BLOCKS", "1")),
            share_pattern=os.environ.get("SHARE_PATTERN", "chunk"),
            model_dim=int(os.environ.get("MODEL_DIM", "1024")),
            embed_dim=int(os.environ.get("EMBED_DIM", "256")),
            num_heads=int(os.environ.get("NUM_HEADS", "16")),
            num_kv_heads=int(os.environ.get("NUM_KV_HEADS", "4")),
            mlp_mult=float(os.environ.get("MLP_MULT", "2.5")),
            mlp_multiple_of=int(os.environ.get("MLP_MULTIPLE_OF", "32")),
            norm_kind=os.environ.get("NORM_KIND", "layernorm"),
            norm_layout=os.environ.get("NORM_LAYOUT", "postnorm"),
            norm_eps=float(os.environ.get("NORM_EPS", "1e-5")),
            tie_embeddings=env_bool("TIE_EMBEDDINGS", True),
            rope_base=float(os.environ.get("ROPE_BASE", "10000.0")),
            qk_norm=env_bool("QK_NORM", True),
            zero_init_residual=env_bool("ZERO_INIT_RESIDUAL", True),
            layer_scale_init=float(os.environ.get("LAYER_SCALE_INIT", "1.0")),
            attn_dropout=float(os.environ.get("ATTN_DROPOUT", "0.0")),
            resid_dropout=float(os.environ.get("RESID_DROPOUT", "0.0")),
            use_bias=env_bool("USE_BIAS", False),
            logit_softcap=float(os.environ.get("LOGIT_SOFTCAP", "30.0")),
            use_final_norm=env_bool("USE_FINAL_NORM", True),
            use_x0_shortcut=env_bool("USE_X0_SHORTCUT", True),
            resid_mix_init=float(os.environ.get("RESID_MIX_INIT", "0.1")),
            report_artifact=env_bool("REPORT_ARTIFACT", True),
            quant_keep_float_numel=int(
                os.environ.get("QUANT_KEEP_FLOAT_NUMEL", "4096")
            ),
            control_tensor_name_patterns=tuple(
                env_csv(
                    "CONTROL_TENSOR_NAME_PATTERNS",
                    "depth_gains,resid_mix_logits",
                )
            ),
            wandb_enable=wandb_enable,
            wandb_project=wandb_project or "param-golf-ablations",
            wandb_entity=os.environ.get("WANDB_ENTITY", ""),
            wandb_group=os.environ.get("WANDB_GROUP", ""),
            wandb_run_name=os.environ.get("WANDB_RUN_NAME", ""),
            wandb_tags=env_csv("WANDB_TAGS", ""),
            wandb_notes=os.environ.get("WANDB_NOTES", ""),
            wandb_mode=os.environ.get("WANDB_MODE", ""),
            wandb_watch=normalize_wandb_watch_mode(
                os.environ.get("WANDB_WATCH", "off")
            ),
            wandb_watch_log_freq=int(os.environ.get("WANDB_WATCH_LOG_FREQ", "100")),
            print_model_summary=env_bool("PRINT_MODEL_SUMMARY", True),
            model_summary_max_depth=int(os.environ.get("MODEL_SUMMARY_MAX_DEPTH", "4")),
            model_summary_show_shapes=env_bool("MODEL_SUMMARY_SHOW_SHAPES", False),
        )


# -----------------------------------------------------------------------------
# DISTRIBUTED / DEVICE
# -----------------------------------------------------------------------------


def init_distributed_and_device(args: Hyperparameters):
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.device == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            requested = "mps"
        else:
            requested = "cpu"
    else:
        requested = args.device

    if distributed:
        backend = "nccl" if requested == "cuda" else "gloo"
        if requested == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(requested)
        dist.init_process_group(backend=backend)
    else:
        if requested == "cuda":
            device = torch.device("cuda", 0)
        else:
            device = torch.device(requested)

    master_process = rank == 0
    return distributed, rank, world_size, local_rank, device, master_process


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------


def infer_backend(args: Hyperparameters) -> str:
    if args.data_backend != "auto":
        return args.data_backend
    if glob.glob(args.train_files):
        return "parameter_golf"
    if Path(args.enwik8_path).exists():
        return "enwik8"
    raise FileNotFoundError(
        "Could not infer DATA_BACKEND. No FineWeb shards matched TRAIN_FILES and no ENWIK8_PATH exists."
    )


def load_parameter_golf_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def count_parameter_golf_shard_tokens(file: Path) -> int:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    return int(header[2])


class FiniteTokenStream:
    def __init__(
        self, files: Optional[list[Path]] = None, tensor: Optional[Tensor] = None
    ):
        if (files is None) == (tensor is None):
            raise ValueError(
                "Provide exactly one of files or tensor to FiniteTokenStream"
            )
        self.files = files
        self.tensor = tensor.contiguous() if tensor is not None else None
        if self.files is not None:
            if not self.files:
                raise FileNotFoundError("No training files were provided")
            self.total_tokens = int(
                sum(count_parameter_golf_shard_tokens(file) for file in self.files)
            )
        else:
            assert self.tensor is not None
            if self.tensor.ndim != 1:
                raise ValueError("FiniteTokenStream tensor mode expects a 1D tensor")
            self.total_tokens = int(self.tensor.numel())
        self.reset()

    def reset(self) -> None:
        self.read_tokens = 0
        self.carry: Optional[Tensor] = None
        if self.files is not None:
            self.file_idx = 0
            self.current_tokens: Optional[Tensor] = None
            self.current_pos = 0
        else:
            self.file_idx = 0
            self.current_tokens = None
            self.current_pos = 0

    def _ensure_file_loaded(self) -> None:
        if self.files is None:
            return
        while self.current_tokens is None and self.file_idx < len(self.files):
            self.current_tokens = load_parameter_golf_shard(self.files[self.file_idx])
            self.current_pos = 0
            if int(self.current_tokens.numel()) == 0:
                self.file_idx += 1
                self.current_tokens = None

    def _advance_file(self) -> None:
        if self.files is None:
            return
        self.file_idx += 1
        self.current_tokens = None
        self.current_pos = 0
        self._ensure_file_loaded()

    def _read_new_exact(self, n: int) -> Tensor:
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if n == 0:
            return torch.empty((0,), dtype=torch.int64)
        if self.read_tokens + n > self.total_tokens:
            raise EOFError(
                f"Requested {n} tokens with only {self.total_tokens - self.read_tokens} unread tokens remaining"
            )
        if self.tensor is not None:
            start = self.read_tokens
            end = start + n
            self.read_tokens = end
            return self.tensor[start:end]
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            self._ensure_file_loaded()
            if self.current_tokens is None:
                raise EOFError(f"Requested {n} tokens but hit end of training files")
            available = int(self.current_tokens.numel() - self.current_pos)
            if available <= 0:
                self._advance_file()
                continue
            k = min(remaining, available)
            chunks.append(self.current_tokens[self.current_pos : self.current_pos + k])
            self.current_pos += k
            self.read_tokens += k
            remaining -= k
            if self.current_pos >= int(self.current_tokens.numel()):
                self._advance_file()
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    def remaining_unread_tokens(self) -> int:
        return max(0, self.total_tokens - self.read_tokens)

    def remaining_target_tokens(self) -> int:
        unread = self.remaining_unread_tokens()
        if self.carry is None:
            return max(0, unread - 1)
        return unread

    def take_targets_exact(self, target_tokens: int) -> Tensor:
        if target_tokens <= 0:
            raise ValueError(f"target_tokens must be positive, got {target_tokens}")
        if target_tokens > self.remaining_target_tokens():
            raise EOFError(
                f"Requested {target_tokens} targets with only {self.remaining_target_tokens()} targets remaining"
            )
        if self.carry is None:
            chunk = self._read_new_exact(target_tokens + 1)
        else:
            fresh = self._read_new_exact(target_tokens)
            chunk = torch.cat([self.carry, fresh], dim=0)
        self.carry = chunk[-1:].clone()
        return chunk


class SequentialDistributedTokenLoader:
    def __init__(
        self,
        stream: FiniteTokenStream,
        rank: int,
        world_size: int,
        device: torch.device,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = stream

    @classmethod
    def from_pattern(
        cls, pattern: str, rank: int, world_size: int, device: torch.device
    ) -> "SequentialDistributedTokenLoader":
        files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        return cls(
            FiniteTokenStream(files=files),
            rank=rank,
            world_size=world_size,
            device=device,
        )

    @classmethod
    def from_tensor(
        cls, tokens: Tensor, rank: int, world_size: int, device: torch.device
    ) -> "SequentialDistributedTokenLoader":
        return cls(
            FiniteTokenStream(tensor=tokens.long().contiguous()),
            rank=rank,
            world_size=world_size,
            device=device,
        )

    @property
    def total_tokens(self) -> int:
        return self.stream.total_tokens

    @property
    def total_target_tokens(self) -> int:
        total = max(0, self.total_tokens - 1)
        if self.world_size > 1:
            total -= total % self.world_size
        return total

    def reset(self) -> None:
        self.stream.reset()

    def remaining_target_tokens(self) -> int:
        remaining = self.stream.remaining_target_tokens()
        if self.world_size > 1:
            remaining -= remaining % self.world_size
        return max(0, remaining)

    def next_batch(self, local_batch_size: int, seq_len: int) -> tuple[Tensor, Tensor]:
        if local_batch_size <= 0 or seq_len <= 0:
            raise ValueError(
                f"local_batch_size and seq_len must be positive, got {local_batch_size}, {seq_len}"
            )
        local_tokens = local_batch_size * seq_len
        global_target_tokens = local_tokens * self.world_size
        if global_target_tokens > self.remaining_target_tokens():
            raise EOFError(
                f"Requested {global_target_tokens} global targets with only {self.remaining_target_tokens()} remaining"
            )
        chunk = self.stream.take_targets_exact(global_target_tokens)
        start = self.rank * local_tokens
        end = start + local_tokens + 1
        local = chunk[start:end].long()
        x = local[:-1].reshape(local_batch_size, seq_len)
        y = local[1:].reshape(local_batch_size, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


def load_enwik8(
    path: Path, train_split: float, byte_limit: int
) -> tuple[Tensor, Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Missing enwik8 file: {path}")
    with gzip.open(path, "rb") as f:
        raw = f.read(byte_limit)
    data = np.frombuffer(raw, dtype=np.uint8).copy()
    split = int(len(data) * train_split)
    train = torch.from_numpy(data[:split])
    val = torch.from_numpy(data[split:])
    return train, val


def load_validation_tokens(pattern: str) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return (
        torch.cat([load_parameter_golf_shard(file) for file in files], dim=0)
        .contiguous()
        .long()
    )


# -----------------------------------------------------------------------------
# BPB EVAL HELPERS
# -----------------------------------------------------------------------------


@dataclass
class BpbMetric:
    kind: str
    base_bytes_lut: Optional[Tensor] = None
    has_leading_space_lut: Optional[Tensor] = None
    is_boundary_token_lut: Optional[Tensor] = None


def build_sentencepiece_luts(
    sp: "spm.SentencePieceProcessor", vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
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


def setup_bpb_metric(
    backend: str, args: Hyperparameters, vocab_size: int, device: torch.device
) -> BpbMetric:
    if backend == "enwik8":
        return BpbMetric(kind="bytes")
    if (
        backend == "parameter_golf"
        and spm is not None
        and Path(args.tokenizer_path).exists()
    ):
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
            build_sentencepiece_luts(sp, vocab_size, device)
        )
        return BpbMetric(
            kind="sentencepiece",
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
    return BpbMetric(kind="none")


# -----------------------------------------------------------------------------
# MODEL UTILITIES
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x32 * rms
        return (y * self.weight.float()).to(dtype=x.dtype)


def make_norm(dim: int, norm_kind: str, eps: float) -> nn.Module:
    kind = norm_kind.lower()
    if kind == "rmsnorm":
        return RMSNorm(dim, eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unknown NORM_KIND={norm_kind}")


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = 0
        self._cos: Optional[Tensor] = None
        self._sin: Optional[Tensor] = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cos is None
            or self._sin is None
            or self._cached_seq_len < seq_len
            or self._cos.device != device
        ):
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
            self._cached_seq_len = seq_len
        assert self._cos is not None and self._sin is not None
        return self._cos[:, :, :seq_len, :].to(dtype=dtype), self._sin[
            :, :, :seq_len, :
        ].to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1)


def probe_sdpa_enable_gqa() -> bool:
    try:
        q = torch.randn(1, 4, 2, 8)
        k = torch.randn(1, 2, 2, 8)
        v = torch.randn(1, 2, 2, 8)
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return True
    except Exception:
        return False


SDPA_ENABLE_GQA = probe_sdpa_enable_gqa()


DeltaSpec = Any


def materialize_delta(delta: DeltaSpec, x: Tensor) -> Optional[Tensor]:
    if delta is None:
        return None
    out = delta
    if callable(out):
        try:
            out = out(x)
        except TypeError:
            out = out()
    if out is None:
        return None
    if not isinstance(out, Tensor):
        raise TypeError(
            f"Delta must be a Tensor, callable returning Tensor, or None; got {type(out)}"
        )
    return out.to(device=x.device, dtype=x.dtype)


def parse_share_pattern(pattern: str) -> tuple[str, int]:
    p = pattern.strip().lower()
    if p in {"chunk", "contiguous"}:
        return "chunk", 0
    if p in {"cycle", "round_robin", "roundrobin"}:
        return "cycle", 0
    repeat_n: Optional[int] = None
    if p.startswith("repeat_"):
        repeat_n = int(p.split("_", 1)[1])
    elif p.startswith("repeat") and len(p) > len("repeat"):
        suffix = p[len("repeat") :]
        if suffix.startswith(":"):
            suffix = suffix[1:]
        repeat_n = int(suffix)
    if repeat_n is not None:
        if repeat_n <= 0:
            raise ValueError(f"repeat_N requires N > 0, got {repeat_n}")
        return "repeat", repeat_n
    raise ValueError(f"Unknown SHARE_PATTERN={pattern}")


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        qk_norm: bool,
        attn_dropout: float,
        resid_dropout: float,
        use_bias: bool,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        if num_heads % num_kv_heads != 0:
            raise ValueError("NUM_HEADS must be divisible by NUM_KV_HEADS")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.qk_norm = qk_norm
        self.attn_dropout = attn_dropout
        self.resid_dropout = nn.Dropout(resid_dropout)

        kv_dim = num_kv_heads * self.head_dim
        self.qkv = nn.Linear(dim, dim + 2 * kv_dim, bias=use_bias)
        self.out_proj = nn.Linear(dim, dim, bias=use_bias)

    def forward(
        self,
        x: Tensor,
        rotary: Rotary,
        q_delta: DeltaSpec = None,
        v_delta: DeltaSpec = None,
    ) -> Tensor:
        bsz, seq_len, _ = x.shape
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = self.qkv(x).split(
            (self.num_heads * self.head_dim, kv_dim, kv_dim), dim=-1
        )

        q_extra = materialize_delta(q_delta, x)
        v_extra = materialize_delta(v_delta, x)
        if q_extra is not None:
            q = q + q_extra
        if v_extra is not None:
            v = v + v_extra

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        cos, sin = rotary(seq_len, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.num_kv_heads != self.num_heads and SDPA_ENABLE_GQA:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                dropout_p=self.attn_dropout if self.training else 0.0,
                enable_gqa=True,
            )
        else:
            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.out_proj(y))


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_mult: float,
        multiple_of: int,
        resid_dropout: float,
        use_bias: bool,
    ):
        super().__init__()
        hidden = int(dim * mlp_mult)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.gate_up = nn.Linear(dim, hidden * 2, bias=use_bias)
        self.down = nn.Linear(hidden, dim, bias=use_bias)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.resid_dropout(self.down(F.silu(gate) * up))


class ALlamaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        mlp_multiple_of: int,
        norm_kind: str,
        norm_layout: str,
        norm_eps: float,
        qk_norm: bool,
        attn_dropout: float,
        resid_dropout: float,
        use_bias: bool,
    ):
        super().__init__()
        self.norm_layout = norm_layout.lower()
        if self.norm_layout not in {"prenorm", "postnorm"}:
            raise ValueError(f"Unknown NORM_LAYOUT={norm_layout}")
        self.norm1 = make_norm(dim, norm_kind, norm_eps)
        self.norm2 = make_norm(dim, norm_kind, norm_eps)
        self.attn = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            use_bias=use_bias,
        )
        self.mlp = SwiGLU(
            dim=dim,
            mlp_mult=mlp_mult,
            multiple_of=mlp_multiple_of,
            resid_dropout=resid_dropout,
            use_bias=use_bias,
        )

    def forward(
        self,
        x: Tensor,
        x0: Optional[Tensor],
        rotary: Rotary,
        attn_gain: Tensor | float = 1.0,
        ffn_gain: Tensor | float = 1.0,
        resid_mix_logits: Optional[Tensor] = None,
        q_delta: DeltaSpec = None,
        v_delta: DeltaSpec = None,
    ) -> Tensor:
        if isinstance(attn_gain, Tensor):
            attn_gain = attn_gain.to(dtype=x.dtype).view(1, 1, 1)
        if isinstance(ffn_gain, Tensor):
            ffn_gain = ffn_gain.to(dtype=x.dtype).view(1, 1, 1)

        if x0 is not None and resid_mix_logits is not None:
            mix = resid_mix_logits.to(dtype=x.dtype).softmax(dim=0)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        if self.norm_layout == "prenorm":
            attn_in = self.norm1(x)
            x = x + attn_gain * self.attn(
                attn_in, rotary, q_delta=q_delta, v_delta=v_delta
            )
            x = x + ffn_gain * self.mlp(self.norm2(x))
            return x

        attn_out = self.attn(x, rotary, q_delta=q_delta, v_delta=v_delta)
        x = self.norm1(x + attn_gain * attn_out)
        x = self.norm2(x + ffn_gain * self.mlp(x))
        return x


class ALlama(nn.Module):
    def __init__(self, args: Hyperparameters, vocab_size: int):
        super().__init__()
        if args.num_layers <= 0:
            raise ValueError("NUM_LAYERS must be positive")
        if args.num_shared_blocks <= 0:
            raise ValueError("NUM_SHARED_BLOCKS must be positive")
        if args.num_shared_blocks > args.num_layers:
            raise ValueError("NUM_SHARED_BLOCKS cannot exceed NUM_LAYERS")
        if args.embed_dim <= 0:
            raise ValueError("EMBED_DIM must be positive")
        if args.model_dim % args.num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")

        self.args = args
        self.vocab_size = vocab_size
        self.model_dim = args.model_dim
        self.embed_dim = args.embed_dim
        self.num_layers = args.num_layers
        self.num_shared_blocks = args.num_shared_blocks
        self.share_pattern, self.share_repeat_n = parse_share_pattern(
            args.share_pattern
        )

        self.token_embedding = nn.Embedding(vocab_size, args.embed_dim)
        self.embed_to_model = (
            nn.Identity()
            if args.embed_dim == args.model_dim
            else nn.Linear(args.embed_dim, args.model_dim, bias=False)
        )
        self.rotary = Rotary(args.model_dim // args.num_heads, base=args.rope_base)
        self.shared_blocks = nn.ModuleList(
            [
                ALlamaBlock(
                    dim=args.model_dim,
                    num_heads=args.num_heads,
                    num_kv_heads=args.num_kv_heads,
                    mlp_mult=args.mlp_mult,
                    mlp_multiple_of=args.mlp_multiple_of,
                    norm_kind=args.norm_kind,
                    norm_layout=args.norm_layout,
                    norm_eps=args.norm_eps,
                    qk_norm=args.qk_norm,
                    attn_dropout=args.attn_dropout,
                    resid_dropout=args.resid_dropout,
                    use_bias=args.use_bias,
                )
                for _ in range(args.num_shared_blocks)
            ]
        )
        self.depth_gains = nn.Parameter(
            torch.full(
                (args.num_layers, 2), float(args.layer_scale_init), dtype=torch.float32
            )
        )
        if args.use_x0_shortcut:
            p = min(max(float(args.resid_mix_init), 1e-4), 1.0 - 1e-4)
            init = torch.tensor([math.log(1.0 - p), math.log(p)], dtype=torch.float32)
            self.resid_mix_logits = nn.Parameter(
                init[:, None]
                .repeat(1, args.model_dim)[None, :, :]
                .repeat(args.num_layers, 1, 1)
            )
        else:
            self.register_parameter("resid_mix_logits", None)
        self.final_norm = (
            make_norm(args.model_dim, args.norm_kind, args.norm_eps)
            if args.use_final_norm
            else nn.Identity()
        )

        if args.tie_embeddings:
            self.model_to_embed = (
                nn.Identity()
                if args.embed_dim == args.model_dim
                else nn.Linear(args.model_dim, args.embed_dim, bias=False)
            )
            self.lm_head = None
        else:
            self.model_to_embed = None
            if args.embed_dim == args.model_dim:
                self.lm_head = nn.Linear(args.model_dim, vocab_size, bias=False)
            else:
                self.lm_head = nn.Sequential(
                    nn.Linear(args.model_dim, args.embed_dim, bias=False),
                    nn.Linear(args.embed_dim, vocab_size, bias=False),
                )

        self._init_weights()

    def _layer_to_block(self, layer_idx: int) -> int:
        if self.share_pattern == "cycle":
            return layer_idx % self.num_shared_blocks
        if self.share_pattern == "repeat":
            return (layer_idx // self.share_repeat_n) % self.num_shared_blocks
        return (layer_idx * self.num_shared_blocks) // self.num_layers

    def layer_to_block_map(self) -> list[int]:
        return [self._layer_to_block(i) for i in range(self.num_layers)]

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.args.zero_init_residual:
            for block in self.shared_blocks:
                nn.init.zeros_(block.attn.out_proj.weight)
                if block.attn.out_proj.bias is not None:
                    nn.init.zeros_(block.attn.out_proj.bias)
                nn.init.zeros_(block.mlp.down.weight)
                if block.mlp.down.bias is not None:
                    nn.init.zeros_(block.mlp.down.bias)

        if isinstance(self.embed_to_model, nn.Linear):
            nn.init.xavier_uniform_(self.embed_to_model.weight)
        if isinstance(self.model_to_embed, nn.Linear):
            nn.init.xavier_uniform_(self.model_to_embed.weight)

    def _resolve_layer_delta(
        self, deltas: DeltaSpec, layer_idx: int, block_idx: int
    ) -> DeltaSpec:
        if deltas is None:
            return None
        if isinstance(deltas, Tensor):
            if deltas.ndim >= 1 and deltas.shape[0] == self.num_layers:
                return deltas[layer_idx]
            if deltas.ndim >= 1 and deltas.shape[0] == self.num_shared_blocks:
                return deltas[block_idx]
            return deltas
        if isinstance(deltas, (list, tuple)):
            if layer_idx < len(deltas):
                return deltas[layer_idx]
            return None
        if isinstance(deltas, Mapping):
            if layer_idx in deltas:
                return deltas[layer_idx]
            if str(layer_idx) in deltas:
                return deltas[str(layer_idx)]
            if block_idx in deltas:
                return deltas[block_idx]
            if str(block_idx) in deltas:
                return deltas[str(block_idx)]
            for key in ("layers", "per_layer"):
                if key in deltas:
                    values = deltas[key]
                    if (
                        isinstance(values, Tensor)
                        and values.ndim >= 1
                        and values.shape[0] == self.num_layers
                    ):
                        return values[layer_idx]
                    if isinstance(values, Sequence) and layer_idx < len(values):
                        return values[layer_idx]
            for key in ("blocks", "per_block"):
                if key in deltas:
                    values = deltas[key]
                    if (
                        isinstance(values, Tensor)
                        and values.ndim >= 1
                        and values.shape[0] == self.num_shared_blocks
                    ):
                        return values[block_idx]
                    if isinstance(values, Sequence) and block_idx < len(values):
                        return values[block_idx]
            return None
        return deltas

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Optional[Tensor] = None,
        *,
        q_deltas: DeltaSpec = None,
        v_deltas: DeltaSpec = None,
        loss_reduction: str = "mean",
    ) -> Tensor:
        x = self.token_embedding(input_ids)
        x = self.embed_to_model(x)
        x0 = x if self.args.use_x0_shortcut else None

        for layer_idx in range(self.num_layers):
            block_idx = self._layer_to_block(layer_idx)
            block = self.shared_blocks[block_idx]
            gains = self.depth_gains[layer_idx]
            resid_mix_logits = (
                self.resid_mix_logits[layer_idx]
                if self.resid_mix_logits is not None
                else None
            )
            q_delta = self._resolve_layer_delta(q_deltas, layer_idx, block_idx)
            v_delta = self._resolve_layer_delta(v_deltas, layer_idx, block_idx)
            x = block(
                x,
                x0=x0,
                rotary=self.rotary,
                attn_gain=gains[0],
                ffn_gain=gains[1],
                resid_mix_logits=resid_mix_logits,
                q_delta=q_delta,
                v_delta=v_delta,
            )

        x = self.final_norm(x)

        if self.args.tie_embeddings:
            assert self.model_to_embed is not None
            x_for_logits = self.model_to_embed(x)
            logits = F.linear(x_for_logits, self.token_embedding.weight)
        else:
            assert self.lm_head is not None
            logits = self.lm_head(x)

        if self.args.logit_softcap > 0.0:
            logits = self.args.logit_softcap * torch.tanh(
                logits / self.args.logit_softcap
            )

        if target_ids is None:
            return logits

        if loss_reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported loss_reduction={loss_reduction}")

        loss_unreduced = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="none",
        ).view_as(target_ids)
        if loss_reduction == "none":
            return loss_unreduced
        if loss_reduction == "sum":
            return loss_unreduced.sum()
        return loss_unreduced.mean()


# -----------------------------------------------------------------------------
# TRAINING UTILITIES
# -----------------------------------------------------------------------------


@dataclass
class _LayerSummary:
    """Summary statistics for a single layer in the model."""

    name: str
    param_shape: Optional[torch.Size]
    inclusive_total_params: int
    inclusive_trainable_params: int


def _unique_param_info(model: nn.Module) -> Dict[int, Tuple[int, bool]]:
    info: Dict[int, Tuple[int, bool]] = {}
    for p in model.parameters(recurse=True):
        pid = id(p)
        if pid not in info:
            info[pid] = (int(p.numel()), bool(p.requires_grad))
    return info


def _format_number(num: int) -> str:
    return f"{num:,}" if num > 0 else "--"


def _format_shape(shape: Optional[torch.Size]) -> str:
    return "x".join(map(str, shape)) if shape else "N/A"


def count_parameters(model: nn.Module) -> int:
    return sum(n for (n, _rg) in _unique_param_info(model).values())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(n for (n, rg) in _unique_param_info(model).values() if rg)


def module_parameter_counts(module: nn.Module) -> tuple[int, int]:
    info = _unique_param_info(module)
    total = sum(n for (n, _rg) in info.values())
    trainable = sum(n for (n, rg) in info.values() if rg)
    return total, trainable


def compute_logical_parameter_counts(model: nn.Module) -> dict[str, float | int]:
    unique_total = count_parameters(model)
    unique_trainable = count_trainable_parameters(model)
    if not isinstance(model, ALlama):
        return {
            "stored_params": unique_total,
            "stored_trainable_params": unique_trainable,
            "functional_params": unique_total,
            "functional_trainable_params": unique_trainable,
            "shared_block_stored_params": 0,
            "shared_block_stored_trainable_params": 0,
            "shared_block_functional_params": 0,
            "shared_block_functional_trainable_params": 0,
            "nonshared_stored_params": unique_total,
            "nonshared_stored_trainable_params": unique_trainable,
            "sharing_ratio": 1.0,
        }

    block_totals: list[int] = []
    block_trainables: list[int] = []
    for block in model.shared_blocks:
        total, trainable = module_parameter_counts(block)
        block_totals.append(total)
        block_trainables.append(trainable)

    shared_unique = int(sum(block_totals))
    shared_unique_trainable = int(sum(block_trainables))
    nonshared_unique = int(unique_total - shared_unique)
    nonshared_unique_trainable = int(unique_trainable - shared_unique_trainable)
    layer_map = model.layer_to_block_map()
    shared_logical = int(sum(block_totals[idx] for idx in layer_map))
    shared_logical_trainable = int(sum(block_trainables[idx] for idx in layer_map))
    logical_total = int(nonshared_unique + shared_logical)
    logical_trainable = int(nonshared_unique_trainable + shared_logical_trainable)
    sharing_ratio = float(logical_total) / float(max(1, unique_total))

    return {
        "stored_params": unique_total,
        "stored_trainable_params": unique_trainable,
        "functional_params": logical_total,
        "functional_trainable_params": logical_trainable,
        "shared_block_stored_params": shared_unique,
        "shared_block_stored_trainable_params": shared_unique_trainable,
        "shared_block_functional_params": shared_logical,
        "shared_block_functional_trainable_params": shared_logical_trainable,
        "nonshared_stored_params": nonshared_unique,
        "nonshared_stored_trainable_params": nonshared_unique_trainable,
        "sharing_ratio": sharing_ratio,
    }


def stored_parameter_bytes(model: nn.Module) -> int:
    seen: Set[int] = set()
    total_bytes = 0
    for p in model.parameters(recurse=True):
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        total_bytes += int(p.numel()) * int(p.element_size())
    return total_bytes


def state_dict_bytes(state_dict: Mapping[str, Tensor]) -> int:
    return int(sum(int(t.numel()) * int(t.element_size()) for t in state_dict.values()))


def serialized_nbytes(obj: object) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return len(buf.getvalue())


def cast_state_dict_floats(
    state_dict: Mapping[str, Tensor], dtype: torch.dtype
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        out[name] = t.to(dtype) if t.is_floating_point() else t
    return out


def payload_tensor_numel(payload: Mapping[str, object]) -> dict[str, int]:
    quantized = payload.get("quantized", {})
    scales = payload.get("scales", {})
    passthrough = payload.get("passthrough", {})
    assert isinstance(quantized, Mapping)
    assert isinstance(scales, Mapping)
    assert isinstance(passthrough, Mapping)
    quantized_numel = int(
        sum(int(t.numel()) for t in quantized.values() if isinstance(t, Tensor))
    )
    scale_numel = int(
        sum(int(t.numel()) for t in scales.values() if isinstance(t, Tensor))
    )
    passthrough_numel = int(
        sum(int(t.numel()) for t in passthrough.values() if isinstance(t, Tensor))
    )
    return {
        "int8_payload_quantized_numel": quantized_numel,
        "int8_payload_scale_numel": scale_numel,
        "int8_payload_passthrough_numel": passthrough_numel,
        "int8_payload_total_tensor_numel": quantized_numel
        + scale_numel
        + passthrough_numel,
    }


def model_footprint_report(
    model: nn.Module, code_path: Path, args: Hyperparameters
) -> dict[str, float | int]:
    state = {
        name: tensor.detach().to("cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    param_report = compute_logical_parameter_counts(model)
    code_bytes = int(code_path.stat().st_size)
    checkpoint = {"config": asdict(args), "model": state}
    payload = build_int8_payload(
        state,
        keep_float_numel=args.quant_keep_float_numel,
        control_patterns=args.control_tensor_name_patterns,
    )
    payload_numel = payload_tensor_numel(payload)
    int8_payload_bytes = serialized_nbytes(payload)
    int8_payload_zlib_bytes = serialized_zlib_nbytes(payload)
    report: dict[str, float | int] = {
        **param_report,
        **payload_numel,
        "stored_parameter_bytes": stored_parameter_bytes(model),
        "state_dict_bytes": state_dict_bytes(state),
        "checkpoint_bytes": serialized_nbytes(checkpoint),
        "checkpoint_zlib_bytes": serialized_zlib_nbytes(checkpoint),
        "int8_payload_bytes": int8_payload_bytes,
        "int8_payload_zlib_bytes": int8_payload_zlib_bytes,
        "code_bytes": code_bytes,
        "artifact_bytes": code_bytes + int8_payload_zlib_bytes,
    }
    return report


def render_model_summary(
    model: nn.Module, max_depth: int = 4, show_param_shapes: bool = False
) -> str:
    param_info = _unique_param_info(model)
    if max_depth <= 0:
        total_params = sum(n for (n, _rg) in param_info.values())
        trainable_params = sum(n for (n, rg) in param_info.values() if rg)
        lines = [
            "=" * 50,
            f"Total params: {_format_number(total_params)}",
            f"Trainable params: {_format_number(trainable_params)}",
            f"Non-trainable params: {_format_number(total_params - trainable_params)}",
            "=" * 50,
        ]
        return "\n".join(lines)

    summary_list: List[_LayerSummary] = []

    def summarize_recursive(module: nn.Module, depth: int, prefix: str) -> Set[int]:
        if depth > max_depth:
            return {id(p) for p in module.parameters(recurse=True)}
        direct_ids: Set[int] = {id(p) for p in module.parameters(recurse=False)}
        child_ids: Set[int] = set()
        for child in module.children():
            child_ids |= summarize_recursive(child, depth + 1, prefix + "  ")
        all_ids = direct_ids | child_ids
        total = sum(param_info[i][0] for i in all_ids)
        trainable = sum(param_info[i][0] for i in all_ids if param_info[i][1])
        param_shape = next(
            (p.shape for p in module.parameters(recurse=False) if p.requires_grad), None
        )
        summary_list.append(
            _LayerSummary(
                name=f"{prefix}{type(module).__name__}",
                param_shape=param_shape,
                inclusive_total_params=total,
                inclusive_trainable_params=trainable,
            )
        )
        return all_ids

    summarize_recursive(model, 1, "")
    total_params = sum(n for (n, _rg) in param_info.values())
    trainable_params = sum(n for (n, rg) in param_info.values() if rg)

    name_col_width = max(len("Layer (type)"), max(len(s.name) for s in summary_list))
    shape_col_width = 0
    if show_param_shapes:
        shape_col_width = max(
            len("Param Shape"),
            max(len(_format_shape(s.param_shape)) for s in summary_list),
        )
    params_col_width = 12
    trainable_col_width = 10
    col_spacing = "  "

    header_parts = [f"{'Layer (type)':<{name_col_width}}"]
    if show_param_shapes:
        header_parts.append(f"{'Param Shape':>{shape_col_width}}")
    header_parts.append(f"{'Param #':>{params_col_width}}")
    header_parts.append(f"{'Trainable':>{trainable_col_width}}")
    header = col_spacing.join(header_parts)
    sep = "=" * len(header)

    lines = [sep, header, sep]
    for entry in summary_list:
        parts = [f"{entry.name:<{name_col_width}}"]
        if show_param_shapes:
            parts.append(f"{_format_shape(entry.param_shape):>{shape_col_width}}")
        parts.append(
            f"{_format_number(entry.inclusive_total_params):>{params_col_width}}"
        )
        parts.append(
            f"{str(entry.inclusive_trainable_params > 0):>{trainable_col_width}}"
        )
        lines.append(col_spacing.join(parts))
    lines.extend(
        [
            sep,
            f"Total params: {_format_number(total_params)}",
            f"Trainable params: {_format_number(trainable_params)}",
            f"Non-trainable params: {_format_number(total_params - trainable_params)}",
            sep,
        ]
    )
    return "\n".join(lines)


def model_summary(
    model: nn.Module, max_depth: int = 4, show_param_shapes: bool = False
) -> None:
    print(
        render_model_summary(
            model, max_depth=max_depth, show_param_shapes=show_param_shapes
        ),
        flush=True,
    )


def get_local_batch_size(args: Hyperparameters, world_size: int) -> int:
    denom = args.train_seq_len * args.grad_accum_steps * world_size
    if args.train_batch_tokens % denom != 0:
        raise ValueError(
            f"TRAIN_BATCH_TOKENS={args.train_batch_tokens} must be divisible by "
            f"TRAIN_SEQ_LEN*GRAD_ACCUM_STEPS*WORLD_SIZE={denom}"
        )
    batch_size = args.train_batch_tokens // denom
    if batch_size <= 0:
        raise ValueError(
            "Local batch size computed as zero; increase TRAIN_BATCH_TOKENS"
        )
    return batch_size


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(
    args: Hyperparameters, device: torch.device
) -> tuple[str, torch.dtype, bool]:
    if args.dtype != "auto":
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        key = args.dtype.lower()
        if key not in mapping:
            raise ValueError(
                f"Unknown DTYPE={args.dtype}. Use bf16/bfloat16 or fp32/float32."
            )
        dtype = mapping[key]
    elif device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    use_autocast = device.type == "cuda" and dtype == torch.bfloat16
    return device.type, dtype, use_autocast


def configure_cuda_precision() -> None:
    if hasattr(torch.backends, "fp32_precision"):
        # PyTorch 2.10+ precision hierarchy: keep IEEE globally, enable TF32 where
        # it is the standard low-risk training fast path.
        torch.backends.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def configure_sdpa_backend(sdpa_backend: str) -> dict[str, bool | str]:
    if sdpa_backend == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    elif sdpa_backend == "efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
    elif sdpa_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    elif sdpa_backend == "cudnn":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(True)
    return {
        "requested": sdpa_backend,
        "flash_available": bool(torch.backends.cuda.is_flash_attention_available()),
        "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "cudnn_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
    }


def autocast_context(device_type: str, amp_dtype: torch.dtype, use_autocast: bool):
    if use_autocast:
        return torch.autocast(device_type=device_type, dtype=amp_dtype)
    return nullcontext()


@dataclass(frozen=True)
class MicroBatchSpec:
    local_batch_size: int
    seq_len: int
    global_target_tokens: int


def pick_microbatch_spec(
    max_global_target_tokens: int,
    max_local_batch_size: int,
    max_seq_len: int,
    world_size: int,
) -> Optional[MicroBatchSpec]:
    best: Optional[MicroBatchSpec] = None
    for local_batch_size in range(1, max_local_batch_size + 1):
        denom = local_batch_size * world_size
        if denom > max_global_target_tokens:
            break
        seq_len = min(max_seq_len, max_global_target_tokens // denom)
        if seq_len < 1:
            continue
        global_target_tokens = denom * seq_len
        if best is None:
            best = MicroBatchSpec(local_batch_size, seq_len, global_target_tokens)
            continue
        if global_target_tokens > best.global_target_tokens:
            best = MicroBatchSpec(local_batch_size, seq_len, global_target_tokens)
            continue
        if global_target_tokens == best.global_target_tokens and seq_len > best.seq_len:
            best = MicroBatchSpec(local_batch_size, seq_len, global_target_tokens)
    return best


def plan_epoch_steps(
    total_tokens: int,
    train_batch_tokens: int,
    max_local_batch_size: int,
    max_seq_len: int,
    world_size: int,
) -> tuple[list[list[MicroBatchSpec]], dict[str, int]]:
    total_target_tokens = max(0, total_tokens - 1)
    effective_target_tokens = total_target_tokens - (
        total_target_tokens % max(1, world_size)
    )
    max_micro_global_tokens = max_local_batch_size * max_seq_len * world_size
    if effective_target_tokens <= 0:
        return [], {
            "total_tokens": total_tokens,
            "total_target_tokens": total_target_tokens,
            "usable_target_tokens": effective_target_tokens,
            "planned_target_tokens": 0,
            "dropped_target_tokens": total_target_tokens,
            "steps_per_epoch": 0,
            "microbatches_per_epoch": 0,
        }
    steps: list[list[MicroBatchSpec]] = []
    remaining_targets = effective_target_tokens
    planned_target_tokens = 0
    microbatch_count = 0
    while remaining_targets > 0:
        step_budget = min(train_batch_tokens, remaining_targets)
        step_specs: list[MicroBatchSpec] = []
        while step_budget > 0 and remaining_targets > 0:
            allowed = min(step_budget, remaining_targets, max_micro_global_tokens)
            spec = pick_microbatch_spec(
                max_global_target_tokens=allowed,
                max_local_batch_size=max_local_batch_size,
                max_seq_len=max_seq_len,
                world_size=world_size,
            )
            if spec is None:
                break
            step_specs.append(spec)
            step_budget -= spec.global_target_tokens
            remaining_targets -= spec.global_target_tokens
            planned_target_tokens += spec.global_target_tokens
            microbatch_count += 1
        if not step_specs:
            break
        steps.append(step_specs)
    dropped_target_tokens = total_target_tokens - planned_target_tokens
    info = {
        "total_tokens": total_tokens,
        "total_target_tokens": total_target_tokens,
        "usable_target_tokens": effective_target_tokens,
        "planned_target_tokens": planned_target_tokens,
        "dropped_target_tokens": dropped_target_tokens,
        "steps_per_epoch": len(steps),
        "microbatches_per_epoch": microbatch_count,
    }
    return steps, info


def get_lr(step: int, total_steps: int, args: Hyperparameters) -> float:
    if step < args.warmup_steps:
        return args.learning_rate * float(step + 1) / float(max(1, args.warmup_steps))
    if total_steps <= args.warmup_steps:
        return args.min_lr
    progress = float(step - args.warmup_steps) / float(
        max(1, total_steps - args.warmup_steps)
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + (args.learning_rate - args.min_lr) * cosine


def resolve_eval_mode(args: Hyperparameters, backend: str) -> str:
    mode = args.eval_mode.strip().lower()
    if mode == "auto":
        return "full" if backend == "parameter_golf" else "sampled"
    if mode not in {"full", "sampled"}:
        raise ValueError(f"Unknown EVAL_MODE={args.eval_mode}")
    return mode


def make_eval_starts(num_tokens: int, seq_len: int, total_sequences: int) -> Tensor:
    max_start = num_tokens - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Validation tokens too short for seq_len={seq_len}")
    if total_sequences <= 1:
        return torch.tensor([0], dtype=torch.long)
    return torch.linspace(0, max_start, steps=total_sequences).long()


@torch.no_grad()
def evaluate_sampled(
    model: nn.Module,
    val_tokens: Tensor,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    device_type: str,
    amp_dtype: torch.dtype,
    use_autocast: bool,
    metric: BpbMetric,
) -> tuple[float, Optional[float]]:
    was_training = model.training
    model.eval()

    total_sequences = max(1, batch_size * num_batches)
    starts = make_eval_starts(int(val_tokens.numel()), seq_len, total_sequences)
    offsets = torch.arange(seq_len + 1)

    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0

    for batch_idx in range(num_batches):
        batch_starts = starts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        if batch_starts.numel() == 0:
            break
        windows = val_tokens[batch_starts[:, None] + offsets[None, :]]
        x = windows[:, :-1].long().to(device, non_blocking=True)
        y = windows[:, 1:].long().to(device, non_blocking=True)

        with autocast_context(device_type, amp_dtype, use_autocast):
            loss_sum = model(x, y, loss_reduction="sum")

        total_loss += float(loss_sum.item())
        total_tokens += int(y.numel())

        if metric.kind == "bytes":
            total_bytes += float(y.numel())
        elif metric.kind == "sentencepiece":
            assert metric.base_bytes_lut is not None
            assert metric.has_leading_space_lut is not None
            assert metric.is_boundary_token_lut is not None
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = metric.base_bytes_lut[tgt_ids].to(torch.float32)
            token_bytes += (
                metric.has_leading_space_lut[tgt_ids]
                & ~metric.is_boundary_token_lut[prev_ids]
            ).to(torch.float32)
            total_bytes += float(token_bytes.sum().item())

    avg_loss = total_loss / float(max(1, total_tokens))
    val_bpb = (
        (avg_loss / math.log(2.0)) * (float(total_tokens) / total_bytes)
        if total_bytes > 0.0
        else None
    )

    if was_training:
        model.train()
    return avg_loss, val_bpb


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    val_tokens: Tensor,
    seq_len: int,
    eval_batch_tokens: int,
    rank: int,
    world_size: int,
    distributed: bool,
    device: torch.device,
    device_type: str,
    amp_dtype: torch.dtype,
    use_autocast: bool,
    metric: BpbMetric,
) -> tuple[float, Optional[float]]:
    was_training = model.training
    model.eval()

    usable = ((int(val_tokens.numel()) - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    tokens = val_tokens[: usable + 1]
    total_seqs = usable // seq_len

    local_batch_tokens = eval_batch_tokens // max(1, world_size)
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"EVAL_BATCH_TOKENS={eval_batch_tokens} is too small for WORLD_SIZE={world_size} and EVAL_SEQ_LEN={seq_len}"
        )
    local_batch_seqs = max(1, local_batch_tokens // seq_len)

    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
        batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
        raw_start = batch_seq_start * seq_len
        raw_end = batch_seq_end * seq_len + 1
        local = tokens[raw_start:raw_end].to(
            device=device, dtype=torch.int64, non_blocking=True
        )
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)

        with autocast_context(device_type, amp_dtype, use_autocast):
            batch_loss_sum = model(x, y, loss_reduction="sum")

        batch_token_count = float(y.numel())
        val_loss_sum += batch_loss_sum.to(torch.float64)
        val_token_count += batch_token_count

        if metric.kind == "bytes":
            val_byte_count += batch_token_count
        elif metric.kind == "sentencepiece":
            assert metric.base_bytes_lut is not None
            assert metric.has_leading_space_lut is not None
            assert metric.is_boundary_token_lut is not None
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = metric.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                metric.has_leading_space_lut[tgt_ids]
                & ~metric.is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if distributed:
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    avg_loss = float((val_loss_sum / val_token_count).item())
    if float(val_byte_count.item()) > 0.0:
        bits_per_token = avg_loss / math.log(2.0)
        tokens_per_byte = float(val_token_count.item() / val_byte_count.item())
        val_bpb = bits_per_token * tokens_per_byte
    else:
        val_bpb = None

    if was_training:
        model.train()
    return avg_loss, val_bpb


# -----------------------------------------------------------------------------
# ARTIFACT SIZE ESTIMATION / EXPORT
# -----------------------------------------------------------------------------


BF16_STORE_DTYPE = torch.bfloat16
FLOAT32_CONTROL_STORE_DTYPE = torch.float32
INT8_PER_ROW_SCALE_DTYPE = torch.bfloat16


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float().contiguous()
    if t32.ndim >= 2:
        flat = t32.view(t32.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp_min(1.0 / 127.0) / 127.0
        q = torch.clamp(torch.round(flat / scale[:, None]), -127, 127).to(torch.int8)
        return q.view_as(t32).contiguous(), scale.to(
            INT8_PER_ROW_SCALE_DTYPE
        ).contiguous()
    scale = torch.tensor(
        max(float(t32.abs().max().item()), 1.0 / 127.0) / 127.0,
        dtype=torch.float32,
    )
    q = torch.clamp(torch.round(t32 / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale.to(INT8_PER_ROW_SCALE_DTYPE)


def should_keep_float(
    name: str, t: Tensor, keep_float_numel: int, control_patterns: Sequence[str]
) -> bool:
    if not t.is_floating_point():
        return True
    if t.numel() <= keep_float_numel:
        return True
    return any(pattern in name for pattern in control_patterns)


def build_int8_payload(
    state_dict: dict[str, Tensor],
    keep_float_numel: int = 4096,
    control_patterns: Sequence[str] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format": "allama_reborn_int8_v1",
        "quantized": {},
        "scales": {},
        "dtypes": {},
        "passthrough": {},
    }
    quantized = payload["quantized"]
    scales = payload["scales"]
    dtypes = payload["dtypes"]
    passthrough = payload["passthrough"]

    assert isinstance(quantized, dict)
    assert isinstance(scales, dict)
    assert isinstance(dtypes, dict)
    assert isinstance(passthrough, dict)

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if should_keep_float(name, t, keep_float_numel, control_patterns):
            store_dtype = (
                FLOAT32_CONTROL_STORE_DTYPE
                if any(pattern in name for pattern in control_patterns)
                else BF16_STORE_DTYPE
            )
            passthrough[name] = t.to(store_dtype) if t.is_floating_point() else t
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            continue
        q, s = quantize_float_tensor(t)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
    return payload


def dequantize_payload(payload: Mapping[str, object]) -> dict[str, Tensor]:
    if payload.get("format") != "allama_reborn_int8_v1":
        raise ValueError(f"Unsupported payload format: {payload.get('format')}")
    state: dict[str, Tensor] = {}
    quantized = payload.get("quantized", {})
    scales = payload.get("scales", {})
    dtypes = payload.get("dtypes", {})
    passthrough = payload.get("passthrough", {})
    assert isinstance(quantized, Mapping)
    assert isinstance(scales, Mapping)
    assert isinstance(dtypes, Mapping)
    assert isinstance(passthrough, Mapping)

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }

    for name, tensor in passthrough.items():
        assert isinstance(name, str)
        assert isinstance(tensor, Tensor)
        target_dtype = dtype_map.get(
            str(dtypes.get(name, str(tensor.dtype).removeprefix("torch."))),
            tensor.dtype,
        )
        state[name] = tensor.to(target_dtype)

    for name, q in quantized.items():
        assert isinstance(name, str)
        assert isinstance(q, Tensor)
        s = scales[name]
        assert isinstance(s, Tensor)
        qf = q.float()
        if s.ndim == 0:
            t = qf * s.float()
        else:
            view_shape = [s.shape[0]] + [1] * (qf.ndim - 1)
            t = qf * s.float().view(*view_shape)
        target_dtype = dtype_map.get(str(dtypes.get(name, "float32")), torch.float32)
        state[name] = t.to(target_dtype)
    return state


def serialized_zlib_nbytes(obj: object) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return len(zlib.compress(buf.getvalue(), level=9))


def artifact_report(
    model: nn.Module, code_path: Path, args: Hyperparameters
) -> dict[str, int]:
    report = model_footprint_report(model, code_path, args)
    return {
        "code_bytes": int(report["code_bytes"]),
        "int8_payload_zlib_bytes": int(report["int8_payload_zlib_bytes"]),
        "artifact_bytes": int(report["artifact_bytes"]),
    }


# -----------------------------------------------------------------------------
# CHECKPOINT IO / LOGGING
# -----------------------------------------------------------------------------


class WandbLogger:
    def __init__(self, args: Hyperparameters, enabled: bool, config: dict[str, Any]):
        self.run = None
        if not enabled:
            return
        if wandb is None:
            print(
                "W&B requested but wandb is not installed. Continuing without W&B.",
                flush=True,
            )
            return
        init_kwargs: dict[str, Any] = {
            "project": args.wandb_project,
            "config": config,
            "name": args.wandb_run_name or args.run_id,
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_group:
            init_kwargs["group"] = args.wandb_group
        if args.wandb_tags:
            init_kwargs["tags"] = args.wandb_tags
        if args.wandb_notes:
            init_kwargs["notes"] = args.wandb_notes
        if args.wandb_mode:
            init_kwargs["mode"] = args.wandb_mode
        self.run = wandb.init(**init_kwargs)

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if self.run is None:
            return
        self.run.log(metrics, step=step)

    def watch(
        self,
        model: nn.Module,
        mode: str,
        log_freq: int,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self.run is None or mode == "off":
            return
        try:
            self.run.watch(model, log=mode, log_freq=max(1, log_freq))
        except Exception as exc:
            if log_fn is not None:
                log_fn(
                    f"wandb_watch_warning mode={mode} log_freq={log_freq} error={exc}"
                )
            return
        if log_fn is not None:
            log_fn(f"wandb_watch mode={mode} log_freq={max(1, log_freq)}")

    def update_config(self, values: dict[str, Any]) -> None:
        if self.run is None:
            return
        self.run.config.update(values, allow_val_change=True)

    def update_summary(self, values: dict[str, Any]) -> None:
        if self.run is None:
            return
        for key, value in values.items():
            self.run.summary[key] = value

    def finish(self) -> None:
        if self.run is not None:
            wandb.finish()


def extract_state_dict_for_load(obj: object) -> dict[str, Tensor]:
    if isinstance(obj, Mapping):
        if obj.get("format") == "allama_reborn_int8_v1":
            return dequantize_payload(obj)
        model_obj = obj.get("model")
        if isinstance(model_obj, Mapping):
            return {str(k): v for k, v in model_obj.items() if isinstance(v, Tensor)}
        if all(isinstance(k, str) and isinstance(v, Tensor) for k, v in obj.items()):
            return {str(k): v for k, v in obj.items()}
    raise ValueError(
        "LOAD_PATH did not contain a supported state dict or compact payload"
    )


def maybe_load_model_weights(
    model: nn.Module, args: Hyperparameters, log: Callable[[str], None]
) -> None:
    if not args.load_path:
        return
    obj = torch.load(args.load_path, map_location="cpu")
    state = extract_state_dict_for_load(obj)
    missing, unexpected = model.load_state_dict(state, strict=args.strict_load)
    log(
        f"loaded_checkpoint={args.load_path} missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        log("missing_keys=" + ",".join(missing))
    if unexpected:
        log("unexpected_keys=" + ",".join(unexpected))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main() -> None:
    cli = parse_cli_overrides(tuple(sys.argv[1:]))
    args = Hyperparameters.from_env(cli)
    distributed, rank, world_size, _local_rank, device, master_process = (
        init_distributed_and_device(args)
    )
    sdpa_status: dict[str, bool | str] = {"requested": args.sdpa_backend}

    if device.type == "cuda":
        configure_cuda_precision()
        sdpa_status = configure_sdpa_backend(args.sdpa_backend)
    elif args.sdpa_backend != "auto":
        raise ValueError(
            f"SDPA_BACKEND={args.sdpa_backend} requires CUDA; got device={device}"
        )

    set_seed(args.seed + rank)
    backend = infer_backend(args)
    eval_mode = resolve_eval_mode(args, backend)

    if backend == "enwik8":
        train_tokens, val_tokens = load_enwik8(
            Path(args.enwik8_path), args.train_split, args.data_bytes_limit
        )
        vocab_size = args.vocab_size if args.vocab_size > 0 else 256
        if vocab_size < 256:
            raise ValueError("VOCAB_SIZE must be >= 256 for enwik8")
        train_loader = SequentialDistributedTokenLoader.from_tensor(
            train_tokens,
            rank=rank,
            world_size=world_size,
            device=device,
        )
    elif backend == "parameter_golf":
        vocab_size = args.vocab_size if args.vocab_size > 0 else 1024
        val_tokens = load_validation_tokens(args.val_files)
        train_loader = SequentialDistributedTokenLoader.from_pattern(
            args.train_files,
            rank=rank,
            world_size=world_size,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported DATA_BACKEND={backend}")

    metric = setup_bpb_metric(backend, args, vocab_size, device)
    device_type, amp_dtype, use_autocast = resolve_dtype(args, device)

    report_model = ALlama(args, vocab_size=vocab_size).to(device)
    model_no_ddp = report_model

    run_dir = Path(args.out_dir) / args.run_id
    log_path = run_dir / "train.log"
    progress_bar: Optional[Any] = None
    if master_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(args), f, indent=2)

    def log(msg: str) -> None:
        if not master_process:
            return
        if progress_bar is not None:
            progress_bar.write(msg)
        else:
            print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if (
        os.environ.get("ITERATIONS")
        and not os.environ.get("MAX_STEPS")
        and master_process
    ):
        log(
            "note=ITERATIONS was provided; in this trainer it is treated only as a MAX_STEPS override. Data traversal is epoch-driven."
        )

    maybe_load_model_weights(report_model, args, log)

    compile_enabled = False
    if args.compile_model:
        if not hasattr(torch, "compile"):
            log("note=torch.compile requested but unavailable in this PyTorch build")
        elif distributed:
            log(
                "note=torch.compile requested but skipped because distributed training is enabled"
            )
        else:
            model_no_ddp = torch.compile(model_no_ddp)  # type: ignore[assignment]
            compile_enabled = True

    if distributed:
        ddp_device_ids = [device.index] if device.type == "cuda" else None
        model: nn.Module = DDP(model_no_ddp, device_ids=ddp_device_ids)
    else:
        model = model_no_ddp

    local_batch_size = get_local_batch_size(args, world_size)
    local_batch_size * world_size
    sampled_eval_batch_size = (
        args.val_batch_size if args.val_batch_size > 0 else local_batch_size
    )
    eval_batch_tokens = (
        args.eval_batch_tokens
        if args.eval_batch_tokens > 0
        else args.train_batch_tokens
    )
    epoch_steps, epoch_plan_info = plan_epoch_steps(
        total_tokens=train_loader.total_tokens,
        train_batch_tokens=args.train_batch_tokens,
        max_local_batch_size=local_batch_size,
        max_seq_len=args.train_seq_len,
        world_size=world_size,
    )
    if not epoch_steps and not args.eval_only:
        raise ValueError(
            f"Not enough training tokens to form even one batch: total_tokens={train_loader.total_tokens}"
        )
    planned_total_steps = len(epoch_steps) * max(1, args.num_epochs)
    if args.max_steps > 0:
        total_training_steps = min(planned_total_steps, args.max_steps)
    else:
        total_training_steps = planned_total_steps
    epoch_step_token_counts = [
        int(sum(spec.global_target_tokens for spec in step_specs))
        for step_specs in epoch_steps
    ]
    nominal_step_tokens = epoch_step_token_counts[0] if epoch_step_token_counts else 0
    final_step_tokens = epoch_step_token_counts[-1] if epoch_step_token_counts else 0
    variable_step_tokens = len(set(epoch_step_token_counts)) > 1

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )

    param_report = compute_logical_parameter_counts(report_model)
    if args.report_artifact:
        footprint_report = model_footprint_report(report_model, Path(__file__), args)
    else:
        footprint_report = dict(param_report)

    physical_param_count = int(param_report["stored_params"])
    logical_param_count = int(param_report["functional_params"])
    sharing_ratio = float(param_report["sharing_ratio"])

    if master_process and args.print_model_summary:
        summary_text = render_model_summary(
            report_model,
            max_depth=args.model_summary_max_depth,
            show_param_shapes=args.model_summary_show_shapes,
        )
        print(summary_text, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(summary_text + "\n")
        with open(run_dir / "model_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")

    log(
        f"run_id={args.run_id} backend={backend} device={device} dtype={amp_dtype} compile={int(compile_enabled)} "
        f"stored_params={physical_param_count} functional_params={logical_param_count} "
        f"sharing_ratio={sharing_ratio:.4f} logical_layers={args.num_layers} shared_blocks={args.num_shared_blocks} "
        f"share_pattern={args.share_pattern} layer_map={report_model.layer_to_block_map()} eval_mode={eval_mode}"
    )
    if device.type == "cuda":
        log(
            f"sdpa_config requested={sdpa_status['requested']} flash_available={int(bool(sdpa_status['flash_available']))} "
            f"flash_enabled={int(bool(sdpa_status['flash_enabled']))} math_enabled={int(bool(sdpa_status['math_enabled']))} "
            f"efficient_enabled={int(bool(sdpa_status['efficient_enabled']))} cudnn_enabled={int(bool(sdpa_status['cudnn_enabled']))}"
        )
    log(
        f"train_plan epochs={args.num_epochs} max_steps={args.max_steps} steps_per_epoch={epoch_plan_info['steps_per_epoch']} "
        f"planned_total_steps={planned_total_steps} total_training_steps={total_training_steps} "
        f"usable_target_tokens_per_epoch={epoch_plan_info['usable_target_tokens']} "
        f"planned_target_tokens_per_epoch={epoch_plan_info['planned_target_tokens']} "
        f"dropped_target_tokens_per_epoch={epoch_plan_info['dropped_target_tokens']}"
    )
    log(
        f"train_static nominal_step_tokens={nominal_step_tokens} final_step_tokens={final_step_tokens} "
        f"variable_step_tokens={int(variable_step_tokens)} grad_accum_steps={args.grad_accum_steps} "
        f"train_log_every={args.train_log_every} val_loss_every={args.val_loss_every}"
    )
    log(
        f"x0_shortcut={args.use_x0_shortcut} resid_mix_init={args.resid_mix_init} "
        f"global_rotary=1 enable_gqa_probe={int(SDPA_ENABLE_GQA)}"
    )
    log(
        "model_init "
        f"stored_params={int(param_report['stored_params'])} "
        f"stored_trainable_params={int(param_report['stored_trainable_params'])} "
        f"functional_params={int(param_report['functional_params'])} "
        f"functional_trainable_params={int(param_report['functional_trainable_params'])} "
        f"shared_block_stored_params={int(param_report['shared_block_stored_params'])} "
        f"shared_block_functional_params={int(param_report['shared_block_functional_params'])} "
        f"nonshared_stored_params={int(param_report['nonshared_stored_params'])} "
        f"sharing_ratio={float(param_report['sharing_ratio']):.6f}"
    )
    if metric.kind == "none":
        log(
            "val_bpb=disabled (no byte accounting available for this backend/environment)"
        )
    else:
        log(f"val_bpb=enabled metric_kind={metric.kind}")

    if args.report_artifact:
        log(
            "size_init "
            f"stored_parameter_bytes={int(footprint_report['stored_parameter_bytes'])} "
            f"state_dict_bytes={int(footprint_report['state_dict_bytes'])} "
            f"checkpoint_bytes={int(footprint_report['checkpoint_bytes'])} "
            f"checkpoint_zlib_bytes={int(footprint_report['checkpoint_zlib_bytes'])} "
            f"int8_payload_bytes={int(footprint_report['int8_payload_bytes'])} "
            f"int8_payload_zlib_bytes={int(footprint_report['int8_payload_zlib_bytes'])} "
            f"code_bytes={int(footprint_report['code_bytes'])} "
            f"artifact_bytes={int(footprint_report['artifact_bytes'])}"
        )
        log(
            "payload_init "
            f"int8_payload_quantized_numel={int(footprint_report['int8_payload_quantized_numel'])} "
            f"int8_payload_scale_numel={int(footprint_report['int8_payload_scale_numel'])} "
            f"int8_payload_passthrough_numel={int(footprint_report['int8_payload_passthrough_numel'])} "
            f"int8_payload_total_tensor_numel={int(footprint_report['int8_payload_total_tensor_numel'])}"
        )

    wandb_config = asdict(args) | {
        "backend": backend,
        "resolved_eval_mode": eval_mode,
        "sdpa_flash_available": bool(sdpa_status.get("flash_available", False)),
        "sdpa_flash_enabled": bool(sdpa_status.get("flash_enabled", False)),
        "sdpa_math_enabled": bool(sdpa_status.get("math_enabled", False)),
        "sdpa_efficient_enabled": bool(sdpa_status.get("efficient_enabled", False)),
        "sdpa_cudnn_enabled": bool(sdpa_status.get("cudnn_enabled", False)),
        "layer_map": report_model.layer_to_block_map(),
        "steps_per_epoch": epoch_plan_info["steps_per_epoch"],
        "planned_total_steps": planned_total_steps,
        "total_training_steps": total_training_steps,
        "train_nominal_step_tokens": nominal_step_tokens,
        "train_final_step_tokens": final_step_tokens,
        "train_variable_step_tokens": variable_step_tokens,
        "model_stored_params": int(param_report["stored_params"]),
        "model_stored_trainable_params": int(param_report["stored_trainable_params"]),
        "model_functional_params": int(param_report["functional_params"]),
        "model_functional_trainable_params": int(
            param_report["functional_trainable_params"]
        ),
        "model_shared_block_stored_params": int(
            param_report["shared_block_stored_params"]
        ),
        "model_shared_block_functional_params": int(
            param_report["shared_block_functional_params"]
        ),
        "model_nonshared_stored_params": int(param_report["nonshared_stored_params"]),
        "sharing_ratio": float(param_report["sharing_ratio"]),
    }
    if args.report_artifact:
        wandb_config |= {
            "model_stored_parameter_bytes_init": int(
                footprint_report["stored_parameter_bytes"]
            ),
            "model_state_dict_bytes_init": int(footprint_report["state_dict_bytes"]),
            "model_checkpoint_bytes_init": int(footprint_report["checkpoint_bytes"]),
            "model_checkpoint_zlib_bytes_init": int(
                footprint_report["checkpoint_zlib_bytes"]
            ),
            "int8_payload_bytes_init": int(footprint_report["int8_payload_bytes"]),
            "model_int8_payload_zlib_bytes_init": int(
                footprint_report["int8_payload_zlib_bytes"]
            ),
            "model_code_bytes": int(footprint_report["code_bytes"]),
            "model_artifact_bytes_init": int(footprint_report["artifact_bytes"]),
            "int8_payload_quantized_numel": int(
                footprint_report["int8_payload_quantized_numel"]
            ),
            "int8_payload_scale_numel": int(
                footprint_report["int8_payload_scale_numel"]
            ),
            "int8_payload_passthrough_numel": int(
                footprint_report["int8_payload_passthrough_numel"]
            ),
            "int8_payload_total_tensor_numel": int(
                footprint_report["int8_payload_total_tensor_numel"]
            ),
        }

    wandb_logger = WandbLogger(
        args,
        enabled=master_process and args.wandb_enable,
        config=wandb_config,
    )
    wandb_logger.watch(
        report_model,
        mode=args.wandb_watch,
        log_freq=args.wandb_watch_log_freq,
        log_fn=log,
    )

    if (
        master_process
        and not args.eval_only
        and total_training_steps > 0
        and tqdm is not None
    ):
        progress_bar = tqdm(
            total=total_training_steps,
            desc="train",
            dynamic_ncols=True,
            leave=True,
        )

    def record_run_footprint(
        step: int,
        checkpoint_path: Optional[Path] = None,
        export_path: Optional[Path] = None,
    ) -> None:
        if not master_process:
            return
        final_report = (
            model_footprint_report(report_model, Path(__file__), args)
            if args.report_artifact
            else compute_logical_parameter_counts(report_model)
        )
        if checkpoint_path is not None and checkpoint_path.exists():
            final_report["saved_checkpoint_bytes"] = int(checkpoint_path.stat().st_size)
        if export_path is not None and export_path.exists():
            final_report["saved_int8_payload_bytes"] = int(export_path.stat().st_size)
        log(
            "model_final "
            f"stored_params={int(final_report['stored_params'])} "
            f"stored_trainable_params={int(final_report['stored_trainable_params'])} "
            f"functional_params={int(final_report['functional_params'])} "
            f"functional_trainable_params={int(final_report['functional_trainable_params'])} "
            f"shared_block_stored_params={int(final_report['shared_block_stored_params'])} "
            f"shared_block_functional_params={int(final_report['shared_block_functional_params'])} "
            f"nonshared_stored_params={int(final_report['nonshared_stored_params'])} "
            f"sharing_ratio={float(final_report['sharing_ratio']):.6f}"
        )
        if args.report_artifact:
            log(
                "size_final "
                f"stored_parameter_bytes={int(final_report['stored_parameter_bytes'])} "
                f"state_dict_bytes={int(final_report['state_dict_bytes'])} "
                f"checkpoint_bytes={int(final_report['checkpoint_bytes'])} "
                f"checkpoint_zlib_bytes={int(final_report['checkpoint_zlib_bytes'])} "
                f"int8_payload_bytes={int(final_report['int8_payload_bytes'])} "
                f"int8_payload_zlib_bytes={int(final_report['int8_payload_zlib_bytes'])} "
                f"code_bytes={int(final_report['code_bytes'])} "
                f"artifact_bytes={int(final_report['artifact_bytes'])} "
                f"saved_checkpoint_bytes={int(final_report.get('saved_checkpoint_bytes', 0))} "
                f"saved_int8_payload_bytes={int(final_report.get('saved_int8_payload_bytes', 0))}"
            )
            final_artifact_summary = {}
            if "saved_checkpoint_bytes" in final_report:
                final_artifact_summary["artifact/saved_checkpoint_bytes"] = int(
                    final_report["saved_checkpoint_bytes"]
                )
            if "saved_int8_payload_bytes" in final_report:
                final_artifact_summary["artifact/saved_int8_payload_bytes"] = int(
                    final_report["saved_int8_payload_bytes"]
                )
            if final_artifact_summary:
                wandb_logger.update_summary(final_artifact_summary)

    def run_eval(step: int) -> None:
        nonlocal eval_batch_tokens
        if distributed and eval_mode == "sampled":
            dist.barrier()
        if eval_mode == "full":
            val_loss, val_bpb = evaluate_full(
                report_model,
                val_tokens=val_tokens,
                seq_len=args.eval_seq_len,
                eval_batch_tokens=eval_batch_tokens,
                rank=rank,
                world_size=world_size,
                distributed=distributed,
                device=device,
                device_type=device_type,
                amp_dtype=amp_dtype,
                use_autocast=use_autocast,
                metric=metric,
            )
            if master_process:
                if val_bpb is None:
                    log(f"eval step={step} eval_mode=full val_loss={val_loss:.6f}")
                    wandb_logger.log({"eval/loss": val_loss}, step=step)
                else:
                    log(
                        f"eval step={step} eval_mode=full val_loss={val_loss:.6f} val_bpb={val_bpb:.6f}"
                    )
                    wandb_logger.log(
                        {"eval/loss": val_loss, "eval/bpb": val_bpb}, step=step
                    )
            if distributed:
                dist.barrier()
            return

        if master_process:
            val_loss, val_bpb = evaluate_sampled(
                report_model if distributed else model_no_ddp,
                val_tokens=val_tokens,
                seq_len=args.eval_seq_len,
                batch_size=sampled_eval_batch_size,
                num_batches=args.val_batches,
                device=device,
                device_type=device_type,
                amp_dtype=amp_dtype,
                use_autocast=use_autocast,
                metric=metric,
            )
            if val_bpb is None:
                log(f"eval step={step} eval_mode=sampled val_loss={val_loss:.6f}")
                wandb_logger.log({"eval/loss": val_loss}, step=step)
            else:
                log(
                    f"eval step={step} eval_mode=sampled val_loss={val_loss:.6f} val_bpb={val_bpb:.6f}"
                )
                wandb_logger.log(
                    {"eval/loss": val_loss, "eval/bpb": val_bpb}, step=step
                )
        if distributed:
            dist.barrier()

    if args.eval_only:
        run_eval(step=0)
        if master_process:
            final_state = report_model.state_dict()
            checkpoint_path: Optional[Path] = None
            export_path: Optional[Path] = None
            if args.save_path:
                checkpoint_path = Path(args.save_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"config": asdict(args), "model": final_state}, checkpoint_path
                )
                log(
                    f"saved_checkpoint={checkpoint_path} bytes={checkpoint_path.stat().st_size}"
                )
            if args.export_int8_path:
                export_path = Path(args.export_int8_path)
                export_path.parent.mkdir(parents=True, exist_ok=True)
                payload = build_int8_payload(
                    final_state,
                    keep_float_numel=args.quant_keep_float_numel,
                    control_patterns=args.control_tensor_name_patterns,
                )
                torch.save(payload, export_path)
                log(
                    f"saved_int8_payload={export_path} bytes={export_path.stat().st_size} "
                    f"zlib_bytes={serialized_zlib_nbytes(payload)}"
                )
            record_run_footprint(
                step=0, checkpoint_path=checkpoint_path, export_path=export_path
            )
        wandb_logger.finish()
        if distributed:
            dist.destroy_process_group()
        return

    start_time = time.time()
    completed_steps = 0
    processed_train_tokens = 0
    last_eval_step = -1
    stop_reason = "end_of_data"

    for epoch in range(1, args.num_epochs + 1):
        train_loader.reset()
        for epoch_step_idx, micro_specs in enumerate(epoch_steps, start=1):
            if args.max_steps > 0 and completed_steps >= args.max_steps:
                stop_reason = f"max_steps={args.max_steps}"
                break
            if (
                args.max_wallclock_seconds > 0.0
                and (time.time() - start_time) >= args.max_wallclock_seconds
            ):
                stop_reason = f"max_wallclock_seconds={args.max_wallclock_seconds}"
                break

            completed_steps += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss_sum = 0.0
            step_token_count = 0
            step_global_token_count = int(
                sum(spec.global_target_tokens for spec in micro_specs)
            )
            step_local_token_count = int(
                sum(spec.local_batch_size * spec.seq_len for spec in micro_specs)
            )

            lr = get_lr(completed_steps - 1, total_training_steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            for micro_step, spec in enumerate(micro_specs):
                x, y = train_loader.next_batch(
                    local_batch_size=spec.local_batch_size, seq_len=spec.seq_len
                )
                is_last_micro = micro_step == len(micro_specs) - 1
                sync_context = (
                    model.no_sync if distributed and not is_last_micro else nullcontext
                )
                with sync_context():
                    with autocast_context(device_type, amp_dtype, use_autocast):
                        loss_sum = model(x, y, loss_reduction="sum")
                        token_count = int(y.numel())
                        loss = loss_sum / float(max(1, step_local_token_count))
                    loss.backward()

                step_loss_sum += float(loss_sum.item())
                step_token_count += token_count

            if args.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            processed_train_tokens += step_global_token_count

            stats = torch.tensor(
                [step_loss_sum, float(step_token_count)],
                device=device,
                dtype=torch.float64,
            )
            if distributed:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            train_loss = float(stats[0].item() / max(1.0, stats[1].item()))
            elapsed = time.time() - start_time
            toks_per_sec = float(processed_train_tokens) / max(elapsed, 1e-6)
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{train_loss:.4f}",
                    lr=f"{lr:.2e}",
                    tok_s=f"{toks_per_sec:.0f}",
                    refresh=False,
                )

            should_log = (
                completed_steps == 1
                or completed_steps % args.train_log_every == 0
                or completed_steps == total_training_steps
            )
            if master_process and should_log:
                log(
                    f"step={completed_steps}/{total_training_steps} epoch_step={epoch_step_idx}/{len(epoch_steps)} "
                    f"lr={lr:.6g} train_loss={train_loss:.6f} "
                    f"processed_tokens={processed_train_tokens} elapsed_s={elapsed:.2f} tokens_per_s={toks_per_sec:.2f}"
                )
                wandb_logger.log(
                    {
                        "train/loss": train_loss,
                        "train/lr": lr,
                        "train/processed_tokens": processed_train_tokens,
                        "train/tokens_per_s": toks_per_sec,
                    },
                    step=completed_steps,
                )

            should_eval = args.val_loss_every > 0 and (
                completed_steps % args.val_loss_every == 0
                or completed_steps == total_training_steps
            )
            if should_eval:
                run_eval(completed_steps)
                last_eval_step = completed_steps

        if stop_reason != "end_of_data":
            break

    if (
        completed_steps > 0
        and args.val_loss_every > 0
        and last_eval_step != completed_steps
    ):
        run_eval(completed_steps)
        last_eval_step = completed_steps
    if master_process:
        log(f"train_stop step={completed_steps} reason={stop_reason}")
        final_state = report_model.state_dict()
        checkpoint_path: Optional[Path] = None
        export_path: Optional[Path] = None
        if args.save_path:
            checkpoint_path = Path(args.save_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": asdict(args), "model": final_state}, checkpoint_path)
            log(
                f"saved_checkpoint={checkpoint_path} bytes={checkpoint_path.stat().st_size}"
            )
        if args.export_int8_path:
            export_path = Path(args.export_int8_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            payload = build_int8_payload(
                final_state,
                keep_float_numel=args.quant_keep_float_numel,
                control_patterns=args.control_tensor_name_patterns,
            )
            torch.save(payload, export_path)
            log(
                f"saved_int8_payload={export_path} bytes={export_path.stat().st_size} "
                f"zlib_bytes={serialized_zlib_nbytes(payload)}"
            )
        record_run_footprint(
            step=completed_steps,
            checkpoint_path=checkpoint_path,
            export_path=export_path,
        )

    wandb_logger.finish()
    if progress_bar is not None:
        progress_bar.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
