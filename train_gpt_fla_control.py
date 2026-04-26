#!/usr/bin/env python3
"""Isolated native-FLA GatedDeltaNet calibration trainer."""

from __future__ import annotations

import argparse
import copy
import io
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from fla.layers import GatedDeltaNet
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from trainer_shared import (
    DistributedTokenLoader,
    build_sentencepiece_luts,
    eval_val,
    load_validation_tokens_from_files,
    make_stream_sync_event,
    resolve_glob_files,
    wait_current_stream,
)
from train_gpt import (
    Muon,
    dequantize_state_dict_int8,
    load_data_shard,
    quantize_state_dict_int8,
)


def env_bool(name: str, default: bool) -> bool:
    """Read one boolean environment flag.

    :param str name: Environment variable name.
    :param bool default: Default value.
    :return bool: Parsed boolean.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Hyperparameters:
    """Environment-backed hyperparameters for the native FLA control."""

    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", f"fla_control_{uuid.uuid4().hex[:8]}")
    seed = int(os.environ.get("SEED", 1337))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 10))
    model_dim = int(os.environ.get("MODEL_DIM", 544))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    head_dim = int(os.environ.get("HEAD_DIM", 64))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = env_bool("TIE_EMBEDDINGS", True)
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    fla_mode = os.environ.get("FLA_MODE", "chunk")
    fla_value_expand = float(os.environ.get("FLA_VALUE_EXPAND", 2.0))
    fla_num_v_heads = os.environ.get("FLA_NUM_V_HEADS")
    fla_num_v_heads = None if fla_num_v_heads in {None, ""} else int(fla_num_v_heads)
    fla_use_gate = env_bool("FLA_USE_GATE", True)
    fla_use_short_conv = env_bool("FLA_USE_SHORT_CONV", True)
    fla_allow_neg_eigval = env_bool("FLA_ALLOW_NEG_EIGVAL", True)
    fla_conv_size = int(os.environ.get("FLA_CONV_SIZE", 4))
    fla_conv_bias = env_bool("FLA_CONV_BIAS", False)
    fla_norm_eps = float(os.environ.get("FLA_NORM_EPS", 1e-5))
    fla_share_qk = env_bool("FLA_SHARE_QK", False)
    fla_share_kv = env_bool("FLA_SHARE_KV", False)

    bigram_hash_buckets = int(os.environ.get("BIGRAM_HASH_BUCKETS", 0))
    bigram_hash_dim = int(os.environ.get("BIGRAM_HASH_DIM", 128))
    trigram_hash_buckets = int(os.environ.get("TRIGRAM_HASH_BUCKETS", 0))
    trigram_hash_dim = int(os.environ.get("TRIGRAM_HASH_DIM", 128))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    log_step0_eval = env_bool("LOG_STEP0_EVAL", False)
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    perf_skip_final_eval = env_bool("PERF_SKIP_FINAL_EVAL", False)

    compile_enabled = env_bool("COMPILE", True)
    compile_fullgraph = env_bool("COMPILE_FULLGRAPH", False)
    storage_dtype = os.environ.get("FLA_STORAGE_DTYPE", "fp32").lower()

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


class CastedLinear(nn.Linear):
    """Linear layer that casts stored weights to the input dtype."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply a linear projection.

        :param Tensor x: Input tensor.
        :return Tensor: Projected tensor.
        """
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class RMSNorm(nn.Module):
    """Parameter-free RMSNorm wrapper."""

    def __init__(self, eps: float = 1e-6):
        """Initialize RMSNorm.

        :param float eps: Numerical epsilon.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the final dimension.

        :param Tensor x: Input activations.
        :return Tensor: Normalized activations.
        """
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class MLP(nn.Module):
    """Baseline-style ReLU-squared MLP."""

    def __init__(self, dim: int, mlp_mult: float):
        """Initialize the feed-forward block.

        :param int dim: Model width.
        :param float mlp_mult: Hidden-width multiplier.
        """
        super().__init__()
        hidden = int(round(dim * mlp_mult))
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP.

        :param Tensor x: Input activations.
        :return Tensor: Projected MLP activations.
        """
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class BigramHashEmbedding(nn.Module):
    """Optional token-pair hash feature from prior records."""

    def __init__(self, buckets: int, dim: int, model_dim: int):
        """Initialize the hash table.

        :param int buckets: Hash bucket count.
        :param int dim: Hash embedding width.
        :param int model_dim: Model width.
        """
        super().__init__()
        self.buckets = buckets
        self.embed = nn.Embedding(buckets, dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = (
            CastedLinear(dim, model_dim, bias=False) if dim != model_dim else None
        )
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return bigram hash features.

        :param Tensor token_ids: Token ids shaped ``[B, T]``.
        :return Tensor: Additive hash features.
        """
        t = token_ids.to(torch.int32)
        mod = self.buckets - 1
        hashed = torch.empty_like(t)
        hashed[..., 0] = mod
        hashed[..., 1:] = (
            torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        )
        out = self.embed(hashed.long())
        if self.proj is not None:
            out = self.proj(out)
        return out * self.scale.to(dtype=out.dtype)


class TrigramHashEmbedding(nn.Module):
    """Optional token-triple hash feature for calibration experiments."""

    def __init__(self, buckets: int, dim: int, model_dim: int):
        """Initialize the trigram hash table.

        :param int buckets: Hash bucket count.
        :param int dim: Hash embedding width.
        :param int model_dim: Model width.
        """
        super().__init__()
        self.buckets = buckets
        self.embed = nn.Embedding(buckets, dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = (
            CastedLinear(dim, model_dim, bias=False) if dim != model_dim else None
        )
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.03, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return trigram hash features.

        :param Tensor token_ids: Token ids shaped ``[B, T]``.
        :return Tensor: Additive hash features.
        """
        t = token_ids.to(torch.int32)
        mod = self.buckets - 1
        hashed = torch.empty_like(t)
        hashed[..., :2] = mod
        mixed = 131071 * t[..., 2:] + 524287 * t[..., 1:-1] + 8191 * t[..., :-2]
        hashed[..., 2:] = mixed % mod
        out = self.embed(hashed.long())
        if self.proj is not None:
            out = self.proj(out)
        return out * self.scale.to(dtype=out.dtype)


class NativeFLABlock(nn.Module):
    """One native FLA GatedDeltaNet block plus MLP."""

    def __init__(self, args: Hyperparameters, layer_idx: int):
        """Initialize one block.

        :param Hyperparameters args: Runtime hyperparameters.
        :param int layer_idx: Layer index.
        """
        super().__init__()
        self.gdn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.gdn = GatedDeltaNet(
            hidden_size=args.model_dim,
            expand_v=args.fla_value_expand,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            num_v_heads=args.fla_num_v_heads,
            mode=args.fla_mode,
            use_gate=args.fla_use_gate,
            use_short_conv=args.fla_use_short_conv,
            allow_neg_eigval=args.fla_allow_neg_eigval,
            conv_size=args.fla_conv_size,
            conv_bias=args.fla_conv_bias,
            layer_idx=layer_idx,
            norm_eps=args.fla_norm_eps,
        )
        self.mlp = MLP(args.model_dim, args.mlp_mult)
        self.gdn_scale = nn.Parameter(torch.ones(args.model_dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(args.model_dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """Apply native GDN and MLP residual updates.

        :param Tensor x: Hidden activations.
        :return Tensor: Updated activations.
        """
        gdn_out = self.gdn(self.gdn_norm(x), use_cache=False)[0]
        x = x + self.gdn_scale.to(dtype=x.dtype)[None, None, :] * gdn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x)
        )
        return x


class NativeFLAGPT(nn.Module):
    """Pure native-FLA GDN language model for calibration."""

    def __init__(self, args: Hyperparameters):
        """Build the isolated native-FLA model.

        :param Hyperparameters args: Runtime hyperparameters.
        """
        super().__init__()
        if args.logit_softcap <= 0.0:
            raise ValueError("LOGIT_SOFTCAP must be positive")
        self.args = args
        self.tie_embeddings = args.tie_embeddings
        self.logit_softcap = args.logit_softcap
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.bigram = (
            BigramHashEmbedding(
                args.bigram_hash_buckets, args.bigram_hash_dim, args.model_dim
            )
            if args.bigram_hash_buckets > 0
            else None
        )
        self.trigram = (
            TrigramHashEmbedding(
                args.trigram_hash_buckets, args.trigram_hash_dim, args.model_dim
            )
            if args.trigram_hash_buckets > 0
            else None
        )
        self.input_norm = RMSNorm()
        self.blocks = nn.ModuleList(
            [NativeFLABlock(args, layer_idx=i) for i in range(args.num_layers)]
        )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None
            if args.tie_embeddings
            else CastedLinear(args.model_dim, args.vocab_size, bias=False)
        )
        self._init_weights()
        self._apply_fla_sharing(args)

    def _init_weights(self) -> None:
        """Initialize embeddings and optional output head."""
        nn.init.normal_(
            self.tok_emb.weight, mean=0.0, std=self.args.tied_embed_init_std
        )
        if self.lm_head is not None:
            nn.init.zeros_(self.lm_head.weight)

    def _apply_fla_sharing(self, args: Hyperparameters) -> None:
        """Apply optional native-FLA projection sharing knobs.

        :param Hyperparameters args: Runtime hyperparameters.
        """
        for block in self.blocks:
            if args.fla_share_qk:
                block.gdn.k_proj.weight = block.gdn.q_proj.weight
            if args.fla_share_kv:
                if block.gdn.k_proj.weight.shape != block.gdn.v_proj.weight.shape:
                    raise ValueError(
                        "FLA_SHARE_KV requires matching k/v projection shapes"
                    )
                block.gdn.v_proj.weight = block.gdn.k_proj.weight

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return next-token logits.

        :param Tensor input_ids: Input token ids.
        :return Tensor: Logits shaped ``[B, T, vocab]``.
        """
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids).to(dtype=x.dtype)
        if self.trigram is not None:
            x = x + self.trigram(input_ids).to(dtype=x.dtype)
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when TIE_EMBEDDINGS=0")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Compute next-token cross entropy.

        :param Tensor input_ids: Input token ids.
        :param Tensor target_ids: Target token ids.
        :return Tensor: Mean cross entropy.
        """
        logits = self.forward_logits(input_ids).reshape(-1, self.args.vocab_size)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")


def restore_control_params_to_fp32(module: nn.Module) -> None:
    """Keep scalar/control parameters in fp32 storage.

    :param nn.Module module: Model to update in place.
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.ndim < 2 or any(
                key in name for key in ("A_log", "dt_bias", "scale", "norm", "bias")
            ):
                if param.dtype != torch.float32:
                    param.data = param.data.float()


def build_optimizers(
    base_model: NativeFLAGPT, args: Hyperparameters
) -> tuple[list[torch.optim.Optimizer], Muon | None]:
    """Create optimizer groups for the native-FLA control.

    :param NativeFLAGPT base_model: Model with owning parameters.
    :param Hyperparameters args: Runtime hyperparameters.
    :return tuple[list[torch.optim.Optimizer], Muon | None]: Optimizers and optional Muon handle.
    """

    def adam(params: Iterable[Tensor], lr: float) -> torch.optim.Adam:
        param_list = list(params)
        opt = torch.optim.Adam(
            [{"params": param_list, "lr": lr, "base_lr": lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        return opt

    token_params: list[Tensor] = [base_model.tok_emb.weight]
    if base_model.bigram is not None:
        token_params.extend(base_model.bigram.parameters())
    if base_model.trigram is not None:
        token_params.extend(base_model.trigram.parameters())

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        param
        for name, param in block_named_params
        if param.ndim == 2
        and not any(key in name for key in ("scale", "A_log", "dt_bias", "norm"))
    ]
    scalar_params = [
        param
        for name, param in block_named_params
        if param.ndim < 2
        or any(key in name for key in ("scale", "A_log", "dt_bias", "norm", "bias"))
    ]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizers: list[torch.optim.Optimizer] = [adam(token_params, token_lr)]
    if base_model.lm_head is not None:
        optimizers.append(adam([base_model.lm_head.weight], args.head_lr))

    optimizer_muon: Muon | None = None
    if matrix_params:
        optimizer_muon = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
        )
        for group in optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr
        optimizers.append(optimizer_muon)
    if scalar_params:
        optimizers.append(adam(scalar_params, args.scalar_lr))
    return optimizers, optimizer_muon


def build_model(args: Hyperparameters, device: torch.device) -> NativeFLAGPT:
    """Build and place the native-FLA model.

    :param Hyperparameters args: Runtime hyperparameters.
    :param torch.device device: Target device.
    :return NativeFLAGPT: Initialized model.
    """
    model = NativeFLAGPT(args).to(device)
    if args.storage_dtype == "bf16":
        model = model.bfloat16()
        restore_control_params_to_fp32(model)
    elif args.storage_dtype != "fp32":
        raise ValueError("FLA_STORAGE_DTYPE must be fp32 or bf16")
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line flags.

    :return argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Run a tiny forward/backward smoke."
    )
    parser.add_argument(
        "--smoke-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device for --smoke.",
    )
    return parser.parse_args()


def run_smoke(cli_args: argparse.Namespace) -> int:
    """Run a tiny native-FLA forward/backward smoke.

    :param argparse.Namespace cli_args: Parsed CLI flags.
    :return int: Exit code.
    """
    args = Hyperparameters()
    if cli_args.smoke_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cli_args.smoke_device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    model = build_model(args, device)
    model.train()
    batch = int(os.environ.get("SMOKE_BATCH_SIZE", 2))
    seq = int(os.environ.get("SMOKE_SEQ_LEN", min(args.train_seq_len, 64)))
    x = torch.randint(args.vocab_size, (batch, seq), device=device, dtype=torch.int32)
    y = torch.randint(args.vocab_size, (batch, seq), device=device, dtype=torch.int64)
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        loss = model(x, y)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(
        f"fla_control_smoke:ok device:{device} loss:{float(loss.detach().cpu()):.6f} "
        f"params:{sum(p.numel() for p in model.parameters())}"
    )
    return 0


def set_cuda_math_knobs() -> None:
    """Enable CUDA math settings used by local trainers."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def main() -> int:
    """Run native-FLA calibration training.

    :return int: Exit code.
    """
    cli_args = parse_args()
    if cli_args.smoke:
        return run_smoke(cli_args)

    args = Hyperparameters()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native FLA calibration training")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    set_cuda_math_knobs()
    stream_sync_event = make_stream_sync_event(device)
    master = rank == 0
    grad_accum_steps = args.grad_accum_steps
    batch_denom = world_size * grad_accum_steps * args.train_seq_len
    if args.train_batch_tokens % batch_denom != 0:
        raise ValueError(
            "TRAIN_BATCH_TOKENS must divide WORLD_SIZE * GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN, "
            f"got {args.train_batch_tokens=} {world_size=} {grad_accum_steps=} "
            f"{args.train_seq_len=}"
        )
    grad_scale = 1.0 / grad_accum_steps

    logfile_handle = None
    code = Path(__file__).read_text(encoding="utf-8")
    if master:
        Path("logs").mkdir(exist_ok=True)
        logfile_handle = Path("logs", f"{args.run_id}.txt").open(
            "a", encoding="utf-8", buffering=1
        )
        print(logfile_handle.name)

    def log0(msg: str, console: bool = True) -> None:
        """Log from rank zero.

        :param str msg: Message.
        :param bool console: Whether to print to stdout.
        """
        if not master:
            return
        if console:
            print(msg)
        if logfile_handle is not None:
            print(msg, file=logfile_handle)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )
    train_files = resolve_glob_files(
        args.train_files, missing_message=f"No files found for: {args.train_files}"
    )
    val_tokens = load_validation_tokens_from_files(
        resolve_glob_files(
            args.val_files, missing_message=f"No validation files for: {args.val_files}"
        ),
        args.train_seq_len,
        load_data_shard=load_data_shard,
        missing_message=f"No validation files for: {args.val_files}",
    )
    log0(
        f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}"
    )
    log0(
        f"train_loader:dataset:{Path(args.data_path).name} train_shards:{len(train_files)}"
    )
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = build_model(args, device)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(
        "native_fla_control:"
        f" layers:{args.num_layers} d_model:{args.model_dim} heads:{args.num_heads} "
        f"head_dim:{args.head_dim} value_expand:{args.fla_value_expand:g} "
        f"mlp_mult:{args.mlp_mult:g} tokenizer_vocab:{args.vocab_size}"
    )
    log0(
        "native_fla_knobs:"
        f" mode:{args.fla_mode} short_conv:{int(args.fla_use_short_conv)} "
        f"gate:{int(args.fla_use_gate)} allow_neg:{int(args.fla_allow_neg_eigval)} "
        f"share_qk:{int(args.fla_share_qk)} "
        f"share_kv:{int(args.fla_share_kv)} bigram:{args.bigram_hash_buckets} "
        f"trigram:{args.trigram_hash_buckets}"
    )
    log0(f"model_params:{n_params}")
    log0(
        f"launch_contract:planned_train_tokens:{args.train_batch_tokens * args.iterations} "
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"grad_accum_steps:{grad_accum_steps} local_batch_size:{args.train_batch_tokens // batch_denom} "
        f"iterations:{args.iterations} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )

    compiled_model: nn.Module = (
        torch.compile(
            base_model,
            dynamic=False,
            fullgraph=args.compile_fullgraph,
        )
        if args.compile_enabled
        else base_model
    )
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )
    optimizers, optimizer_muon = build_optimizers(base_model, args)
    train_loader = DistributedTokenLoader(
        train_files, rank, world_size, device, load_data_shard=load_data_shard
    )

    def zero_grad_all() -> None:
        """Clear gradients for all optimizers."""
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step: int, elapsed_ms: float) -> float:
        """Compute warmdown scale.

        :param int step: Current step.
        :param float elapsed_ms: Elapsed training milliseconds.
        :return float: LR multiplier.
        """
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if start <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        warmdown_ms = args.warmdown_iters * step_ms
        return (
            remaining_ms / max(warmdown_ms, 1e-9)
            if remaining_ms <= warmdown_ms
            else 1.0
        )

    if args.warmup_steps > 0:
        init_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in base_model.state_dict().items()
        }
        init_optimizers = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        zero_grad_all()
        warmup_x, warmup_y = train_loader.next_batch(
            args.train_batch_tokens, args.train_seq_len, grad_accum_steps
        )
        for warmup_step in range(args.warmup_steps):
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = (
                        micro_step == grad_accum_steps - 1
                    )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(warmup_x, warmup_y)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_state, strict=True)
        for opt, state in zip(optimizers, init_optimizers, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            train_files, rank, world_size, device, load_data_shard=load_data_shard
        )

    training_time_ms = 0.0
    stop_after_step: int | None = None
    zero_grad_all()
    wait_current_stream(stream_sync_event)
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )
        should_validate = (
            last_step
            or (step == 0 and args.log_step0_eval)
            or (
                args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0
            )
        )
        if should_validate:
            wait_current_stream(stream_sync_event)
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                model=model,
                rank=rank,
                world_size=world_size,
                device=device,
                grad_accum_steps=grad_accum_steps,
                train_seq_len=args.train_seq_len,
                val_batch_size=args.val_batch_size,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
                use_inference_mode=True,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            wait_current_stream(stream_sync_event)
            t0 = time.perf_counter()
        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        if optimizer_muon is not None:
            frac = (
                min(step / args.muon_momentum_warmup_steps, 1.0)
                if args.muon_momentum_warmup_steps > 0
                else 1.0
            )
            muon_momentum = (
                1 - frac
            ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
            for group in optimizer_muon.param_groups:
                group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (
            step <= 10 or step % args.train_log_every == 0
        ):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms"
            )
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    if args.perf_skip_final_eval:
        log0("perf_mode: skipping serialization and final roundtrip eval")
        if logfile_handle is not None:
            logfile_handle.close()
        if distributed:
            dist.destroy_process_group()
        return 0

    if master:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    if master:
        Path("final_model.int8.ptz").write_bytes(quant_blob)
        q_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        log0(
            f"Serialized model int8+zlib: {q_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']})"
        )
        log0(f"Total submission size int8+zlib: {q_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()

    quant_state = torch.load(
        io.BytesIO(zlib.decompress(Path("final_model.int8.ptz").read_bytes())),
        map_location="cpu",
    )
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    q_start = torch.cuda.Event(enable_timing=True)
    q_end = torch.cuda.Event(enable_timing=True)
    q_start.record()
    q_val_loss, q_val_bpb = eval_val(
        model=model,
        rank=rank,
        world_size=world_size,
        device=device,
        grad_accum_steps=grad_accum_steps,
        train_seq_len=args.train_seq_len,
        val_batch_size=args.val_batch_size,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        use_inference_mode=True,
    )
    q_end.record()
    q_end.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{q_start.elapsed_time(q_end):.0f}ms"
    )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}"
    )

    if logfile_handle is not None:
        logfile_handle.close()
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
