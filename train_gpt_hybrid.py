"""
train_gpt_hybrid.py — Parameter Golf submission with GDN Hybrid model.
Drop-in replacement for train_gpt.py. Same data loading, eval, quantization,
DDP, Muon optimizer. Adds wandb logging + GDN-specific hyperparameters.

Usage (single GPU):
  RUN_ID=hybrid_test torchrun --standalone --nproc_per_node=1 train_gpt_hybrid.py

Usage (8xH100):
  RUN_ID=hybrid_8gpu torchrun --standalone --nproc_per_node=8 train_gpt_hybrid.py

Sweep (see sweep.sh):
  WANDB_SWEEP=1 GDN_EXPAND_V=1.5 GDN_RATIO=3 ... torchrun ...
"""

from __future__ import annotations

import copy
import gc
import glob
import io
import math
import os
import random
import sys
import time
import uuid
import zlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from model import _HAS_FLA, SCALAR_PARAM_PATTERNS, HybridGPT
from profiler_report import (
    build_profile_report,
    build_profile_rows,
    write_profile_report,
)

# ── Optional wandb ────────────────────────────────────────────────────
_USE_WANDB = bool(int(os.environ.get("USE_WANDB", "1")))
try:
    import wandb
except ImportError:
    _USE_WANDB = False
    wandb = None


# =====================================================================
# HYPERPARAMETERS
# =====================================================================


class Hyperparameters:
    """Environment-backed hyperparameter surface for the hybrid trainer."""

    # Data
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    log_step0_eval = bool(int(os.environ.get("LOG_STEP0_EVAL", "0")))
    wandb_watch = os.environ.get("WANDB_WATCH", "none").lower()
    wandb_watch_log_freq = int(os.environ.get("WANDB_WATCH_LOG_FREQ", 100))

    # Training
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(
        os.environ.get("TRAIN_SEQ_LEN", 2048)
    )  # 2048 default: GDN benefits from longer context
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    compile = bool(int(os.environ.get("COMPILE", "1"))) and not bool(
        int(os.environ.get("TORCH_COMPILE_DISABLE", "0"))
    )
    compile_strategy = os.environ.get("COMPILE_STRATEGY", "model").lower()
    perf_timing = bool(int(os.environ.get("PERF_TIMING", "0")))
    perf_ignore_steps = int(os.environ.get("PERF_IGNORE_STEPS", 0))
    perf_skip_final_eval = bool(int(os.environ.get("PERF_SKIP_FINAL_EVAL", "0")))
    artifact_limit_bytes = int(os.environ.get("ARTIFACT_LIMIT_BYTES", 16_000_000))
    profile = bool(int(os.environ.get("PROFILE", "0")))
    profile_dir = os.environ.get("PROFILE_DIR", "./profiles")
    profile_wait = int(os.environ.get("PROFILE_WAIT", 5))
    profile_warmup = int(os.environ.get("PROFILE_WARMUP", 3))
    profile_active = int(os.environ.get("PROFILE_ACTIVE", 4))
    profile_repeat = int(os.environ.get("PROFILE_REPEAT", 1))
    profile_record_shapes = bool(int(os.environ.get("PROFILE_RECORD_SHAPES", "1")))
    profile_memory = bool(int(os.environ.get("PROFILE_MEMORY", "1")))
    profile_with_stack = bool(int(os.environ.get("PROFILE_WITH_STACK", "0")))
    profile_with_flops = bool(int(os.environ.get("PROFILE_WITH_FLOPS", "0")))
    profile_with_modules = bool(int(os.environ.get("PROFILE_WITH_MODULES", "0")))
    profile_row_limit = int(os.environ.get("PROFILE_ROW_LIMIT", 60))
    profile_sort_by = os.environ.get("PROFILE_SORT_BY", "self_cuda_time_total")
    profile_stop_on_complete = bool(
        int(os.environ.get("PROFILE_STOP_ON_COMPLETE", "1"))
    )

    # Model — shared
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 16))
    model_dim = int(os.environ.get("MODEL_DIM", 384))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    leaky_slope = float(os.environ.get("LEAKY_SLOPE", 0.5))
    norm_style = os.environ.get("NORM_STYLE", "pre").lower()
    residual_alpha = (
        float(os.environ["RESIDUAL_ALPHA"]) if "RESIDUAL_ALPHA" in os.environ else None
    )

    # Model — attention blocks
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))

    # Model — GDN blocks
    gdn_n_heads = int(os.environ.get("GDN_N_HEADS", 8))
    gdn_expand_v = float(os.environ.get("GDN_EXPAND_V", 1.0))
    gdn_head_k_dim = int(os.environ.get("GDN_HEAD_K_DIM", 48))

    gdn_allow_neg_eigval = bool(int(os.environ.get("GDN_ALLOW_NEG_EIGVAL", "1")))
    gdn_conv_size = int(os.environ.get("GDN_CONV_SIZE", 4))
    gdn_use_q_conv = bool(int(os.environ.get("GDN_USE_Q_CONV", "1")))
    gdn_use_k_conv = bool(int(os.environ.get("GDN_USE_K_CONV", "1")))
    gdn_use_v_conv = bool(int(os.environ.get("GDN_USE_V_CONV", "1")))
    gdn_use_packed_qkv_conv = bool(int(os.environ.get("GDN_USE_PACKED_QKV_CONV", "0")))
    gdn_use_packed_qkv_proj = bool(int(os.environ.get("GDN_USE_PACKED_QKV_PROJ", "0")))
    gdn_conv_output_contiguous = bool(
        int(os.environ.get("GDN_CONV_OUTPUT_CONTIGUOUS", "0"))
    )
    gdn_q_conv_output_contiguous = bool(
        int(
            os.environ.get(
                "GDN_Q_CONV_OUTPUT_CONTIGUOUS",
                "1" if gdn_conv_output_contiguous else "0",
            )
        )
    )
    gdn_k_conv_output_contiguous = bool(
        int(
            os.environ.get(
                "GDN_K_CONV_OUTPUT_CONTIGUOUS",
                "1" if gdn_conv_output_contiguous else "0",
            )
        )
    )
    gdn_v_conv_output_contiguous = bool(
        int(
            os.environ.get(
                "GDN_V_CONV_OUTPUT_CONTIGUOUS",
                "1" if gdn_conv_output_contiguous else "0",
            )
        )
    )
    gdn_control_proj_fp32 = bool(int(os.environ.get("GDN_CONTROL_PROJ_FP32", "1")))
    gdn_gates_fp32 = bool(int(os.environ.get("GDN_GATES_FP32", "1")))
    gdn_output_norm_fp32 = bool(int(os.environ.get("GDN_OUTPUT_NORM_FP32", "1")))
    gdn_use_cuda_frontend_nct = bool(
        int(os.environ.get("GDN_USE_CUDA_FRONTEND_NCT", "0"))
    )
    gdn_use_cuda_packed_conv = bool(
        int(os.environ.get("GDN_USE_CUDA_PACKED_CONV", "0"))
    )
    gdn_use_cuda_packed_conv_aten_backward = bool(
        int(os.environ.get("GDN_USE_CUDA_PACKED_CONV_ATEN_BACKWARD", "0"))
    )
    gdn_use_cuda_packed_conv_aten_weight_backward = bool(
        int(os.environ.get("GDN_USE_CUDA_PACKED_CONV_ATEN_WEIGHT_BACKWARD", "0"))
    )
    gdn_use_cuda_fused_frontend = bool(
        int(os.environ.get("GDN_USE_CUDA_FUSED_FRONTEND", "0"))
    )
    gdn_use_cuda_fused_frontend_lib = bool(
        int(os.environ.get("GDN_USE_CUDA_FUSED_FRONTEND_LIB", "0"))
    )
    gdn_use_cuda_fused_output = bool(
        int(os.environ.get("GDN_USE_CUDA_FUSED_OUTPUT", "0"))
    )
    gdn_use_cuda_split_norm = bool(int(os.environ.get("GDN_USE_CUDA_SPLIT_NORM", "0")))
    gdn_use_cuda_split_norm_lib = bool(
        int(os.environ.get("GDN_USE_CUDA_SPLIT_NORM_LIB", "0"))
    )
    gdn_use_packed_qkv_conv_custom_backward = bool(
        int(os.environ.get("GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD", "0"))
    )
    gdn_use_packed_qkv_single_contig = bool(
        int(os.environ.get("GDN_PACKED_QKV_SINGLE_CONTIG", "0"))
    )
    gdn_use_packed_qkv_split_copy = bool(
        int(os.environ.get("GDN_PACKED_QKV_SPLIT_COPY", "0"))
    )
    cudnn_benchmark = bool(int(os.environ.get("CUDNN_BENCHMARK", "0")))
    gdn_ratio = int(os.environ.get("GDN_RATIO", 3))  # 3 GDN : 1 Attn

    # Optimizer
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
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.04))


def normalize_wandb_watch_mode(mode: str) -> str:
    """Normalize the requested W&B watch mode.

    :param str mode: Raw watch mode string.
    :raises ValueError: If the mode is unsupported.
    :return str: Normalized watch mode.
    """
    normalized = mode.lower()
    if normalized in {"0", "false", "off", "none"}:
        return "none"
    if normalized in {"gradients", "all"}:
        return normalized
    raise ValueError(f"WANDB_WATCH must be one of: none, gradients, all (got {mode!r})")


def build_wandb_config(
    args: Hyperparameters,
    *,
    compile_stats: dict[str, int | str],
    local_batch_size: int,
    grad_accum_steps: int,
    planned_train_tokens: int,
    n_attn: int,
    n_gdn: int,
    n_params: int,
    sdp_backend: str,
    wandb_watch_mode: str,
) -> dict[str, Any]:
    """Build the canonical W&B config schema for HGDN runs.

    This schema is intentionally explicit and uses env-style uppercase keys only.
    Do not mirror lowercase Python attribute names into W&B config.

    :param Hyperparameters args: Trainer hyperparameters.
    :param dict[str, int | str] compile_stats: Compile status summary.
    :param int local_batch_size: Per-rank microbatch size.
    :param int grad_accum_steps: Gradient accumulation steps.
    :param int planned_train_tokens: Planned training-token budget.
    :param int n_attn: Number of attention blocks in the instantiated model.
    :param int n_gdn: Number of GDN blocks in the instantiated model.
    :param int n_params: Total parameter count.
    :param str sdp_backend: Active SDPA backend label.
    :param str wandb_watch_mode: Normalized W&B watch mode.
    :return dict[str, Any]: Canonical HGDN W&B config mapping.
    """
    return {
        "NUM_LAYERS": args.num_layers,
        "MODEL_DIM": args.model_dim,
        "MLP_MULT": args.mlp_mult,
        "GDN_RATIO": args.gdn_ratio,
        "ATTN_BLOCKS": n_attn,
        "CONV_BLOCKS": n_gdn,
        "N_PARAMS": n_params,
        "NUM_HEADS": args.num_heads,
        "NUM_KV_HEADS": args.num_kv_heads,
        "GDN_N_HEADS": args.gdn_n_heads,
        "GDN_EXPAND_V": args.gdn_expand_v,
        "GDN_HEAD_K_DIM": args.gdn_head_k_dim,
        "GDN_CONV_SIZE": args.gdn_conv_size,
        "NORM_STYLE": args.norm_style,
        "RESIDUAL_ALPHA": args.residual_alpha,
        "TIE_EMBEDDINGS": args.tie_embeddings,
        "TRAIN_BATCH_TOKENS": args.train_batch_tokens,
        "TRAIN_SEQ_LEN": args.train_seq_len,
        "GRAD_ACCUM_STEPS": grad_accum_steps,
        "LOCAL_BATCH_SIZE": local_batch_size,
        "ITERATIONS": args.iterations,
        "WARMDOWN_ITERS": args.warmdown_iters,
        "WARMUP_STEPS": args.warmup_steps,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "VAL_LOSS_EVERY": args.val_loss_every,
        "TRAIN_LOG_EVERY": args.train_log_every,
        "MAX_WALLCLOCK_SECONDS": args.max_wallclock_seconds,
        "PLANNED_TRAIN_TOKENS": planned_train_tokens,
        "ARTIFACT_LIMIT_BYTES": args.artifact_limit_bytes,
        "COMPILE": args.compile,
        "COMPILE_STRATEGY": args.compile_strategy,
        "COMPILE_GDN_DISABLED": compile_stats["gdn_disabled"],
        "COMPILE_GDN_MLPS_COMPILED": compile_stats["gdn_mlps_compiled"],
        "COMPILE_ATTN_BLOCKS_COMPILED": compile_stats["attn_blocks_compiled"],
        "COMPILE_MODEL_COMPILED": compile_stats["model_compiled"],
        "SDPA_BACKEND": sdp_backend,
        "CUDNN_BENCHMARK": args.cudnn_benchmark,
        "FLA_AVAILABLE": _HAS_FLA,
        "EMBED_LR": args.embed_lr,
        "HEAD_LR": args.head_lr,
        "TIED_EMBED_LR": args.tied_embed_lr,
        "MATRIX_LR": args.matrix_lr,
        "SCALAR_LR": args.scalar_lr,
        "MUON_MOMENTUM": args.muon_momentum,
        "MUON_BACKEND_STEPS": args.muon_backend_steps,
        "MUON_MOMENTUM_WARMUP_START": args.muon_momentum_warmup_start,
        "MUON_MOMENTUM_WARMUP_STEPS": args.muon_momentum_warmup_steps,
        "BETA1": args.beta1,
        "BETA2": args.beta2,
        "ADAM_EPS": args.adam_eps,
        "GRAD_CLIP_NORM": args.grad_clip_norm,
        "WEIGHT_DECAY": args.weight_decay,
        "GDN_USE_Q_CONV": args.gdn_use_q_conv,
        "GDN_USE_K_CONV": args.gdn_use_k_conv,
        "GDN_USE_V_CONV": args.gdn_use_v_conv,
        "GDN_USE_PACKED_QKV_CONV": args.gdn_use_packed_qkv_conv,
        "GDN_USE_PACKED_QKV_PROJ": args.gdn_use_packed_qkv_proj,
        "GDN_CONV_OUTPUT_CONTIGUOUS": args.gdn_conv_output_contiguous,
        "GDN_Q_CONV_OUTPUT_CONTIGUOUS": args.gdn_q_conv_output_contiguous,
        "GDN_K_CONV_OUTPUT_CONTIGUOUS": args.gdn_k_conv_output_contiguous,
        "GDN_V_CONV_OUTPUT_CONTIGUOUS": args.gdn_v_conv_output_contiguous,
        "GDN_CONTROL_PROJ_FP32": args.gdn_control_proj_fp32,
        "GDN_GATES_FP32": args.gdn_gates_fp32,
        "GDN_OUTPUT_NORM_FP32": args.gdn_output_norm_fp32,
        "GDN_USE_CUDA_FRONTEND_NCT": args.gdn_use_cuda_frontend_nct,
        "GDN_USE_CUDA_PACKED_CONV": args.gdn_use_cuda_packed_conv,
        "GDN_USE_CUDA_PACKED_CONV_ATEN_BACKWARD": args.gdn_use_cuda_packed_conv_aten_backward,
        "GDN_USE_CUDA_PACKED_CONV_ATEN_WEIGHT_BACKWARD": args.gdn_use_cuda_packed_conv_aten_weight_backward,
        "GDN_USE_CUDA_FUSED_FRONTEND": args.gdn_use_cuda_fused_frontend,
        "GDN_USE_CUDA_FUSED_FRONTEND_LIB": args.gdn_use_cuda_fused_frontend_lib,
        "GDN_USE_CUDA_FUSED_OUTPUT": args.gdn_use_cuda_fused_output,
        "GDN_USE_CUDA_SPLIT_NORM": args.gdn_use_cuda_split_norm,
        "GDN_USE_CUDA_SPLIT_NORM_LIB": args.gdn_use_cuda_split_norm_lib,
        "GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD": args.gdn_use_packed_qkv_conv_custom_backward,
        "GDN_PACKED_QKV_SINGLE_CONTIG": args.gdn_use_packed_qkv_single_contig,
        "GDN_PACKED_QKV_SPLIT_COPY": args.gdn_use_packed_qkv_split_copy,
        "WANDB_WATCH": wandb_watch_mode,
        "WANDB_WATCH_LOG_FREQ": args.wandb_watch_log_freq,
    }


def build_profiler(
    args: Hyperparameters,
    run_id: str,
    master_process: bool,
) -> tuple[Any, Path | None, int]:
    """Create a scheduled torch profiler and its output directory.

    :param Hyperparameters args: Trainer hyperparameters.
    :param str run_id: Logical run identifier used for trace paths.
    :param bool master_process: Whether the current rank owns profiler output.
    :return tuple[Any, Path | None, int]: Profiler instance or `None`, output directory, and total scheduled train steps.
    """
    if not args.profile or not master_process:
        return None, None, 0
    if args.profile_active <= 0:
        raise ValueError("PROFILE_ACTIVE must be > 0 when PROFILE=1")
    if args.profile_repeat <= 0:
        raise ValueError("PROFILE_REPEAT must be > 0 when PROFILE=1")
    output_dir = Path(args.profile_dir).resolve() / run_id
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    total_steps = (
        args.profile_wait + args.profile_warmup + args.profile_active
    ) * args.profile_repeat
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
            repeat=args.profile_repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(trace_dir),
            worker_name=run_id,
            use_gzip=True,
        ),
        record_shapes=args.profile_record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.profile_with_stack,
        with_flops=args.profile_with_flops,
        with_modules=args.profile_with_modules,
    )
    return profiler, output_dir, total_steps


# =====================================================================
# MUON OPTIMIZER (from modded-nanogpt via baseline train_gpt.py)
# =====================================================================


def _zeropower_via_newtonschulz5_wide(X: Tensor, steps: int = 10) -> Tensor:
    """Run Newton-Schulz iterations for matrices with rows <= columns.

    :param Tensor X: Normalized matrix update with ``rows <= columns``.
    :param int steps: Iteration count, defaults to 10.
    :return Tensor: Approximate zeroth-power normalized update.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def _zeropower_via_newtonschulz5_tall(X: Tensor, steps: int = 10) -> Tensor:
    """Run Newton-Schulz iterations for matrices with rows > columns.

    :param Tensor X: Normalized matrix update with ``rows > columns``.
    :param int steps: Iteration count, defaults to 10.
    :return Tensor: Approximate zeroth-power normalized update.
    """
    return _zeropower_via_newtonschulz5_wide(X.T, steps=steps).T


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
    """Orthogonalize a matrix gradient with Newton-Schulz iterations.

    Keep the shape-dependent orientation branch outside the compiled helpers so
    Dynamo does not need to specialize on the symbolic ``rows > cols`` guard.

    :param Tensor G: Matrix-shaped gradient update.
    :param int steps: Iteration count, defaults to 10.
    :param float eps: Numerical stability epsilon, defaults to 1e-7.
    :return Tensor: Approximate zeroth-power normalized update.
    """
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        return _zeropower_via_newtonschulz5_tall(X, steps=steps)
    return _zeropower_via_newtonschulz5_wide(X, steps=steps)


class Muon(torch.optim.Optimizer):
    """Minimal Muon optimizer used for matrix-shaped parameters."""

    def __init__(
        self,
        params: list[Tensor],
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ):
        """Initialize Muon.

        :param list[Tensor] params: Matrix parameters assigned to Muon.
        :param float lr: Learning rate.
        :param float momentum: Momentum coefficient.
        :param int backend_steps: Newton-Schulz iteration count.
        :param bool nesterov: Whether to use Nesterov momentum, defaults to True.
        :param float weight_decay: Decoupled weight decay, defaults to 0.0.
        """
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
            ),
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Apply one Muon optimization step.

        :param Optional[Callable[[], Tensor]] closure: Optional reevaluation closure, defaults to None.
        :return Optional[Tensor]: Closure loss when provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# =====================================================================
# DATA LOADING (from baseline)
# =====================================================================


def load_data_shard(file: Path) -> Tensor:
    """Load one tokenized FineWeb shard from disk.

    :param Path file: Path to the `.bin` shard.
    :return Tensor: Flat `uint16` token tensor.
    """
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def resolve_token_files(
    pattern: str, *, missing_message: str | None = None
) -> list[Path]:
    """Resolve a token-shard glob into a sorted file list.

    :param str pattern: Glob pattern for token shards.
    :param str | None missing_message: Optional custom error when no files match, defaults to None.
    :raises FileNotFoundError: If the pattern matches no files.
    :return list[Path]: Sorted shard paths.
    """
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(missing_message or f"No files for: {pattern}")
    return files


class TokenStream:
    """Sequential shard reader that wraps around at the end of the file list."""

    def __init__(self, files: list[Path]):
        """Initialize a streaming shard reader.

        :param list[Path] files: Ordered token shard paths.
        """
        self.files = list(files)
        if not self.files:
            raise ValueError("TokenStream requires at least one shard file")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        """Advance to the next shard, wrapping around when needed."""
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        """Take a contiguous token span across shard boundaries.

        :param int n: Number of tokens to read.
        :return Tensor: Requested token span.
        """
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    """Shard loader that slices each global batch by distributed rank."""

    def __init__(
        self, files: list[Path], rank: int, world_size: int, device: torch.device
    ):
        """Initialize the distributed token loader.

        :param list[Path] files: Ordered token shard paths.
        :param int rank: Distributed rank.
        :param int world_size: Distributed world size.
        :param torch.device device: Target device for batches.
        """
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(files)

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
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# =====================================================================
# EVAL (from baseline — BPB calculation)
# =====================================================================


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Build lookup tables for tokenizer-aware byte counting.

    :param spm.SentencePieceProcessor sp: Loaded SentencePiece tokenizer.
    :param int vocab_size: Expected vocabulary size.
    :param torch.device device: Target device for the lookup tensors.
    :return tuple[Tensor, Tensor, Tensor]: Base bytes, leading-space flags, and boundary flags.
    """
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros(table_size, dtype=np.int16)
    has_leading_space = np.zeros(table_size, dtype=np.bool_)
    is_boundary = np.ones(table_size, dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary, dtype=torch.bool, device=device),
    )


def load_validation_tokens(files: list[Path], seq_len: int) -> Tensor:
    """Load and trim validation tokens to an integer number of sequences.

    :param list[Path] files: Ordered validation shard paths.
    :param int seq_len: Sequence length used for eval reshaping.
    :return Tensor: Concatenated validation tokens.
    """
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_lut: Tensor,
) -> tuple[float, float]:
    """Evaluate validation loss and tokenizer-agnostic BPB.

    :param Hyperparameters args: Trainer hyperparameters.
    :param nn.Module model: Model or DDP-wrapped model.
    :param int rank: Distributed rank.
    :param int world_size: Distributed world size.
    :param torch.device device: Evaluation device.
    :param int grad_accum_steps: Gradient accumulation factor used to derive local batch size.
    :param Tensor val_tokens: Flat validation token buffer.
    :param Tensor base_bytes_lut: Base byte-count lookup table.
    :param Tensor has_leading_space_lut: Leading-space lookup table.
    :param Tensor is_boundary_lut: Boundary-token lookup table.
    :return tuple[float, float]: Validation loss and bits-per-byte.
    """
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            rs, re = bss * args.train_seq_len, bse * args.train_seq_len + 1
            local = val_tokens[rs:re].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                bl = model(x, y).detach()
            btc = float(y.numel())
            val_loss_sum += bl.to(torch.float64) * btc
            val_token_count += btc
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).to(
                torch.int16
            )
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (val_loss_sum, val_token_count, val_byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)


# =====================================================================
# QUANTIZATION (from baseline)
# =====================================================================

INT8_CLIP_Q = 99.99984 / 100.0
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536


def quantize_state_dict_int8(
    state_dict: dict[str, Tensor],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Quantize a state dict with per-row int8 weights where possible.

    :param dict[str, Tensor] state_dict: Floating-point state dict.
    :return tuple[dict[str, Any], dict[str, int]]: Quantized payload object and serialization stats.
    """
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += t.numel() * t.element_size()
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if any(p in name for p in SCALAR_PARAM_PATTERNS):
                kept = t.float().contiguous()
            else:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(torch.float16).contiguous()
            passthrough[name] = kept
            stats["int8_payload_bytes"] += kept.numel() * kept.element_size()
            continue
        t32 = t.float()
        if t32.ndim == 2:
            ca = (
                torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                if t32.numel()
                else torch.empty(t32.shape[0])
            )
            clipped = torch.clamp(t32, -ca[:, None], ca[:, None])
            scale = (ca / 127.0).clamp_min(1 / 127.0)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(
                torch.int8
            )
            scales[name] = scale.to(torch.float16).contiguous()
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        else:
            ca = (
                float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
                if t32.numel()
                else 0.0
            )
            scale = torch.tensor(ca / 127.0 if ca > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(
                torch.round(torch.clamp(t32, -ca, ca) / scale), -127, 127
            ).to(torch.int8)
            scales[name] = scale
        quantized[name] = q.contiguous()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += (
            q.numel() + scales[name].numel() * scales[name].element_size()
        )
    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, Any]) -> dict[str, Tensor]:
    """Restore a quantized state dict payload to dense tensors.

    :param dict[str, Any] obj: Serialized quantized payload.
    :return dict[str, Tensor]: Dequantized state dict.
    """
    out = {}
    qmeta = obj.get("qmeta", {})
    pod = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (
                q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
            ).to(dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(dtype)
    for name, t in obj["passthrough"].items():
        ot = t.detach().cpu().contiguous()
        orig = pod.get(name)
        if isinstance(orig, str):
            ot = ot.to(getattr(torch, orig))
        out[name] = ot
    return out


def serialize_quantized_state_dict_int8(
    state_dict: dict[str, Tensor], zlib_level: int = 9
) -> tuple[dict[str, Any], bytes, dict[str, int | float]]:
    """Quantize, serialize, and compress a state dict with byte-level audit stats.

    :param dict[str, Tensor] state_dict: Floating-point state dict to pack.
    :param int zlib_level: Compression level for the final zlib payload, defaults to 9.
    :return tuple[dict[str, Any], bytes, dict[str, int | float]]: Quantized object, compressed blob, and byte audit metrics.
    """
    quant_obj, quant_stats = quantize_state_dict_int8(state_dict)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=zlib_level)
    quant_raw_bytes = len(quant_raw)
    quant_zlib_bytes = len(quant_blob)
    int8_payload_bytes = quant_stats["int8_payload_bytes"]
    baseline_tensor_bytes = quant_stats["baseline_tensor_bytes"]
    audit = {
        **quant_stats,
        "quant_raw_torch_bytes": quant_raw_bytes,
        "quant_raw_overhead_bytes": quant_raw_bytes - int8_payload_bytes,
        "quant_zlib_bytes": quant_zlib_bytes,
        "baseline_to_payload_ratio": baseline_tensor_bytes / max(int8_payload_bytes, 1),
        "payload_to_zlib_ratio": int8_payload_bytes / max(quant_zlib_bytes, 1),
        "raw_quant_to_zlib_ratio": quant_raw_bytes / max(quant_zlib_bytes, 1),
    }
    return quant_obj, quant_blob, audit


# =====================================================================
# UTILITY
# =====================================================================


def restore_low_dim_params_to_fp32(
    module: nn.Module, *, gdn_control_proj_fp32: bool = True
) -> None:
    """Restore low-dimensional or scalar parameters to fp32.

    :param nn.Module module: Module whose parameters should be restored in place.
    :param bool gdn_control_proj_fp32: Whether to also restore HGDN control
        projections `w_a/w_b/w_g` to fp32, defaults to True.
    """
    gdn_control_patterns = ("w_a.", "w_b.", "w_g.")
    with torch.no_grad():
        for name, param in module.named_parameters():
            if not gdn_control_proj_fp32 and any(
                pattern in name for pattern in gdn_control_patterns
            ):
                continue
            if (
                param.ndim < 2 or any(p in name for p in SCALAR_PARAM_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


def maybe_compile(
    obj: Any, *, enabled: bool, dynamic: bool = False, fullgraph: bool = False
) -> Any:
    """Conditionally wrap an object with `torch.compile`.

    Uses ``nn.Module.compile()`` for top-level modules to match the current
    PyTorch guidance, and falls back to ``torch.compile`` for plain callables.

    :param Any obj: Callable or module to compile.
    :param bool enabled: Whether compilation is enabled.
    :param bool dynamic: Whether to enable dynamic shapes, defaults to False.
    :param bool fullgraph: Whether to require full-graph capture, defaults to False.
    :return Any: Compiled object when enabled, otherwise the original object.
    """
    if not enabled:
        return obj
    if isinstance(obj, nn.Module):
        obj.compile(dynamic=dynamic, fullgraph=fullgraph)
        return obj
    return torch.compile(obj, dynamic=dynamic, fullgraph=fullgraph)


def maybe_disable_compile(obj: Any, *, enabled: bool, reason: str | None = None) -> Any:
    """Conditionally mark an object as eager-only for `torch.compile`.

    :param Any obj: Callable or module to exclude from Dynamo tracing.
    :param bool enabled: Whether compilation is enabled.
    :param str | None reason: Optional reason string for debugging, defaults to None.
    :return Any: Wrapped object when enabled, otherwise the original object.
    """
    if not enabled:
        return obj
    return torch.compiler.disable(obj, reason=reason)


def prepare_hybrid_compile(
    model: HybridGPT, *, enabled: bool, strategy: str
) -> tuple[nn.Module, dict[str, int | str]]:
    """Apply selective compile boundaries for the hybrid stack.

    `hybrid` is the default and most aggressive safe option on torch 2.11:
    keep GDN layers eager because FLA already dispatches Triton kernels, compile
    pure attention blocks full-graph, compile GDN-block MLPs full-graph, then
    compile the top-level model with `fullgraph=False` so the remaining eager
    GDN boundaries become clean graph breaks instead of trace-time failures.

    :param HybridGPT model: Hybrid language model to prepare.
    :param bool enabled: Whether compilation is enabled.
    :param str strategy: One of `model`, `selective`, or `hybrid`.
    :raises ValueError: If `strategy` is not recognized.
    :return tuple[nn.Module, dict[str, int | str]]: Prepared model and compile stats.
    """
    compile_stats: dict[str, int | str] = {
        "strategy": "off" if not enabled else strategy,
        "gdn_disabled": 0,
        "gdn_mlps_compiled": 0,
        "attn_blocks_compiled": 0,
        "model_compiled": 0,
    }
    if not enabled:
        return model, compile_stats

    if strategy not in {"model", "selective", "hybrid"}:
        raise ValueError(
            f"Unsupported COMPILE_STRATEGY={strategy!r}; expected model, selective, or hybrid"
        )

    if strategy in {"selective", "hybrid"}:
        for idx, block_type in enumerate(model.block_types):
            block = model.blocks[idx]
            if block_type == "gdn" and hasattr(block, "gdn") and hasattr(block, "mlp"):
                block.gdn = maybe_disable_compile(
                    block.gdn,
                    enabled=True,
                    reason="FLA GDN path already dispatches Triton kernels",
                )
                block.mlp = maybe_compile(
                    block.mlp, enabled=True, dynamic=False, fullgraph=True
                )
                compile_stats["gdn_disabled"] += 1
                compile_stats["gdn_mlps_compiled"] += 1
            elif block_type == "attn":
                model.blocks[idx] = maybe_compile(
                    block, enabled=True, dynamic=False, fullgraph=True
                )
                compile_stats["attn_blocks_compiled"] += 1

    if strategy in {"model", "hybrid"}:
        model = maybe_compile(model, enabled=True, dynamic=False, fullgraph=False)
        compile_stats["model_compiled"] = 1

    return model, compile_stats


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    """Run hybrid training, validation, and roundtrip artifact checks."""
    global _zeropower_via_newtonschulz5_tall
    global _zeropower_via_newtonschulz5_wide
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    wandb_watch_mode = normalize_wandb_watch_mode(args.wandb_watch)
    _zeropower_via_newtonschulz5_wide = maybe_compile(
        _zeropower_via_newtonschulz5_wide, enabled=args.compile, dynamic=True
    )
    _zeropower_via_newtonschulz5_tall = maybe_compile(
        _zeropower_via_newtonschulz5_tall, enabled=args.compile, dynamic=True
    )
    zeropower_via_newtonschulz5 = maybe_compile(
        zeropower_via_newtonschulz5, enabled=False, dynamic=True
    )

    # ── Distributed + CUDA ────────────────────────────────────────────
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(
            f"WORLD_SIZE must evenly divide 8, got WORLD_SIZE={world_size}"
        )
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    batch_denom = world_size * grad_accum_steps * args.train_seq_len
    if args.train_batch_tokens % batch_denom != 0:
        raise ValueError(
            "TRAIN_BATCH_TOKENS must be divisible by WORLD_SIZE * GRAD_ACCUM_STEPS * "
            f"TRAIN_SEQ_LEN, got {args.train_batch_tokens=} {world_size=} "
            f"{grad_accum_steps=} {args.train_seq_len=}"
        )
    local_batch_size = args.train_batch_tokens // batch_denom
    planned_train_tokens = args.train_batch_tokens * args.iterations

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    sdp_backend = "flash"

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt" if master_process else None

    def log0(msg: str, console: bool = True) -> None:
        """Log a line on rank 0 to stdout and the run logfile.

        :param str msg: Message to log.
        :param bool console: Whether to print to stdout, defaults to True.
        """
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    profiler, profile_output_dir, profile_total_steps = build_profiler(
        args, args.run_id, master_process
    )
    profile_ctx: Callable[[str], Any]
    profile_ctx = (
        torch.autograd.profiler.record_function
        if args.profile
        else lambda _n: nullcontext()
    )

    # ── wandb init ────────────────────────────────────────────────────
    if master_process and _USE_WANDB:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "pg-hgdn-ablations"),
            name=args.run_id,
            tags=[
                f"gdn_ratio={args.gdn_ratio}",
                f"layers={args.num_layers}",
                f"gdn_nh={args.gdn_n_heads}",
                f"seq={args.train_seq_len}",
            ],
        )

    log0(code, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)
    log0(
        f"compile:{int(args.compile)} compile_strategy:{args.compile_strategy} "
        f"seed:{args.seed} pythonhashseed:{os.environ.get('PYTHONHASHSEED', '<unset>')} "
        f"sdp_backend:{sdp_backend} fla_available:{int(_HAS_FLA)} "
        f"cudnn_benchmark:{int(args.cudnn_benchmark)} "
        f"wandb_watch:{wandb_watch_mode} wandb_watch_log_freq:{args.wandb_watch_log_freq} "
        f"perf_timing:{int(args.perf_timing)} perf_ignore_steps:{args.perf_ignore_steps} "
        f"perf_skip_final_eval:{int(args.perf_skip_final_eval)} "
        f"profile:{int(args.profile)} "
        f"gdn_convs:q={int(args.gdn_use_q_conv)} k={int(args.gdn_use_k_conv)} "
        f"v={int(args.gdn_use_v_conv)} "
        f"packed_conv={int(args.gdn_use_packed_qkv_conv)} "
        f"packed_proj={int(args.gdn_use_packed_qkv_proj)} "
        f"packed_custom_bwd={int(args.gdn_use_packed_qkv_conv_custom_backward)} "
        f"packed_single_contig={int(args.gdn_use_packed_qkv_single_contig)} "
        f"packed_split_copy={int(args.gdn_use_packed_qkv_split_copy)} "
        f"output_contiguous={int(args.gdn_conv_output_contiguous)} "
        f"q_output_contiguous={int(args.gdn_q_conv_output_contiguous)} "
        f"k_output_contiguous={int(args.gdn_k_conv_output_contiguous)} "
        f"v_output_contiguous={int(args.gdn_v_conv_output_contiguous)} "
        f"control_proj_fp32={int(args.gdn_control_proj_fp32)} "
        f"gates_fp32={int(args.gdn_gates_fp32)} "
        f"output_norm_fp32={int(args.gdn_output_norm_fp32)} "
        f"cuda_frontend_nct={int(args.gdn_use_cuda_frontend_nct)} "
        f"cuda_packed_conv={int(args.gdn_use_cuda_packed_conv)} "
        f"cuda_packed_conv_aten_bwd={int(args.gdn_use_cuda_packed_conv_aten_backward)} "
        f"cuda_packed_conv_aten_weight_bwd={int(args.gdn_use_cuda_packed_conv_aten_weight_backward)} "
        f"cuda_fused_frontend={int(args.gdn_use_cuda_fused_frontend)} "
        f"cuda_fused_frontend_lib={int(args.gdn_use_cuda_fused_frontend_lib)} "
        f"cuda_fused_output={int(args.gdn_use_cuda_fused_output)} "
        f"cuda_split_norm={int(args.gdn_use_cuda_split_norm)} "
        f"cuda_split_norm_lib={int(args.gdn_use_cuda_split_norm_lib)}",
        console=False,
    )
    if args.profile:
        log0(
            f"profile_plan:dir:{profile_output_dir} wait:{args.profile_wait} "
            f"warmup:{args.profile_warmup} active:{args.profile_active} "
            f"repeat:{args.profile_repeat} total_steps:{profile_total_steps} "
            f"record_shapes:{int(args.profile_record_shapes)} "
            f"profile_memory:{int(args.profile_memory)} with_stack:{int(args.profile_with_stack)} "
            f"with_flops:{int(args.profile_with_flops)} "
            f"with_modules:{int(args.profile_with_modules)} "
            f"stop_on_complete:{int(args.profile_stop_on_complete)}"
        )

    # ── Tokenizer + data ──────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(
            f"Script only supports SentencePiece .model tokenizers: {args.tokenizer_path}"
        )
    tokenizer_path = Path(args.tokenizer_path).resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            "Tokenizer file not found. Download the published sp1024 assets with:\n"
            "  python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1\n"
            f"Expected tokenizer at: {tokenizer_path}"
        )
    dataset_dir = Path(args.data_path).resolve()
    missing_data_message = (
        "FineWeb shards not found for the hybrid trainer. Download the published "
        "sp1024 assets with:\n"
        "  python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1\n"
        f"Expected dataset under: {dataset_dir}"
    )
    train_files = resolve_token_files(
        args.train_files, missing_message=missing_data_message
    )
    val_files = resolve_token_files(
        args.val_files, missing_message=missing_data_message
    )

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    val_tokens = load_validation_tokens(val_files, args.train_seq_len)
    base_bytes_lut, has_ls_lut, is_bnd_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(
        f"train_loader:dataset:{dataset_dir.name} train_shards:{len(train_files)} "
        f"val_shards:{len(val_files)}"
    )
    log0(
        f"launch_contract:planned_train_tokens:{planned_train_tokens} "
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"grad_accum_steps:{grad_accum_steps} local_batch_size:{local_batch_size}"
    )

    # ── Model ─────────────────────────────────────────────────────────
    base_model = (
        HybridGPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            d_model=args.model_dim,
            attn_heads=args.num_heads,
            attn_kv_heads=args.num_kv_heads,
            gdn_n_heads=args.gdn_n_heads,
            gdn_head_k_dim=args.gdn_head_k_dim,
            gdn_expand_v=args.gdn_expand_v,
            gdn_allow_neg_eigval=args.gdn_allow_neg_eigval,
            gdn_conv_size=args.gdn_conv_size,
            gdn_use_q_conv=args.gdn_use_q_conv,
            gdn_use_k_conv=args.gdn_use_k_conv,
            gdn_use_v_conv=args.gdn_use_v_conv,
            gdn_use_packed_qkv_conv=args.gdn_use_packed_qkv_conv,
            gdn_use_packed_qkv_proj=args.gdn_use_packed_qkv_proj,
            gdn_conv_output_contiguous=args.gdn_conv_output_contiguous,
            gdn_q_conv_output_contiguous=args.gdn_q_conv_output_contiguous,
            gdn_k_conv_output_contiguous=args.gdn_k_conv_output_contiguous,
            gdn_v_conv_output_contiguous=args.gdn_v_conv_output_contiguous,
            gdn_gates_fp32=args.gdn_gates_fp32,
            gdn_output_norm_fp32=args.gdn_output_norm_fp32,
            gdn_use_cuda_frontend_nct=args.gdn_use_cuda_frontend_nct,
            gdn_use_cuda_packed_conv=args.gdn_use_cuda_packed_conv,
            gdn_use_cuda_packed_conv_aten_backward=args.gdn_use_cuda_packed_conv_aten_backward,
            gdn_use_cuda_packed_conv_aten_weight_backward=args.gdn_use_cuda_packed_conv_aten_weight_backward,
            gdn_use_cuda_fused_frontend=args.gdn_use_cuda_fused_frontend,
            gdn_use_cuda_fused_frontend_lib=args.gdn_use_cuda_fused_frontend_lib,
            gdn_use_cuda_fused_output=args.gdn_use_cuda_fused_output,
            gdn_use_cuda_split_norm=args.gdn_use_cuda_split_norm,
            gdn_use_cuda_split_norm_lib=args.gdn_use_cuda_split_norm_lib,
            gdn_use_packed_qkv_conv_custom_backward=args.gdn_use_packed_qkv_conv_custom_backward,
            gdn_use_packed_qkv_single_contig=args.gdn_use_packed_qkv_single_contig,
            gdn_use_packed_qkv_split_copy=args.gdn_use_packed_qkv_split_copy,
            mlp_mult=args.mlp_mult,
            leaky_slope=args.leaky_slope,
            gdn_ratio=args.gdn_ratio,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            logit_softcap=args.logit_softcap,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            norm_style=args.norm_style,
            residual_alpha=args.residual_alpha,
        )
        .to(device)
        .bfloat16()
    )

    restore_low_dim_params_to_fp32(
        base_model, gdn_control_proj_fp32=args.gdn_control_proj_fp32
    )
    compiled_model, compile_stats = prepare_hybrid_compile(
        base_model, enabled=args.compile, strategy=args.compile_strategy
    )
    model = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    n_params = sum(p.numel() for p in base_model.parameters())
    n_gdn = sum(1 for t in base_model.block_types if t == "gdn")
    n_attn = sum(1 for t in base_model.block_types if t == "attn")
    log0(
        f"model_params:{n_params} blocks:{n_gdn}G+{n_attn}A "
        f"norm_style:{base_model.norm_style} residual_alpha:{base_model.residual_alpha:g}"
    )
    log0(
        "compile_plan:"
        f"strategy:{compile_stats['strategy']} "
        f"gdn_disabled:{compile_stats['gdn_disabled']} "
        f"gdn_mlps_compiled:{compile_stats['gdn_mlps_compiled']} "
        f"attn_blocks_compiled:{compile_stats['attn_blocks_compiled']} "
        f"model_compiled:{compile_stats['model_compiled']}"
    )
    log0(
        f"world_size:{world_size} grad_accum_steps:{grad_accum_steps} "
        f"sdp_backends:cudnn=False flash=True mem_efficient=False math=False"
    )

    if master_process and _USE_WANDB:
        wandb.config.update(
            build_wandb_config(
                args,
                compile_stats=compile_stats,
                local_batch_size=local_batch_size,
                grad_accum_steps=grad_accum_steps,
                planned_train_tokens=planned_train_tokens,
                n_attn=n_attn,
                n_gdn=n_gdn,
                n_params=n_params,
                sdp_backend=sdp_backend,
                wandb_watch_mode=wandb_watch_mode,
            )
        )

    # ── Optimizers ────────────────────────────────────────────────────
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for n, p in block_named_params
        if p.ndim == 2 and not any(pat in n for pat in SCALAR_PARAM_PATTERNS)
    ]
    scalar_params = [
        p
        for n, p in block_named_params
        if p.ndim < 2 or any(pat in n for pat in SCALAR_PARAM_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [
            {
                "params": [base_model.tok_emb.weight],
                "lr": token_lr,
                "base_lr": token_lr,
            }
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.weight_decay,
    )
    for g in optimizer_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [
                {
                    "params": [base_model.lm_head.weight],
                    "lr": args.head_lr,
                    "base_lr": args.head_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    # ── Data loader & warmup ──────────────────────────────────────────
    train_loader = DistributedTokenLoader(train_files, rank, world_size, device)
    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def zero_grad_all() -> None:
        """Clear gradients across all optimizer groups."""
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def lr_mul(step: int, elapsed_ms: float) -> float:
        """Compute the warmdown multiplier for the current step.

        :param int step: Completed optimizer steps.
        :param float elapsed_ms: Elapsed training time in milliseconds.
        :return float: Learning-rate scale multiplier.
        """
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if ws <= step < args.iterations
                else 1.0
            )
        sms = elapsed_ms / max(step, 1)
        wms = args.warmdown_iters * sms
        rms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

    # Warmup (compile priming)
    if args.warmup_steps > 0:
        init_state = {
            n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()
        }
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for o in optimizers:
                o.step()
            zero_grad_all()
            if ws + 1 == args.warmup_steps or (ws + 1) % 10 == 0:
                log0(f"warmup_step:{ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_state, strict=True)
        for o, s in zip(optimizers, init_opt):
            o.load_state_dict(s)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(train_files, rank, world_size, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if master_process and _USE_WANDB and wandb_watch_mode != "none":
        wandb.watch(
            base_model,
            log=wandb_watch_mode,
            log_freq=args.wandb_watch_log_freq,
            log_graph=False,
        )

    # ── Training loop ─────────────────────────────────────────────────
    training_time_ms = 0.0
    perf_time_ms = 0.0
    perf_measured_steps = 0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    if profiler is not None:
        profiler.__enter__()

    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )
        should_validate = (
            (last_step and not args.perf_skip_final_eval)
            or (step == 0 and args.log_step0_eval)
            or (
                args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0
            )
        )

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            with profile_ctx("eval.val"):
                val_loss, val_bpb = eval_val(
                    args,
                    model,
                    rank,
                    world_size,
                    device,
                    grad_accum_steps,
                    val_tokens,
                    base_bytes_lut,
                    has_ls_lut,
                    is_bnd_lut,
                )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms"
            )
            if master_process and _USE_WANDB:
                wandb.log({"eval/loss": val_loss, "eval/bpb": val_bpb}, step=step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        perf_t0 = time.perf_counter() if args.perf_timing else None

        with profile_ctx("train.step"):
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                with profile_ctx("train.load_batch"):
                    x, y = train_loader.next_batch(
                        args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                    )
                with profile_ctx("train.forward_backward"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(x, y)
                    train_loss += loss.detach()
                    (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        mm = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in optimizer_muon.param_groups:
            g["momentum"] = mm

        with profile_ctx("train.optimizer_step"):
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * scale
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), args.grad_clip_norm
                )
            for opt in optimizers:
                opt.step()
            zero_grad_all()

        step += 1
        perf_step_ms = float("nan")
        perf_tokens_per_s = float("nan")
        if args.perf_timing:
            torch.cuda.synchronize()
            assert perf_t0 is not None
            measured_ms = 1000.0 * (time.perf_counter() - perf_t0)
            if step > args.perf_ignore_steps:
                perf_time_ms += measured_ms
                perf_measured_steps += 1
            if perf_measured_steps > 0:
                perf_step_ms = perf_time_ms / perf_measured_steps
                perf_tokens_per_s = args.train_batch_tokens / max(
                    perf_step_ms / 1000.0, 1e-9
                )
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        step_ms = approx_ms / step
        tokens_per_s = args.train_batch_tokens / max(step_ms / 1000.0, 1e-9)
        processed_tokens = step * args.train_batch_tokens
        should_log = args.train_log_every > 0 and (
            step <= 10
            or step % args.train_log_every == 0
            or stop_after_step is not None
        )
        if should_log:
            perf_suffix = ""
            if args.perf_timing:
                perf_suffix = (
                    f" perf_avg:{perf_step_ms:.2f}ms"
                    if perf_measured_steps > 0
                    else f" perf_avg:pending(ignore<{args.perf_ignore_steps + 1})"
                )
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_ms:.0f}ms step_avg:{step_ms:.2f}ms{perf_suffix}"
            )
            if master_process and _USE_WANDB:
                metrics = {
                    "train/lr": args.matrix_lr * scale,
                    "train/lr_scale": scale,
                    "train/loss": train_loss.item(),
                    "train/processed_tokens": processed_tokens,
                    "train/step_ms": perf_step_ms
                    if args.perf_timing and perf_measured_steps > 0
                    else step_ms,
                    "train/tokens_per_s": perf_tokens_per_s
                    if args.perf_timing and perf_measured_steps > 0
                    else tokens_per_s,
                }
                wandb.log(metrics, step=step)

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
        if profiler is not None:
            profiler.step()
            if args.profile_stop_on_complete and step >= profile_total_steps:
                stop_after_step = step

    if profiler is not None:
        profiler.__exit__(None, None, None)
        if profile_output_dir is not None:
            profile_rows = build_profile_rows(profiler.key_averages())
            report = build_profile_report(
                profile_rows,
                sort_by=args.profile_sort_by,
                metadata={
                    "run_id": args.run_id,
                    "profile_output_dir": str(profile_output_dir),
                    "profile_row_limit": args.profile_row_limit,
                    "profile_total_steps": profile_total_steps,
                    "train_batch_tokens": args.train_batch_tokens,
                    "train_seq_len": args.train_seq_len,
                    "model_dim": args.model_dim,
                    "num_layers": args.num_layers,
                    "gdn_ratio": args.gdn_ratio,
                    "mlp_mult": args.mlp_mult,
                    "compile": args.compile,
                    "compile_strategy": args.compile_strategy,
                },
            )
            write_profile_report(
                profile_output_dir,
                report=report,
            )
            log0(f"profile_summary:{profile_output_dir / 'key_averages.json'}")
            if master_process and _USE_WANDB:
                wandb.summary["profile_dir"] = str(profile_output_dir)

    peak_mem_alloc_mib = torch.cuda.max_memory_allocated() // 1024 // 1024
    peak_mem_reserved_mib = torch.cuda.max_memory_reserved() // 1024 // 1024
    log0(
        f"peak memory allocated: {peak_mem_alloc_mib} MiB reserved: {peak_mem_reserved_mib} MiB"
    )
    if args.perf_timing and perf_measured_steps > 0:
        perf_step_ms_final = perf_time_ms / perf_measured_steps
        perf_tokens_per_s_final = args.train_batch_tokens / max(
            perf_step_ms_final / 1000.0, 1e-9
        )
        log0(
            f"perf_summary ignore_steps:{args.perf_ignore_steps} measured_steps:{perf_measured_steps} "
            f"step_ms:{perf_step_ms_final:.2f} tokens_per_s:{perf_tokens_per_s_final:.2f}"
        )
        if master_process and _USE_WANDB:
            wandb.summary["perf_ignore_steps_final"] = args.perf_ignore_steps
            wandb.summary["perf_measured_steps_final"] = perf_measured_steps
            wandb.summary["perf_step_ms_final"] = perf_step_ms_final
            wandb.summary["perf_tokens_per_s_final"] = perf_tokens_per_s_final
    if args.perf_skip_final_eval:
        log0("perf_mode: skipping serialization and final roundtrip eval")
        if master_process and _USE_WANDB:
            wandb.summary["perf_skip_final_eval"] = True
            wandb.finish()
        if distributed:
            dist.destroy_process_group()
        return

    # ── Serialize + quantize + roundtrip ──────────────────────────────
    qfb = None
    raw_model_bytes = None
    code_bytes = len(code.encode("utf-8"))
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        raw_model_bytes = os.path.getsize("final_model.pt")
        log0(f"raw_model_bytes:{raw_model_bytes}")

    quant_obj, quant_blob, quant_audit = serialize_quantized_state_dict_int8(
        base_model.state_dict()
    )
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        qfb = len(quant_blob)
        log0(
            "quant_audit "
            f"baseline_tensor_bytes:{quant_audit['baseline_tensor_bytes']} "
            f"int8_payload_bytes:{quant_audit['int8_payload_bytes']} "
            f"raw_torch_bytes:{quant_audit['quant_raw_torch_bytes']} "
            f"raw_torch_overhead_bytes:{quant_audit['quant_raw_overhead_bytes']} "
            f"baseline_to_payload_ratio:{quant_audit['baseline_to_payload_ratio']:.2f}x "
            f"payload_to_zlib_ratio:{quant_audit['payload_to_zlib_ratio']:.2f}x "
            f"raw_quant_to_zlib_ratio:{quant_audit['raw_quant_to_zlib_ratio']:.2f}x"
        )
        log0(f"int8_zlib_bytes:{qfb} code_bytes:{code_bytes} total:{qfb + code_bytes}")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    torch.cuda.synchronize()
    t_q = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_ls_lut,
        is_bnd_lut,
    )
    torch.cuda.synchronize()
    q_eval_ms = 1000 * (time.perf_counter() - t_q)
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{q_eval_ms:.0f}ms"
    )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}"
    )

    artifact_bytes = None
    artifact_headroom = None
    artifact_status = None
    artifact_warning = None
    if master_process:
        artifact_bytes = qfb + code_bytes
        artifact_headroom = args.artifact_limit_bytes - artifact_bytes
        if artifact_headroom < 0:
            artifact_status = "OVER_LIMIT"
            artifact_warning = "DISQUALIFIED"
        elif artifact_headroom > 0:
            artifact_status = "UNDER_LIMIT"
            artifact_warning = "LEFT_ON_TABLE"
        else:
            artifact_status = "ON_BUDGET"
            artifact_warning = ""
        log0(
            f"artifact_status:{artifact_status} artifact_warning:{artifact_warning or 'NONE'} "
            f"headroom_bytes:{artifact_headroom}"
        )

    if master_process and _USE_WANDB:
        wandb.summary["system/peak_mem_alloc_mib"] = peak_mem_alloc_mib
        wandb.summary["system/peak_mem_reserved_mib"] = peak_mem_reserved_mib
        wandb.summary["roundtrip_val_loss_final"] = q_val_loss
        wandb.summary["roundtrip_val_bpb_final"] = q_val_bpb
        wandb.summary["roundtrip_eval_time_ms_final"] = q_eval_ms
        wandb.summary["artifact_status_final"] = artifact_status
        wandb.summary["artifact_warning_final"] = artifact_warning
        wandb.summary["artifact_bytes_final"] = artifact_bytes
        wandb.summary["artifact_headroom_bytes_final"] = artifact_headroom
        wandb.summary["artifact/raw_state_dict_bytes_final"] = raw_model_bytes
        wandb.summary["artifact/quant_baseline_tensor_bytes_final"] = quant_audit[
            "baseline_tensor_bytes"
        ]
        wandb.summary["artifact/quant_int8_payload_bytes_final"] = quant_audit[
            "int8_payload_bytes"
        ]
        wandb.summary["artifact/quant_raw_torch_bytes_final"] = quant_audit[
            "quant_raw_torch_bytes"
        ]
        wandb.summary["artifact/quant_raw_torch_overhead_bytes_final"] = quant_audit[
            "quant_raw_overhead_bytes"
        ]
        wandb.summary["artifact/quant_baseline_to_payload_ratio_final"] = quant_audit[
            "baseline_to_payload_ratio"
        ]
        wandb.summary["artifact/quant_payload_to_zlib_ratio_final"] = quant_audit[
            "payload_to_zlib_ratio"
        ]
        wandb.summary["artifact/quant_raw_to_zlib_ratio_final"] = quant_audit[
            "raw_quant_to_zlib_ratio"
        ]
        wandb.summary["artifact/code_bytes_final"] = code_bytes
        wandb.summary["artifact/int8_payload_zlib_bytes_final"] = qfb
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
