"""
train_gpt_hybrid.py — Parameter Golf submission with GDN Hybrid model.
Drop-in replacement for train_gpt.py. Same data loading, eval, quantization,
DDP, Muon optimizer. Adds wandb logging + GDN-specific hyperparameters.

Usage (single GPU):
  RUN_ID=hybrid_test torchrun --standalone --nproc_per_node=1 train_gpt_hybrid.py

Usage (8xH100):
  RUN_ID=hybrid_8gpu torchrun --standalone --nproc_per_node=8 train_gpt_hybrid.py

Config-driven runs:
  RUN_ID=hybrid_test NUM_LAYERS=8 MODEL_DIM=512 ... torchrun ...
"""

from __future__ import annotations

import copy
import gc
import io
import os
import random
import sys
import time
import uuid
import zlib
import atexit
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from hgdn_runtime_utils import (
    dequantize_state_dict_int8,
    maybe_compile,
    normalize_wandb_watch_mode,
    prepare_hybrid_compile,
    restore_low_dim_params_to_fp32,
    serialize_quantized_state_dict_int8,
)
from model import (
    _HAS_FLA,
    GDN_FLA_RECURRENCE_MODES,
    HybridGPT,
    attention_backend_name,
    is_gdn_conv_param,
    is_gdn_decay_param,
    uses_scalar_optimizer,
    validate_gdn_w_g_optimizer,
)
from profiler_report import (
    build_profile_report,
    build_profile_rows,
    write_profile_report,
)
from trainer_shared import (
    DistributedTokenLoader,
    build_sentencepiece_luts,
    eval_val,
    load_validation_tokens_from_files,
    make_stream_sync_event,
    resolve_glob_files,
    stage_eval_batch_views,
    wait_current_stream,
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
    min_val_seqs = int(os.environ.get("MIN_VAL_SEQS", 1))
    val_max_seqs = int(os.environ.get("VAL_MAX_SEQS", 0))
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
    compile_strategy = os.environ.get("COMPILE_STRATEGY", "hybrid").lower()
    distributed_mode = os.environ.get("DISTRIBUTED_MODE", "ddp").lower()
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
    attn_use_flash_attn3 = bool(int(os.environ.get("ATTN_USE_FLASH_ATTN3", "1")))

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
    gdn_use_packed_qkv_conv_custom_backward = bool(
        int(os.environ.get("GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD", "0"))
    )
    gdn_use_packed_qkv_single_contig = bool(
        int(os.environ.get("GDN_PACKED_QKV_SINGLE_CONTIG", "0"))
    )
    gdn_use_packed_qkv_split_copy = bool(
        int(os.environ.get("GDN_PACKED_QKV_SPLIT_COPY", "0"))
    )
    gdn_fla_recurrence_mode = (
        "direct_fused"
        if bool(int(os.environ.get("GDN_USE_DIRECT_FLA_LAYER_SEMANTICS", "0")))
        else os.environ.get("GDN_FLA_RECURRENCE_MODE", "direct").lower()
    )
    cudnn_benchmark = bool(int(os.environ.get("CUDNN_BENCHMARK", "0")))
    gdn_ratio = int(os.environ.get("GDN_RATIO", 3))  # 3 GDN : 1 Attn
    block_pattern = os.environ.get("BLOCK_PATTERN", "").strip()
    gdn_w_g_optimizer = os.environ.get("GDN_W_G_OPTIMIZER", "scalar").lower()

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
    muon_distributed_mode = os.environ.get(
        "MUON_DISTRIBUTED_MODE", "packed_allreduce"
    ).lower()


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
    attention_backend: str,
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
    :param str attention_backend: Active standard-attention kernel label.
    :param str wandb_watch_mode: Normalized W&B watch mode.
    :return dict[str, Any]: Canonical HGDN W&B config mapping.
    """
    return {
        "NUM_LAYERS": args.num_layers,
        "MODEL_DIM": args.model_dim,
        "MLP_MULT": args.mlp_mult,
        "GDN_RATIO": args.gdn_ratio,
        "BLOCK_PATTERN": args.block_pattern or None,
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
        "DISTRIBUTED_MODE": args.distributed_mode,
        "MUON_DISTRIBUTED_MODE": args.muon_distributed_mode,
        "GDN_W_G_OPTIMIZER": args.gdn_w_g_optimizer,
        "COMPILE_GDN_DISABLED": compile_stats["gdn_disabled"],
        "COMPILE_GDN_FLA_BLOCKS_COMPILED": compile_stats["gdn_fla_blocks_compiled"],
        "COMPILE_GDN_MLPS_COMPILED": compile_stats["gdn_mlps_compiled"],
        "COMPILE_ATTN_BLOCKS_COMPILED": compile_stats["attn_blocks_compiled"],
        "COMPILE_MODEL_COMPILED": compile_stats["model_compiled"],
        "SDPA_BACKEND": sdp_backend,
        "ATTENTION_BACKEND": attention_backend,
        "CUDNN_BENCHMARK": args.cudnn_benchmark,
        "FLA_AVAILABLE": _HAS_FLA,
        "GDN_FLA_RECURRENCE_MODE": args.gdn_fla_recurrence_mode,
        "ATTN_USE_FLASH_ATTN3": args.attn_use_flash_attn3,
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


def prewarm_muon_backend_shapes(params: list[Tensor], *, backend_steps: int) -> int:
    """Precompile Muon Newton-Schulz helpers on the live matrix-shape family.

    This keeps shape-family compilation out of the real training loop by running
    one representative call per unique 2D parameter shape before the training
    timer starts.

    :param list[Tensor] params: Matrix parameters assigned to Muon.
    :param int backend_steps: Newton-Schulz iteration count.
    :return int: Number of unique matrix shapes prewarmed.
    """
    shape_to_param: dict[tuple[int, int], Tensor] = {}
    for param in params:
        if param.ndim == 2:
            shape_to_param.setdefault((int(param.shape[0]), int(param.shape[1])), param)
    ordered = sorted(
        shape_to_param.items(),
        key=lambda item: (
            item[0][0] == item[0][1],
            item[0][0] > item[0][1],
            item[0][0],
            item[0][1],
        ),
    )
    count = 0
    with torch.no_grad():
        for shape, param in ordered:
            sample = torch.randn(shape, device=param.device, dtype=param.dtype)
            _ = zeropower_via_newtonschulz5(sample, steps=backend_steps)
            count += 1
    if params and params[0].is_cuda:
        torch.cuda.synchronize()
    return count


def prewarm_rotary_caches(
    module: nn.Module, *, seq_len: int, dtype: torch.dtype
) -> int:
    """Populate attention rotary caches before model compilation.

    This avoids the lazy `_cos/_sin` cache mutation from landing inside the
    compiled training forward the first time attention runs.

    :param nn.Module module: Module tree containing attention rotary modules.
    :param int seq_len: Sequence length to materialize.
    :param torch.dtype dtype: Target cache dtype.
    :return int: Number of rotary modules prewarmed.
    """
    try:
        device = next(module.parameters()).device
    except StopIteration:
        return 0
    count = 0
    for submodule in module.modules():
        rotary = getattr(submodule, "rotary", None)
        if rotary is None or not callable(rotary):
            continue
        head_dim = getattr(submodule, "head_dim", None)
        if head_dim is None:
            continue
        rotary(seq_len, device, dtype)
        count += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    return count


class Muon(torch.optim.Optimizer):
    """Parallel-capable Muon for matrix-shaped parameters.

    In distributed runs this can overlap bank reduce-scatter/all-gather with the
    scalar/token Adam steps when the trainer uses the no-DDP distributed path.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        distributed_mode: str = "packed_allreduce",
    ):
        """Initialize Muon.

        :param list[Tensor] params: Matrix parameters assigned to Muon.
        :param float lr: Learning rate.
        :param float momentum: Momentum coefficient.
        :param int backend_steps: Newton-Schulz iteration count.
        :param bool nesterov: Whether to use Nesterov momentum, defaults to True.
        :param float weight_decay: Decoupled weight decay, defaults to 0.0.
        :param str distributed_mode: Distributed update mode. Either
            `"packed_allreduce"` or `"sharded_rsag"`, defaults to
            `"packed_allreduce"`.
        """
        if distributed_mode not in {"packed_allreduce", "sharded_rsag"}:
            raise ValueError(
                "distributed_mode must be 'packed_allreduce' or 'sharded_rsag', "
                f"got {distributed_mode!r}"
            )
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
        self._built = False
        self._distributed_mode = distributed_mode

    def _build(self) -> None:
        """Allocate distributed Muon state for the current parameter groups."""
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        self._bank_meta: list[dict[str, object]] = []
        total_numel = 0
        for group in self.param_groups:
            for param in group["params"]:
                rows = int(param.shape[0])
                padded_rows = ((rows + ws - 1) // ws) * ws
                shard_rows = padded_rows // ws
                tail = tuple(param.shape[1:])
                device = param.device
                numel = int(param.numel())
                self._bank_meta.append(
                    {
                        "p": param,
                        "rows": rows,
                        "numel": numel,
                        "offset": total_numel,
                        "padded_grad": torch.zeros(
                            padded_rows, *tail, device=device, dtype=torch.bfloat16
                        ),
                        "shard": torch.zeros(
                            shard_rows, *tail, device=device, dtype=torch.bfloat16
                        ),
                        "shard_mom": torch.zeros(
                            shard_rows, *tail, device=device, dtype=torch.bfloat16
                        ),
                        "full_update": torch.zeros(
                            padded_rows, *tail, device=device, dtype=torch.bfloat16
                        ),
                        "scale": max(1, param.shape[-2] / param.shape[-1]) ** 0.5,
                    }
                )
                total_numel += numel
        self._bank_meta.sort(key=lambda meta: -cast(Tensor, meta["p"]).numel())
        if self._bank_meta:
            offset = 0
            for meta in self._bank_meta:
                meta["offset"] = offset
                offset += cast(int, meta["numel"])
            self._packed_updates = torch.zeros(
                offset,
                device=cast(Tensor, self._bank_meta[0]["p"]).device,
                dtype=torch.bfloat16,
            )
        self._built = True

    def launch_reduce_scatters(self) -> None:
        """Launch async reduce-scatter ops for matrix params after backward."""
        if not self._built:
            self._build()
        if not self._distributed or self._distributed_mode != "sharded_rsag":
            return
        self._rs_futures: list[dist.Work | None] = []
        for meta in self._bank_meta:
            param = cast(Tensor, meta["p"])
            if param.grad is None:
                self._rs_futures.append(None)
                continue
            padded_grad = cast(Tensor, meta["padded_grad"])
            rows = cast(int, meta["rows"])
            padded_grad[:rows].copy_(param.grad.bfloat16())
            if padded_grad.shape[0] > rows:
                padded_grad[rows:].zero_()
            shard = cast(Tensor, meta["shard"])
            work = dist.reduce_scatter_tensor(
                shard,
                padded_grad,
                op=dist.ReduceOp.AVG,
                async_op=True,
            )
            self._rs_futures.append(work)

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

        if not self._built:
            self._build()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)

            prev_ag_handle: dist.Work | None = None
            prev_meta: dict[str, object] | None = None
            sharded = (
                self._distributed
                and self._distributed_mode == "sharded_rsag"
                and hasattr(self, "_rs_futures")
            )
            packed_distributed = self._distributed and not sharded

            if packed_distributed:
                packed_updates = self._packed_updates
                packed_updates.zero_()
                for index, meta in enumerate(self._bank_meta):
                    param = cast(Tensor, meta["p"])
                    if index % self._world_size != self._rank or param.grad is None:
                        continue
                    grad = param.grad
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = cast(Tensor, state["momentum_buffer"])
                    buf.mul_(momentum).add_(grad)
                    update = grad.add(buf, alpha=momentum) if nesterov else buf
                    update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                    offset = cast(int, meta["offset"])
                    numel = cast(int, meta["numel"])
                    packed_updates[offset : offset + numel].copy_(update.reshape(-1))
                dist.all_reduce(packed_updates, op=dist.ReduceOp.SUM)
                for meta in self._bank_meta:
                    param = cast(Tensor, meta["p"])
                    if wd > 0.0:
                        param.data.mul_(1.0 - lr * wd)
                    offset = cast(int, meta["offset"])
                    numel = cast(int, meta["numel"])
                    update = packed_updates[offset : offset + numel].view_as(param)
                    param.add_(
                        update.to(dtype=param.dtype),
                        alpha=-lr * cast(float, meta["scale"]),
                    )
                continue

            for index, meta in enumerate(self._bank_meta):
                param = cast(Tensor, meta["p"])
                if param.grad is None:
                    continue

                if prev_ag_handle is not None and prev_meta is not None:
                    prev_ag_handle.wait()
                    prev_param = cast(Tensor, prev_meta["p"])
                    prev_rows = cast(int, prev_meta["rows"])
                    update = cast(Tensor, prev_meta["full_update"])[:prev_rows]
                    if wd > 0.0:
                        prev_param.data.mul_(1.0 - lr * wd)
                    prev_param.add_(
                        update.to(dtype=prev_param.dtype),
                        alpha=-lr * cast(float, prev_meta["scale"]),
                    )

                if sharded and self._rs_futures[index] is not None:
                    self._rs_futures[index].wait()
                    grad = cast(Tensor, meta["shard"])
                    buf = cast(Tensor, meta["shard_mom"])
                else:
                    grad = param.grad.bfloat16()
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = cast(Tensor, state["momentum_buffer"])

                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)

                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        cast(Tensor, meta["full_update"]), update, async_op=True
                    )
                    prev_meta = meta
                else:
                    if wd > 0.0:
                        param.data.mul_(1.0 - lr * wd)
                    param.add_(
                        update.to(dtype=param.dtype),
                        alpha=-lr * cast(float, meta["scale"]),
                    )
                    prev_ag_handle = None
                    prev_meta = None

            if prev_ag_handle is not None and prev_meta is not None:
                prev_ag_handle.wait()
                prev_param = cast(Tensor, prev_meta["p"])
                prev_rows = cast(int, prev_meta["rows"])
                update = cast(Tensor, prev_meta["full_update"])[:prev_rows]
                if wd > 0.0:
                    prev_param.data.mul_(1.0 - lr * wd)
                prev_param.add_(
                    update.to(dtype=prev_param.dtype),
                    alpha=-lr * cast(float, prev_meta["scale"]),
                )

        if hasattr(self, "_rs_futures"):
            del self._rs_futures
        return loss


class ReplicatedGradSync:
    """Bucket replicated grads so non-Muon sync uses fewer NCCL collectives."""

    def __init__(self, params: list[Tensor]):
        """Initialize the replicated-gradient synchronization helper.

        :param list[Tensor] params: Replicated parameters whose grads must be averaged.
        """
        self._params = [param for param in params if param.requires_grad]
        self._built = False

    def _build(self) -> None:
        """Create per-device flat buckets for replicated-gradient averaging."""
        self._distributed = dist.is_available() and dist.is_initialized()
        self._buckets: list[dict[str, object]] = []
        if not self._distributed or not self._params:
            self._built = True
            return
        seen: set[tuple[torch.device, torch.dtype]] = set()
        ordered_keys: list[tuple[torch.device, torch.dtype]] = []
        for param in self._params:
            key = (param.device, param.dtype)
            if key in seen:
                continue
            seen.add(key)
            ordered_keys.append(key)
        for key in ordered_keys:
            bucket_params = [
                param for param in self._params if (param.device, param.dtype) == key
            ]
            bucket_params.sort(key=lambda param: -param.numel())
            total_numel = sum(int(param.numel()) for param in bucket_params)
            if total_numel <= 0:
                continue
            self._buckets.append(
                {
                    "params": bucket_params,
                    "flat": torch.empty(
                        total_numel,
                        device=key[0],
                        dtype=key[1],
                    ),
                }
            )
        self._built = True

    def launch_all_reduces(self) -> None:
        """Pack available grads into flat buckets and launch async all-reduces."""
        if not self._built:
            self._build()
        self._pending: list[dict[str, object]] = []
        if not self._distributed:
            return
        for bucket in self._buckets:
            flat = cast(Tensor, bucket["flat"])
            active: list[tuple[Tensor, int, int]] = []
            offset = 0
            for param in cast(list[Tensor], bucket["params"]):
                grad = param.grad
                if grad is None:
                    continue
                numel = int(grad.numel())
                flat[offset : offset + numel].copy_(grad.reshape(-1))
                active.append((grad, offset, numel))
                offset += numel
            if offset <= 0:
                continue
            work = dist.all_reduce(
                flat[:offset],
                op=dist.ReduceOp.AVG,
                async_op=True,
            )
            self._pending.append(
                {
                    "work": work,
                    "flat": flat,
                    "active": active,
                }
            )

    def wait(self) -> None:
        """Wait for in-flight all-reduces and unpack averaged grads in-place."""
        if not hasattr(self, "_pending"):
            return
        for item in self._pending:
            cast(dist.Work, item["work"]).wait()
            flat = cast(Tensor, item["flat"])
            for grad, offset, numel in cast(
                list[tuple[Tensor, int, int]], item["active"]
            ):
                grad.copy_(flat[offset : offset + numel].view_as(grad))
        self._pending.clear()


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


def prewarm_eval_forward(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
) -> bool:
    """Precompile one representative eval/inference forward before timed training.

    This keeps the first `model.eval()` + `torch.inference_mode()` graph build
    out of the measured training loop and avoids a one-time grad-mode
    recompile on the first validation step.

    :param Hyperparameters args: Runtime hyperparameters.
    :param nn.Module model: Compiled training model or DDP wrapper.
    :param int rank: Distributed rank.
    :param int world_size: Distributed world size.
    :param torch.device device: Active CUDA device.
    :param int grad_accum_steps: Gradient accumulation steps.
    :param Tensor val_tokens: Validation token stream.
    :return bool: `True` when a representative eval batch was prewarmed.
    """

    local_batch_tokens = args.val_batch_size // (grad_accum_steps * world_size)
    if local_batch_tokens < args.train_seq_len:
        return False
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (int(val_tokens.numel()) - 1) // args.train_seq_len
    if total_seqs <= 0:
        return False
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    if seq_start >= seq_end:
        return False
    batch_seq_end = min(seq_start + local_batch_seqs, seq_end)
    raw_start = seq_start * args.train_seq_len
    raw_end = batch_seq_end * args.train_seq_len + 1
    local = val_tokens[raw_start:raw_end]
    x, y, _, _ = stage_eval_batch_views(
        local[:-1].reshape(-1, args.train_seq_len),
        local[1:].reshape(-1, args.train_seq_len),
        device,
        None,
        None,
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            _ = model(x, y)
    if was_training:
        model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return True


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
    args.gdn_w_g_optimizer = validate_gdn_w_g_optimizer(args.gdn_w_g_optimizer)
    if args.gdn_fla_recurrence_mode not in GDN_FLA_RECURRENCE_MODES:
        raise ValueError(
            "GDN_FLA_RECURRENCE_MODE must be one of "
            f"{sorted(GDN_FLA_RECURRENCE_MODES)}, got {args.gdn_fla_recurrence_mode!r}"
        )
    if args.muon_distributed_mode not in {"packed_allreduce", "sharded_rsag"}:
        raise ValueError(
            "MUON_DISTRIBUTED_MODE must be 'packed_allreduce' or 'sharded_rsag', "
            f"got {args.muon_distributed_mode!r}"
        )
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
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got WORLD_SIZE={world_size}")
    grad_accum_steps_override = int(os.environ.get("GRAD_ACCUM_STEPS", "0"))
    if grad_accum_steps_override < 0:
        raise ValueError(
            f"GRAD_ACCUM_STEPS must be non-negative, got {grad_accum_steps_override}"
        )
    if grad_accum_steps_override > 0:
        grad_accum_steps = grad_accum_steps_override
    else:
        if 8 % world_size != 0:
            raise ValueError(
                "WORLD_SIZE must evenly divide 8 when GRAD_ACCUM_STEPS is not "
                f"explicitly set, got WORLD_SIZE={world_size}"
            )
        grad_accum_steps = 8 // world_size
    if grad_accum_steps < 1:
        raise ValueError(
            f"Resolved GRAD_ACCUM_STEPS must be >= 1, got {grad_accum_steps}"
        )
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
    use_parallel_muon_distributed = (
        distributed and args.distributed_mode == "parallel_muon"
    )
    uses_ddp = distributed and not use_parallel_muon_distributed
    master_process = rank == 0
    stream_sync_event = make_stream_sync_event(device)
    perf_start_event = (
        torch.cuda.Event(enable_timing=True)
        if args.perf_timing and device.type == "cuda"
        else None
    )
    perf_end_event = (
        torch.cuda.Event(enable_timing=True)
        if args.perf_timing and device.type == "cuda"
        else None
    )

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
    attention_backend = attention_backend_name()

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt" if master_process else None
    logfile_handle = None
    if logfile is not None:
        logfile_handle = open(logfile, "a", encoding="utf-8", buffering=1)
        atexit.register(logfile_handle.close)

    def log0(msg: str, console: bool = True) -> None:
        """Log a line on rank 0 to stdout and the run logfile.

        :param str msg: Message to log.
        :param bool console: Whether to print to stdout, defaults to True.
        """
        if not master_process:
            return
        if console:
            print(msg)
        if logfile_handle is not None:
            print(msg, file=logfile_handle)

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
    if args.gdn_ratio > 0 and not _HAS_FLA:
        raise RuntimeError(
            "Packed HGDN training requires the FLA gated-delta fast path. "
            "Refusing to silently fall back to the naive recurrence. "
            "Install/repair FLA or use GDN_RATIO=0."
        )
    log0(
        f"compile:{int(args.compile)} compile_strategy:{args.compile_strategy} "
        f"seed:{args.seed} pythonhashseed:{os.environ.get('PYTHONHASHSEED', '<unset>')} "
        f"sdp_backend:{sdp_backend} attention_backend:{attention_backend} "
        f"fla_available:{int(_HAS_FLA)} "
        f"gdn_fla_recurrence_mode:{args.gdn_fla_recurrence_mode} "
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
        f"output_norm_fp32={int(args.gdn_output_norm_fp32)} ",
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
    train_files = resolve_glob_files(
        args.train_files, missing_message=missing_data_message
    )
    val_files = resolve_glob_files(args.val_files, missing_message=missing_data_message)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    val_tokens = load_validation_tokens_from_files(
        val_files,
        args.train_seq_len,
        load_data_shard=load_data_shard,
        missing_message="No validation shards provided",
        min_seqs=args.min_val_seqs,
        max_seqs=args.val_max_seqs,
    )
    base_bytes_lut, has_ls_lut, is_bnd_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(
        f"train_loader:dataset:{dataset_dir.name} train_shards:{len(train_files)} "
        f"val_shards:{len(val_files)}"
    )
    log0(
        f"val_loader:pattern:{args.val_files} tokens:{val_tokens.numel() - 1} "
        f"seqs:{(val_tokens.numel() - 1) // args.train_seq_len} "
        f"min_seqs:{args.min_val_seqs} max_seqs:{args.val_max_seqs}"
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
            gdn_use_packed_qkv_conv_custom_backward=args.gdn_use_packed_qkv_conv_custom_backward,
            gdn_use_packed_qkv_single_contig=args.gdn_use_packed_qkv_single_contig,
            gdn_use_packed_qkv_split_copy=args.gdn_use_packed_qkv_split_copy,
            gdn_fla_recurrence_mode=args.gdn_fla_recurrence_mode,
            mlp_mult=args.mlp_mult,
            leaky_slope=args.leaky_slope,
            gdn_ratio=args.gdn_ratio,
            block_pattern=args.block_pattern or None,
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
        base_model,
        gdn_control_proj_fp32=args.gdn_control_proj_fp32,
        gdn_w_g_optimizer=args.gdn_w_g_optimizer,
    )
    rotary_prewarm_count = 0
    if args.compile:
        rotary_prewarm_count = prewarm_rotary_caches(
            base_model,
            seq_len=args.train_seq_len,
            dtype=torch.bfloat16,
        )
    compile_top_level = not distributed
    compiled_model, compile_stats = prepare_hybrid_compile(
        base_model,
        enabled=args.compile,
        strategy=args.compile_strategy,
        compile_top_level=compile_top_level,
    )
    model = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if uses_ddp
        else compiled_model
    )
    if uses_ddp and args.compile and args.compile_strategy in {"model", "hybrid"}:
        model = maybe_compile(model, enabled=True, dynamic=False, fullgraph=False)
        compile_stats["model_compiled"] = 1

    n_params = sum(p.numel() for p in base_model.parameters())
    n_gdn = sum(1 for t in base_model.block_types if t == "gdn")
    n_attn = sum(1 for t in base_model.block_types if t == "attn")
    log0(
        f"model_params:{n_params} blocks:{n_gdn}G+{n_attn}A "
        f"norm_style:{base_model.norm_style} residual_alpha:{base_model.residual_alpha:g} "
        f"block_pattern:{args.block_pattern or '<ratio>'}"
    )
    decay_stats = [
        module.decay_init_stats()
        for module in base_model.modules()
        if hasattr(module, "decay_init_stats")
    ]
    if decay_stats:
        alpha_means = [row["alpha_mean"] for row in decay_stats]
        alpha_mins = [row["alpha_min"] for row in decay_stats]
        alpha_maxs = [row["alpha_max"] for row in decay_stats]
        log0(
            "gdn_decay_init:"
            f"layers:{len(decay_stats)} "
            f"A_mean:{sum(row['A_mean'] for row in decay_stats) / len(decay_stats):.4g} "
            f"dt_mean:{sum(row['dt_mean'] for row in decay_stats) / len(decay_stats):.4g} "
            f"alpha_mean:{sum(alpha_means) / len(alpha_means):.4g} "
            f"alpha_min:{min(alpha_mins):.4g} "
            f"alpha_max:{max(alpha_maxs):.4g}"
        )
    log0(
        "compile_plan:"
        f"strategy:{compile_stats['strategy']} "
        f"gdn_disabled:{compile_stats['gdn_disabled']} "
        f"gdn_blocks_compiled:{compile_stats['gdn_blocks_compiled']} "
        f"gdn_fla_blocks_compiled:{compile_stats['gdn_fla_blocks_compiled']} "
        f"gdn_mlps_compiled:{compile_stats['gdn_mlps_compiled']} "
        f"attn_blocks_compiled:{compile_stats['attn_blocks_compiled']} "
        f"model_compiled:{compile_stats['model_compiled']}"
    )
    log0(
        f"world_size:{world_size} grad_accum_steps:{grad_accum_steps} "
        f"distributed_mode:{args.distributed_mode} uses_ddp:{int(uses_ddp)} "
        f"muon_distributed_mode:{args.muon_distributed_mode} "
        f"gdn_w_g_optimizer:{args.gdn_w_g_optimizer} "
        f"sdp_backends:cudnn=False flash=True mem_efficient=False math=False"
    )
    if args.compile and distributed and args.compile_strategy in {"model", "hybrid"}:
        log0(
            "compile_top_level_suppressed:distributed_launch "
            "top-level model compile disabled on distributed launches"
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
                attention_backend=attention_backend,
                wandb_watch_mode=wandb_watch_mode,
            )
        )

    # ── Optimizers ────────────────────────────────────────────────────
    block_named_params = list(base_model.blocks.named_parameters())
    gdn_conv_params = [p for n, p in block_named_params if is_gdn_conv_param(n)]
    matrix_params = [
        p
        for n, p in block_named_params
        if p.ndim == 2
        and not uses_scalar_optimizer(
            n,
            param_ndim=p.ndim,
            gdn_w_g_optimizer=args.gdn_w_g_optimizer,
        )
        and not is_gdn_conv_param(n)
    ]
    scalar_params = [
        p
        for n, p in block_named_params
        if uses_scalar_optimizer(
            n,
            param_ndim=p.ndim,
            gdn_w_g_optimizer=args.gdn_w_g_optimizer,
        )
        and not is_gdn_decay_param(n)
        and not is_gdn_conv_param(n)
    ]
    gdn_decay_params = [p for n, p in block_named_params if is_gdn_decay_param(n)]
    optimizer_owned_ids = {
        id(p)
        for params in (matrix_params, scalar_params, gdn_conv_params, gdn_decay_params)
        for p in params
    }
    missing_optimizer_params = [
        n
        for n, p in block_named_params
        if p.requires_grad and id(p) not in optimizer_owned_ids
    ]
    if missing_optimizer_params:
        raise RuntimeError(
            "Trainable block params missing optimizer coverage: "
            + ", ".join(missing_optimizer_params)
        )
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
        distributed_mode=args.muon_distributed_mode,
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
    optimizer_gdn_conv = None
    if gdn_conv_params:
        optimizer_gdn_conv = torch.optim.Adam(
            [
                {
                    "params": gdn_conv_params,
                    "lr": args.scalar_lr,
                    "base_lr": args.scalar_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            fused=True,
        )
    optimizer_gdn_decay = None
    if gdn_decay_params:
        optimizer_gdn_decay = torch.optim.Adam(
            [
                {
                    "params": gdn_decay_params,
                    "lr": args.scalar_lr,
                    "base_lr": args.scalar_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=0.0,
            fused=True,
        )
    optimizer_head = None
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_gdn_conv is not None:
        optimizers.append(optimizer_gdn_conv)
    if optimizer_gdn_decay is not None:
        optimizers.append(optimizer_gdn_decay)
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
    replicated_params: list[Tensor] = []
    replicated_grad_sync = None
    if use_parallel_muon_distributed:
        replicated_params.extend(optimizer_tok.param_groups[0]["params"])
        replicated_params.extend(scalar_params)
        if optimizer_gdn_conv is not None:
            replicated_params.extend(gdn_conv_params)
        if optimizer_gdn_decay is not None:
            replicated_params.extend(gdn_decay_params)
        if optimizer_head is not None:
            replicated_params.extend(optimizer_head.param_groups[0]["params"])
        replicated_grad_sync = ReplicatedGradSync(replicated_params)
    replicated_optimizers = [optimizer_tok, optimizer_scalar]
    if optimizer_gdn_conv is not None:
        replicated_optimizers.append(optimizer_gdn_conv)
    if optimizer_gdn_decay is not None:
        replicated_optimizers.append(optimizer_gdn_decay)
    if optimizer_head is not None:
        replicated_optimizers.append(optimizer_head)

    muon_prewarm_count = 0
    if args.compile and device.type == "cuda":
        muon_prewarm_count = prewarm_muon_backend_shapes(
            matrix_params,
            backend_steps=args.muon_backend_steps,
        )

    # ── Data loader & warmup ──────────────────────────────────────────
    train_loader = DistributedTokenLoader(
        train_files,
        rank,
        world_size,
        device,
        load_data_shard=load_data_shard,
    )
    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def zero_grad_all() -> None:
        """Clear gradients across all optimizer groups."""
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def apply_optimizer_step() -> None:
        """Step optimizers using the active distributed synchronization strategy."""
        if use_parallel_muon_distributed:
            assert replicated_grad_sync is not None
            replicated_grad_sync.launch_all_reduces()
            if args.muon_distributed_mode == "sharded_rsag":
                optimizer_muon.launch_reduce_scatters()
            replicated_grad_sync.wait()
            for opt in replicated_optimizers:
                opt.step()
            optimizer_muon.step()
            return
        for opt in optimizers:
            opt.step()

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
        zero_grad_all()
        warmup_x, warmup_y = train_loader.next_batch(
            args.train_batch_tokens, args.train_seq_len, grad_accum_steps
        )
        for ws in range(args.warmup_steps):
            for ms in range(grad_accum_steps):
                if uses_ddp:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wl = model(warmup_x, warmup_y)
                (wl * grad_scale).backward()
            apply_optimizer_step()
            zero_grad_all()
            if ws + 1 == args.warmup_steps or (ws + 1) % 10 == 0:
                log0(f"warmup_step:{ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_state, strict=True)
        for o, s in zip(optimizers, init_opt):
            o.load_state_dict(s)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            train_files,
            rank,
            world_size,
            device,
            load_data_shard=load_data_shard,
        )
    eval_prewarm_count = 0
    if args.compile and (
        args.log_step0_eval or args.val_loss_every > 0 or not args.perf_skip_final_eval
    ):
        eval_prewarm_count = int(
            prewarm_eval_forward(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
            )
        )
    if muon_prewarm_count > 0 or rotary_prewarm_count > 0 or eval_prewarm_count > 0:
        log0(
            "compile_prewarm:"
            f" muon_shapes:{muon_prewarm_count}"
            f" rotary_modules:{rotary_prewarm_count}"
            f" eval_graph:{eval_prewarm_count}"
        )
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
    zero_grad_all()
    wait_current_stream(stream_sync_event)
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
            wait_current_stream(stream_sync_event)
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            with profile_ctx("eval.val"):
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
                    has_leading_space_lut=has_ls_lut,
                    is_boundary_token_lut=is_bnd_lut,
                    use_inference_mode=False,
                )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms"
            )
            if master_process and _USE_WANDB:
                wandb.log({"eval/loss": val_loss, "eval/bpb": val_bpb}, step=step)
            wait_current_stream(stream_sync_event)
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        train_loss = torch.zeros((), device=device)
        perf_t0 = (
            time.perf_counter() if args.perf_timing and device.type != "cuda" else None
        )
        if perf_start_event is not None:
            perf_start_event.record()

        with profile_ctx("train.step"):
            for ms in range(grad_accum_steps):
                if uses_ddp:
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
            apply_optimizer_step()
            zero_grad_all()

        step += 1
        perf_step_ms = float("nan")
        perf_tokens_per_s = float("nan")
        if args.perf_timing:
            if perf_start_event is not None and perf_end_event is not None:
                perf_end_event.record()
                perf_end_event.synchronize()
                measured_ms = perf_start_event.elapsed_time(perf_end_event)
            else:
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
        base_model.state_dict(),
        gdn_w_g_optimizer=args.gdn_w_g_optimizer,
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

    if device.type == "cuda":
        q_eval_start = torch.cuda.Event(enable_timing=True)
        q_eval_end = torch.cuda.Event(enable_timing=True)
        q_eval_start.record()
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
            has_leading_space_lut=has_ls_lut,
            is_boundary_token_lut=is_bnd_lut,
            use_inference_mode=False,
        )
        q_eval_end.record()
        q_eval_end.synchronize()
        q_eval_ms = q_eval_start.elapsed_time(q_eval_end)
    else:
        t_q = time.perf_counter()
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
            has_leading_space_lut=has_ls_lut,
            is_boundary_token_lut=is_bnd_lut,
            use_inference_mode=False,
        )
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
            artifact_warning = ""
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
