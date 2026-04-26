"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import io
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
import atexit
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
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

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap


class Hyperparameters:
    """Runtime knobs for data, model, optimizer, and training length."""

    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. VAL_MAX_SEQS=0 uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    min_val_seqs = int(os.environ.get("MIN_VAL_SEQS", 1))
    val_max_seqs = int(os.environ.get("VAL_MAX_SEQS", 0))
    log_step0_eval = bool(int(os.environ.get("LOG_STEP0_EVAL", "0")))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    artifact_limit_bytes = int(os.environ.get("ARTIFACT_LIMIT_BYTES", 16_000_000))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    norm_kind = os.environ.get("NORM_KIND", "rms").lower()
    norm_style = os.environ.get("NORM_STYLE", "pre").lower()
    residual_alpha = os.environ.get("RESIDUAL_ALPHA")
    residual_alpha = None if residual_alpha in {None, ""} else float(residual_alpha)
    use_input_norm = bool(int(os.environ.get("INPUT_NORM", "1")))
    use_final_norm = bool(int(os.environ.get("FINAL_NORM", "1")))
    use_residual_mix = bool(int(os.environ.get("USE_RESIDUAL_MIX", "1")))
    use_skip_weights = bool(int(os.environ.get("USE_SKIP_WEIGHTS", "1")))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
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


# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
    """Orthogonalize a matrix gradient with a short Newton-Schulz iteration.

    :param Tensor G: Input 2D gradient matrix.
    :param int steps: Number of Newton-Schulz iterations to run.
    :param float eps: Small norm floor for numerical stability.
    :return Tensor: Orthogonalized update matrix.
    """

    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix-shaped parameters."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
    ):
        """Initialize Muon with the reference hyperparameters.

        :param Iterable[Tensor] params: Parameter iterable to optimize.
        :param float lr: Learning rate.
        :param float momentum: Momentum coefficient.
        :param int backend_steps: Newton-Schulz steps for the matrix backend.
        :param bool nesterov: Whether to use Nesterov momentum.
        """

        super().__init__(
            params,
            dict(
                lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov
            ),
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Apply one Muon update step.

        :param Callable[[], Tensor] | None closure: Optional reevaluation closure.
        :return Tensor | None: The closure loss, if one was provided.
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
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

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
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    """Return the number of bytes occupied by a tensor.

    :param Tensor t: Tensor to measure.
    :return int: Tensor storage size in bytes.
    """

    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(
    name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]
) -> Tensor:
    """Keep selected float tensors in a smaller passthrough form for export.

    :param str name: Parameter name.
    :param Tensor t: Tensor to export.
    :param dict[str, str] passthrough_orig_dtypes: Original dtype registry.
    :return Tensor: Tensor to retain in the export payload.
    """

    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a float tensor to int8 with a matching scale tensor.

    :param Tensor t: Float tensor to quantize.
    :return tuple[Tensor, Tensor]: Quantized tensor and scale tensor.
    """

    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(
            torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None]
        )
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = (
            torch.clamp(torch.round(clipped / scale[:, None]), -127, 127)
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = (
        float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
        if t32.numel()
        else 0.0
    )
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = (
        torch.clamp(
            torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127
        )
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


def quantize_state_dict_int8(
    state_dict: dict[str, Tensor],
) -> tuple[dict[str, object], dict[str, int]]:
    """Quantize a state dict into the clean int8 export bundle format.

    :param dict[str, Tensor] state_dict: Model state dict to compress.
    :return tuple[dict[str, object], dict[str, int]]: Bundle object and summary stats.
    """

    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
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


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    """Reconstruct float tensors from a clean int8 export bundle.

    :param dict[str, object] obj: Serialized int8 export bundle.
    :return dict[str, Tensor]: Reconstructed float state dict.
    """

    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (
                (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1))))
                .to(dtype=dtype)
                .contiguous()
            )
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------


def load_data_shard(file: Path) -> Tensor:
    """Load one binary token shard from disk.

    :param Path file: Shard path.
    :return Tensor: Token tensor loaded from the shard.
    """

    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

NormStyle = Literal["pre", "post", "keel"]
NormKind = Literal["rms", "layer"]


def validate_norm_style(norm_style: str) -> NormStyle:
    """Validate and normalize the residual norm placement.

    :param str norm_style: Norm-style string to validate.
    :return NormStyle: Normalized norm-style value.
    """

    normalized = norm_style.lower()
    if normalized not in {"pre", "post", "keel"}:
        raise ValueError(
            f"NORM_STYLE must be 'pre', 'post', or 'keel', got {norm_style!r}"
        )
    return normalized


def validate_norm_kind(norm_kind: str) -> NormKind:
    """Validate and normalize the normalization family.

    :param str norm_kind: Norm-kind string to validate.
    :return NormKind: Normalized norm-kind value.
    """

    normalized = norm_kind.lower()
    if normalized not in {"rms", "layer"}:
        raise ValueError(f"NORM_KIND must be 'rms' or 'layer', got {norm_kind!r}")
    return normalized


class RMSNorm(nn.Module):
    """RMSNorm wrapper that keeps the epsilon configurable."""

    def __init__(self, eps: float | None = None):
        """Create an RMSNorm module.

        :param float | None eps: Normalization epsilon.
        """

        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the last dimension of ``x`` with RMS scaling.

        :param Tensor x: Input tensor.
        :return Tensor: RMS-normalized tensor.
        """

        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


def build_norm(kind: NormKind, dim: int, eps: float = 1e-6) -> nn.Module:
    """Construct the configured normalization module.

    :param NormKind kind: Normalization family.
    :param int dim: Hidden dimension.
    :param float eps: Normalization epsilon.
    :return nn.Module: Requested normalization module.
    """

    kind = validate_norm_kind(kind)
    if kind == "rms":
        return RMSNorm(eps)
    return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)


class CastedLinear(nn.Linear):
    """Linear layer that stores weights in fp32 and casts at compute time."""

    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        """Apply a linear projection in the input dtype.

        :param Tensor x: Input tensor.
        :return Tensor: Linear projection output.
        """

        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    """Restore control and low-rank parameters to fp32 storage.

    :param nn.Module module: Module to normalize in-place.
    """

    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2
                or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    """Cache RoPE cos/sin tables for a given sequence length."""

    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        """Create the rotary embedding cache.

        :param int dim: Head dimension.
        :param float base: RoPE base frequency.
        """

        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_key: tuple[int, str, int | None, torch.dtype] | None = None
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Return cached RoPE tables in the requested dtype.

        :param int seq_len: Sequence length.
        :param torch.device device: Target device.
        :param torch.dtype dtype: Requested output dtype.
        :return tuple[Tensor, Tensor]: Cosine and sine RoPE tables.
        """

        cache_key = (seq_len, device.type, device.index, dtype)
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._cache_key != cache_key
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            cos = freqs.cos()[None, None, :, :]
            sin = freqs.sin()[None, None, :, :]
            if dtype != torch.float32:
                cos = cos.to(dtype=dtype)
                sin = sin.to(dtype=dtype)
            self._cos_cached = cos
            self._sin_cached = sin
            self._cache_key = cache_key
        return self._cos_cached, self._sin_cached


def prewarm_rotary_caches(
    module: nn.Module, *, seq_len: int, dtype: torch.dtype
) -> int:
    """Populate attention rotary caches before model compilation.

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
        rotary(seq_len, device, dtype)
        count += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    return count


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to a head dimension.

    :param Tensor x: Head tensor to rotate.
    :param Tensor cos: Cached cosine table.
    :param Tensor sin: Cached sine table.
    :return Tensor: Rotary-embedded tensor.
    """

    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """GQA attention block with RMS-normalized RoPE projections."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        """Initialize the attention block.

        :param int dim: Model width.
        :param int num_heads: Number of query heads.
        :param int num_kv_heads: Number of key/value heads.
        :param float rope_base: RoPE base frequency.
        :param float qk_gain_init: Initial query gain.
        """

        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        """Run causal attention over a ``[B, T, D]`` input tensor.

        :param Tensor x: Input activations.
        :return Tensor: Attention output.
        """

        bsz, seqlen, dim = x.shape
        q = (
            self.c_q(x)
            .reshape(bsz, seqlen, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.c_k(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.c_v(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    """ReLU-squared MLP block from the baseline transformer."""

    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        """Initialize the feed-forward block.

        :param int dim: Model width.
        :param int mlp_mult: Hidden width multiplier.
        """

        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP nonlinearity and output projection.

        :param Tensor x: Input activations.
        :return Tensor: MLP output.
        """

        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    """Transformer block that alternates residual-attention and MLP updates."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        norm_kind: NormKind = "rms",
        norm_style: NormStyle = "pre",
        residual_alpha: float = 1.0,
        use_residual_mix: bool = True,
        is_first_block: bool = False,
    ):
        """Initialize a transformer block.

        :param int dim: Model width.
        :param int num_heads: Attention head count.
        :param int num_kv_heads: Key/value head count.
        :param int mlp_mult: MLP width multiplier.
        :param float rope_base: RoPE base frequency.
        :param float qk_gain_init: Initial query gain.
        :param NormKind norm_kind: Normalization family.
        :param NormStyle norm_style: Residual norm placement.
        :param float residual_alpha: Residual blend coefficient.
        :param bool use_residual_mix: Whether to use residual mixing.
        :param bool is_first_block: Whether this is the first block.
        """

        super().__init__()
        self.norm_style = validate_norm_style(norm_style)
        self.residual_alpha = residual_alpha
        self.use_residual_mix = use_residual_mix
        self.is_first_block = is_first_block
        self.attn_in_norm = build_norm(norm_kind, dim)
        self.attn_out_norm = build_norm(norm_kind, dim)
        self.mlp_in_norm = build_norm(norm_kind, dim)
        self.mlp_out_norm = build_norm(norm_kind, dim)
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        if self.use_residual_mix:
            self.resid_mix = nn.Parameter(
                torch.stack((torch.ones(dim), torch.zeros(dim))).float()
            )
        else:
            self.register_parameter("resid_mix", None)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        """Apply the block in the configured norm style.

        :param Tensor x: Current hidden state.
        :param Tensor x0: Initial hidden state.
        :return Tensor: Updated hidden state.
        """

        if self.resid_mix is not None:
            mix = self.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_scale = self.attn_scale.to(dtype=x.dtype)[None, None, :]
        mlp_scale = self.mlp_scale.to(dtype=x.dtype)[None, None, :]
        if self.norm_style == "pre":
            x = x + attn_scale * self.attn(self.attn_in_norm(x))
            x = x + mlp_scale * self.mlp(self.mlp_in_norm(x))
            return x
        if self.norm_style == "post":
            x = self.attn_out_norm(x + attn_scale * self.attn(x))
            x = self.mlp_out_norm(x + mlp_scale * self.mlp(x))
            return x
        if self.is_first_block:
            x = x + attn_scale * self.attn(self.attn_in_norm(x))
            x = x + mlp_scale * self.mlp(self.mlp_in_norm(x))
            return x
        x = self.attn_out_norm(
            self.residual_alpha * x + attn_scale * self.attn(self.attn_in_norm(x))
        )
        x = self.mlp_out_norm(
            self.residual_alpha * x + mlp_scale * self.mlp(self.mlp_in_norm(x))
        )
        return x


class GPT(nn.Module):
    """Baseline GPT model used by the reference trainer."""

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        norm_kind: NormKind = "rms",
        norm_style: NormStyle = "pre",
        residual_alpha: float | None = None,
        use_input_norm: bool = True,
        use_final_norm: bool = True,
        use_residual_mix: bool = True,
        use_skip_weights: bool = True,
    ):
        """Build the GPT stack, embeddings, and output head.

        :param int vocab_size: Vocabulary size.
        :param int num_layers: Number of transformer blocks.
        :param int model_dim: Model width.
        :param int num_heads: Attention head count.
        :param int num_kv_heads: Key/value head count.
        :param int mlp_mult: MLP width multiplier.
        :param bool tie_embeddings: Whether to tie input and output embeddings.
        :param float tied_embed_init_std: Stddev for tied embeddings.
        :param float logit_softcap: Logit softcap value.
        :param float rope_base: RoPE base frequency.
        :param float qk_gain_init: Initial query gain.
        :param NormKind norm_kind: Normalization family.
        :param NormStyle norm_style: Residual norm placement.
        :param float | None residual_alpha: Residual blend coefficient.
        :param bool use_input_norm: Whether to normalize inputs.
        :param bool use_final_norm: Whether to normalize the final hidden state.
        :param bool use_residual_mix: Whether to use residual mixing.
        :param bool use_skip_weights: Whether to use skip connections.
        """

        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.norm_kind = validate_norm_kind(norm_kind)
        self.norm_style = validate_norm_style(norm_style)
        self.residual_alpha = (
            float(2 * num_layers)
            if residual_alpha is None and self.norm_style == "keel"
            else float(1.0 if residual_alpha is None else residual_alpha)
        )
        self.use_input_norm = use_input_norm
        self.use_final_norm = use_final_norm
        self.use_residual_mix = use_residual_mix
        self.use_skip_weights = use_skip_weights
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.input_norm = (
            build_norm(self.norm_kind, model_dim)
            if self.use_input_norm
            else nn.Identity()
        )
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        if self.use_skip_weights and self.num_skip_weights > 0:
            self.skip_weights = nn.Parameter(
                torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
            )
        else:
            self.register_parameter("skip_weights", None)
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    self.norm_kind,
                    self.norm_style,
                    self.residual_alpha,
                    self.use_residual_mix,
                    i == 0,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = (
            build_norm(self.norm_kind, model_dim)
            if self.use_final_norm
            else nn.Identity()
        )
        self.lm_head = (
            None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize tied embeddings and zero-init marked projections."""

        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Compute the training loss for a batch of token ids.

        :param Tensor input_ids: Input token ids.
        :param Tensor target_ids: Next-token targets.
        :return Tensor: Mean cross-entropy loss.
        """

        x = self.tok_emb(input_ids)
        x = self.input_norm(x)
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            if self.skip_weights is not None:
                skips.append(x)
        for i in range(self.num_decoder_layers):
            if self.skip_weights is not None and skips:
                x = (
                    x
                    + self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------


def main() -> None:
    """Run training, validation, logging, and checkpoint/export plumbing."""

    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
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
                "WORLD_SIZE must divide 8 when GRAD_ACCUM_STEPS is not explicitly "
                f"set, got WORLD_SIZE={world_size}"
            )
        grad_accum_steps = 8 // world_size
    if grad_accum_steps < 1:
        raise ValueError(
            f"Resolved GRAD_ACCUM_STEPS must be >= 1, got {grad_accum_steps}"
        )
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    stream_sync_event = make_stream_sync_event(device)

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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

    logfile = None
    logfile_handle = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        logfile_handle = open(logfile, "a", encoding="utf-8", buffering=1)
        atexit.register(logfile_handle.close)
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        """Log from rank zero to stdout and the run logfile.

        :param str msg: Message to log.
        :param bool console: Whether to print to stdout.
        """

        if not master_process:
            return
        if console:
            print(msg)
        if logfile_handle is not None:
            print(msg, file=logfile_handle)

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

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(
            f"Script only setup for SentencePiece .model file: {args.tokenizer_path}"
        )
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    train_files = resolve_glob_files(
        args.train_files,
        missing_message=f"No files found for pattern: {args.train_files}",
    )
    val_files = resolve_glob_files(
        args.val_files,
        missing_message=f"No files found for pattern: {args.val_files}",
    )
    actual_train_files = len(train_files)
    val_tokens = load_validation_tokens_from_files(
        val_files,
        args.train_seq_len,
        load_data_shard=load_data_shard,
        missing_message=f"No files found for pattern: {args.val_files}",
        min_seqs=args.min_val_seqs,
        max_seqs=args.val_max_seqs,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )
    log0(
        f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}"
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(
        f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1} "
        f"seqs:{(val_tokens.numel() - 1) // args.train_seq_len} "
        f"min_seqs:{args.min_val_seqs} max_seqs:{args.val_max_seqs}"
    )

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = (
        GPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            norm_kind=args.norm_kind,
            norm_style=args.norm_style,
            residual_alpha=args.residual_alpha,
            use_input_norm=args.use_input_norm,
            use_final_norm=args.use_final_norm,
            use_residual_mix=args.use_residual_mix,
            use_skip_weights=args.use_skip_weights,
        )
        .to(device)
        .bfloat16()
    )
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    rotary_prewarm_count = prewarm_rotary_caches(
        base_model,
        seq_len=args.train_seq_len,
        dtype=torch.bfloat16,
    )
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2
        or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights is not None:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [
        optimizer_tok,
        optimizer_muon,
        optimizer_scalar,
    ]
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
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(
        "norm_contract:"
        f" kind:{base_model.norm_kind}"
        f" style:{base_model.norm_style}"
        f" residual_alpha:{base_model.residual_alpha:g}"
        f" input_norm:{int(base_model.use_input_norm)}"
        f" final_norm:{int(base_model.use_final_norm)}"
        f" residual_mix:{int(base_model.use_residual_mix)}"
        f" skip_weights:{int(base_model.use_skip_weights)}"
    )
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    if rotary_prewarm_count > 0:
        log0(f"compile_prewarm: rotary_modules:{rotary_prewarm_count}")
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(
        train_files,
        rank,
        world_size,
        device,
        load_data_shard=load_data_shard,
    )

    def zero_grad_all() -> None:
        """Zero all optimizer gradients with ``set_to_none=True``."""

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step: int, elapsed_ms: float) -> float:
        """Compute the warmdown multiplier for the current step.

        :param int step: Current optimization step.
        :param float elapsed_ms: Elapsed training milliseconds.
        :return float: Learning-rate multiplier.
        """

        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return (
            remaining_ms / max(warmdown_ms, 1e-9)
            if remaining_ms <= warmdown_ms
            else 1.0
        )

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
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
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    warmup_loss = model(warmup_x, warmup_y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (
                args.warmup_steps <= 20
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == args.warmup_steps
            ):
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
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

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

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
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

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
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (
            step <= 10
            or step % args.train_log_every == 0
            or stop_after_step is not None
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
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

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"raw_model_bytes:{model_bytes}")
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    quant_file_bytes = len(quant_blob)
    code_bytes = len(code.encode("utf-8"))
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        ratio = quant_stats["baseline_tensor_bytes"] / max(
            quant_stats["int8_payload_bytes"], 1
        )
        raw_overhead_bytes = quant_raw_bytes - quant_stats["int8_payload_bytes"]
        payload_to_zlib_ratio = quant_stats["int8_payload_bytes"] / max(
            quant_file_bytes, 1
        )
        raw_quant_to_zlib_ratio = quant_raw_bytes / max(quant_file_bytes, 1)
        log0(
            "quant_audit "
            f"baseline_tensor_bytes:{quant_stats['baseline_tensor_bytes']} "
            f"int8_payload_bytes:{quant_stats['int8_payload_bytes']} "
            f"raw_torch_bytes:{quant_raw_bytes} "
            f"raw_torch_overhead_bytes:{raw_overhead_bytes} "
            f"baseline_to_payload_ratio:{ratio:.2f}x "
            f"payload_to_zlib_ratio:{payload_to_zlib_ratio:.2f}x "
            f"raw_quant_to_zlib_ratio:{raw_quant_to_zlib_ratio:.2f}x"
        )
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")
        log0(
            f"int8_zlib_bytes:{quant_file_bytes} code_bytes:{code_bytes} "
            f"total:{quant_file_bytes + code_bytes}"
        )

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu"
    )
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
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            use_inference_mode=True,
        )
        q_eval_end.record()
        q_eval_end.synchronize()
        q_eval_ms = q_eval_start.elapsed_time(q_eval_end)
    else:
        t_qeval = time.perf_counter()
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
        q_eval_ms = 1000.0 * (time.perf_counter() - t_qeval)
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{q_eval_ms:.0f}ms"
    )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}"
    )
    if master_process:
        artifact_bytes = quant_file_bytes + code_bytes
        artifact_headroom = args.artifact_limit_bytes - artifact_bytes
        if artifact_headroom < 0:
            artifact_status = "OVER_LIMIT"
            artifact_warning = "DISQUALIFIED"
        elif artifact_headroom > 0:
            artifact_status = "UNDER_LIMIT"
            artifact_warning = "NONE"
        else:
            artifact_status = "ON_BUDGET"
            artifact_warning = "NONE"
        log0(
            f"artifact_status:{artifact_status} artifact_warning:{artifact_warning} "
            f"headroom_bytes:{artifact_headroom}"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
