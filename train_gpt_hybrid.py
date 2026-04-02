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
import glob
import io
import math
import os
import random
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from model import _HAS_FLA, SCALAR_PARAM_PATTERNS, CastedLinear, HybridGPT

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
    wandb_watch_log_freq = int(os.environ.get("WANDB_WATCH_LOG_FREQ", 25))

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
    artifact_limit_bytes = int(os.environ.get("ARTIFACT_LIMIT_BYTES", 16_000_000))

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

    # Model — attention blocks
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))

    # Model — GDN blocks
    gdn_n_heads = int(os.environ.get("GDN_N_HEADS", 8))
    gdn_expand_v = float(os.environ.get("GDN_EXPAND_V", 1.0))
    gdn_head_k_dim = int(os.environ.get("GDN_HEAD_K_DIM", 48))

    gdn_allow_neg_eigval = bool(int(os.environ.get("GDN_ALLOW_NEG_EIGVAL", "1")))
    gdn_conv_size = int(os.environ.get("GDN_CONV_SIZE", 4))
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

    def to_dict(self) -> dict:
        return {
            k: v
            for k, v in vars(type(self)).items()
            if not k.startswith("_") and not callable(v) and k != "to_dict"
        }


# =====================================================================
# MUON OPTIMIZER (from modded-nanogpt via baseline train_gpt.py)
# =====================================================================


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
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
    def __init__(
        self,
        params,
        lr,
        momentum,
        backend_steps,
        nesterov=True,
        weight_decay=0.0,
    ):
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
    def step(self, closure=None):
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
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
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
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
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


def build_sentencepiece_luts(sp, vocab_size, device):
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


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files for: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def eval_val(
    args,
    model,
    rank,
    world_size,
    device,
    grad_accum_steps,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_lut,
):
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


def quantize_state_dict_int8(state_dict):
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


def dequantize_state_dict_int8(obj):
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


# =====================================================================
# UTILITY
# =====================================================================


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2 or any(p in name for p in SCALAR_PARAM_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


def maybe_compile(
    obj, *, enabled: bool, dynamic: bool = False, fullgraph: bool = False
):
    if not enabled:
        return obj
    return torch.compile(obj, dynamic=dynamic, fullgraph=fullgraph)


# =====================================================================
# MAIN
# =====================================================================


def main():
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = maybe_compile(
        zeropower_via_newtonschulz5, enabled=args.compile
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

    def log0(msg, console=True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    # ── wandb init ────────────────────────────────────────────────────
    if master_process and _USE_WANDB:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "param-golf-hybrid"),
            name=args.run_id,
            config=args.to_dict(),
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
        f"compile:{int(args.compile)} sdp_backend:{sdp_backend} fla_available:{int(_HAS_FLA)}",
        console=False,
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
    train_files = sorted(dataset_dir.glob("fineweb_train_*.bin"))
    val_files = sorted(dataset_dir.glob("fineweb_val_*.bin"))
    if not train_files or not val_files:
        raise FileNotFoundError(
            "FineWeb shards not found for the hybrid trainer. Download the published "
            "sp1024 assets with:\n"
            "  python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1\n"
            f"Expected dataset under: {dataset_dir}"
        )

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
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
            mlp_mult=args.mlp_mult,
            leaky_slope=args.leaky_slope,
            gdn_ratio=args.gdn_ratio,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            logit_softcap=args.logit_softcap,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
        )
        .to(device)
        .bfloat16()
    )

    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = maybe_compile(
        base_model, enabled=args.compile, dynamic=False, fullgraph=True
    )
    model = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    n_params = sum(p.numel() for p in base_model.parameters())
    n_gdn = sum(1 for t in base_model.block_types if t == "gdn")
    n_attn = sum(1 for t in base_model.block_types if t == "attn")
    log0(f"model_params:{n_params} blocks:{n_gdn}G+{n_attn}A")
    log0(
        f"world_size:{world_size} grad_accum_steps:{grad_accum_steps} "
        f"sdp_backends:cudnn=False flash=True mem_efficient=False math=False"
    )

    if master_process and _USE_WANDB:
        wandb.config.update(
            {
                "artifact_limit_bytes": args.artifact_limit_bytes,
                "compile": args.compile,
                "fla_available": _HAS_FLA,
                "local_batch_size": local_batch_size,
                "n_attn_blocks": n_attn,
                "n_gdn_blocks": n_gdn,
                "n_params": n_params,
                "planned_train_tokens": planned_train_tokens,
                "sdp_backend": sdp_backend,
            }
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
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def lr_mul(step, elapsed_ms):
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
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device
        )
    if master_process and _USE_WANDB:
        wandb.watch(
            base_model,
            log="gradients",
            log_freq=args.wandb_watch_log_freq,
            log_graph=False,
        )

    # ── Training loop ─────────────────────────────────────────────────
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
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
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
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

        for ms in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
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

        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
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
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_ms:.0f}ms step_avg:{step_ms:.2f}ms"
            )
            if master_process and _USE_WANDB:
                wandb.log(
                    {
                        "train/lr": args.matrix_lr * scale,
                        "train/lr_scale": scale,
                        "train/loss": train_loss.item(),
                        "train/processed_tokens": processed_tokens,
                        "train/step_ms": step_ms,
                        "train/tokens_per_s": tokens_per_s,
                    },
                    step=step,
                )

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    peak_mem_alloc_mib = torch.cuda.max_memory_allocated() // 1024 // 1024
    peak_mem_reserved_mib = torch.cuda.max_memory_reserved() // 1024 // 1024
    log0(
        f"peak memory allocated: {peak_mem_alloc_mib} MiB reserved: {peak_mem_reserved_mib} MiB"
    )

    # ── Serialize + quantize + roundtrip ──────────────────────────────
    qfb = None
    code_bytes = len(code.encode("utf-8"))
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"raw_model_bytes:{os.path.getsize('final_model.pt')}")

    quant_obj, _ = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        qfb = os.path.getsize("final_model.int8.ptz")
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
        wandb.summary["artifact/code_bytes_final"] = code_bytes
        wandb.summary["artifact/int8_payload_zlib_bytes_final"] = qfb
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
