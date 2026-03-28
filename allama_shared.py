"""Compile-friendly shared-parameter ALlama components.

This is the v4 rewrite for the Parameter Golf setting. The design deliberately
backs away from the v3 custom Triton/autograd path and from the per-forward
Python controller objects that were blocking `torch.compile` from doing useful
fusion and unrolling.

Key choices in this version:
- plain PyTorch elementwise ops only; no custom `autograd.Function` wrappers
- additive x0 reinjection + RMSNorm in a direct `F.rms_norm` path
- simple learned per-layer modulation tensors instead of a controller network
- fixed layer-to-block assignment computed once at init
- Muon + Adam optimizer split retained from the v3 work
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


# -----------------------------------------------------------------------------
# MUON OPTIMIZER
# -----------------------------------------------------------------------------


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
    """Orthogonalize a 2D update with the Muon Newton-Schulz iteration."""
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
    """Muon optimizer for matrix-valued parameters."""

    def __init__(
        self,
        params: list[Tensor],
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
            ),
        )

    def _updates_buffer(self, group: dict[str, Any], params: list[Tensor]) -> Tensor:
        """Reuse one flat bfloat16 update buffer per parameter group."""
        total_params = sum(int(p.numel()) for p in params)
        device = params[0].device
        updates_flat = group.get("_updates_flat")
        if (
            updates_flat is None
            or not isinstance(updates_flat, Tensor)
            or updates_flat.numel() != total_params
            or updates_flat.device != device
        ):
            updates_flat = torch.zeros(
                total_params, device=device, dtype=torch.bfloat16
            )
            group["_updates_flat"] = updates_flat
        else:
            updates_flat.zero_()
        return updates_flat

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            backend_steps = int(group["backend_steps"])
            nesterov = bool(group["nesterov"])
            updates_flat = self._updates_buffer(group, params)

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


# -----------------------------------------------------------------------------
# CORE UTILITIES
# -----------------------------------------------------------------------------


class CastedLinear(nn.Linear):
    """Linear layer that casts weights to the activation dtype at matmul time."""

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class RMSNormWeight(nn.Module):
    """Minimal RMSNorm storing only the learned scale vector."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight.to(dtype=x.dtype), eps=self.eps)


class Rotary(nn.Module):
    """RoPE cache keyed by sequence length and device."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Optional[Tensor] = None
        self._sin_cached: Optional[Tensor] = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        assert self._cos_cached is not None and self._sin_cached is not None
        return (
            self._cos_cached[:, :, :seq_len, :].to(dtype=dtype),
            self._sin_cached[:, :, :seq_len, :].to(dtype=dtype),
        )


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to a tensor whose last dimension is split into two halves."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1)


def _probe_sdpa_enable_gqa() -> bool:
    """Probe whether the current PyTorch build accepts ``enable_gqa=True``."""
    try:
        q = torch.randn(1, 4, 2, 8)
        k = torch.randn(1, 2, 2, 8)
        v = torch.randn(1, 2, 2, 8)
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return True
    except Exception:
        return False


_SDPA_ENABLE_GQA: Optional[bool] = None


def sdpa_enable_gqa_available() -> bool:
    """Return whether the current runtime supports ``enable_gqa=True`` in SDPA."""
    global _SDPA_ENABLE_GQA
    if _SDPA_ENABLE_GQA is None:
        _SDPA_ENABLE_GQA = _probe_sdpa_enable_gqa()
    return _SDPA_ENABLE_GQA


def parse_share_pattern(pattern: str) -> tuple[str, int]:
    """Parse a share-pattern string such as ``chunk`` or ``repeat_2``."""
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
    raise ValueError(f"Unknown share pattern {pattern!r}")


def make_layer_to_block(
    num_layers: int, num_shared_blocks: int, pattern: str
) -> Tensor:
    """Create the fixed virtual-layer to shared-block assignment tensor."""
    share_pattern, share_repeat_n = parse_share_pattern(pattern)
    out = torch.empty(num_layers, dtype=torch.long)
    for layer_idx in range(num_layers):
        if share_pattern == "cycle":
            block_idx = layer_idx % num_shared_blocks
        elif share_pattern == "repeat":
            block_idx = (layer_idx // share_repeat_n) % num_shared_blocks
        else:
            block_idx = (layer_idx * num_shared_blocks) // num_layers
        out[layer_idx] = int(block_idx)
    return out


def prenorm_with_x0(
    x: Tensor,
    x0: Optional[Tensor],
    gate: Optional[Tensor],
    weight: Tensor,
    eps: float,
) -> Tensor:
    """Additively mix in ``x0`` and apply RMSNorm using plain PyTorch ops.

    This intentionally stays as simple PyTorch so `torch.compile` can fuse the
    elementwise pieces instead of being blocked by a custom autograd boundary.
    """
    if x0 is not None and gate is not None:
        x = x + torch.sigmoid(gate).to(dtype=x.dtype)[None, None, :] * x0
    return F.rms_norm(x, (x.size(-1),), weight.to(dtype=x.dtype), eps=eps)


# -----------------------------------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------------------------------


@dataclass
class HyperSharedConfig:
    """Configuration for the compile-friendly shared ALlama model."""

    vocab_size: int
    model_dim: int = 768
    embed_dim: int = 768
    num_layers: int = 24
    num_shared_blocks: int = 4
    share_pattern: str = "repeat_2"
    num_heads: int = 12
    num_kv_heads: int = 4
    mlp_mult: float = 2.5
    mlp_multiple_of: int = 64
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    norm_kind: str = "rmsnorm"
    norm_layout: str = "prenorm"
    qk_norm: bool = True
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    logit_softcap: float = 30.0
    q_gain_init: float = 1.5
    x0_gate_init: float = -6.0
    use_x0_shortcut: bool = True
    use_final_norm: bool = True
    zero_init_residual: bool = True
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    use_bias: bool = False
    cast_linears: bool = True


# -----------------------------------------------------------------------------
# SHARED BLOCKS
# -----------------------------------------------------------------------------


class HyperSharedAttention(nn.Module):
    """Shared attention block with simple per-layer q-gain modulation."""

    def __init__(self, cfg: HyperSharedConfig):
        super().__init__()
        if cfg.model_dim % cfg.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if cfg.num_heads % cfg.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = int(cfg.num_heads)
        self.num_kv_heads = int(cfg.num_kv_heads)
        self.head_dim = int(cfg.model_dim // cfg.num_heads)
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qk_norm = bool(cfg.qk_norm)
        self.attn_dropout = float(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(float(cfg.resid_dropout))
        linear_cls = CastedLinear if cfg.cast_linears else nn.Linear
        self.qkv = linear_cls(
            cfg.model_dim,
            cfg.model_dim + 2 * self.kv_dim,
            bias=cfg.use_bias,
        )
        self.proj = linear_cls(cfg.model_dim, cfg.model_dim, bias=cfg.use_bias)
        self.proj._zero_init = True
        self.rotary = Rotary(self.head_dim, base=cfg.rope_base)

    def forward(
        self,
        x: Tensor,
        q_gain: Tensor,
    ) -> Tensor:
        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split((dim, self.kv_dim, self.kv_dim), dim=-1)

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * q_gain.to(dtype=q.dtype)[None, :, None, None]

        if self.num_kv_heads != self.num_heads and sdpa_enable_gqa_available():
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

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.resid_dropout(self.proj(y))


class HyperSharedSwiGLU(nn.Module):
    """Shared SwiGLU MLP."""

    def __init__(self, cfg: HyperSharedConfig):
        super().__init__()
        hidden = int(cfg.model_dim * cfg.mlp_mult)
        hidden = cfg.mlp_multiple_of * (
            (hidden + cfg.mlp_multiple_of - 1) // cfg.mlp_multiple_of
        )
        self.hidden = hidden
        self.resid_dropout = nn.Dropout(float(cfg.resid_dropout))
        linear_cls = CastedLinear if cfg.cast_linears else nn.Linear
        self.gate_up = linear_cls(cfg.model_dim, hidden * 2, bias=cfg.use_bias)
        self.down = linear_cls(hidden, cfg.model_dim, bias=cfg.use_bias)
        self.down._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.resid_dropout(self.down(F.silu(gate) * up))


class HyperSharedBlock(nn.Module):
    """One shared transformer block with simple per-layer modulations."""

    def __init__(self, cfg: HyperSharedConfig):
        super().__init__()
        if cfg.norm_kind.strip().lower() != "rmsnorm":
            raise ValueError(f"v4 shared model expects RMSNorm; got {cfg.norm_kind!r}")
        self.norm_layout = cfg.norm_layout.strip().lower()
        if self.norm_layout not in {"prenorm", "postnorm"}:
            raise ValueError(
                f"v4 shared model expects prenorm or postnorm layout; got {cfg.norm_layout!r}"
            )
        self.attn_norm = RMSNormWeight(cfg.model_dim, eps=cfg.norm_eps)
        self.mlp_norm = RMSNormWeight(cfg.model_dim, eps=cfg.norm_eps)
        self.attn = HyperSharedAttention(cfg)
        self.mlp = HyperSharedSwiGLU(cfg)

    def forward(
        self,
        x: Tensor,
        x0: Optional[Tensor],
        attn_scale: Tensor,
        mlp_scale: Tensor,
        q_gain: Tensor,
        x0_gate: Optional[Tensor],
    ) -> Tensor:
        attn_scale = attn_scale.to(dtype=x.dtype)[None, None, :]
        mlp_scale = mlp_scale.to(dtype=x.dtype)[None, None, :]

        if self.norm_layout == "prenorm":
            attn_in = prenorm_with_x0(
                x,
                x0,
                x0_gate,
                self.attn_norm.weight,
                self.attn_norm.eps,
            )
            attn_out = self.attn(attn_in, q_gain=q_gain)
            x = x + attn_scale * attn_out
            mlp_out = self.mlp(self.mlp_norm(x))
            x = x + mlp_scale * mlp_out
            return x

        attn_in = x
        if x0 is not None and x0_gate is not None:
            attn_in = attn_in + x0_gate.to(dtype=x.dtype)[None, None, :] * x0
        attn_out = self.attn(attn_in, q_gain=q_gain)
        x = self.attn_norm(x + attn_scale * attn_out)
        mlp_out = self.mlp(x)
        x = self.mlp_norm(x + mlp_scale * mlp_out)
        return x


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------


class HyperSharedALlama(nn.Module):
    """Decoder-only compile-friendly shared-parameter ALlama model."""

    def __init__(self, cfg: HyperSharedConfig):
        super().__init__()
        if cfg.num_shared_blocks <= 0:
            raise ValueError("num_shared_blocks must be positive")
        if cfg.num_shared_blocks > cfg.num_layers:
            raise ValueError("num_shared_blocks cannot exceed num_layers")
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.embed_to_model = (
            nn.Identity()
            if cfg.embed_dim == cfg.model_dim
            else nn.Linear(cfg.embed_dim, cfg.model_dim, bias=False)
        )
        self.stem_norm = RMSNormWeight(cfg.model_dim, eps=cfg.norm_eps)
        self.shared_blocks = nn.ModuleList(
            [HyperSharedBlock(cfg) for _ in range(cfg.num_shared_blocks)]
        )
        layer_to_block = make_layer_to_block(
            cfg.num_layers,
            cfg.num_shared_blocks,
            cfg.share_pattern,
        )
        self.register_buffer("layer_to_block_idx", layer_to_block, persistent=False)
        self._layer_to_block_tuple = tuple(int(idx) for idx in layer_to_block.tolist())
        self._layer_blocks = tuple(
            self.shared_blocks[idx] for idx in self._layer_to_block_tuple
        )
        self.attn_scale = nn.Parameter(
            torch.ones(cfg.num_layers, cfg.model_dim, dtype=torch.float32)
        )
        self.mlp_scale = nn.Parameter(
            torch.ones(cfg.num_layers, cfg.model_dim, dtype=torch.float32)
        )
        self.q_gain = nn.Parameter(
            torch.full(
                (cfg.num_layers, cfg.num_heads),
                float(cfg.q_gain_init),
                dtype=torch.float32,
            )
        )
        if cfg.use_x0_shortcut:
            self.x0_gate = nn.Parameter(
                torch.full(
                    (cfg.num_layers, cfg.model_dim),
                    float(cfg.x0_gate_init),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("x0_gate", None)
        self.final_norm = (
            RMSNormWeight(cfg.model_dim, eps=cfg.norm_eps)
            if cfg.use_final_norm
            else nn.Identity()
        )
        if cfg.tie_embeddings:
            self.model_to_embed = (
                nn.Identity()
                if cfg.embed_dim == cfg.model_dim
                else nn.Linear(cfg.model_dim, cfg.embed_dim, bias=False)
            )
            self.lm_head = None
        else:
            self.model_to_embed = None
            self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.cfg.tie_embeddings:
            nn.init.normal_(
                self.token_embedding.weight,
                mean=0.0,
                std=float(self.cfg.tied_embed_init_std),
            )
        else:
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False) and self.cfg.zero_init_residual:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if isinstance(self.embed_to_model, nn.Linear):
            nn.init.xavier_uniform_(self.embed_to_model.weight)
        if isinstance(self.model_to_embed, nn.Linear):
            nn.init.xavier_uniform_(self.model_to_embed.weight)

    def layer_to_block_map(self) -> list[int]:
        """Return the complete virtual-layer to shared-block assignment map."""
        return list(self._layer_to_block_tuple)

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Optional[Tensor] = None,
        *,
        loss_reduction: str = "mean",
    ) -> Tensor:
        x = self.token_embedding(input_ids)
        x = self.embed_to_model(x)
        x = self.stem_norm(x)
        x0 = x if self.cfg.use_x0_shortcut else None
        x0_gate = self.x0_gate

        for layer_idx, block in enumerate(self._layer_blocks):
            x = block(
                x,
                x0=x0,
                attn_scale=self.attn_scale[layer_idx],
                mlp_scale=self.mlp_scale[layer_idx],
                q_gain=self.q_gain[layer_idx],
                x0_gate=(x0_gate[layer_idx] if x0_gate is not None else None),
            )

        x = self.final_norm(x)
        if self.cfg.tie_embeddings:
            assert self.model_to_embed is not None
            logits_proj = self.model_to_embed(x)
            logits = F.linear(logits_proj, self.token_embedding.weight)
        else:
            assert self.lm_head is not None
            logits = self.lm_head(x)

        if self.cfg.logit_softcap > 0.0:
            logits = self.cfg.logit_softcap * torch.tanh(
                logits / self.cfg.logit_softcap
            )

        if target_ids is None:
            return logits

        if loss_reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"unsupported loss_reduction {loss_reduction!r}")

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
# TRAINING HELPERS
# -----------------------------------------------------------------------------


CONTROL_NAME_PATTERNS = (
    "norm",
    "attn_scale",
    "mlp_scale",
    "q_gain",
    "x0_gate",
)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    """Keep low-dimensional and control parameters in fp32 storage."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2
                or any(pattern in name for pattern in CONTROL_NAME_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


@dataclass
class OptimizerBundle:
    """Container for the multi-optimizer training stack."""

    optimizers: list[torch.optim.Optimizer]
    token_lr: float
    head_lr: float
    matrix_lr: float
    scalar_lr: float

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def set_lr_multiplier(self, lr_mul: float) -> None:
        for opt in self.optimizers:
            for group in opt.param_groups:
                base_lr = float(group.get("base_lr", group["lr"]))
                group["lr"] = base_lr * lr_mul

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "token_lr": float(self.token_lr),
            "head_lr": float(self.head_lr),
            "matrix_lr": float(self.matrix_lr),
            "scalar_lr": float(self.scalar_lr),
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        states = state_dict.get("optimizers", [])
        if len(states) != len(self.optimizers):
            raise ValueError(
                f"optimizer state count mismatch: expected {len(self.optimizers)}, got {len(states)}"
            )
        for opt, opt_state in zip(self.optimizers, states):
            opt.load_state_dict(opt_state)


def _is_control_param(name: str, param: Tensor) -> bool:
    return param.ndim < 2 or any(pattern in name for pattern in CONTROL_NAME_PATTERNS)


def _adam_kwargs(beta1: float, beta2: float, eps: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"betas": (beta1, beta2), "eps": eps}
    if torch.cuda.is_available():
        kwargs["fused"] = True
    return kwargs


def build_allama_optimizers(
    model: HyperSharedALlama,
    *,
    tied_embed_lr: float = 0.03,
    head_lr: float = 0.01,
    matrix_lr: float = 0.02,
    scalar_lr: float = 0.04,
    beta1: float = 0.9,
    beta2: float = 0.95,
    adam_eps: float = 1e-8,
    muon_momentum: float = 0.95,
    muon_backend_steps: int = 5,
) -> OptimizerBundle:
    """Build the Muon + Adam split tailored to the shared ALlama model."""
    token_params: list[Tensor] = []
    head_params: list[Tensor] = []
    matrix_params: list[Tensor] = []
    scalar_params: list[Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "token_embedding.weight":
            token_params.append(param)
            continue
        if name.startswith("lm_head") or name.startswith("model_to_embed"):
            head_params.append(param)
            continue
        if _is_control_param(name, param):
            scalar_params.append(param)
            continue
        if param.ndim == 2:
            matrix_params.append(param)
        else:
            scalar_params.append(param)

    optimizers: list[torch.optim.Optimizer] = []
    adam_kwargs = _adam_kwargs(beta1, beta2, adam_eps)

    if token_params:
        opt_tok = torch.optim.Adam(
            [{"params": token_params, "lr": tied_embed_lr, "base_lr": tied_embed_lr}],
            **adam_kwargs,
        )
        optimizers.append(opt_tok)

    if head_params:
        opt_head = torch.optim.Adam(
            [{"params": head_params, "lr": head_lr, "base_lr": head_lr}],
            **adam_kwargs,
        )
        optimizers.append(opt_head)

    if matrix_params:
        opt_muon = Muon(
            matrix_params,
            lr=matrix_lr,
            momentum=muon_momentum,
            backend_steps=muon_backend_steps,
        )
        for group in opt_muon.param_groups:
            group["base_lr"] = float(matrix_lr)
        optimizers.append(opt_muon)

    if scalar_params:
        opt_scalar = torch.optim.Adam(
            [{"params": scalar_params, "lr": scalar_lr, "base_lr": scalar_lr}],
            **adam_kwargs,
        )
        optimizers.append(opt_scalar)

    return OptimizerBundle(
        optimizers=optimizers,
        token_lr=float(tied_embed_lr),
        head_lr=float(head_lr),
        matrix_lr=float(matrix_lr),
        scalar_lr=float(scalar_lr),
    )


# -----------------------------------------------------------------------------
# OPTIONAL QAT HELPER
# -----------------------------------------------------------------------------


def ste_fake_quant_symmetric(
    w: Tensor,
    bits: int = 6,
    per_row: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Symmetric STE fake quantizer."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    if per_row and w.ndim == 2:
        scale = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax
    else:
        scale = w.detach().abs().amax().clamp_min(eps) / qmax
    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    deq = q * scale
    return w + (deq - w).detach()


class FakeQuantLinear(CastedLinear):
    """CastedLinear that fake-quantizes its weight during training."""

    def __init__(
        self, *args: Any, qat_bits: int = 6, qat_per_row: bool = True, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.qat_bits = int(qat_bits)
        self.qat_per_row = bool(qat_per_row)

    def forward(self, x: Tensor) -> Tensor:
        w = (
            ste_fake_quant_symmetric(
                self.weight,
                bits=self.qat_bits,
                per_row=self.qat_per_row,
            )
            if self.training
            else self.weight
        )
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


ALlamaV4 = HyperSharedALlama


# -----------------------------------------------------------------------------
# TRAINER INTEGRATION
# -----------------------------------------------------------------------------


def configure_cuda_fastpath(sdpa_backend: str = "flash") -> dict[str, bool | str]:
    """Set fast CUDA defaults and an explicit SDPA backend.

    :param str sdpa_backend: Requested SDPA backend name.
    :return dict[str, bool | str]: Effective CUDA SDPA capability flags.
    """
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    backend = sdpa_backend.strip().lower()
    if backend == "auto":
        pass
    elif backend == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    elif backend == "efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
    elif backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    elif backend == "cudnn":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(True)
    else:
        raise ValueError(f"unsupported sdpa backend {sdpa_backend!r}")

    return {
        "requested": backend,
        "flash_available": bool(torch.backends.cuda.is_flash_attention_available()),
        "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "cudnn_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
    }


def compile_model_for_train(
    model: nn.Module,
    enabled: bool,
    *,
    fullgraph: bool = True,
) -> nn.Module:
    """Compile the training graph with static shapes.

    :param nn.Module model: Model to compile.
    :param bool enabled: Whether compilation is enabled.
    :param bool fullgraph: Whether to require fullgraph capture.
    :return nn.Module: Compiled model when enabled, else the original model.
    """
    if not enabled or not hasattr(torch, "compile"):
        return model
    return torch.compile(model, dynamic=False, fullgraph=fullgraph)


def build_hyper_shared_model_from_args(
    args: Any,
    vocab_size: int,
    device: torch.device,
) -> HyperSharedALlama:
    """Map a trainer args object into the shared-model config.

    :param Any args: Trainer argument object exposing the shared-model fields.
    :param int vocab_size: Token vocabulary size.
    :param torch.device device: Target device for the initialized model.
    :return HyperSharedALlama: Initialized shared ALlama model.
    """
    cfg = HyperSharedConfig(
        vocab_size=vocab_size,
        model_dim=int(args.model_dim),
        embed_dim=int(args.embed_dim),
        num_layers=int(args.num_layers),
        num_shared_blocks=int(args.num_shared_blocks),
        share_pattern=str(args.share_pattern),
        num_heads=int(args.num_heads),
        num_kv_heads=int(args.num_kv_heads),
        mlp_mult=float(args.mlp_mult),
        mlp_multiple_of=int(getattr(args, "mlp_multiple_of", 64)),
        rope_base=float(getattr(args, "rope_base", 10000.0)),
        norm_eps=float(getattr(args, "norm_eps", 1e-5)),
        norm_kind=str(getattr(args, "norm_kind", "rmsnorm")),
        norm_layout=str(getattr(args, "norm_layout", "prenorm")),
        qk_norm=bool(getattr(args, "qk_norm", True)),
        tie_embeddings=bool(getattr(args, "tie_embeddings", True)),
        tied_embed_init_std=float(getattr(args, "tied_embed_init_std", 0.005)),
        logit_softcap=float(getattr(args, "logit_softcap", 30.0)),
        q_gain_init=float(getattr(args, "q_gain_init", 1.5)),
        x0_gate_init=float(getattr(args, "x0_gate_init", -6.0)),
        use_x0_shortcut=bool(getattr(args, "use_x0_shortcut", True)),
        use_final_norm=bool(getattr(args, "use_final_norm", True)),
        zero_init_residual=bool(getattr(args, "zero_init_residual", True)),
        attn_dropout=float(getattr(args, "attn_dropout", 0.0)),
        resid_dropout=float(getattr(args, "resid_dropout", 0.0)),
        use_bias=bool(getattr(args, "use_bias", False)),
        cast_linears=bool(getattr(args, "cast_linears", True)),
    )

    model = HyperSharedALlama(cfg).to(device)
    dtype_name = str(getattr(args, "dtype", "auto")).lower()
    if device.type == "cuda" and dtype_name in {"bf16", "bfloat16", "auto"}:
        model = model.bfloat16()
        restore_low_dim_params_to_fp32(model)
    elif device.type == "cuda" and dtype_name in {"fp16", "float16"}:
        model = model.half()
        restore_low_dim_params_to_fp32(model)
    return model


def swap_selected_linears_to_qat(
    model: nn.Module,
    *,
    qat_bits: int = 6,
    include_patterns: tuple[str, ...] = (
        "qkv",
        "proj",
        "gate_up",
        "down",
        "embed_to_model",
        "model_to_embed",
    ),
) -> nn.Module:
    """Recursively replace selected linears with QAT variants.

    :param nn.Module model: Model to rewrite in-place.
    :param int qat_bits: Target fake-quant bit width.
    :param tuple[str, ...] include_patterns: Name patterns identifying eligible layers.
    :return nn.Module: The rewritten model.
    """

    def replace(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (CastedLinear, nn.Linear)) and any(
                pat in full_name for pat in include_patterns
            ):
                qat = FakeQuantLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    qat_bits=qat_bits,
                )
                qat.weight.data.copy_(child.weight.data)
                if child.bias is not None and qat.bias is not None:
                    qat.bias.data.copy_(child.bias.data)
                setattr(parent, name, qat)
            else:
                replace(child, full_name)

    replace(model)
    return model


def build_hyper_shared_optimizers_from_args(
    model: HyperSharedALlama, args: Any
) -> OptimizerBundle:
    """Build the Muon + Adam optimizer stack from a trainer args object.

    :param HyperSharedALlama model: Shared model whose parameters will be optimized.
    :param Any args: Trainer argument object exposing optimizer hyperparameters.
    :return OptimizerBundle: Multi-optimizer bundle for shared-model training.
    """
    return build_allama_optimizers(
        model,
        tied_embed_lr=float(
            getattr(args, "tied_embed_lr", getattr(args, "embed_lr", 0.03))
        ),
        head_lr=float(getattr(args, "head_lr", 0.01)),
        matrix_lr=float(
            getattr(args, "matrix_lr", getattr(args, "learning_rate", 0.02))
        ),
        scalar_lr=float(getattr(args, "scalar_lr", 0.04)),
        beta1=float(getattr(args, "beta1", 0.9)),
        beta2=float(getattr(args, "beta2", 0.95)),
        adam_eps=float(getattr(args, "adam_eps", 1e-8)),
        muon_momentum=float(getattr(args, "muon_momentum", 0.95)),
        muon_backend_steps=int(getattr(args, "muon_backend_steps", 5)),
    )
