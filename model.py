"""
GDN Hybrid Model for Parameter Golf — v3
=========================================
P0 fixes from code review:
  - Single n_heads for Q/K/V (matches FLA chunk_gated_delta_rule API)
  - Split projections: w_q/w_k/w_v (Muon) vs w_a/w_b/w_g (Adam)
  - Correct naive recurrence (no broken grouped-key averaging)
  - kernels.py removed from submission
  - depth_control config added
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_HAS_FLA = False
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    _HAS_FLA = True
except ImportError:
    pass

# Parameters routed to Adam (not Muon). Muon is for 2D feature maps only.
SCALAR_PARAM_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weight",
    "w_a.",
    "w_b.",
    "w_g.",  # GDN control projections → Adam
    "A_log",
    "dt_bias",  # GDN decay params → Adam
)

NormStyle = Literal["pre", "post", "keel"]


def validate_norm_style(norm_style: str) -> NormStyle:
    """Normalize and validate a residual norm-placement string.

    :param str norm_style: Requested normalization style.
    :raises ValueError: If the style is unsupported.
    :return NormStyle: Validated normalization style.
    """
    normalized = norm_style.lower()
    if normalized not in {"pre", "post", "keel"}:
        raise ValueError(
            f"norm_style must be 'pre', 'post', or 'keel', got {norm_style!r}"
        )
    return normalized


def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Apply RMS normalization over the last dimension.

    :param Tensor x: Input activations.
    :param float eps: Numerical stability epsilon, defaults to 1e-6.
    :return Tensor: Normalized activations.
    """
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def l2_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Apply L2 normalization over the last dimension.

    :param Tensor x: Input activations.
    :param float eps: Numerical stability epsilon, defaults to 1e-6.
    :return Tensor: Unit-normalized activations.
    """
    return F.normalize(x, p=2, dim=-1, eps=eps)


class CastedLinear(nn.Linear):
    """Linear projection that casts stored weights to the input dtype."""

    def forward(self, x: Tensor) -> Tensor:
        """Project activations using weights cast to the input dtype.

        :param Tensor x: Input activations.
        :return Tensor: Projected activations.
        """
        return F.linear(
            x,
            self.weight.to(x.dtype),
            self.bias.to(x.dtype) if self.bias is not None else None,
        )


class RMSNorm(nn.Module):
    """Parameter-free RMSNorm wrapper used throughout the hybrid stack."""

    def __init__(self, eps: float = 1e-6):
        """Initialize the RMSNorm wrapper.

        :param float eps: Numerical stability epsilon, defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize activations with RMSNorm.

        :param Tensor x: Input activations.
        :return Tensor: Normalized activations.
        """
        return rms_norm(x, self.eps)


class CausalConv1d(nn.Module):
    """Depthwise causal convolution used to preprocess sequence features."""

    def __init__(self, dim: int, kernel_size: int = 4):
        """Initialize the causal depthwise convolution.

        :param int dim: Channel count.
        :param int kernel_size: Convolution kernel width, defaults to 4.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size - 1, groups=dim, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the causal convolution over the sequence axis.

        :param Tensor x: Input activations shaped `(batch, seq, dim)`.
        :return Tensor: Convolved activations with causal trimming applied.
        """
        x = x.transpose(1, 2)
        x = self.conv(x)[..., : x.size(-1)]
        x = F.silu(x)
        return x.transpose(1, 2)


# ── Naive GDN recurrence ─────────────────────────────────────────────


def gdn_recurrent_naive(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    initial_state: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Run the reference gated delta recurrence on CPU or fallback paths.

    :param Tensor q: Normalized queries shaped `(batch, seq, heads, d_k)`.
    :param Tensor k: Normalized keys shaped `(batch, seq, heads, d_k)`.
    :param Tensor v: Values shaped `(batch, seq, heads, d_v)`.
    :param Tensor alpha: Decay multipliers shaped `(batch, seq, heads)`.
    :param Tensor beta: Write gates shaped `(batch, seq, heads)`.
    :param Optional[Tensor] initial_state: Optional initial state shaped `(batch, heads, d_k, d_v)`, defaults to None.
    :return tuple[Tensor, Tensor]: Sequence outputs and final recurrence state.
    """
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    q, k, v = q.float(), k.float(), v.float()
    S = (
        initial_state.float()
        if initial_state is not None
        else torch.zeros(B, H, Dk, Dv, device=q.device)
    )
    outputs = []
    for t in range(T):
        q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
        a_t = alpha[:, t, :, None, None]
        b_t = beta[:, t, :, None, None]
        kS = torch.einsum("bhd,bhdv->bhv", k_t, S)
        kkS = torch.einsum("bhd,bhv->bhdv", k_t, kS)
        S = a_t * (S - b_t * kkS) + torch.einsum("bhd,bhv->bhdv", k_t, v_t)
        outputs.append(torch.einsum("bhdv,bhd->bhv", S, q_t))
    return torch.stack(outputs, dim=1), S


# ── Gated DeltaNet layer ─────────────────────────────────────────────
#
# Matches FLA API: q, k, v all have n_heads heads.
# head_v_dim = head_k_dim * expand_v controls state capacity.
# Projections split: w_q/w_k/w_v are proper feature maps (→ Muon),
# w_a/w_b are tiny scalar controls, w_g is a sigmoid gate (→ Adam).


class GatedDeltaNet(nn.Module):
    """Hybrid sequence mixer that wraps the FLA Gated DeltaNet kernel."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        head_k_dim: int = 48,
        expand_v: float = 2.0,
        allow_neg_eigval: bool = True,
        conv_size: int = 4,
        use_fla: bool = True,
    ):
        """Initialize a Gated DeltaNet block.

        :param int d_model: Model width.
        :param int n_heads: Number of recurrence heads, defaults to 4.
        :param int head_k_dim: Per-head key width, defaults to 48.
        :param float expand_v: Value expansion relative to `head_k_dim`, defaults to 2.0.
        :param bool allow_neg_eigval: Whether to allow negative eigenvalues in the write gate, defaults to True.
        :param int conv_size: Depthwise causal convolution width, defaults to 4.
        :param bool use_fla: Whether to use the FLA kernel when available, defaults to True.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = int(head_k_dim * expand_v)
        assert self.head_v_dim % 2 == 0, f"head_v_dim={self.head_v_dim} must be even"
        self.allow_neg_eigval = allow_neg_eigval
        self.use_fla = use_fla and _HAS_FLA

        total_qk = n_heads * head_k_dim
        total_v = n_heads * self.head_v_dim

        # Feature-map projections → Muon
        self.w_q = CastedLinear(d_model, total_qk, bias=False)
        self.w_k = CastedLinear(d_model, total_qk, bias=False)
        self.w_v = CastedLinear(d_model, total_v, bias=False)

        # Control projections → Adam (tiny: d_model → n_heads each)
        self.w_a = CastedLinear(d_model, n_heads, bias=False)
        self.w_b = CastedLinear(d_model, n_heads, bias=False)

        # Output gate → Adam (sigmoid path, not a feature map)
        self.w_g = CastedLinear(d_model, total_v, bias=False)

        # Output projection → Muon
        self.w_out = CastedLinear(total_v, d_model, bias=False)
        self.w_out._zero_init = True

        # Learnable decay params → Adam
        self.A_log = nn.Parameter(torch.zeros(n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))

        # Short causal convolutions
        self.q_conv = CausalConv1d(total_qk, conv_size)
        self.k_conv = CausalConv1d(total_qk, conv_size)
        self.v_conv = CausalConv1d(total_v, conv_size)

    def _project_recurrence_inputs(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Project activations into recurrence inputs with aligned dtypes.

        :param Tensor x: Input activations shaped `(batch, seq, d_model)`.
        :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Normalized q/k, values, decay logits, and write gates.
        """
        B, T, D = x.shape
        dt = x.dtype
        H, Dk, Dv = self.n_heads, self.head_k_dim, self.head_v_dim

        # Project (separate matmuls — correct Muon/Adam routing)
        q = self.q_conv(self.w_q(x))
        k = self.k_conv(self.w_k(x))
        v = self.v_conv(self.w_v(x))

        # Reshape and normalize
        q = l2_norm(q.view(B, T, H, Dk)).to(dt)
        k = l2_norm(k.view(B, T, H, Dk)).to(dt)
        v = v.view(B, T, H, Dv)

        # Decay and write gate
        g = -self.A_log.exp() * F.softplus(
            self.w_a(x) + self.dt_bias
        )  # (B,T,H) log-space
        beta = torch.sigmoid(self.w_b(x))  # (B,T,H)
        if self.allow_neg_eigval:
            beta = beta * 2.0
        g = g.to(dtype=dt)
        beta = beta.to(dtype=dt)
        return q, k, v, g, beta

    def forward(self, x: Tensor) -> Tensor:
        """Apply the Gated DeltaNet layer to a sequence batch.

        :param Tensor x: Input activations shaped `(batch, seq, d_model)`.
        :return Tensor: Updated activations shaped `(batch, seq, d_model)`.
        """
        B, T, _ = x.shape
        dt = x.dtype
        H, Dv = self.n_heads, self.head_v_dim
        q, k, v, g, beta = self._project_recurrence_inputs(x)

        # Recurrence
        if self.use_fla and x.is_cuda:
            o, _ = chunk_gated_delta_rule(
                q, k, v, g, beta, scale=1.0, output_final_state=False
            )
        else:
            o, _ = gdn_recurrent_naive(q, k, v, g.exp(), beta)

        # Output gating
        g_out = self.w_g(x).view(B, T, H, Dv)
        o = rms_norm(o.float()).to(dt) * F.silu(g_out)
        return self.w_out(o.reshape(B, T, -1))


# ── Standard attention ────────────────────────────────────────────────


class Rotary(nn.Module):
    """Rotary embedding cache for attention heads."""

    def __init__(self, dim: int, base: float = 10000.0):
        """Initialize the rotary cache.

        :param int dim: Per-head dimension.
        :param float base: Rotary frequency base, defaults to 10000.0.
        """
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)),
            persistent=False,
        )
        self._cache_len = 0
        self._cos: Optional[Tensor] = None
        self._sin: Optional[Tensor] = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Return cached cosine and sine tables for a sequence length.

        :param int seq_len: Sequence length to cover.
        :param torch.device device: Target device.
        :param torch.dtype dtype: Target dtype for the cache.
        :return tuple[Tensor, Tensor]: Cosine and sine rotary tables.
        """
        if (
            self._cos is None
            or self._cache_len != seq_len
            or self._cos.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
            self._cache_len = seq_len
        return self._cos.to(dtype), self._sin.to(dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to the last dimension of a head tensor.

    :param Tensor x: Input tensor with paired rotary dimensions.
    :param Tensor cos: Cached cosine table.
    :param Tensor sin: Cached sine table.
    :return Tensor: Rotated tensor.
    """
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """Grouped-query causal self-attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
    ):
        """Initialize the causal attention module.

        :param int dim: Model width.
        :param int num_heads: Attention query head count.
        :param int num_kv_heads: Key/value head count for GQA.
        :param float rope_base: Rotary frequency base, defaults to 10000.0.
        :param float qk_gain_init: Initial per-head query gain, defaults to 1.5.
        """
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
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
        """Apply causal grouped-query attention.

        :param Tensor x: Input activations shaped `(batch, seq, dim)`.
        :return Tensor: Attention outputs shaped `(batch, seq, dim)`.
        """
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads)
        )
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, D))


# ── MLP ───────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """Squared LeakyReLU feed-forward block."""

    def __init__(self, dim: int, mult: float = 3.0, leaky_slope: float = 0.5):
        """Initialize the MLP block.

        :param int dim: Model width.
        :param float mult: Hidden width multiplier, defaults to 3.0.
        :param float leaky_slope: LeakyReLU slope, defaults to 0.5.
        """
        super().__init__()
        hidden = int(dim * mult)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.leaky_slope = leaky_slope

    def forward(self, x: Tensor) -> Tensor:
        """Apply the squared LeakyReLU MLP.

        :param Tensor x: Input activations.
        :return Tensor: MLP outputs.
        """
        h = F.leaky_relu(self.fc(x), negative_slope=self.leaky_slope)
        return self.proj(h * h)


# ── Blocks ────────────────────────────────────────────────────────────


class GDNBlock(nn.Module):
    """Residual transformer block with Gated DeltaNet mixing."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        mlp_mult: float = 3.0,
        head_k_dim: int = 48,
        expand_v: float = 2.0,
        allow_neg_eigval: bool = True,
        conv_size: int = 4,
        leaky_slope: float = 0.5,
        norm_style: NormStyle = "pre",
        residual_alpha: float = 1.0,
        is_first_block: bool = False,
    ):
        """Initialize the GDN residual block.

        :param int dim: Model width.
        :param int n_heads: GDN head count, defaults to 4.
        :param float mlp_mult: MLP expansion factor, defaults to 3.0.
        :param int head_k_dim: GDN key width per head, defaults to 48.
        :param float expand_v: GDN value expansion factor, defaults to 2.0.
        :param bool allow_neg_eigval: Whether to allow negative eigenvalues, defaults to True.
        :param int conv_size: Causal convolution width, defaults to 4.
        :param float leaky_slope: LeakyReLU slope, defaults to 0.5.
        :param NormStyle norm_style: Residual norm placement, defaults to "pre".
        :param float residual_alpha: Residual scaling factor used by KEEL-style blocks, defaults to 1.0.
        :param bool is_first_block: Whether this is the first block in the stack, defaults to False.
        """
        super().__init__()
        self.norm_style = validate_norm_style(norm_style)
        self.residual_alpha = residual_alpha
        self.is_first_block = is_first_block
        self.attn_in_norm, self.attn_out_norm = RMSNorm(), RMSNorm()
        self.mlp_in_norm, self.mlp_out_norm = RMSNorm(), RMSNorm()
        self.gdn = GatedDeltaNet(
            dim, n_heads, head_k_dim, expand_v, allow_neg_eigval, conv_size
        )
        self.mlp = MLP(dim, mlp_mult, leaky_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack([torch.ones(dim), torch.zeros(dim)]).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        """Apply the residual GDN block.

        :param Tensor x: Current activations.
        :param Tensor x0: Block-stack input used by residual mixing.
        :return Tensor: Updated activations.
        """
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_scale = self.attn_scale.to(dtype=x.dtype)[None, None, :]
        mlp_scale = self.mlp_scale.to(dtype=x.dtype)[None, None, :]
        if self.norm_style == "pre":
            x = x + attn_scale * self.gdn(self.attn_in_norm(x))
            x = x + mlp_scale * self.mlp(self.mlp_in_norm(x))
            return x
        if self.norm_style == "post":
            x = self.attn_out_norm(x + attn_scale * self.gdn(x))
            x = self.mlp_out_norm(x + mlp_scale * self.mlp(x))
            return x
        if self.is_first_block:
            x = x + attn_scale * self.gdn(self.attn_in_norm(x))
            x = x + mlp_scale * self.mlp(self.mlp_in_norm(x))
            return x
        x = self.attn_out_norm(
            self.residual_alpha * x + attn_scale * self.gdn(self.attn_in_norm(x))
        )
        x = self.mlp_out_norm(
            self.residual_alpha * x + mlp_scale * self.mlp(self.mlp_in_norm(x))
        )
        return x


class AttnBlock(nn.Module):
    """Residual transformer block with causal attention mixing."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float = 3.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        leaky_slope: float = 0.5,
        norm_style: NormStyle = "pre",
        residual_alpha: float = 1.0,
        is_first_block: bool = False,
    ):
        """Initialize the attention residual block.

        :param int dim: Model width.
        :param int num_heads: Query head count.
        :param int num_kv_heads: Key/value head count.
        :param float mlp_mult: MLP expansion factor, defaults to 3.0.
        :param float rope_base: Rotary frequency base, defaults to 10000.0.
        :param float qk_gain_init: Initial query gain, defaults to 1.5.
        :param float leaky_slope: LeakyReLU slope, defaults to 0.5.
        :param NormStyle norm_style: Residual norm placement, defaults to "pre".
        :param float residual_alpha: Residual scaling factor used by KEEL-style blocks, defaults to 1.0.
        :param bool is_first_block: Whether this is the first block in the stack, defaults to False.
        """
        super().__init__()
        self.norm_style = validate_norm_style(norm_style)
        self.residual_alpha = residual_alpha
        self.is_first_block = is_first_block
        self.attn_in_norm, self.attn_out_norm = RMSNorm(), RMSNorm()
        self.mlp_in_norm, self.mlp_out_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init
        )
        self.mlp = MLP(dim, mlp_mult, leaky_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack([torch.ones(dim), torch.zeros(dim)]).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        """Apply the residual attention block.

        :param Tensor x: Current activations.
        :param Tensor x0: Block-stack input used by residual mixing.
        :return Tensor: Updated activations.
        """
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


# ── Hybrid GPT ────────────────────────────────────────────────────────


class HybridGPT(nn.Module):
    """Interleaved GDN-attention language model used by the hybrid trainer."""

    def __init__(
        self,
        vocab_size: int = 1024,
        num_layers: int = 16,
        d_model: int = 384,
        # Attention config
        attn_heads: int = 8,
        attn_kv_heads: int = 4,
        # GDN config (single n_heads, expand_v controls state)
        gdn_n_heads: int = 4,
        gdn_head_k_dim: int = 48,
        gdn_expand_v: float = 2.0,
        gdn_allow_neg_eigval: bool = True,
        gdn_conv_size: int = 4,
        # Shared
        mlp_mult: float = 3.0,
        leaky_slope: float = 0.5,
        gdn_ratio: int = 3,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        logit_softcap: float = 30.0,
        tie_embeddings: bool = True,
        tied_embed_init_std: float = 0.005,
        norm_style: NormStyle = "pre",
        residual_alpha: float | None = None,
    ):
        """Initialize the hybrid language model.

        :param int vocab_size: Token vocabulary size, defaults to 1024.
        :param int num_layers: Total block count, defaults to 16.
        :param int d_model: Model width, defaults to 384.
        :param int attn_heads: Attention query head count, defaults to 8.
        :param int attn_kv_heads: Attention key/value head count, defaults to 4.
        :param int gdn_n_heads: GDN head count, defaults to 4.
        :param int gdn_head_k_dim: GDN key width per head, defaults to 48.
        :param float gdn_expand_v: GDN value expansion factor, defaults to 2.0.
        :param bool gdn_allow_neg_eigval: Whether to allow negative eigenvalues, defaults to True.
        :param int gdn_conv_size: GDN convolution width, defaults to 4.
        :param float mlp_mult: MLP expansion factor, defaults to 3.0.
        :param float leaky_slope: LeakyReLU slope, defaults to 0.5.
        :param int gdn_ratio: Number of GDN layers per attention layer, defaults to 3.
        :param float rope_base: Rotary frequency base, defaults to 10000.0.
        :param float qk_gain_init: Initial attention query gain, defaults to 1.5.
        :param float logit_softcap: Tanh logit softcap, defaults to 30.0.
        :param bool tie_embeddings: Whether to tie embedding and output weights, defaults to True.
        :param float tied_embed_init_std: Embedding init std when tied, defaults to 0.005.
        :param NormStyle norm_style: Residual norm placement inside each block, defaults to "pre".
        :param float | None residual_alpha: Optional residual scaling for KEEL-style blocks, defaults to None.
        """
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.d_model = d_model
        self.norm_style = validate_norm_style(norm_style)
        self.residual_alpha = (
            float(2 * num_layers)
            if residual_alpha is None and self.norm_style == "keel"
            else float(1.0 if residual_alpha is None else residual_alpha)
        )
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        period = gdn_ratio + 1
        blocks, self.block_types = [], []
        for i in range(num_layers):
            if (i + 1) % period == 0:
                blocks.append(
                    AttnBlock(
                        d_model,
                        attn_heads,
                        attn_kv_heads,
                        mlp_mult,
                        rope_base,
                        qk_gain_init,
                        leaky_slope,
                        self.norm_style,
                        self.residual_alpha,
                        i == 0,
                    )
                )
                self.block_types.append("attn")
            else:
                blocks.append(
                    GDNBlock(
                        d_model,
                        gdn_n_heads,
                        mlp_mult,
                        gdn_head_k_dim,
                        gdn_expand_v,
                        gdn_allow_neg_eigval,
                        gdn_conv_size,
                        leaky_slope,
                        self.norm_style,
                        self.residual_alpha,
                        i == 0,
                    )
                )
                self.block_types.append("gdn")
        self.blocks = nn.ModuleList(blocks)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, d_model, dtype=torch.float32)
        )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings else CastedLinear(d_model, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights(tied_embed_init_std)

    def _init_weights(self, tied_std: float) -> None:
        """Initialize embeddings and zero-init marked projections.

        :param float tied_std: Standard deviation for tied embedding init.
        """
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Compute the autoregressive training loss.

        :param Tensor input_ids: Input token ids shaped `(batch, seq)`.
        :param Tensor target_ids: Next-token targets shaped `(batch, seq)`.
        :return Tensor: Mean cross-entropy loss.
        """
        x = rms_norm(self.tok_emb(input_ids))
        x0, skips = x, []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = (
                    x
                    + self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = (
            F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        )
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")


# ── Presets ──────────────────────────────────────────────────────────
#
# These constructors mirror the historical sweep families. Earlier artifact-size
# comments here were based on a stale proxy estimate; use the trainer's final
# quantized artifact audit or docs/README.md for current measured bytes.


def make_hybrid_tight(vocab_size: int = 1024) -> HybridGPT:
    """Build the primary budget-filling hybrid preset.

    :param int vocab_size: Vocabulary size, defaults to 1024.
    :return HybridGPT: Hybrid preset with matched `Dk=Dv=48`.
    """
    return HybridGPT(
        vocab_size=vocab_size,
        num_layers=16,
        d_model=384,
        gdn_n_heads=8,
        gdn_head_k_dim=48,
        gdn_expand_v=1.0,
        gdn_ratio=3,
        mlp_mult=3.0,
    )


def make_hybrid_wide(vocab_size: int = 1024) -> HybridGPT:
    """Build the wider-state hybrid preset.

    :param int vocab_size: Vocabulary size, defaults to 1024.
    :return HybridGPT: Hybrid preset with wider per-head state.
    """
    return HybridGPT(
        vocab_size=vocab_size,
        num_layers=16,
        d_model=384,
        gdn_n_heads=4,
        gdn_head_k_dim=48,
        gdn_expand_v=2.0,
        gdn_ratio=3,
        mlp_mult=3.25,
    )


def make_baseline_fill(vocab_size: int = 1024) -> HybridGPT:
    """Build the width-matched pure-attention baseline.

    :param int vocab_size: Vocabulary size, defaults to 1024.
    :return HybridGPT: Pure-attention baseline preset.
    """
    return HybridGPT(
        vocab_size=vocab_size, num_layers=11, d_model=512, gdn_ratio=0, mlp_mult=2.75
    )


def make_depth_control(vocab_size: int = 1024) -> HybridGPT:
    """Build the depth-matched pure-attention control preset.

    :param int vocab_size: Vocabulary size, defaults to 1024.
    :return HybridGPT: Pure-attention control with matched depth.
    """
    return HybridGPT(
        vocab_size=vocab_size, num_layers=16, d_model=384, gdn_ratio=0, mlp_mult=3.75
    )
