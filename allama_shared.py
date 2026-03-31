"""Compile-friendly shared-parameter ALlama components.

This is the v4 rewrite for the Parameter Golf setting. The design deliberately
backs away from the v3 custom Triton/autograd path and from the per-forward
Python controller objects that were blocking `torch.compile` from doing useful
fusion and unrolling.

Key choices in this version:
- plain PyTorch elementwise ops only; no custom `autograd.Function` wrappers
- additive x0 reinjection + direct PyTorch norm ops
- simple learned per-layer modulation tensors instead of a controller network
- fixed layer-to-block assignment computed once at init
- Muon + Adam optimizer split retained from the v3 work
- optional larger Triton block kernels behind `ATTN_KERNEL` / `MLP_KERNEL`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


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


class LayerNormWeightBias(nn.Module):
    """Minimal LayerNorm storing learned scale and bias vectors."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x,
            (x.size(-1),),
            self.weight.to(dtype=x.dtype),
            self.bias.to(dtype=x.dtype),
            eps=self.eps,
        )


def build_norm(dim: int, kind: str, eps: float) -> nn.Module:
    """Construct the requested normalization layer."""
    norm_kind = kind.strip().lower()
    if norm_kind == "rmsnorm":
        return RMSNormWeight(dim, eps=eps)
    if norm_kind == "layernorm":
        return LayerNormWeightBias(dim, eps=eps)
    raise ValueError(f"expected norm_kind in {{'rmsnorm', 'layernorm'}}, got {kind!r}")


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


def _compiler_disable(fn):
    """Return a callable excluded from Dynamo/Inductor capture."""
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
        return torch.compiler.disable(fn)
    return torch._dynamo.disable(fn)


@_compiler_disable
def _probe_sdpa_enable_gqa() -> bool:
    """Probe whether the current runtime backend accepts ``enable_gqa=True``."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        q = torch.randn(1, 4, 2, 8, device=device, dtype=dtype)
        k = torch.randn(1, 2, 2, 8, device=device, dtype=dtype)
        v = torch.randn(1, 2, 2, 8, device=device, dtype=dtype)
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return True
    except Exception:
        return False


_SDPA_ENABLE_GQA: dict[tuple[bool, bool, bool, bool, bool], bool] = {}


def _sdpa_backend_signature() -> tuple[bool, bool, bool, bool, bool]:
    """Return a cache key for the active SDPA backend configuration."""
    if not torch.cuda.is_available():
        return (False, False, False, False, False)
    return (
        True,
        bool(torch.backends.cuda.flash_sdp_enabled()),
        bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        bool(torch.backends.cuda.math_sdp_enabled()),
        bool(torch.backends.cuda.cudnn_sdp_enabled()),
    )


@_compiler_disable
def sdpa_enable_gqa_available() -> bool:
    """Return whether the current runtime supports ``enable_gqa=True`` in SDPA."""
    signature = _sdpa_backend_signature()
    cached = _SDPA_ENABLE_GQA.get(signature)
    if cached is None:
        cached = _probe_sdpa_enable_gqa()
        _SDPA_ENABLE_GQA[signature] = cached
    return cached


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


def mix_x0(
    x: Tensor,
    x0: Optional[Tensor],
    gate: Optional[Tensor],
) -> Tensor:
    """Additively mix in ``x0`` with a bounded gate using plain PyTorch ops.

    This intentionally stays as simple PyTorch so `torch.compile` can fuse the
    elementwise pieces instead of being blocked by a custom autograd boundary.
    The same bounded gate is used in both prenorm and postnorm paths so the
    layout comparison is not confounded by different shortcut parameterization.
    """
    if x0 is not None and gate is not None:
        x = x + torch.sigmoid(gate).to(dtype=x.dtype)[None, None, :] * x0
    return x


def rms_norm_backward_total(
    mixed: Tensor,
    weight: Tensor,
    grad_normed_out: Optional[Tensor],
    grad_mixed_out: Optional[Tensor],
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Combine direct residual grads with RMSNorm backward.

    :param Tensor mixed: Mixed residual activation ``[M, D]``.
    :param Tensor weight: RMSNorm weight vector ``[D]``.
    :param Optional[Tensor] grad_normed_out: Gradient on the normalized output.
    :param Optional[Tensor] grad_mixed_out: Direct gradient on the mixed output.
    :param float eps: RMSNorm epsilon.
    :return tuple[Tensor, Tensor]: Total gradient on ``mixed`` and grad on ``weight``.
    """
    grad_total_f = (
        torch.zeros_like(mixed, dtype=torch.float32)
        if grad_mixed_out is None
        else grad_mixed_out.float()
    )
    grad_weight = torch.zeros_like(weight, dtype=torch.float32)
    if grad_normed_out is None:
        return grad_total_f.to(dtype=mixed.dtype), grad_weight.to(dtype=weight.dtype)

    mixed_f = mixed.float()
    weight_f = weight.float()[None, :]
    grad_normed_f = grad_normed_out.float()
    inv_rms = (mixed_f.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    grad_h = grad_normed_f * weight_f
    dot = (grad_h * mixed_f).sum(dim=-1, keepdim=True)
    grad_total_f = (
        grad_total_f
        + inv_rms * grad_h
        - inv_rms.pow(3) * mixed_f * dot / float(mixed.size(-1))
    )
    grad_weight = (grad_normed_f * mixed_f * inv_rms).sum(dim=0)
    return grad_total_f.to(dtype=mixed.dtype), grad_weight.to(dtype=weight.dtype)


_ALLAMA_MLP_GATEUP_OP: Optional[Any] = None
_ALLAMA_MLP_GATEUP_BACKWARD_PARTS_OP: Optional[Any] = None
_ALLAMA_ATTN_RMS_BRIDGE_OP: Optional[Any] = None
_ALLAMA_MLP_FULL_OP: Optional[Any] = None


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=4,
                num_stages=3,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=4,
                num_stages=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=8,
                num_stages=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                num_warps=8,
                num_stages=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
                num_warps=8,
                num_stages=5,
            ),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def gateup_swiglu_kernel(
        x_ptr,
        gate_weight_t_ptr,
        up_weight_t_ptr,
        hidden_ptr,
        M,
        N,
        K,
        x_stride_m,
        x_stride_k,
        gate_stride_k,
        gate_stride_n,
        up_stride_k,
        up_stride_n,
        hidden_stride_m,
        hidden_stride_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Fuse gate/up projections with the SwiGLU epilogue."""
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_base in range(0, K, BLOCK_K):
            k_idx = k_base + offs_k
            mask_k = k_idx < K
            x_ptrs = x_ptr + offs_m[:, None] * x_stride_m + k_idx[None, :] * x_stride_k
            x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            gate_ptrs = (
                gate_weight_t_ptr
                + k_idx[:, None] * gate_stride_k
                + offs_n[None, :] * gate_stride_n
            )
            up_ptrs = (
                up_weight_t_ptr
                + k_idx[:, None] * up_stride_k
                + offs_n[None, :] * up_stride_n
            )
            gate_weight = tl.load(
                gate_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            )
            up_weight = tl.load(
                up_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            )
            acc_gate += tl.dot(x_vals, gate_weight)
            acc_up += tl.dot(x_vals, up_weight)

        hidden = acc_gate * tl.sigmoid(acc_gate) * acc_up
        hidden_ptrs = (
            hidden_ptr
            + offs_m[:, None] * hidden_stride_m
            + offs_n[None, :] * hidden_stride_n
        )
        tl.store(
            hidden_ptrs,
            hidden.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=4,
                num_stages=3,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=4,
                num_stages=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                num_warps=8,
                num_stages=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                num_warps=8,
                num_stages=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
                num_warps=8,
                num_stages=5,
            ),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def gateup_swiglu_backward_parts_kernel(
        x_ptr,
        gate_weight_t_ptr,
        up_weight_t_ptr,
        grad_hidden_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        M,
        N,
        K,
        x_stride_m,
        x_stride_k,
        gate_stride_k,
        gate_stride_n,
        up_stride_k,
        up_stride_n,
        grad_hidden_stride_m,
        grad_hidden_stride_n,
        grad_gate_stride_m,
        grad_gate_stride_n,
        grad_up_stride_m,
        grad_up_stride_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Fuse gate/up recompute with the SwiGLU derivative epilogue."""
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_base in range(0, K, BLOCK_K):
            k_idx = k_base + offs_k
            mask_k = k_idx < K
            x_ptrs = x_ptr + offs_m[:, None] * x_stride_m + k_idx[None, :] * x_stride_k
            x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            gate_ptrs = (
                gate_weight_t_ptr
                + k_idx[:, None] * gate_stride_k
                + offs_n[None, :] * gate_stride_n
            )
            up_ptrs = (
                up_weight_t_ptr
                + k_idx[:, None] * up_stride_k
                + offs_n[None, :] * up_stride_n
            )
            gate_weight = tl.load(
                gate_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            )
            up_weight = tl.load(
                up_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            )
            acc_gate += tl.dot(x_vals, gate_weight)
            acc_up += tl.dot(x_vals, up_weight)

        grad_hidden_ptrs = (
            grad_hidden_ptr
            + offs_m[:, None] * grad_hidden_stride_m
            + offs_n[None, :] * grad_hidden_stride_n
        )
        grad_hidden = tl.load(
            grad_hidden_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
        ).to(tl.float32)
        sigmoid_gate = tl.sigmoid(acc_gate)
        silu_gate = acc_gate * sigmoid_gate
        silu_prime = sigmoid_gate * (1.0 + acc_gate * (1.0 - sigmoid_gate))
        grad_gate = grad_hidden * acc_up * silu_prime
        grad_up = grad_hidden * silu_gate

        grad_gate_ptrs = (
            grad_gate_ptr
            + offs_m[:, None] * grad_gate_stride_m
            + offs_n[None, :] * grad_gate_stride_n
        )
        grad_up_ptrs = (
            grad_up_ptr
            + offs_m[:, None] * grad_up_stride_m
            + offs_n[None, :] * grad_up_stride_n
        )
        tl.store(
            grad_gate_ptrs,
            grad_gate.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )
        tl.store(
            grad_up_ptrs,
            grad_up.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def gateup_swiglu_triton(
    x_norm: Tensor,
    gate_weight_t: Tensor,
    up_weight_t: Tensor,
) -> Tensor:
    """Run the fused Triton gate/up projection and SwiGLU kernel."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    hidden = torch.empty(
        x_norm.size(0),
        gate_weight_t.size(1),
        device=x_norm.device,
        dtype=x_norm.dtype,
    )

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (
            triton.cdiv(x_norm.size(0), meta["BLOCK_M"])
            * triton.cdiv(gate_weight_t.size(1), meta["BLOCK_N"]),
        )

    gateup_swiglu_kernel[grid](
        x_norm,
        gate_weight_t,
        up_weight_t,
        hidden,
        x_norm.size(0),
        gate_weight_t.size(1),
        x_norm.size(1),
        x_norm.stride(0),
        x_norm.stride(1),
        gate_weight_t.stride(0),
        gate_weight_t.stride(1),
        up_weight_t.stride(0),
        up_weight_t.stride(1),
        hidden.stride(0),
        hidden.stride(1),
    )
    return hidden


def gateup_swiglu_backward_parts_triton(
    x_norm: Tensor,
    gate_weight_t: Tensor,
    up_weight_t: Tensor,
    grad_hidden: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run the Triton fused gate/up derivative kernel."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    grad_gate = torch.empty_like(grad_hidden)
    grad_up = torch.empty_like(grad_hidden)

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (
            triton.cdiv(x_norm.size(0), meta["BLOCK_M"])
            * triton.cdiv(gate_weight_t.size(1), meta["BLOCK_N"]),
        )

    gateup_swiglu_backward_parts_kernel[grid](
        x_norm,
        gate_weight_t,
        up_weight_t,
        grad_hidden,
        grad_gate,
        grad_up,
        x_norm.size(0),
        gate_weight_t.size(1),
        x_norm.size(1),
        x_norm.stride(0),
        x_norm.stride(1),
        gate_weight_t.stride(0),
        gate_weight_t.stride(1),
        up_weight_t.stride(0),
        up_weight_t.stride(1),
        grad_hidden.stride(0),
        grad_hidden.stride(1),
        grad_gate.stride(0),
        grad_gate.stride(1),
        grad_up.stride(0),
        grad_up.stride(1),
    )
    return grad_gate, grad_up


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def attn_outproj_residual_kernel(
        attn_y_ptr,
        proj_weight_t_ptr,
        residual_x_ptr,
        scale_ptr,
        out_ptr,
        M,
        N,
        K,
        seq_len,
        head_dim,
        attn_stride_b,
        attn_stride_h,
        attn_stride_t,
        attn_stride_d,
        proj_stride_k,
        proj_stride_n,
        residual_stride_m,
        residual_stride_n,
        out_stride_m,
        out_stride_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fuse head-major attention output projection and residual add."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        b_idx = offs_m // seq_len
        t_idx = offs_m % seq_len

        for k_base in range(0, K, BLOCK_K):
            k_idx = k_base + offs_k
            mask_k = k_idx < K
            h_idx = k_idx // head_dim
            d_idx = k_idx % head_dim

            attn_ptrs = (
                attn_y_ptr
                + b_idx[:, None] * attn_stride_b
                + h_idx[None, :] * attn_stride_h
                + t_idx[:, None] * attn_stride_t
                + d_idx[None, :] * attn_stride_d
            )
            attn_vals = tl.load(
                attn_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
            )
            weight_ptrs = (
                proj_weight_t_ptr
                + k_idx[:, None] * proj_stride_k
                + offs_n[None, :] * proj_stride_n
            )
            weight = tl.load(
                weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            )
            acc += tl.dot(attn_vals, weight)

        residual_ptrs = (
            residual_x_ptr
            + offs_m[:, None] * residual_stride_m
            + offs_n[None, :] * residual_stride_n
        )
        residual = tl.load(
            residual_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
        ).to(tl.float32)
        scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        out = residual + acc * scale[None, :]
        out_ptrs = (
            out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n
        )
        tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
        ],
        key=["M", "N", "K"],
        reset_to_zero=["grad_scale_ptr"],
    )
    @triton.jit
    def attn_grad_scale_kernel(
        attn_y_ptr,
        proj_weight_t_ptr,
        grad_out_ptr,
        grad_scale_ptr,
        M,
        N,
        K,
        seq_len,
        head_dim,
        attn_stride_b,
        attn_stride_h,
        attn_stride_t,
        attn_stride_d,
        proj_stride_k,
        proj_stride_n,
        grad_out_stride_m,
        grad_out_stride_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fuse branch recompute with the reduction for ``grad_scale``."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        b_idx = offs_m // seq_len
        t_idx = offs_m % seq_len

        for k_base in range(0, K, BLOCK_K):
            k_idx = k_base + offs_k
            mask_k = k_idx < K
            h_idx = k_idx // head_dim
            d_idx = k_idx % head_dim

            attn_ptrs = (
                attn_y_ptr
                + b_idx[:, None] * attn_stride_b
                + h_idx[None, :] * attn_stride_h
                + t_idx[:, None] * attn_stride_t
                + d_idx[None, :] * attn_stride_d
            )
            attn_vals = tl.load(
                attn_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
            )
            weight_ptrs = (
                proj_weight_t_ptr
                + k_idx[:, None] * proj_stride_k
                + offs_n[None, :] * proj_stride_n
            )
            weight = tl.load(
                weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            )
            acc += tl.dot(attn_vals, weight)

        grad_out_ptrs = (
            grad_out_ptr
            + offs_m[:, None] * grad_out_stride_m
            + offs_n[None, :] * grad_out_stride_n
        )
        grad_out = tl.load(
            grad_out_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
        ).to(tl.float32)
        partial = tl.sum(acc * grad_out, axis=0)
        tl.atomic_add(grad_scale_ptr + offs_n, partial, mask=mask_n)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def attn_grad_attn_y_kernel(
        grad_branch_ptr,
        proj_weight_t_ptr,
        grad_attn_y_ptr,
        M,
        N,
        K,
        seq_len,
        head_dim,
        grad_branch_stride_m,
        grad_branch_stride_n,
        proj_stride_k,
        proj_stride_n,
        grad_attn_stride_b,
        grad_attn_stride_h,
        grad_attn_stride_t,
        grad_attn_stride_d,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute ``grad_attn_y`` directly into head-major layout."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_k = offs_k < K
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for n_base in range(0, N, BLOCK_N):
            n_idx = n_base + offs_n
            mask_n = n_idx < N
            grad_branch_ptrs = (
                grad_branch_ptr
                + offs_m[:, None] * grad_branch_stride_m
                + n_idx[None, :] * grad_branch_stride_n
            )
            grad_branch_vals = tl.load(
                grad_branch_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
            )
            weight_ptrs = (
                proj_weight_t_ptr
                + offs_k[:, None] * proj_stride_k
                + n_idx[None, :] * proj_stride_n
            )
            weight = tl.load(
                weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            )
            acc += tl.dot(grad_branch_vals, tl.trans(weight))

        b_idx = offs_m // seq_len
        t_idx = offs_m % seq_len
        h_idx = offs_k // head_dim
        d_idx = offs_k % head_dim
        out_ptrs = (
            grad_attn_y_ptr
            + b_idx[:, None] * grad_attn_stride_b
            + h_idx[None, :] * grad_attn_stride_h
            + t_idx[:, None] * grad_attn_stride_t
            + d_idx[None, :] * grad_attn_stride_d
        )
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_k[None, :])

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def attn_grad_proj_weight_t_kernel(
        attn_y_ptr,
        grad_branch_ptr,
        grad_proj_weight_t_ptr,
        M,
        N,
        K,
        seq_len,
        head_dim,
        attn_stride_b,
        attn_stride_h,
        attn_stride_t,
        attn_stride_d,
        grad_branch_stride_m,
        grad_branch_stride_n,
        grad_proj_stride_k,
        grad_proj_stride_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute ``grad_proj_weight_t`` without flattening the head-major input."""
        pid_k = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        mask_k = offs_k < K
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

        h_idx = offs_k // head_dim
        d_idx = offs_k % head_dim

        for m_base in range(0, M, BLOCK_M):
            m_idx = m_base + offs_m
            mask_m = m_idx < M
            b_idx = m_idx // seq_len
            t_idx = m_idx % seq_len
            attn_ptrs = (
                attn_y_ptr
                + b_idx[None, :] * attn_stride_b
                + h_idx[:, None] * attn_stride_h
                + t_idx[None, :] * attn_stride_t
                + d_idx[:, None] * attn_stride_d
            )
            attn_vals = tl.load(
                attn_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0
            )
            grad_branch_ptrs = (
                grad_branch_ptr
                + m_idx[:, None] * grad_branch_stride_m
                + offs_n[None, :] * grad_branch_stride_n
            )
            grad_branch_vals = tl.load(
                grad_branch_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
            )
            acc += tl.dot(attn_vals, grad_branch_vals)

        out_ptrs = (
            grad_proj_weight_t_ptr
            + offs_k[:, None] * grad_proj_stride_k
            + offs_n[None, :] * grad_proj_stride_n
        )
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_k[:, None] & mask_n[None, :])


def attn_outproj_residual_triton(
    residual_x: Tensor,
    attn_y: Tensor,
    proj_weight_t: Tensor,
    scale: Tensor,
) -> Tensor:
    """Run the fused Triton attention-outproj-residual kernel."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, num_heads, seq_len, head_dim = attn_y.shape
    model_dim = num_heads * head_dim
    out = torch.empty_like(residual_x)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(residual_x.size(0), meta["BLOCK_M"]),
            triton.cdiv(residual_x.size(1), meta["BLOCK_N"]),
        )

    attn_outproj_residual_kernel[grid](
        attn_y,
        proj_weight_t,
        residual_x,
        scale,
        out,
        residual_x.size(0),
        residual_x.size(1),
        model_dim,
        seq_len,
        head_dim,
        attn_y.stride(0),
        attn_y.stride(1),
        attn_y.stride(2),
        attn_y.stride(3),
        proj_weight_t.stride(0),
        proj_weight_t.stride(1),
        residual_x.stride(0),
        residual_x.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def attn_grad_scale_triton(
    attn_y: Tensor,
    proj_weight_t: Tensor,
    grad_out: Tensor,
) -> Tensor:
    """Run the fused branch-recompute reduction for ``grad_scale``."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, num_heads, seq_len, head_dim = attn_y.shape
    model_dim = num_heads * head_dim
    grad_scale = torch.zeros(model_dim, device=grad_out.device, dtype=torch.float32)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(grad_out.size(0), meta["BLOCK_M"]),
            triton.cdiv(grad_out.size(1), meta["BLOCK_N"]),
        )

    attn_grad_scale_kernel[grid](
        attn_y,
        proj_weight_t,
        grad_out,
        grad_scale,
        grad_out.size(0),
        grad_out.size(1),
        model_dim,
        seq_len,
        head_dim,
        attn_y.stride(0),
        attn_y.stride(1),
        attn_y.stride(2),
        attn_y.stride(3),
        proj_weight_t.stride(0),
        proj_weight_t.stride(1),
        grad_out.stride(0),
        grad_out.stride(1),
    )
    return grad_scale


def attn_grad_attn_y_triton(
    grad_branch: Tensor,
    proj_weight_t: Tensor,
    seq_len: int,
    head_dim: int,
) -> Tensor:
    """Run the Triton backward kernel for ``grad_attn_y``."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    model_dim = proj_weight_t.size(0)
    batch = grad_branch.size(0) // seq_len
    num_heads = model_dim // head_dim
    grad_attn_y = torch.empty(
        batch,
        num_heads,
        seq_len,
        head_dim,
        device=grad_branch.device,
        dtype=grad_branch.dtype,
    )

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(grad_branch.size(0), meta["BLOCK_M"]),
            triton.cdiv(model_dim, meta["BLOCK_K"]),
        )

    attn_grad_attn_y_kernel[grid](
        grad_branch,
        proj_weight_t,
        grad_attn_y,
        grad_branch.size(0),
        grad_branch.size(1),
        model_dim,
        seq_len,
        head_dim,
        grad_branch.stride(0),
        grad_branch.stride(1),
        proj_weight_t.stride(0),
        proj_weight_t.stride(1),
        grad_attn_y.stride(0),
        grad_attn_y.stride(1),
        grad_attn_y.stride(2),
        grad_attn_y.stride(3),
    )
    return grad_attn_y


def attn_grad_proj_weight_t_triton(
    attn_y: Tensor,
    grad_branch: Tensor,
) -> Tensor:
    """Run the Triton backward kernel for ``grad_proj_weight_t``."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, num_heads, seq_len, head_dim = attn_y.shape
    model_dim = num_heads * head_dim
    grad_proj_weight_t = torch.empty(
        model_dim,
        grad_branch.size(1),
        device=grad_branch.device,
        dtype=grad_branch.dtype,
    )

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(model_dim, meta["BLOCK_K"]),
            triton.cdiv(grad_branch.size(1), meta["BLOCK_N"]),
        )

    attn_grad_proj_weight_t_kernel[grid](
        attn_y,
        grad_branch,
        grad_proj_weight_t,
        grad_branch.size(0),
        grad_branch.size(1),
        model_dim,
        seq_len,
        head_dim,
        attn_y.stride(0),
        attn_y.stride(1),
        attn_y.stride(2),
        attn_y.stride(3),
        grad_branch.stride(0),
        grad_branch.stride(1),
        grad_proj_weight_t.stride(0),
        grad_proj_weight_t.stride(1),
    )
    return grad_proj_weight_t


def register_allama_mlp_gateup_custom_op() -> Optional[Any]:
    """Register the optional Triton gate/up custom op once."""
    global _ALLAMA_MLP_GATEUP_OP, _ALLAMA_MLP_GATEUP_BACKWARD_PARTS_OP
    if _ALLAMA_MLP_GATEUP_OP is not None:
        return _ALLAMA_MLP_GATEUP_OP
    if triton is None:
        return None

    @torch.library.custom_op(
        "allama_triton::gateup_swiglu_backward_parts",
        mutates_args=(),
    )
    def gateup_swiglu_backward_parts_op(
        x_norm: Tensor,
        gate_weight_t: Tensor,
        up_weight_t: Tensor,
        grad_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return gateup_swiglu_backward_parts_triton(
            x_norm, gate_weight_t, up_weight_t, grad_hidden
        )

    @gateup_swiglu_backward_parts_op.register_fake
    def _gateup_swiglu_backward_parts_op_fake(
        x_norm: Tensor,
        gate_weight_t: Tensor,
        up_weight_t: Tensor,
        grad_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del x_norm, gate_weight_t, up_weight_t
        return grad_hidden.new_empty(grad_hidden.shape), grad_hidden.new_empty(
            grad_hidden.shape
        )

    @torch.library.custom_op(
        "allama_triton::gateup_swiglu",
        mutates_args=(),
    )
    def gateup_swiglu_op(
        x_norm: Tensor,
        gate_weight_t: Tensor,
        up_weight_t: Tensor,
    ) -> Tensor:
        return gateup_swiglu_triton(x_norm, gate_weight_t, up_weight_t)

    @gateup_swiglu_op.register_fake
    def _gateup_swiglu_op_fake(
        x_norm: Tensor,
        gate_weight_t: Tensor,
        up_weight_t: Tensor,
    ) -> Tensor:
        del up_weight_t
        return x_norm.new_empty((x_norm.size(0), gate_weight_t.size(1)))

    def setup_context(
        ctx: Any,
        inputs: tuple[Tensor, Tensor, Tensor],
        output: Tensor,
    ) -> None:
        del output
        x_norm, gate_weight_t, up_weight_t = inputs
        ctx.save_for_backward(x_norm, gate_weight_t, up_weight_t)

    def backward(ctx: Any, grad_hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_norm, gate_weight_t, up_weight_t = ctx.saved_tensors
        grad_gate, grad_up = gateup_swiglu_backward_parts_op(
            x_norm, gate_weight_t, up_weight_t, grad_hidden
        )
        grad_x = grad_gate @ gate_weight_t.transpose(0, 1)
        grad_x = grad_x + grad_up @ up_weight_t.transpose(0, 1)
        grad_gate_weight_t = x_norm.transpose(0, 1) @ grad_gate
        grad_up_weight_t = x_norm.transpose(0, 1) @ grad_up
        return grad_x, grad_gate_weight_t, grad_up_weight_t

    torch.library.register_autograd(
        "allama_triton::gateup_swiglu",
        backward,
        setup_context=setup_context,
    )
    _ALLAMA_MLP_GATEUP_BACKWARD_PARTS_OP = gateup_swiglu_backward_parts_op
    _ALLAMA_MLP_GATEUP_OP = gateup_swiglu_op
    return _ALLAMA_MLP_GATEUP_OP


def register_allama_attn_rms_bridge_custom_op() -> Optional[Any]:
    """Register the fused attention out-proj plus next-RMSNorm bridge op."""
    global _ALLAMA_ATTN_RMS_BRIDGE_OP
    if _ALLAMA_ATTN_RMS_BRIDGE_OP is not None:
        return _ALLAMA_ATTN_RMS_BRIDGE_OP
    if triton is None:
        return None

    @torch.library.custom_op(
        "allama_triton::attn_grad_scale",
        mutates_args=(),
    )
    def attn_grad_scale_op(
        attn_y: Tensor,
        proj_weight_t: Tensor,
        grad_out: Tensor,
    ) -> Tensor:
        return attn_grad_scale_triton(attn_y, proj_weight_t, grad_out)

    @attn_grad_scale_op.register_fake
    def _attn_grad_scale_op_fake(
        attn_y: Tensor,
        proj_weight_t: Tensor,
        grad_out: Tensor,
    ) -> Tensor:
        del attn_y, proj_weight_t
        return grad_out.new_empty((grad_out.size(1),), dtype=torch.float32)

    @torch.library.custom_op(
        "allama_triton::attn_grad_attn_y",
        mutates_args=(),
    )
    def attn_grad_attn_y_op(
        grad_branch: Tensor,
        proj_weight_t: Tensor,
        seq_len: int,
        head_dim: int,
    ) -> Tensor:
        return attn_grad_attn_y_triton(grad_branch, proj_weight_t, seq_len, head_dim)

    @attn_grad_attn_y_op.register_fake
    def _attn_grad_attn_y_op_fake(
        grad_branch: Tensor,
        proj_weight_t: Tensor,
        seq_len: int,
        head_dim: int,
    ) -> Tensor:
        model_dim = proj_weight_t.size(0)
        batch = grad_branch.size(0) // seq_len
        num_heads = model_dim // head_dim
        return grad_branch.new_empty((batch, num_heads, seq_len, head_dim))

    @torch.library.custom_op(
        "allama_triton::attn_grad_proj_weight_t",
        mutates_args=(),
    )
    def attn_grad_proj_weight_t_op(
        attn_y: Tensor,
        grad_branch: Tensor,
    ) -> Tensor:
        return attn_grad_proj_weight_t_triton(attn_y, grad_branch)

    @attn_grad_proj_weight_t_op.register_fake
    def _attn_grad_proj_weight_t_op_fake(
        attn_y: Tensor,
        grad_branch: Tensor,
    ) -> Tensor:
        model_dim = attn_y.size(1) * attn_y.size(3)
        return grad_branch.new_empty((model_dim, grad_branch.size(1)))

    @torch.library.custom_op(
        "allama_triton::attn_rms_bridge",
        mutates_args=(),
    )
    def attn_rms_bridge_op(
        residual_x: Tensor,
        attn_y: Tensor,
        proj_weight_t: Tensor,
        scale: Tensor,
        norm_weight: Tensor,
        eps: float,
    ) -> tuple[Tensor, Tensor]:
        mixed = attn_outproj_residual_triton(residual_x, attn_y, proj_weight_t, scale)
        normed = F.rms_norm(
            mixed,
            (mixed.size(-1),),
            norm_weight.to(dtype=mixed.dtype),
            eps=eps,
        )
        return mixed, normed

    @attn_rms_bridge_op.register_fake
    def _attn_rms_bridge_op_fake(
        residual_x: Tensor,
        attn_y: Tensor,
        proj_weight_t: Tensor,
        scale: Tensor,
        norm_weight: Tensor,
        eps: float,
    ) -> tuple[Tensor, Tensor]:
        del attn_y, proj_weight_t, scale, norm_weight, eps
        return residual_x.new_empty(residual_x.shape), residual_x.new_empty(
            residual_x.shape
        )

    def setup_context(
        ctx: Any,
        inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, float],
        output: tuple[Tensor, Tensor],
    ) -> None:
        residual_x, attn_y, proj_weight_t, scale, norm_weight, eps = inputs
        del residual_x
        mixed, _ = output
        ctx.save_for_backward(mixed, attn_y, proj_weight_t, scale, norm_weight)
        ctx.eps = float(eps)

    def backward(
        ctx: Any,
        grad_mixed_out: Optional[Tensor],
        grad_normed_out: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, None]:
        mixed, attn_y, proj_weight_t, scale, norm_weight = ctx.saved_tensors
        grad_total, grad_norm_weight = rms_norm_backward_total(
            mixed,
            norm_weight,
            grad_normed_out,
            grad_mixed_out,
            ctx.eps,
        )
        scale_cast = scale.to(dtype=grad_total.dtype)[None, :]
        grad_x = grad_total
        grad_scale = attn_grad_scale_op(attn_y, proj_weight_t, grad_total).to(
            dtype=scale.dtype
        )
        grad_branch = grad_total * scale_cast
        grad_proj_weight_t = attn_grad_proj_weight_t_op(attn_y, grad_branch)
        grad_attn_y = attn_grad_attn_y_op(
            grad_branch,
            proj_weight_t,
            attn_y.size(2),
            attn_y.size(3),
        )
        return (
            grad_x,
            grad_attn_y,
            grad_proj_weight_t,
            grad_scale,
            grad_norm_weight,
            None,
        )

    torch.library.register_autograd(
        "allama_triton::attn_rms_bridge",
        backward,
        setup_context=setup_context,
    )
    _ALLAMA_ATTN_RMS_BRIDGE_OP = attn_rms_bridge_op
    return _ALLAMA_ATTN_RMS_BRIDGE_OP


def register_allama_mlp_full_custom_op() -> Optional[Any]:
    """Register the larger MLP block custom op once."""
    global _ALLAMA_MLP_FULL_OP
    if _ALLAMA_MLP_FULL_OP is not None:
        return _ALLAMA_MLP_FULL_OP
    gateup_op = register_allama_mlp_gateup_custom_op()
    backward_parts_op = _ALLAMA_MLP_GATEUP_BACKWARD_PARTS_OP
    if gateup_op is None or backward_parts_op is None:
        return None

    @torch.library.custom_op(
        "allama_triton::mlp_full",
        mutates_args=(),
    )
    def mlp_full_op(
        residual_x: Tensor,
        x_norm: Tensor,
        gate_weight_t: Tensor,
        up_weight_t: Tensor,
        down_weight_t: Tensor,
        scale: Tensor,
    ) -> Tensor:
        hidden = gateup_op(x_norm, gate_weight_t, up_weight_t)
        branch = hidden @ down_weight_t
        return residual_x + branch * scale.to(dtype=residual_x.dtype)[None, :]

    @mlp_full_op.register_fake
    def _mlp_full_op_fake(
        residual_x: Tensor,
        x_norm: Tensor,
        gate_weight_t: Tensor,
        up_weight_t: Tensor,
        down_weight_t: Tensor,
        scale: Tensor,
    ) -> Tensor:
        del x_norm, gate_weight_t, up_weight_t, down_weight_t, scale
        return residual_x.new_empty(residual_x.shape)

    def setup_context(
        ctx: Any,
        inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        output: Tensor,
    ) -> None:
        del output
        residual_x, x_norm, gate_weight_t, up_weight_t, down_weight_t, scale = inputs
        del residual_x
        ctx.save_for_backward(x_norm, gate_weight_t, up_weight_t, down_weight_t, scale)

    def backward(
        ctx: Any, grad_out: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        x_norm, gate_weight_t, up_weight_t, down_weight_t, scale = ctx.saved_tensors
        hidden = gateup_op(x_norm, gate_weight_t, up_weight_t)
        scale_cast = scale.to(dtype=grad_out.dtype)[None, :]
        grad_residual_x = grad_out
        grad_branch = grad_out * scale_cast
        branch = hidden @ down_weight_t
        grad_scale = (
            (grad_out.float() * branch.float()).sum(dim=0).to(dtype=scale.dtype)
        )
        grad_down_weight_t = hidden.transpose(0, 1) @ grad_branch
        grad_hidden = grad_branch @ down_weight_t.transpose(0, 1)
        grad_gate, grad_up = backward_parts_op(
            x_norm, gate_weight_t, up_weight_t, grad_hidden
        )
        grad_x_norm = grad_gate @ gate_weight_t.transpose(0, 1)
        grad_x_norm = grad_x_norm + grad_up @ up_weight_t.transpose(0, 1)
        grad_gate_weight_t = x_norm.transpose(0, 1) @ grad_gate
        grad_up_weight_t = x_norm.transpose(0, 1) @ grad_up
        return (
            grad_residual_x,
            grad_x_norm,
            grad_gate_weight_t,
            grad_up_weight_t,
            grad_down_weight_t,
            grad_scale,
        )

    torch.library.register_autograd(
        "allama_triton::mlp_full",
        backward,
        setup_context=setup_context,
    )
    _ALLAMA_MLP_FULL_OP = mlp_full_op
    return _ALLAMA_MLP_FULL_OP


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
    attn_kernel: str = "pytorch"
    mlp_kernel: str = "pytorch"


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
        self.use_sdpa_gqa = (
            self.num_kv_heads != self.num_heads and sdpa_enable_gqa_available()
        )
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

    def forward_heads(
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

        if self.use_sdpa_gqa:
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
        return y

    def forward(
        self,
        x: Tensor,
        q_gain: Tensor,
    ) -> Tensor:
        bsz, seqlen, dim = x.shape
        y = self.forward_heads(x, q_gain=q_gain)
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
        self.mlp_kernel = cfg.mlp_kernel.strip().lower()
        if self.mlp_kernel not in {"pytorch", "triton_gateup", "triton_full"}:
            raise ValueError(
                f"Unsupported mlp_kernel={cfg.mlp_kernel!r}; expected pytorch, triton_gateup, or triton_full."
            )
        self.gateup_op = (
            register_allama_mlp_gateup_custom_op()
            if self.mlp_kernel in {"triton_gateup", "triton_full"}
            else None
        )
        self.full_op = (
            register_allama_mlp_full_custom_op()
            if self.mlp_kernel == "triton_full"
            else None
        )
        self.resid_dropout = nn.Dropout(float(cfg.resid_dropout))
        linear_cls = CastedLinear if cfg.cast_linears else nn.Linear
        self.gate_up = linear_cls(cfg.model_dim, hidden * 2, bias=cfg.use_bias)
        self.down = linear_cls(hidden, cfg.model_dim, bias=cfg.use_bias)
        self.down._zero_init = True

    def project_hidden(self, x: Tensor) -> Tensor:
        """Return the hidden SwiGLU activation before the down projection."""
        if (
            self.gateup_op is not None
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and self.gate_up.bias is None
        ):
            weight_t = self.gate_up.weight.to(dtype=x.dtype).t()
            return self.gateup_op(
                x.reshape(-1, x.size(-1)),
                weight_t[:, : self.hidden],
                weight_t[:, self.hidden :],
            ).view(*x.shape[:-1], self.hidden)
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return F.silu(gate) * up

    def forward_full(self, residual_x: Tensor, x_norm: Tensor, scale: Tensor) -> Tensor:
        """Run the larger MLP boundary that includes down-proj and residual add."""
        if (
            self.mlp_kernel == "triton_full"
            and self.full_op is not None
            and residual_x.is_cuda
            and residual_x.dtype == torch.bfloat16
            and self.gate_up.bias is None
            and self.down.bias is None
            and self.resid_dropout.p == 0.0
        ):
            gate_weight_t = (
                self.gate_up.weight[: self.hidden].to(dtype=x_norm.dtype).t()
            )
            up_weight_t = self.gate_up.weight[self.hidden :].to(dtype=x_norm.dtype).t()
            down_weight_t = self.down.weight.to(dtype=x_norm.dtype).t()
            out = self.full_op(
                residual_x.reshape(-1, residual_x.size(-1)),
                x_norm.reshape(-1, x_norm.size(-1)),
                gate_weight_t,
                up_weight_t,
                down_weight_t,
                scale,
            )
            return out.view_as(residual_x)

        hidden = self.project_hidden(x_norm)
        branch = self.resid_dropout(self.down(hidden))
        return residual_x + scale.to(dtype=residual_x.dtype)[None, None, :] * branch

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.project_hidden(x)
        return self.resid_dropout(self.down(hidden))


class HyperSharedBlock(nn.Module):
    """One shared transformer block with simple per-layer modulations."""

    def __init__(self, cfg: HyperSharedConfig):
        super().__init__()
        self.norm_kind = cfg.norm_kind.strip().lower()
        if self.norm_kind not in {"rmsnorm", "layernorm"}:
            raise ValueError(
                f"v4 shared model expects RMSNorm or LayerNorm; got {cfg.norm_kind!r}"
            )
        self.norm_layout = cfg.norm_layout.strip().lower()
        if self.norm_layout not in {"prenorm", "postnorm"}:
            raise ValueError(
                f"v4 shared model expects prenorm or postnorm layout; got {cfg.norm_layout!r}"
            )
        self.attn_kernel = cfg.attn_kernel.strip().lower()
        if self.attn_kernel not in {"pytorch", "triton_bridge"}:
            raise ValueError(
                f"Unsupported attn_kernel={cfg.attn_kernel!r}; expected pytorch or triton_bridge."
            )
        self.attn_norm = build_norm(cfg.model_dim, cfg.norm_kind, cfg.norm_eps)
        self.mlp_norm = build_norm(cfg.model_dim, cfg.norm_kind, cfg.norm_eps)
        self.attn_bridge_op = (
            register_allama_attn_rms_bridge_custom_op()
            if self.attn_kernel == "triton_bridge"
            else None
        )
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
            attn_in = self.attn_norm(mix_x0(x, x0, x0_gate))
            if (
                self.attn_kernel == "triton_bridge"
                and self.attn_bridge_op is not None
                and x.is_cuda
                and x.dtype == torch.bfloat16
                and self.norm_kind == "rmsnorm"
                and isinstance(self.mlp_norm, RMSNormWeight)
                and self.attn.proj.bias is None
                and self.attn.resid_dropout.p == 0.0
            ):
                attn_y = self.attn.forward_heads(attn_in, q_gain=q_gain)
                mixed, mlp_in = self.attn_bridge_op(
                    x.reshape(-1, x.size(-1)),
                    attn_y,
                    self.attn.proj.weight.to(dtype=x.dtype).t(),
                    attn_scale.reshape(-1),
                    self.mlp_norm.weight,
                    float(self.mlp_norm.eps),
                )
                x = mixed.view_as(x)
                mlp_in = mlp_in.view_as(x)
            else:
                attn_out = self.attn(attn_in, q_gain=q_gain)
                x = x + attn_scale * attn_out
                mlp_in = self.mlp_norm(x)
            if self.mlp.mlp_kernel == "triton_full":
                x = self.mlp.forward_full(x, mlp_in, mlp_scale.reshape(-1))
            else:
                mlp_out = self.mlp(mlp_in)
                x = x + mlp_scale * mlp_out
            return x

        attn_in = mix_x0(x, x0, x0_gate)
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
        self.stem_norm = build_norm(cfg.model_dim, cfg.norm_kind, cfg.norm_eps)
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
            build_norm(cfg.model_dim, cfg.norm_kind, cfg.norm_eps)
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
        attn_kernel=str(getattr(args, "attn_kernel", "pytorch")),
        mlp_kernel=str(getattr(args, "mlp_kernel", "pytorch")),
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
