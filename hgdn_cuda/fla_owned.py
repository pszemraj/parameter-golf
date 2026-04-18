"""Owned HGDN recurrence orchestration over upstream FLA Triton kernels.

This module keeps the HGDN recurrence on a compile-safe boundary by bypassing
the upstream backend registry and its Python-side dispatch machinery. The
underlying Triton kernels still come from the installed FLA package; the
orchestration and operator boundary live here.
"""

from __future__ import annotations

import torch
import triton
from torch import Tensor

try:
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
    from fla.ops.common.chunk_o import (
        chunk_bwd_dqkwg,
        chunk_bwd_dv_local,
        chunk_fwd_o,
    )
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.gated_delta_rule.naive import naive_chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.wy_fast import (
        prepare_wy_repr_bwd,
        recompute_w_u_fwd,
    )
    from fla.ops.utils import chunk_local_cumsum, solve_tril
except ImportError:
    chunk_gated_delta_rule_bwd_dhu = None
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64 = None
    chunk_bwd_dqkwg = None
    chunk_bwd_dv_local = None
    chunk_fwd_o = None
    chunk_scaled_dot_kkt_fwd = None
    naive_chunk_gated_delta_rule = None
    prepare_wy_repr_bwd = None
    recompute_w_u_fwd = None
    chunk_local_cumsum = None
    solve_tril = None


HAS_OWNED_FLA_GATED_DELTA_RULE = all(
    fn is not None
    for fn in (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        chunk_bwd_dqkwg,
        chunk_bwd_dv_local,
        chunk_fwd_o,
        chunk_scaled_dot_kkt_fwd,
        naive_chunk_gated_delta_rule,
        prepare_wy_repr_bwd,
        recompute_w_u_fwd,
        chunk_local_cumsum,
        solve_tril,
    )
)

_CHUNK_SIZE = 64


def _require_owned_fla() -> None:
    """Raise when the owned HGDN recurrence stack is unavailable."""

    if not HAS_OWNED_FLA_GATED_DELTA_RULE:
        raise RuntimeError(
            "Owned HGDN gated-delta recurrence is unavailable; required FLA "
            "kernels or helpers could not be imported."
        )


def _chunk_gated_delta_rule_fwd_h_equal_length(
    k: Tensor,
    w: Tensor,
    u: Tensor,
    *,
    g: Tensor | None = None,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    save_new_value: bool = True,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Run the raw equal-length recurrence-state kernel without backend dispatch.

    :param Tensor k: Key tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor w: WY helper tensor shaped like ``k``.
    :param Tensor u: Updated value tensor shaped ``(batch, seq, heads, head_v)``.
    :param Tensor | None g: Cumulative log gates, defaults to `None`.
    :param Tensor | None initial_state: Optional initial state, defaults to `None`.
    :param bool output_final_state: Whether to emit the final state, defaults to `False`.
    :param bool save_new_value: Whether to emit ``v_new``, defaults to `True`.
    :param bool use_exp2: Whether to use exp2 math, defaults to `False`.
    :param bool transpose_state_layout: Whether to transpose state layout, defaults to `False`.
    :return tuple[Tensor, Tensor, Tensor | None]: State tensor, transformed value tensor, final state.
    """
    _require_owned_fla()

    B, T, H, K, V = *k.shape, u.shape[-1]
    NT = triton.cdiv(T, _CHUNK_SIZE)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if transpose_state_layout:
        h = k.new_empty(B, NT, H, V, K)
        final_state = (
            k.new_zeros(B, H, V, K, dtype=torch.float32) if output_final_state else None
        )
    else:
        h = k.new_empty(B, NT, H, K, V)
        final_state = (
            k.new_zeros(B, H, K, V, dtype=torch.float32) if output_final_state else None
        )

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        """Return the Triton launch grid for the equal-length state kernel.

        :param dict[str, int] meta: Triton compile-time launch metadata.
        :return tuple[int, int]: Launch grid over value tiles and batch-head rows.
        """
        return (triton.cdiv(V, meta["BV"]), B * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=None,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=None,
        chunk_offsets=None,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=_CHUNK_SIZE,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return h, v_new, final_state


def run_chunk_gated_delta_rule_owned(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    scale: float = 1.0,
) -> Tensor:
    """Run the owned HGDN recurrence orchestration.

    CUDA uses the upstream Triton kernels directly without the FLA backend
    registry. CPU falls back to the upstream naive reference.

    :param Tensor q: Query tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor k: Key tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor v: Value tensor shaped ``(batch, seq, heads, head_v)``.
    :param Tensor g: Log-space gate tensor shaped ``(batch, seq, heads)``.
    :param Tensor beta: Beta tensor shaped ``(batch, seq, heads)``.
    :param float scale: Attention scale, defaults to `1.0`.
    :return Tensor: Recurrence output shaped like ``v``.
    """
    _require_owned_fla()

    if q.device.type != "cuda":
        out, _ = naive_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=_CHUNK_SIZE,
            scale=scale,
            output_final_state=False,
        )
        return out.to(dtype=v.dtype).contiguous()

    g_cumsum = chunk_local_cumsum(g, chunk_size=_CHUNK_SIZE)
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_cumsum,
        beta=beta,
        output_dtype=torch.float32,
        chunk_size=_CHUNK_SIZE,
    )
    A = solve_tril(
        A=A,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g_cumsum,
    )
    h, v_new, _ = _chunk_gated_delta_rule_fwd_h_equal_length(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        output_final_state=False,
    )
    return chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cumsum,
        scale=scale,
        chunk_size=_CHUNK_SIZE,
    ).contiguous()


def run_chunk_gated_delta_rule_owned_backward(
    grad_output: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run the owned HGDN recurrence backward orchestration.

    :param Tensor grad_output: Gradient for the recurrence output.
    :param Tensor q: Query tensor.
    :param Tensor k: Key tensor.
    :param Tensor v: Value tensor.
    :param Tensor g: Log-space gate tensor.
    :param Tensor beta: Beta tensor.
    :param float scale: Attention scale, defaults to `1.0`.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Gradients for ``q, k, v, g, beta``.
    """
    _require_owned_fla()

    if q.device.type != "cuda":
        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            k_req = k.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)
            g_req = g.detach().requires_grad_(True)
            beta_req = beta.detach().requires_grad_(True)
            out = run_chunk_gated_delta_rule_owned(
                q_req,
                k_req,
                v_req,
                g_req,
                beta_req,
                scale=scale,
            )
        grad_q, grad_k, grad_v, grad_g, grad_beta = torch.autograd.grad(
            out,
            (q_req, k_req, v_req, g_req, beta_req),
            grad_outputs=grad_output,
            allow_unused=False,
        )
        return (
            grad_q.contiguous(),
            grad_k.contiguous(),
            grad_v.contiguous(),
            grad_g.contiguous(),
            grad_beta.contiguous(),
        )

    g_cumsum = chunk_local_cumsum(g, chunk_size=_CHUNK_SIZE)
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_cumsum,
        beta=beta,
        output_dtype=torch.float32,
        chunk_size=_CHUNK_SIZE,
    )
    A = solve_tril(
        A=A,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g_cumsum,
    )
    h, v_new, _ = _chunk_gated_delta_rule_fwd_h_equal_length(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        output_final_state=False,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g_cumsum,
        do=grad_output,
        scale=scale,
        chunk_size=_CHUNK_SIZE,
    )
    dh, _dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g_cumsum,
        h0=None,
        dht=None,
        do=grad_output,
        dv=dv,
        scale=scale,
        chunk_size=_CHUNK_SIZE,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g_cumsum,
        h=h,
        do=grad_output,
        dh=dh,
        dv=dv,
        scale=scale,
        chunk_size=_CHUNK_SIZE,
    )
    dk2, grad_v, grad_beta, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g_cumsum,
        A=A,
        dw=dw,
        du=dv,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    grad_g = chunk_local_cumsum(
        dg,
        chunk_size=_CHUNK_SIZE,
        reverse=True,
    )
    return (
        dq.contiguous(),
        dk.contiguous(),
        grad_v.contiguous(),
        grad_g.contiguous(),
        grad_beta.contiguous(),
    )
