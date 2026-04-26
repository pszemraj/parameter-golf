"""Compile-visible HGDN recurrence boundary over public FLA ops.

The installed FLA package owns the gated-delta kernel implementation. This
module keeps the HGDN trainer behind a stable `torch.library` operator boundary
without copying private FLA Triton-kernel signatures into the repo.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None
    naive_chunk_gated_delta_rule = None


HAS_OWNED_FLA_GATED_DELTA_RULE = chunk_gated_delta_rule is not None

_CHUNK_SIZE = 64


def _require_owned_fla() -> None:
    """Raise when the public FLA recurrence stack is unavailable."""
    if not HAS_OWNED_FLA_GATED_DELTA_RULE:
        raise RuntimeError(
            "HGDN gated-delta recurrence is unavailable; public FLA ops could "
            "not be imported."
        )


def run_chunk_gated_delta_rule_owned(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Run the HGDN recurrence through the installed public FLA op.

    :param Tensor q: Query tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor k: Key tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor v: Value tensor shaped ``(batch, seq, heads, head_v)``.
    :param Tensor g: Log-space gate tensor shaped ``(batch, seq, heads)``.
    :param Tensor beta: Beta tensor shaped ``(batch, seq, heads)``.
    :param float | None scale: Optional recurrence scale. `None` keeps the
        public FLA default of ``1 / sqrt(head_k)``.
    :return Tensor: Recurrence output shaped like ``v``.
    """
    _require_owned_fla()
    if q.device.type == "cuda":
        out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            output_final_state=False,
        )
        return out.contiguous()
    if naive_chunk_gated_delta_rule is None:
        raise RuntimeError("FLA naive gated-delta reference is unavailable on CPU.")
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


def run_chunk_gated_delta_rule_owned_fused_gate_norm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Run public FLA with in-kernel q/k L2 norm and decay-gate activation.

    :param Tensor q: Raw query tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor k: Raw key tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor v: Value tensor shaped ``(batch, seq, heads, head_v)``.
    :param Tensor g: Raw gate tensor shaped ``(batch, seq, heads)``.
    :param Tensor beta: Beta tensor shaped ``(batch, seq, heads)``.
    :param Tensor A_log: Decay scale parameter shaped ``(heads,)``.
    :param Tensor dt_bias: Decay dt-bias parameter shaped ``(heads,)``.
    :param float | None scale: Optional recurrence scale. `None` keeps the
        public FLA default of ``1 / sqrt(head_k)``.
    :raises RuntimeError: If CUDA is unavailable for this upstream-semantics path.
    :return Tensor: Recurrence output shaped like ``v``.
    """
    _require_owned_fla()
    if q.device.type != "cuda":
        raise RuntimeError(
            "The fused-gate/norm FLA path requires CUDA; the CPU naive reference "
            "does not expose use_qk_l2norm_in_kernel/use_gate_in_kernel."
        )
    out, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    return out.contiguous()


def run_chunk_gated_delta_rule_owned_backward(
    grad_output: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    scale: float | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Recompute the public FLA recurrence and differentiate it.

    :param Tensor grad_output: Gradient for the recurrence output.
    :param Tensor q: Query tensor.
    :param Tensor k: Key tensor.
    :param Tensor v: Value tensor.
    :param Tensor g: Log-space gate tensor.
    :param Tensor beta: Beta tensor.
    :param float | None scale: Optional recurrence scale. `None` keeps the
        public FLA default of ``1 / sqrt(head_k)``.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Gradients for inputs.
    """
    _require_owned_fla()
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
