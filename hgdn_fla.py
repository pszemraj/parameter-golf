"""Public-FLA recurrence wrappers used by the sparse HGDN trainer."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None
    naive_chunk_gated_delta_rule = None


HAS_FLA_GATED_DELTA_RULE = chunk_gated_delta_rule is not None
_CHUNK_SIZE = 64


def _dynamo_disable(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Disable TorchDynamo tracing for a function when available.

    :param Callable[..., Any] fn: Function to wrap.
    :return Callable[..., Any]: Wrapped function.
    """
    disable = getattr(torch._dynamo, "disable", None)
    if disable is None:
        return fn
    return disable(fn)


def _require_fla() -> None:
    """Raise when the public FLA recurrence stack is unavailable."""
    if not HAS_FLA_GATED_DELTA_RULE:
        raise RuntimeError(
            "HGDN gated-delta recurrence is unavailable; public FLA ops could "
            "not be imported."
        )


def _run_chunk_gated_delta_rule(
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
    :param float | None scale: Optional recurrence scale. `None` keeps public
        FLA's default of ``1 / sqrt(head_k)``.
    :return Tensor: Recurrence output shaped like ``v``.
    """
    _require_fla()
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


def _run_chunk_gated_delta_rule_fused_gate_norm(
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
    :param float | None scale: Optional recurrence scale. `None` keeps public
        FLA's default of ``1 / sqrt(head_k)``.
    :raises RuntimeError: If CUDA is unavailable for this upstream-semantics path.
    :return Tensor: Recurrence output shaped like ``v``.
    """
    _require_fla()
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


def _run_chunk_gated_delta_rule_backward(
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
    :param float | None scale: Optional recurrence scale. `None` keeps public
        FLA's default of ``1 / sqrt(head_k)``.
    :return tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Gradients for inputs.
    """
    _require_fla()
    with torch.enable_grad():
        q_req = q.detach().requires_grad_(True)
        k_req = k.detach().requires_grad_(True)
        v_req = v.detach().requires_grad_(True)
        g_req = g.detach().requires_grad_(True)
        beta_req = beta.detach().requires_grad_(True)
        out = _run_chunk_gated_delta_rule(
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


_FLA_V1_LIB = torch.library.Library("hgdn_fla_v1", "DEF")
_FLA_V1_LIB.define(
    "chunk_gated_delta_rule(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta) -> Tensor"
)
_FLA_V1_LIB.define(
    "chunk_gated_delta_rule_backward("
    "Tensor grad_output, Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta"
    ") -> (Tensor grad_q, Tensor grad_k, Tensor grad_v, Tensor grad_g, Tensor grad_beta)"
)


@torch.library.impl("hgdn_fla_v1::chunk_gated_delta_rule", "CPU")
def _fla_chunk_gated_delta_rule_cpu(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tensor:
    """CPU implementation for the compile-visible recurrence op."""
    return _run_chunk_gated_delta_rule(q, k, v, g, beta)


@torch.library.impl("hgdn_fla_v1::chunk_gated_delta_rule", "CUDA")
def _fla_chunk_gated_delta_rule_cuda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tensor:
    """CUDA implementation for the compile-visible recurrence op."""
    return _run_chunk_gated_delta_rule(q, k, v, g, beta)


@torch.library.impl("hgdn_fla_v1::chunk_gated_delta_rule_backward", "CPU")
def _fla_chunk_gated_delta_rule_backward_cpu(
    grad_output: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CPU implementation for the compile-visible recurrence backward op."""
    return _run_chunk_gated_delta_rule_backward(grad_output, q, k, v, g, beta)


@torch.library.impl("hgdn_fla_v1::chunk_gated_delta_rule_backward", "CUDA")
def _fla_chunk_gated_delta_rule_backward_cuda(
    grad_output: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CUDA implementation for the compile-visible recurrence backward op."""
    return _run_chunk_gated_delta_rule_backward(grad_output, q, k, v, g, beta)


@torch.library.register_fake("hgdn_fla_v1::chunk_gated_delta_rule")
def _fla_chunk_gated_delta_rule_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tensor:
    """Fake tensor implementation for recurrence compile tracing."""
    del q, k, g, beta
    return torch.empty_like(v)


@torch.library.register_fake("hgdn_fla_v1::chunk_gated_delta_rule_backward")
def _fla_chunk_gated_delta_rule_backward_fake(
    grad_output: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fake tensor implementation for recurrence backward compile tracing."""
    del grad_output
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g),
        torch.empty_like(beta),
    )


def _setup_fla_chunk_gated_delta_rule_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    output: Tensor,
) -> None:
    """Save recurrence inputs for custom autograd.

    :param torch.autograd.function.FunctionCtx ctx: Autograd context.
    :param tuple[Tensor, Tensor, Tensor, Tensor, Tensor] inputs: q/k/v/g/beta inputs.
    :param Tensor output: Forward output, unused.
    """
    del output
    ctx.save_for_backward(*inputs)


def _fla_chunk_gated_delta_rule_backward_formula(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Autograd formula for the compile-visible recurrence op."""
    q, k, v, g, beta = ctx.saved_tensors
    return torch.ops.hgdn_fla_v1.chunk_gated_delta_rule_backward(
        grad_output,
        q,
        k,
        v,
        g,
        beta,
    )


torch.library.register_autograd(
    "hgdn_fla_v1::chunk_gated_delta_rule",
    _fla_chunk_gated_delta_rule_backward_formula,
    setup_context=_setup_fla_chunk_gated_delta_rule_context,
)


def fla_chunk_gated_delta_rule_compile_visible(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tensor:
    """Run the FLA gated-delta recurrence through a compile-visible library op.

    :param Tensor q: Query tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor k: Key tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor v: Value tensor shaped ``(batch, seq, heads, head_v)``.
    :param Tensor g: Log-space gate tensor shaped ``(batch, seq, heads)``.
    :param Tensor beta: Beta tensor shaped ``(batch, seq, heads)``.
    :return Tensor: Recurrence output shaped like ``v``.
    """
    return torch.ops.hgdn_fla_v1.chunk_gated_delta_rule(q, k, v, g, beta)


@_dynamo_disable
def fla_chunk_gated_delta_rule_direct(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tensor:
    """Run the public FLA recurrence directly, preserving native autograd.

    :param Tensor q: Query tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor k: Key tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor v: Value tensor shaped ``(batch, seq, heads, head_v)``.
    :param Tensor g: Log-space gate tensor shaped ``(batch, seq, heads)``.
    :param Tensor beta: Beta tensor shaped ``(batch, seq, heads)``.
    :return Tensor: Recurrence output shaped like ``v``.
    """
    return _run_chunk_gated_delta_rule(q, k, v, g, beta)


@_dynamo_disable
def fla_chunk_gated_delta_rule_direct_fused_gate_norm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
) -> Tensor:
    """Run public FLA with upstream-style q/k norm and gate activation.

    :param Tensor q: Raw query tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor k: Raw key tensor shaped ``(batch, seq, heads, head_k)``.
    :param Tensor v: Value tensor shaped ``(batch, seq, heads, head_v)``.
    :param Tensor g: Raw gate tensor shaped ``(batch, seq, heads)``.
    :param Tensor beta: Beta tensor shaped ``(batch, seq, heads)``.
    :param Tensor A_log: Decay scale parameter shaped ``(heads,)``.
    :param Tensor dt_bias: Decay dt-bias parameter shaped ``(heads,)``.
    :return Tensor: Recurrence output shaped like ``v``.
    """
    return _run_chunk_gated_delta_rule_fused_gate_norm(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale=None,
    )
