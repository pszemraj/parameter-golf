"""CUDA parity check for the optional HGDN fused extension.

Usage:
  python scripts/hgdn_cuda_parity.py
  DTYPE=fp32 python scripts/hgdn_cuda_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hgdn_cuda import (  # noqa: E402
    extension_status,
    fused_packed_qkv_frontend,
    fused_rmsnorm_silu_gate,
    packed_qkv_frontend_reference,
    rmsnorm_silu_gate_reference,
)


def parse_dtype() -> torch.dtype:
    """Parse the requested CUDA dtype from the environment.

    :raises ValueError: If ``DTYPE`` is unsupported.
    :return torch.dtype: Requested dtype.
    """
    raw = os.environ.get("DTYPE", "bf16").lower()
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported DTYPE={raw!r}; expected bf16, fp16, or fp32")


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return forward/backward comparison tolerances for one dtype.

    :param torch.dtype dtype: CUDA dtype under test.
    :return tuple[float, float]: Forward/backward tolerances.
    """
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 5e-2, 5e-2


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the maximum absolute difference as a Python float.

    :param torch.Tensor a: First tensor.
    :param torch.Tensor b: Second tensor.
    :return float: Maximum absolute difference.
    """
    return float((a - b).abs().max().item())


def check_frontend(dtype: torch.dtype) -> None:
    """Check fused packed front-end forward and backward parity.

    :param torch.dtype dtype: CUDA dtype under test.
    """
    bsz, seq, n_heads, head_k_dim, head_v_dim = 2, 32, 4, 8, 8
    channels = n_heads * (2 * head_k_dim + head_v_dim)
    kernel = 4
    atol, rtol = tolerances(dtype)

    torch.manual_seed(1337)
    qkv_ref = torch.randn(
        bsz, seq, channels, device="cuda", dtype=dtype, requires_grad=True
    )
    weight_ref = torch.randn(
        channels, kernel, device="cuda", dtype=dtype, requires_grad=True
    )
    qkv_ext = qkv_ref.detach().clone().requires_grad_(True)
    weight_ext = weight_ref.detach().clone().requires_grad_(True)

    q_ref, k_ref, v_ref = packed_qkv_frontend_reference(
        qkv_ref,
        weight_ref,
        n_heads=n_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )
    q_ext, k_ext, v_ext = fused_packed_qkv_frontend(
        qkv_ext,
        weight_ext,
        n_heads=n_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        enabled=True,
    )

    torch.testing.assert_close(q_ext.float(), q_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(k_ext.float(), k_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(v_ext.float(), v_ref.float(), atol=atol, rtol=rtol)

    grad_q = torch.randn_like(q_ref)
    grad_k = torch.randn_like(k_ref)
    grad_v = torch.randn_like(v_ref)
    (q_ref * grad_q).sum().backward(retain_graph=True)
    (k_ref * grad_k).sum().backward(retain_graph=True)
    (v_ref * grad_v).sum().backward()
    grad_qkv_ref = qkv_ref.grad.detach().clone()
    grad_weight_ref = weight_ref.grad.detach().clone()

    (q_ext * grad_q).sum().backward(retain_graph=True)
    (k_ext * grad_k).sum().backward(retain_graph=True)
    (v_ext * grad_v).sum().backward()
    grad_qkv_ext = qkv_ext.grad.detach().clone()
    grad_weight_ext = weight_ext.grad.detach().clone()

    torch.testing.assert_close(
        grad_qkv_ext.float(), grad_qkv_ref.float(), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        grad_weight_ext.float(), grad_weight_ref.float(), atol=atol, rtol=rtol
    )

    print(
        "frontend_parity:"
        f" dtype={dtype}"
        f" q={max_abs_diff(q_ext.float(), q_ref.float()):.6f}"
        f" k={max_abs_diff(k_ext.float(), k_ref.float()):.6f}"
        f" v={max_abs_diff(v_ext.float(), v_ref.float()):.6f}"
        f" grad_qkv={max_abs_diff(grad_qkv_ext.float(), grad_qkv_ref.float()):.6f}"
        f" grad_weight={max_abs_diff(grad_weight_ext.float(), grad_weight_ref.float()):.6f}"
    )


def check_output(dtype: torch.dtype) -> None:
    """Check fused output RMSNorm*SiLU(gate) forward and backward parity.

    :param torch.dtype dtype: CUDA dtype under test.
    """
    atol, rtol = tolerances(dtype)

    torch.manual_seed(2026)
    o_ref = torch.randn(2, 32, 4, 8, device="cuda", dtype=dtype, requires_grad=True)
    gate_ref = torch.randn(2, 32, 4, 8, device="cuda", dtype=dtype, requires_grad=True)
    o_ext = o_ref.detach().clone().requires_grad_(True)
    gate_ext = gate_ref.detach().clone().requires_grad_(True)

    out_ref = rmsnorm_silu_gate_reference(o_ref, gate_ref, fp32_accum=True)
    out_ext = fused_rmsnorm_silu_gate(
        o_ext,
        gate_ext,
        fp32_accum=True,
        enabled=True,
    )

    torch.testing.assert_close(out_ext.float(), out_ref.float(), atol=atol, rtol=rtol)

    grad = torch.randn_like(out_ref)
    (out_ref * grad).sum().backward()
    grad_o_ref = o_ref.grad.detach().clone()
    grad_gate_ref = gate_ref.grad.detach().clone()

    (out_ext * grad).sum().backward()
    grad_o_ext = o_ext.grad.detach().clone()
    grad_gate_ext = gate_ext.grad.detach().clone()

    torch.testing.assert_close(
        grad_o_ext.float(), grad_o_ref.float(), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        grad_gate_ext.float(), grad_gate_ref.float(), atol=atol, rtol=rtol
    )

    print(
        "output_parity:"
        f" dtype={dtype}"
        f" out={max_abs_diff(out_ext.float(), out_ref.float()):.6f}"
        f" grad_o={max_abs_diff(grad_o_ext.float(), grad_o_ref.float()):.6f}"
        f" grad_gate={max_abs_diff(grad_gate_ext.float(), grad_gate_ref.float()):.6f}"
    )


def main() -> None:
    """Run the CUDA parity suite."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HGDN CUDA parity checks")

    status = extension_status()
    print(f"hgdn_cuda_extension:{status}")
    if not status["loaded"]:
        raise RuntimeError(
            "HGDN CUDA extension is not loaded. Build it with "
            "`python setup_hgdn_cuda.py build_ext --inplace` or enable "
            "`GDN_CUDA_ALLOW_JIT_BUILD=1`."
        )

    dtype = parse_dtype()
    check_frontend(dtype)
    check_output(dtype)
    print("HGDN CUDA parity passed")


if __name__ == "__main__":
    main()
