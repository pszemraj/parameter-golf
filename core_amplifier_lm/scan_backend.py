"""Parallel first-order scan helpers for the maintained Core/Amplifier path.

This module centralizes the repo-local associative scan implementation plus the
direct ``accelerated-scan`` CUDA integration used by the parallel minGRU path
and by experimental temporal tap modes. The goal is to keep recurrence
semantics stable while making the active backend explicit, fast by default on
CUDA, and impossible to silently downgrade when the active Core/Amplifier path
expects acceleration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

import torch

SCAN_BACKEND_CHOICES = ("auto", "heinsen", "assoc", "assoc_accel", "sequential")
ScalarScanFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def normalize_scan_backend(backend: str, *, allow_heinsen: bool) -> str:
    """Normalize and validate a scan backend name.

    :param str backend: Requested backend name.
    :param bool allow_heinsen: Whether the Heinsen backend is allowed.
    :return str: Normalized backend name.
    """
    normalized = str(backend).strip().lower()
    valid = set(SCAN_BACKEND_CHOICES)
    if not allow_heinsen:
        valid.discard("heinsen")
    if normalized not in valid:
        expected = ", ".join(sorted(valid))
        raise ValueError(f"unknown scan backend {backend!r}; expected one of {expected}")
    return normalized


def accelerated_scan_installed() -> bool:
    """Return whether the ``accelerated-scan`` package is importable.

    :return bool: ``True`` when ``accelerated-scan`` is available.
    """
    import importlib.util

    return importlib.util.find_spec("accelerated_scan") is not None


def resolve_scan_backend(
    requested: str,
    *,
    device: torch.device,
    allow_heinsen: bool,
) -> str:
    """Resolve a requested backend to an active backend for a device.

    :param str requested: Requested backend name.
    :param torch.device device: Runtime device.
    :param bool allow_heinsen: Whether the Heinsen backend is allowed.
    :return str: Active backend name.
    """
    requested = normalize_scan_backend(requested, allow_heinsen=allow_heinsen)
    accel_available = accelerated_scan_installed()
    if requested == "auto":
        if device.type == "cuda":
            if not accel_available:
                raise RuntimeError(
                    "scan_backend=auto on CUDA requires accelerated-scan. "
                    "Install the repo requirements or choose an explicit slower backend."
                )
            return "assoc_accel"
        return "assoc"
    if requested == "assoc":
        return "assoc"
    if requested == "assoc_accel":
        if device.type != "cuda":
            raise RuntimeError("scan_backend=assoc_accel requires a CUDA device; use assoc on CPU.")
        if not accel_available:
            raise RuntimeError(
                "scan_backend=assoc_accel requires accelerated-scan. Install the repo requirements."
            )
        return "assoc_accel"
    return requested


@lru_cache(maxsize=1)
def _get_accelerated_scalar_scan() -> ScalarScanFn:
    """Return the direct accelerated scalar scan function.

    The maintained CUDA path uses ``accelerated_scan.scalar.scan`` directly with
    explicit carry-prefix handling in repo code. This keeps the fast path tight
    and avoids extra wrapper behavior from third-party convenience layers.

    :return callable: Direct accelerated scalar scan function.
    """
    from accelerated_scan.scalar import scan as scalar_scan

    return scalar_scan


def sequential_affine_scan(
    gates: torch.Tensor,
    inputs: torch.Tensor,
    *,
    prev: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a simple sequential affine scan.

    :param torch.Tensor gates: Forget gates of shape ``[B, T, ...]``.
    :param torch.Tensor inputs: Input contributions of shape ``[B, T, ...]``.
    :param Optional[torch.Tensor] prev: Optional previous state.
    :return torch.Tensor: Scanned outputs with the same shape as ``inputs``.
    """
    if gates.shape != inputs.shape:
        gates = gates.expand_as(inputs)
    if prev is None:
        state = torch.zeros_like(inputs[:, 0])
    else:
        state = prev
    outputs = []
    for t in range(inputs.shape[1]):
        state = gates[:, t] * state + inputs[:, t]
        outputs.append(state[:, None])
    return torch.cat(outputs, dim=1)


def accelerated_scalar_affine_scan(
    gates: torch.Tensor,
    inputs: torch.Tensor,
    *,
    prev: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the direct accelerated scalar scan with manual carry handling.

    The recurrence is ``x[t] = gates[t] * x[t-1] + inputs[t]`` over tensors of
    shape ``[B, T, ...]``. The upstream scalar kernel expects ``[B, C, T]`` and
    already supports arbitrary sequence lengths, so we flatten the feature axes,
    prepend ``prev`` manually when present, and avoid the extra convenience
    wrapper logic that was written around a different backend contract.

    :param torch.Tensor gates: Forget gates of shape ``[B, T, ...]``.
    :param torch.Tensor inputs: Input contributions of shape ``[B, T, ...]``.
    :param Optional[torch.Tensor] prev: Optional previous state of shape
        ``[B, ...]``.
    :return torch.Tensor: Scanned outputs with the same shape as ``inputs``.
    """
    if gates.shape != inputs.shape:
        gates = gates.expand_as(inputs)

    batch_size, seq_len = inputs.shape[:2]
    flat_dim = int(inputs[0, 0].numel())

    gates_bct = gates.reshape(batch_size, seq_len, flat_dim).permute(0, 2, 1).contiguous()
    inputs_bct = inputs.reshape(batch_size, seq_len, flat_dim).permute(0, 2, 1).contiguous()

    if prev is not None:
        prev_bct = (
            prev.reshape(batch_size, flat_dim).unsqueeze(-1).to(inputs_bct.dtype).contiguous()
        )
        prefix_gates = torch.ones(
            (batch_size, flat_dim, 1), dtype=gates_bct.dtype, device=gates_bct.device
        )
        gates_bct = torch.cat((prefix_gates, gates_bct), dim=-1)
        inputs_bct = torch.cat((prev_bct, inputs_bct), dim=-1)

    scalar_scan = _get_accelerated_scalar_scan()
    outputs_bct = scalar_scan(gates_bct, inputs_bct)

    if prev is not None:
        outputs_bct = outputs_bct[..., 1:]

    outputs_btd = outputs_bct.permute(0, 2, 1).contiguous()
    return outputs_btd.reshape_as(inputs)


def torch_assoc_affine_scan(
    gates: torch.Tensor,
    inputs: torch.Tensor,
    *,
    prev: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a repo-local Hillis-Steele affine scan in Torch.

    This is the maintained slow/reference associative path for the recurrence
    ``x[t] = gates[t] * x[t-1] + inputs[t]`` over tensors of shape
    ``[B, T, ...]``. It uses iterative doubling over affine-composition pairs
    and requires no external scan package.

    :param torch.Tensor gates: Forget gates of shape ``[B, T, ...]``.
    :param torch.Tensor inputs: Input contributions of shape ``[B, T, ...]``.
    :param Optional[torch.Tensor] prev: Optional previous state of shape
        ``[B, ...]``.
    :return torch.Tensor: Scanned outputs with the same shape as ``inputs``.
    """
    if gates.shape != inputs.shape:
        gates = gates.expand_as(inputs)

    batch_size, seq_len = inputs.shape[:2]
    flat_dim = int(inputs[0, 0].numel())

    coeff = gates.reshape(batch_size, seq_len, flat_dim).contiguous()
    values = inputs.reshape(batch_size, seq_len, flat_dim).contiguous()

    if prev is not None:
        prefix_coeff = torch.ones((batch_size, 1, flat_dim), dtype=coeff.dtype, device=coeff.device)
        prefix_values = (
            prev.reshape(batch_size, flat_dim).unsqueeze(1).to(values.dtype).contiguous()
        )
        coeff = torch.cat((prefix_coeff, coeff), dim=1)
        values = torch.cat((prefix_values, values), dim=1)

    offset = 1
    total_steps = coeff.shape[1]
    while offset < total_steps:
        coeff_cur = coeff[:, offset:]
        values_cur = values[:, offset:]
        coeff_prev = coeff[:, :-offset]
        values_prev = values[:, :-offset]
        coeff_next = coeff.clone()
        values_next = values.clone()
        values_next[:, offset:] = values_cur + coeff_cur * values_prev
        coeff_next[:, offset:] = coeff_cur * coeff_prev
        coeff, values = coeff_next, values_next
        offset *= 2

    if prev is not None:
        values = values[:, 1:]

    return values.reshape_as(inputs)


def apply_affine_scan(
    gates: torch.Tensor,
    inputs: torch.Tensor,
    *,
    prev: Optional[torch.Tensor] = None,
    backend: str,
) -> tuple[torch.Tensor, str]:
    """Apply a first-order affine scan.

    The recurrence is ``x[t] = gates[t] * x[t-1] + inputs[t]``.

    :param torch.Tensor gates: Forget gates of shape ``[B, T, ...]``.
    :param torch.Tensor inputs: Input contributions of shape ``[B, T, ...]``.
    :param Optional[torch.Tensor] prev: Optional previous state.
    :param str backend: Requested active backend; may not be ``heinsen``.
    :return tuple[torch.Tensor, str]: Scanned outputs and the backend actually used.
    """

    backend = normalize_scan_backend(backend, allow_heinsen=False)
    if backend == "sequential":
        return sequential_affine_scan(gates, inputs, prev=prev), "sequential"
    if backend == "assoc":
        return torch_assoc_affine_scan(gates, inputs, prev=prev), "assoc"
    if backend == "assoc_accel":
        try:
            return accelerated_scalar_affine_scan(gates, inputs, prev=prev), "assoc_accel"
        except Exception as exc:
            raise RuntimeError(
                "scan backend 'assoc_accel' failed via accelerated_scan.scalar. "
                "The active Core/Amplifier path no longer falls back silently; "
                "either fix the accelerated scalar install or choose an explicit slower backend."
            ) from exc

    raise RuntimeError(f"unsupported affine scan backend {backend!r}")
