"""Parallel first-order scan helpers with required accelerated CUDA backends.

This module centralizes the optional ``assoc-scan`` / ``accelerated-scan``
integration used by the parallel minGRU path and by experimental temporal tap
modes. The goal is to keep recurrence semantics stable while making the active
backend explicit, fast by default on CUDA, and impossible to silently
downgrade when the active Core/Amplifier path expects acceleration.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
from typing import Optional

import torch

SCAN_BACKEND_CHOICES = ("auto", "heinsen", "assoc", "assoc_accel", "sequential")


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


def assoc_scan_installed() -> bool:
    """Return whether the ``assoc-scan`` package is importable.

    :return bool: ``True`` when ``assoc-scan`` is available.
    """
    return importlib.util.find_spec("assoc_scan") is not None


def accelerated_scan_installed() -> bool:
    """Return whether the ``accelerated-scan`` package is importable.

    :return bool: ``True`` when ``accelerated-scan`` is available.
    """
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
    assoc_available = assoc_scan_installed()
    accel_available = accelerated_scan_installed()
    if requested == "auto":
        if device.type == "cuda":
            if not assoc_available or not accel_available:
                raise RuntimeError(
                    "scan_backend=auto on CUDA requires both assoc-scan and accelerated-scan. "
                    "Install the repo requirements or choose an explicit slower backend."
                )
            return "assoc_accel"
        if not assoc_available:
            raise RuntimeError(
                "scan_backend=auto requires assoc-scan on non-CUDA devices. "
                "Install the repo requirements or choose an explicit backend."
            )
        return "assoc"
    if requested == "assoc":
        if not assoc_available:
            raise RuntimeError(
                "scan_backend=assoc requires assoc-scan. Install the repo requirements."
            )
        return "assoc"
    if requested == "assoc_accel":
        if device.type != "cuda":
            raise RuntimeError("scan_backend=assoc_accel requires a CUDA device; use assoc on CPU.")
        if not assoc_available or not accel_available:
            raise RuntimeError(
                "scan_backend=assoc_accel requires both assoc-scan and accelerated-scan. "
                "Install the repo requirements."
            )
        return "assoc_accel"
    return requested


@lru_cache(maxsize=2)
def _get_assoc_scan_module(use_accelerated: bool) -> torch.nn.Module:
    """Return a cached ``AssocScan`` instance.

    :param bool use_accelerated: Whether to prefer the accelerated backend.
    :return torch.nn.Module: Cached ``AssocScan`` module.
    """
    from assoc_scan import AssocScan

    return AssocScan(use_accelerated=use_accelerated)


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

    use_accelerated = backend == "assoc_accel"
    try:
        scan = _get_assoc_scan_module(use_accelerated)
        return scan(gates, inputs, prev=prev), backend
    except Exception as exc:
        raise RuntimeError(
            f"scan backend {backend!r} failed. "
            "The active Core/Amplifier path no longer falls back silently; "
            "either fix the accelerated scan install or choose an explicit slower backend."
        ) from exc
