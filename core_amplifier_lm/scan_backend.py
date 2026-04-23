"""Parallel first-order scan helpers with optional accelerated backends.

This module centralizes the optional ``assoc-scan`` / ``accelerated-scan``
integration used by the parallel minGRU path and by experimental temporal tap
modes. The goal is to keep recurrence semantics stable while making the active
backend explicit and easy to summarize.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
from typing import Optional
import warnings

import torch

SCAN_BACKEND_CHOICES = ("auto", "heinsen", "assoc", "assoc_accel", "sequential")

_WARNED_ACCEL_FALLBACK = False
_WARNED_ASSOC_FALLBACK = False


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
    if requested == "auto":
        if assoc_scan_installed():
            if device.type == "cuda" and accelerated_scan_installed():
                return "assoc_accel"
            return "assoc"
        return "heinsen" if allow_heinsen else "sequential"
    if requested == "assoc_accel" and device.type != "cuda":
        return "assoc" if assoc_scan_installed() else ("heinsen" if allow_heinsen else "sequential")
    if requested in {"assoc", "assoc_accel"} and not assoc_scan_installed():
        return "heinsen" if allow_heinsen else "sequential"
    if requested == "assoc_accel" and not accelerated_scan_installed():
        return "assoc" if assoc_scan_installed() else ("heinsen" if allow_heinsen else "sequential")
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
    """Apply a first-order affine scan with backend fallback.

    The recurrence is ``x[t] = gates[t] * x[t-1] + inputs[t]``.

    :param torch.Tensor gates: Forget gates of shape ``[B, T, ...]``.
    :param torch.Tensor inputs: Input contributions of shape ``[B, T, ...]``.
    :param Optional[torch.Tensor] prev: Optional previous state.
    :param str backend: Requested active backend; may not be ``heinsen``.
    :return tuple[torch.Tensor, str]: Scanned outputs and the backend actually used.
    """
    global _WARNED_ACCEL_FALLBACK, _WARNED_ASSOC_FALLBACK

    backend = normalize_scan_backend(backend, allow_heinsen=False)
    if backend == "sequential":
        return sequential_affine_scan(gates, inputs, prev=prev), "sequential"

    use_accelerated = backend == "assoc_accel"
    try:
        scan = _get_assoc_scan_module(use_accelerated)
        return scan(gates, inputs, prev=prev), backend
    except Exception as exc:
        if use_accelerated:
            if not _WARNED_ACCEL_FALLBACK:
                warnings.warn(
                    f"accelerated scan backend failed ({exc!r}); falling back to assoc-scan",
                    stacklevel=2,
                )
                _WARNED_ACCEL_FALLBACK = True
            try:
                scan = _get_assoc_scan_module(False)
                return scan(gates, inputs, prev=prev), "assoc"
            except Exception as ref_exc:
                if not _WARNED_ASSOC_FALLBACK:
                    warnings.warn(
                        f"assoc-scan fallback failed ({ref_exc!r}); using sequential scan",
                        stacklevel=2,
                    )
                    _WARNED_ASSOC_FALLBACK = True
                return sequential_affine_scan(gates, inputs, prev=prev), "sequential"
        if not _WARNED_ASSOC_FALLBACK:
            warnings.warn(
                f"assoc-scan backend failed ({exc!r}); using sequential scan",
                stacklevel=2,
            )
            _WARNED_ASSOC_FALLBACK = True
        return sequential_affine_scan(gates, inputs, prev=prev), "sequential"
