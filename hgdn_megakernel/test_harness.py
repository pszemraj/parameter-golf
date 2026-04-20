"""Shared validation harness helpers for HGDN kernel tests."""

from __future__ import annotations

from pathlib import Path

import torch

from hgdn_megakernel.generate_test_data import (
    CASE_FORMAT_VERSION,
    RECURRENCE_CONTRACT,
    generate_case,
)
from model import _HAS_FLA


def diff_stats(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float | int]:
    """Return error statistics for one tensor pair.

    :param torch.Tensor got: Candidate tensor.
    :param torch.Tensor ref: Reference tensor.
    :return dict[str, float | int]: Error summary.
    """
    got_f = got.detach().float().cpu().flatten()
    ref_f = ref.detach().float().cpu().flatten()
    delta = got_f - ref_f
    abs_diff = delta.abs()
    rel_diff = abs_diff / ref_f.abs().clamp_min(1e-6)
    flat_index = int(abs_diff.argmax().item())
    rmse = float(delta.square().mean().sqrt().item())
    norm_rel = float(delta.norm().item() / ref_f.norm().clamp_min(1e-6).item())
    return {
        "max_abs": float(abs_diff.max().item()),
        "max_rel": float(rel_diff.max().item()),
        "rmse": rmse,
        "norm_rel": norm_rel,
        "flat_index": flat_index,
    }


def tensor_window(
    tensor: torch.Tensor, flat_index: int, radius: int = 2
) -> list[float]:
    """Return a small flat window around one failing element.

    :param torch.Tensor tensor: Tensor to inspect.
    :param int flat_index: Flat failing index.
    :param int radius: Window radius, defaults to 2.
    :return list[float]: Neighboring flat values.
    """
    flat = tensor.detach().float().cpu().flatten()
    lo = max(0, flat_index - radius)
    hi = min(flat.numel(), flat_index + radius + 1)
    return [float(value) for value in flat[lo:hi].tolist()]


def check_close(
    name: str,
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    enforce: bool = True,
) -> bool:
    """Check one tensor pair and print mismatch diagnostics on failure.

    :param str name: Display label.
    :param torch.Tensor got: Candidate tensor.
    :param torch.Tensor ref: Reference tensor.
    :param float atol: Absolute tolerance.
    :param float rtol: Relative tolerance.
    :param bool enforce: Whether to raise on mismatch, defaults to `True`.
    :raises AssertionError: If the tensors do not match and `enforce=True`.
    :return bool: Whether the tensors matched within tolerance.
    """
    stats = diff_stats(got, ref)
    ok = torch.allclose(
        got.detach().float().cpu(),
        ref.detach().float().cpu(),
        atol=atol,
        rtol=rtol,
    )
    print(
        f"{name:>24s}: max_abs={stats['max_abs']:.6g} "
        f"max_rel={stats['max_rel']:.6g} rmse={stats['rmse']:.6g} "
        f"norm_rel={stats['norm_rel']:.6g} idx={stats['flat_index']} ok={ok}"
    )
    if ok:
        return True
    print(f"{name} got_window={tensor_window(got, int(stats['flat_index']))}")
    print(f"{name} ref_window={tensor_window(ref, int(stats['flat_index']))}")
    if enforce:
        raise AssertionError(f"{name} mismatch")
    return False


def load_payload(path: Path) -> dict[str, object]:
    """Load one serialized reference payload from disk.

    :param Path path: Case file path.
    :return dict[str, object]: Serialized payload.
    """
    return torch.load(path, map_location="cpu")


def payload_references(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return named reference dictionaries for one payload.

    :param dict[str, object] payload: Loaded payload.
    :raises KeyError: If the payload predates the supported structured format.
    :return dict[str, dict[str, object]]: Reference outputs and gradients.
    """
    references = payload.get("references")
    if not isinstance(references, dict):
        raise KeyError("payload is missing structured 'references'")
    return references


def case_needs_regen(path: Path) -> bool:
    """Return whether a serialized case should be regenerated.

    :param Path path: Case file path.
    :return bool: Whether the case is missing or stale.
    """
    if not path.exists():
        return True
    try:
        payload = load_payload(path)
    except Exception:
        return True
    if payload.get("format_version") != CASE_FORMAT_VERSION:
        return True
    if "inputs" not in payload or "references" not in payload:
        return True
    if payload["meta"].get("recurrence_contract") != RECURRENCE_CONTRACT:
        return True
    references = payload_references(payload)
    if "eager" not in references:
        return True
    if _HAS_FLA and payload["meta"].get("has_fla_reference", False) != (
        "fla" in references
    ):
        return True
    if _HAS_FLA and "fla" not in references:
        return True
    return False


def ensure_case(path: Path, *, batch: int, seq: int, seed: int) -> None:
    """Generate a missing reference case in place.

    :param Path path: Output case path.
    :param int batch: Batch size.
    :param int seq: Sequence length.
    :param int seed: RNG seed.
    """
    if not case_needs_regen(path):
        return
    generate_case(
        out=path,
        batch=batch,
        seq=seq,
        d_model=384,
        n_heads=8,
        head_k_dim=48,
        expand_v=1.0,
        conv_size=4,
        seed=seed,
    )
