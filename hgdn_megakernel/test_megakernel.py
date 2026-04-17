"""Parity, launch-count, and timing checks for the HGDN megakernel."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hgdn_megakernel import device_report, extension_status, hgdn_megakernel  # noqa: E402
from hgdn_megakernel.generate_test_data import (  # noqa: E402
    CASE_FORMAT_VERSION,
    RECURRENCE_CONTRACT,
    generate_case,
)
from model import _HAS_FLA  # noqa: E402

TENSOR_NAMES = (
    "x",
    "w_qkv",
    "w_a",
    "w_b",
    "w_g",
    "w_out",
    "conv_w",
    "A_log",
    "dt_bias",
)
REFERENCE_NAMES = (
    "y",
    "loss",
    "grad_x",
    "grad_w_qkv",
    "grad_w_a",
    "grad_w_b",
    "grad_w_g",
    "grad_w_out",
    "grad_conv_w",
    "grad_A_log",
    "grad_dt_bias",
)


def _diff_stats(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float | int]:
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


def _tensor_window(
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
    return [float(v) for v in flat[lo:hi].tolist()]


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
    stats = _diff_stats(got, ref)
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
    print(f"{name} got_window={_tensor_window(got, int(stats['flat_index']))}")
    print(f"{name} ref_window={_tensor_window(ref, int(stats['flat_index']))}")
    if enforce:
        raise AssertionError(f"{name} mismatch")
    return False


def load_payload(path: Path) -> dict[str, object]:
    """Load one serialized reference payload from disk.

    :param Path path: Case file path.
    :return dict[str, object]: Serialized payload.
    """
    return torch.load(path, map_location="cpu")


def load_case(
    path: Path,
) -> tuple[dict[str, object], dict[str, object], list[torch.Tensor]]:
    """Load one serialized reference case onto CUDA.

    :param Path path: Case file path.
    :return tuple[dict[str, object], dict[str, object], list[torch.Tensor]]:
        Metadata, full payload, and CUDA tensors.
    """
    payload = load_payload(path)
    meta = payload["meta"]
    inputs = payload["inputs"] if "inputs" in payload else payload
    tensors = []
    for name in TENSOR_NAMES:
        tensor = inputs[name].cuda()
        if tensor.dtype.is_floating_point:
            tensor = tensor.requires_grad_(True)
        tensors.append(tensor)
    return meta, payload, tensors


def _legacy_references(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """Adapt the old flat payload shape into the new reference structure.

    :param dict[str, object] payload: Loaded payload.
    :return dict[str, dict[str, object]]: Named references.
    """
    return {"eager": {name: payload[name] for name in REFERENCE_NAMES}}


def payload_references(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return the named reference dictionaries for one payload.

    :param dict[str, object] payload: Loaded payload.
    :return dict[str, dict[str, object]]: Reference outputs and gradients.
    """
    if "references" in payload:
        return payload["references"]
    return _legacy_references(payload)


def forward_backward_case(
    path: Path,
    *,
    atol: float,
    rtol: float,
    grad_scale: float,
    rec_chunk_t: int | None,
    timing_warmups: int,
    timing_repeats: int,
) -> dict[str, float | int]:
    """Run forward/backward parity for one serialized case.

    :param Path path: Case file path.
    :param float atol: Forward tolerance.
    :param float rtol: Forward tolerance.
    :param float grad_scale: Multiplier for backward tolerances.
    :param int | None rec_chunk_t: Runtime checkpoint cadence override.
    :param int timing_warmups: Number of timing warmup iterations.
    :param int timing_repeats: Number of measured timing iterations.
    :return dict[str, float | int]: Event timing metrics.
    """
    meta, payload, tensors = load_case(path)
    references = payload_references(payload)
    grad_tolerances = (atol * grad_scale, rtol * grad_scale)
    reference_tensor_tolerances: dict[str, tuple[float, float]] = {
        "y": (atol, rtol),
        "loss": (atol, rtol),
        "grad_x": grad_tolerances,
        "grad_w_qkv": grad_tolerances,
        "grad_w_a": grad_tolerances,
        "grad_w_b": grad_tolerances,
        "grad_w_g": grad_tolerances,
        "grad_w_out": grad_tolerances,
        "grad_conv_w": grad_tolerances,
        "grad_A_log": grad_tolerances,
        "grad_dt_bias": grad_tolerances,
    }
    control_enforce = {name: True for name in REFERENCE_NAMES}
    if "eager" in references and "fla" in references:
        print("reference_control:eager_vs_fla")
        for name in REFERENCE_NAMES:
            tol_atol, tol_rtol = reference_tensor_tolerances[name]
            control_enforce[name] = check_close(
                f"control/fla_vs_eager/{name}",
                references["fla"][name],
                references["eager"][name],
                atol=tol_atol,
                rtol=tol_rtol,
                enforce=False,
            )
        if not all(control_enforce.values()):
            print(
                "reference_control: FLA differs from eager on this case; "
                "FLA comparisons remain diagnostic only where the control drifts."
            )
    y = hgdn_megakernel(
        *tensors,
        n_heads=int(meta["H"]),
        head_k_dim=int(meta["Dk"]),
        head_v_dim=int(meta["Dv"]),
        conv_size=int(meta["K"]),
        rec_chunk_t=rec_chunk_t,
        allow_neg_eigval=bool(meta["allow_neg_eigval"]),
    )
    torch.cuda.synchronize()
    for reference_name, reference in references.items():
        enforce = reference_name != "fla" or control_enforce["y"]
        check_close(
            f"{reference_name}/forward_y",
            y,
            reference["y"],
            atol=atol,
            rtol=rtol,
            enforce=enforce,
        )

    loss = y.float().square().mean() + 0.01 * y.float().sum()
    loss.backward()
    torch.cuda.synchronize()
    for reference_name, reference in references.items():
        enforce = reference_name != "fla" or control_enforce["loss"]
        check_close(
            f"{reference_name}/loss",
            loss.detach().cpu(),
            reference["loss"],
            atol=atol,
            rtol=rtol,
            enforce=enforce,
        )
    names = [
        "grad_x",
        "grad_w_qkv",
        "grad_w_a",
        "grad_w_b",
        "grad_w_g",
        "grad_w_out",
        "grad_conv_w",
        "grad_A_log",
        "grad_dt_bias",
    ]
    for reference_name, reference in references.items():
        for tensor, name in zip(tensors, names):
            enforce = reference_name != "fla" or control_enforce[name]
            check_close(
                f"{reference_name}/{name}",
                tensor.grad,
                reference[name],
                atol=grad_tolerances[0],
                rtol=grad_tolerances[1],
                enforce=enforce,
            )

    for tensor in tensors:
        tensor.grad = None
    grad_out = torch.randn_like(y)
    for _ in range(timing_warmups):
        out = hgdn_megakernel(
            *tensors,
            n_heads=int(meta["H"]),
            head_k_dim=int(meta["Dk"]),
            head_v_dim=int(meta["Dv"]),
            conv_size=int(meta["K"]),
            rec_chunk_t=rec_chunk_t,
            allow_neg_eigval=bool(meta["allow_neg_eigval"]),
        )
        torch.autograd.backward((out,), (grad_out,))
        for tensor in tensors:
            tensor.grad = None
    torch.cuda.synchronize()

    forward_times: list[float] = []
    forward_backward_times: list[float] = []
    for _ in range(timing_repeats):
        for tensor in tensors:
            tensor.grad = None
        start = torch.cuda.Event(enable_timing=True)
        mid = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = hgdn_megakernel(
            *tensors,
            n_heads=int(meta["H"]),
            head_k_dim=int(meta["Dk"]),
            head_v_dim=int(meta["Dv"]),
            conv_size=int(meta["K"]),
            rec_chunk_t=rec_chunk_t,
            allow_neg_eigval=bool(meta["allow_neg_eigval"]),
        )
        mid.record()
        torch.autograd.backward((out,), (grad_out,))
        end.record()
        torch.cuda.synchronize()
        forward_times.append(float(start.elapsed_time(mid)))
        forward_backward_times.append(float(start.elapsed_time(end)))

    result = {
        "timing_repeats": timing_repeats,
        "forward_ms": float(statistics.median(forward_times)),
        "forward_backward_ms": float(statistics.median(forward_backward_times)),
    }
    if timing_repeats > 1:
        result.update(
            {
                "forward_ms_min": float(min(forward_times)),
                "forward_ms_max": float(max(forward_times)),
                "forward_backward_ms_min": float(min(forward_backward_times)),
                "forward_backward_ms_max": float(max(forward_backward_times)),
            }
        )
    return result


def count_launches(path: Path, *, rec_chunk_t: int | None) -> dict[str, object]:
    """Count megakernel launches for one reference case with torch profiler.

    :param Path path: Case file path.
    :param int | None rec_chunk_t: Runtime checkpoint cadence override.
    :return dict[str, object]: Launch-count summary.
    """
    meta, _payload, tensors = load_case(path)
    grad_out = torch.randn(
        int(meta["B"]),
        int(meta["T"]),
        int(meta["D"]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    warmup_out = hgdn_megakernel(
        *tensors,
        n_heads=int(meta["H"]),
        head_k_dim=int(meta["Dk"]),
        head_v_dim=int(meta["Dv"]),
        conv_size=int(meta["K"]),
        rec_chunk_t=rec_chunk_t,
        allow_neg_eigval=bool(meta["allow_neg_eigval"]),
    )
    torch.autograd.backward((warmup_out,), (grad_out,))
    for tensor in tensors:
        tensor.grad = None
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        out = hgdn_megakernel(
            *tensors,
            n_heads=int(meta["H"]),
            head_k_dim=int(meta["Dk"]),
            head_v_dim=int(meta["Dv"]),
            conv_size=int(meta["K"]),
            rec_chunk_t=rec_chunk_t,
            allow_neg_eigval=bool(meta["allow_neg_eigval"]),
        )
        torch.autograd.backward((out,), (grad_out,))
        torch.cuda.synchronize()
    events = list(prof.events())
    cuda_events = [
        event
        for event in events
        if "cuda" in str(getattr(event, "device_type", "")).lower()
    ]
    cuda_keys = sorted(
        {
            event.name
            for event in cuda_events
            if "kernel" in event.name.lower()
            or "memset" in event.name.lower()
            or "memcpy" in event.name.lower()
        }
    )
    forward_count = sum(
        "hgdn_forward_bf16_kernel" in event.name.lower() for event in events
    )
    backward_count = sum(
        "hgdn_backward_bf16_kernel" in event.name.lower() for event in events
    )
    suspicious = [
        event.name
        for event in cuda_events
        if "memset" in event.name.lower()
        or "memcpy" in event.name.lower()
        or ("kernel" in event.name.lower() and "hgdn_" not in event.name.lower())
    ]
    summary = {
        "forward_launch_count": int(forward_count),
        "backward_launch_count": int(backward_count),
        "cuda_keys": cuda_keys,
        "suspicious_cuda_keys": suspicious,
    }
    print(json.dumps(summary, indent=2))
    if forward_count != 1 or backward_count != 1:
        raise AssertionError(
            "megakernel launch count is not exactly one forward + one backward"
        )
    return summary


def _case_needs_regen(path: Path) -> bool:
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
    if not _case_needs_regen(path):
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


def main() -> None:
    """Run the local HGDN megakernel validation suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("hgdn_megakernel/cases"),
    )
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--grad-scale", type=float, default=4.0)
    parser.add_argument(
        "--include-b2-t512",
        action="store_true",
        help="Also validate the optional B=2,T=512 parity case.",
    )
    parser.add_argument(
        "--include-b1-t2048",
        action="store_true",
        help="Also validate the H100-gate B=1,T=2048 parity case.",
    )
    parser.add_argument(
        "--timing-warmups",
        type=int,
        default=3,
        help="Number of warmup forward/backward timing iterations per case.",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=1,
        help="Number of measured forward/backward timing iterations per case.",
    )
    parser.add_argument(
        "--rec-chunk-t",
        type=int,
        default=None,
        help="Optional runtime checkpoint cadence override for the megakernel.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the HGDN megakernel test harness")
    status = extension_status()
    print(f"hgdn_megakernel_extension:{json.dumps(status, sort_keys=True)}")
    if not status["loaded"]:
        raise RuntimeError(
            "HGDN megakernel extension is not loaded. Build it with "
            "`python setup_hgdn_megakernel.py build_ext --inplace` "
            "or enable `GDN_MEGAKERNEL_ALLOW_JIT_BUILD=1`."
        )
    if args.rec_chunk_t is not None:
        print(f"runtime_rec_chunk_t:{int(args.rec_chunk_t)}")

    case_dir = args.case_dir
    case_dir.mkdir(parents=True, exist_ok=True)
    print(device_report().strip())
    case_specs = [
        ("B1_T8", 1, 8, 1337),
        ("B1_T32", 1, 32, 2026),
        ("B1_T128", 1, 128, 2048),
        ("B1_T512", 1, 512, 4096),
    ]
    if args.include_b2_t512:
        case_specs.append(("B2_T512", 2, 512, 8192))
    if args.include_b1_t2048:
        case_specs.append(("B1_T2048", 1, 2048, 16384))

    results: dict[str, object] = {}
    case_32 = case_dir / "case_b1_t32.pt"
    for label, batch, seq, seed in case_specs:
        path = case_dir / f"case_b{batch}_t{seq}.pt"
        ensure_case(path, batch=batch, seq=seq, seed=seed)
        print(f"running_case:{label}")
        results[label] = forward_backward_case(
            path,
            atol=args.atol,
            rtol=args.rtol,
            grad_scale=args.grad_scale,
            rec_chunk_t=args.rec_chunk_t,
            timing_warmups=args.timing_warmups,
            timing_repeats=args.timing_repeats,
        )
        if label == "B1_T32":
            case_32 = path
    results["launch_counts"] = count_launches(case_32, rec_chunk_t=args.rec_chunk_t)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
