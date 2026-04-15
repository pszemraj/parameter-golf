"""Parity, launch-count, and timing checks for the HGDN megakernel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hgdn_megakernel import device_report, extension_status, hgdn_megakernel  # noqa: E402
from hgdn_megakernel.generate_test_data import generate_case  # noqa: E402


def _max_diff(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, int]:
    """Return max abs, max rel, and flat index for one tensor pair.

    :param torch.Tensor got: Candidate tensor.
    :param torch.Tensor ref: Reference tensor.
    :return tuple[float, float, int]: Max abs diff, max rel diff, and flat index.
    """
    got_f = got.detach().float().cpu().flatten()
    ref_f = ref.detach().float().cpu().flatten()
    abs_diff = (got_f - ref_f).abs()
    rel_diff = abs_diff / ref_f.abs().clamp_min(1e-6)
    flat_index = int(abs_diff.argmax().item())
    return float(abs_diff.max().item()), float(rel_diff.max().item()), flat_index


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
) -> None:
    """Check one tensor pair and print mismatch diagnostics on failure.

    :param str name: Display label.
    :param torch.Tensor got: Candidate tensor.
    :param torch.Tensor ref: Reference tensor.
    :param float atol: Absolute tolerance.
    :param float rtol: Relative tolerance.
    :raises AssertionError: If the tensors do not match.
    """
    abs_diff, rel_diff, flat_index = _max_diff(got, ref)
    ok = torch.allclose(
        got.detach().float().cpu(),
        ref.detach().float().cpu(),
        atol=atol,
        rtol=rtol,
    )
    print(
        f"{name:>14s}: abs={abs_diff:.6g} rel={rel_diff:.6g} idx={flat_index} ok={ok}"
    )
    if ok:
        return
    print(f"{name} got_window={_tensor_window(got, flat_index)}")
    print(f"{name} ref_window={_tensor_window(ref, flat_index)}")
    raise AssertionError(f"{name} mismatch")


def load_case(path: Path) -> tuple[dict[str, object], list[torch.Tensor]]:
    """Load one serialized reference case onto CUDA.

    :param Path path: Case file path.
    :return tuple[dict[str, object], list[torch.Tensor]]: Metadata and CUDA tensors.
    """
    payload = torch.load(path, map_location="cpu")
    meta = payload["meta"]
    names = [
        "x",
        "w_qkv",
        "w_a",
        "w_b",
        "w_g",
        "w_out",
        "conv_w",
        "A_log",
        "dt_bias",
    ]
    tensors = []
    for name in names:
        tensor = payload[name].cuda()
        if tensor.dtype.is_floating_point:
            tensor = tensor.requires_grad_(True)
        tensors.append(tensor)
    return meta, tensors


def forward_backward_case(
    path: Path,
    *,
    atol: float,
    rtol: float,
    grad_scale: float,
) -> dict[str, float]:
    """Run forward/backward parity for one serialized case.

    :param Path path: Case file path.
    :param float atol: Forward tolerance.
    :param float rtol: Forward tolerance.
    :param float grad_scale: Multiplier for backward tolerances.
    :return dict[str, float]: Event timing metrics.
    """
    payload = torch.load(path, map_location="cpu")
    meta, tensors = load_case(path)
    print(device_report().strip())
    y = hgdn_megakernel(
        *tensors,
        n_heads=int(meta["H"]),
        head_k_dim=int(meta["Dk"]),
        head_v_dim=int(meta["Dv"]),
        conv_size=int(meta["K"]),
        allow_neg_eigval=bool(meta["allow_neg_eigval"]),
    )
    torch.cuda.synchronize()
    check_close("forward_y", y, payload["y"], atol=atol, rtol=rtol)

    loss = y.float().square().mean() + 0.01 * y.float().sum()
    loss.backward()
    torch.cuda.synchronize()
    check_close("loss", loss.detach().cpu(), payload["loss"], atol=atol, rtol=rtol)
    grad_tolerances = (atol * grad_scale, rtol * grad_scale)
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
    for tensor, name in zip(tensors, names):
        check_close(
            name,
            tensor.grad,
            payload[name],
            atol=grad_tolerances[0],
            rtol=grad_tolerances[1],
        )

    for tensor in tensors:
        tensor.grad = None
    warmups = 3
    grad_out = torch.randn_like(y)
    for _ in range(warmups):
        out = hgdn_megakernel(
            *tensors,
            n_heads=int(meta["H"]),
            head_k_dim=int(meta["Dk"]),
            head_v_dim=int(meta["Dv"]),
            conv_size=int(meta["K"]),
            allow_neg_eigval=bool(meta["allow_neg_eigval"]),
        )
        torch.autograd.backward((out,), (grad_out,))
        for tensor in tensors:
            tensor.grad = None
    torch.cuda.synchronize()

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
        allow_neg_eigval=bool(meta["allow_neg_eigval"]),
    )
    mid.record()
    torch.autograd.backward((out,), (grad_out,))
    end.record()
    torch.cuda.synchronize()
    return {
        "forward_ms": float(start.elapsed_time(mid)),
        "forward_backward_ms": float(start.elapsed_time(end)),
    }


def count_launches(path: Path) -> dict[str, object]:
    """Count megakernel launches for one reference case with torch profiler.

    :param Path path: Case file path.
    :return dict[str, object]: Launch-count summary.
    """
    meta, tensors = load_case(path)
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
            allow_neg_eigval=bool(meta["allow_neg_eigval"]),
        )
        torch.autograd.backward((out,), (grad_out,))
        torch.cuda.synchronize()
    events = list(prof.events())
    cuda_keys = sorted(
        {
            event.name
            for event in events
            if "cuda" in str(getattr(event, "device_type", "")).lower()
            or "kernel" in event.name.lower()
            or "memset" in event.name.lower()
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
        for event in events
        if "memset" in event.name.lower()
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


def ensure_case(path: Path, *, seq: int, seed: int) -> None:
    """Generate a missing reference case in place.

    :param Path path: Output case path.
    :param int seq: Sequence length.
    :param int seed: RNG seed.
    """
    if path.exists():
        return
    generate_case(
        out=path,
        batch=1,
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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the HGDN megakernel test harness")
    status = extension_status()
    print(f"hgdn_megakernel_extension:{json.dumps(status, sort_keys=True)}")
    if not status["loaded"]:
        raise RuntimeError(
            "HGDN megakernel extension is not loaded. Build it with "
            "`conda run -s --name pg python setup_hgdn_megakernel.py build_ext --inplace` "
            "or enable `GDN_MEGAKERNEL_ALLOW_JIT_BUILD=1`."
        )

    case_dir = args.case_dir
    case_dir.mkdir(parents=True, exist_ok=True)
    case_8 = case_dir / "case_b1_t8.pt"
    case_32 = case_dir / "case_b1_t32.pt"
    ensure_case(case_8, seq=8, seed=1337)
    ensure_case(case_32, seq=32, seed=2026)

    results = {
        "B1_T8": forward_backward_case(
            case_8,
            atol=args.atol,
            rtol=args.rtol,
            grad_scale=args.grad_scale,
        ),
        "B1_T32": forward_backward_case(
            case_32,
            atol=args.atol,
            rtol=args.rtol,
            grad_scale=args.grad_scale,
        ),
        "launch_counts": count_launches(case_32),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
