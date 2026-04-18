"""Parity, owned-launch, and timing checks for the HGDN core-kernel path."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from _repo_bootstrap import ensure_repo_root_on_sys_path

REPO_ROOT = ensure_repo_root_on_sys_path()

from hgdn_megakernel import device_report, extension_status  # noqa: E402
from hgdn_megakernel.generate_test_data import (  # noqa: E402
    CASE_FORMAT_VERSION,
    RECURRENCE_CONTRACT,
    generate_case,
    hydrate_module_from_inputs,
    make_module,
)
from model import _HAS_FLA  # noqa: E402

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


def _make_case_module(
    meta: dict[str, object],
    *,
    use_fla: bool,
    use_cuda_corekernel: bool,
) -> torch.nn.Module:
    """Instantiate one live HGDN module for a serialized case.

    :param dict[str, object] meta: Serialized case metadata.
    :param bool use_fla: Whether to enable the FLA recurrence path.
    :param bool use_cuda_corekernel: Whether to enable the owned CUDA core
        kernel path.
    :return torch.nn.Module: Prepared HGDN module on CUDA.
    """
    return make_module(
        d_model=int(meta["D"]),
        n_heads=int(meta["H"]),
        head_k_dim=int(meta["Dk"]),
        expand_v=float(meta["expand_v"]),
        conv_size=int(meta["K"]),
        use_fla=use_fla,
        use_cuda_corekernel=use_cuda_corekernel,
    )


def _run_module(
    module: torch.nn.Module,
    x_seed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run one live-module forward/backward pass and capture outputs.

    :param torch.nn.Module module: Live HGDN module.
    :param torch.Tensor x_seed: Shared input activations without gradients.
    :return dict[str, torch.Tensor]: Output and gradient tensors.
    """
    module.zero_grad(set_to_none=True)
    x = x_seed.detach().clone().requires_grad_(True)
    y = module(x)
    loss = y.float().square().mean() + 0.01 * y.float().sum()
    loss.backward()
    conv_grad = module.qkv_conv.conv.weight.grad.detach().view(
        module.qkv_conv.conv.weight.shape[0], -1
    )
    return {
        "y": y.detach(),
        "loss": loss.detach(),
        "grad_x": x.grad.detach(),
        "grad_w_qkv": module.w_qkv.weight.grad.detach(),
        "grad_w_a": module.w_a.weight.grad.detach(),
        "grad_w_b": module.w_b.weight.grad.detach(),
        "grad_w_g": module.w_g.weight.grad.detach(),
        "grad_w_out": module.w_out.weight.grad.detach(),
        "grad_conv_w": conv_grad.detach(),
        "grad_A_log": module.A_log.grad.detach(),
        "grad_dt_bias": module.dt_bias.grad.detach(),
    }


def _benchmark_module(
    module: torch.nn.Module,
    x_seed: torch.Tensor,
    *,
    timing_warmups: int,
    timing_repeats: int,
) -> dict[str, float]:
    """Measure one module path with CUDA events.

    :param torch.nn.Module module: Module to benchmark.
    :param torch.Tensor x_seed: Shared input activations.
    :param int timing_warmups: Number of warmup iterations.
    :param int timing_repeats: Number of measured iterations.
    :return dict[str, float]: Median and min/max timing summary.
    """
    x = x_seed.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(x)
    for _ in range(timing_warmups):
        module.zero_grad(set_to_none=True)
        x.grad = None
        out = module(x)
        torch.autograd.backward((out,), (grad_out,))
    torch.cuda.synchronize()

    forward_times: list[float] = []
    forward_backward_times: list[float] = []
    for _ in range(timing_repeats):
        module.zero_grad(set_to_none=True)
        x.grad = None
        start = torch.cuda.Event(enable_timing=True)
        mid = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = module(x)
        mid.record()
        torch.autograd.backward((out,), (grad_out,))
        end.record()
        torch.cuda.synchronize()
        forward_times.append(float(start.elapsed_time(mid)))
        forward_backward_times.append(float(start.elapsed_time(end)))

    summary = {
        "forward_ms": float(statistics.median(forward_times)),
        "forward_backward_ms": float(statistics.median(forward_backward_times)),
    }
    if timing_repeats > 1:
        summary.update(
            {
                "forward_ms_min": float(min(forward_times)),
                "forward_ms_max": float(max(forward_times)),
                "forward_backward_ms_min": float(min(forward_backward_times)),
                "forward_backward_ms_max": float(max(forward_backward_times)),
            }
        )
    return summary


def forward_backward_case(
    path: Path,
    *,
    atol: float,
    rtol: float,
    grad_scale: float,
    timing_warmups: int,
    timing_repeats: int,
) -> dict[str, float | int]:
    """Run forward/backward parity and timing for one serialized case.

    :param Path path: Case file path.
    :param float atol: Forward tolerance.
    :param float rtol: Forward tolerance.
    :param float grad_scale: Multiplier for backward tolerances.
    :param int timing_warmups: Number of timing warmup iterations.
    :param int timing_repeats: Number of measured timing iterations.
    :return dict[str, float | int]: Event timing metrics.
    """
    payload = load_payload(path)
    meta = payload["meta"]
    references = payload_references(payload)
    inputs = payload["inputs"]
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

    core_module = _make_case_module(
        meta,
        use_fla=False,
        use_cuda_corekernel=True,
    )
    eager_module = _make_case_module(
        meta,
        use_fla=False,
        use_cuda_corekernel=False,
    )
    hydrate_module_from_inputs(core_module, inputs)
    hydrate_module_from_inputs(eager_module, inputs)
    x_seed = inputs["x"].cuda()

    result = _run_module(core_module, x_seed)
    for reference_name, reference in references.items():
        for name in REFERENCE_NAMES:
            tol_atol, tol_rtol = reference_tensor_tolerances[name]
            enforce = reference_name != "fla" or control_enforce[name]
            label = "forward_y" if name == "y" else name
            check_close(
                f"{reference_name}/{label}",
                result[name],
                reference[name],
                atol=tol_atol,
                rtol=tol_rtol,
                enforce=enforce,
            )

    core_timing = _benchmark_module(
        core_module,
        x_seed,
        timing_warmups=timing_warmups,
        timing_repeats=timing_repeats,
    )
    eager_timing = _benchmark_module(
        eager_module,
        x_seed,
        timing_warmups=timing_warmups,
        timing_repeats=timing_repeats,
    )
    return {
        "timing_repeats": timing_repeats,
        "core_forward_ms": core_timing["forward_ms"],
        "core_forward_backward_ms": core_timing["forward_backward_ms"],
        "eager_forward_ms": eager_timing["forward_ms"],
        "eager_forward_backward_ms": eager_timing["forward_backward_ms"],
        "core_vs_eager_forward_ratio": core_timing["forward_ms"]
        / max(eager_timing["forward_ms"], 1e-9),
        "core_vs_eager_forward_backward_ratio": core_timing["forward_backward_ms"]
        / max(eager_timing["forward_backward_ms"], 1e-9),
        **{
            key: value
            for key, value in {
                "core_forward_ms_min": core_timing.get("forward_ms_min"),
                "core_forward_ms_max": core_timing.get("forward_ms_max"),
                "core_forward_backward_ms_min": core_timing.get(
                    "forward_backward_ms_min"
                ),
                "core_forward_backward_ms_max": core_timing.get(
                    "forward_backward_ms_max"
                ),
                "eager_forward_ms_min": eager_timing.get("forward_ms_min"),
                "eager_forward_ms_max": eager_timing.get("forward_ms_max"),
                "eager_forward_backward_ms_min": eager_timing.get(
                    "forward_backward_ms_min"
                ),
                "eager_forward_backward_ms_max": eager_timing.get(
                    "forward_backward_ms_max"
                ),
            }.items()
            if value is not None
        },
    }


def count_launches(path: Path) -> dict[str, object]:
    """Count owned core-kernel launches for one reference case.

    The dense projections intentionally stay outside the owned core-kernel path,
    so this check only asserts one `hgdn_core_forward_*` and one
    `hgdn_core_backward_*` launch inside the profiled region.

    :param Path path: Case file path.
    :return dict[str, object]: Launch-count summary.
    """
    payload = load_payload(path)
    meta = payload["meta"]
    inputs = payload["inputs"]
    module = _make_case_module(
        meta,
        use_fla=False,
        use_cuda_corekernel=True,
    )
    hydrate_module_from_inputs(module, inputs)
    x_seed = inputs["x"].cuda()
    x_warmup = x_seed.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(x_warmup)

    warmup_out = module(x_warmup)
    torch.autograd.backward((warmup_out,), (grad_out,))
    module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    x = x_seed.detach().clone().requires_grad_(True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        out = module(x)
        torch.autograd.backward((out,), (grad_out,))
        torch.cuda.synchronize()

    events = list(prof.events())
    cuda_events = [
        event
        for event in events
        if "cuda" in str(getattr(event, "device_type", "")).lower()
    ]
    owned_cuda_keys = sorted(
        {event.name for event in cuda_events if "hgdn_core_" in event.name.lower()}
    )
    forward_count = sum(
        "hgdn_core_forward_bf16_kernel" in event.name.lower() for event in events
    )
    backward_count = sum(
        "hgdn_core_backward_bf16_kernel" in event.name.lower() for event in events
    )
    unexpected_owned = sorted(
        {
            event.name
            for event in cuda_events
            if "hgdn_core_" in event.name.lower()
            and "hgdn_core_forward_bf16_kernel" not in event.name.lower()
            and "hgdn_core_backward_bf16_kernel" not in event.name.lower()
        }
    )
    summary = {
        "core_forward_launch_count": int(forward_count),
        "core_backward_launch_count": int(backward_count),
        "owned_cuda_keys": owned_cuda_keys,
        "unexpected_owned_cuda_keys": unexpected_owned,
    }
    print(json.dumps(summary, indent=2))
    if forward_count != 1 or backward_count != 1 or unexpected_owned:
        raise AssertionError(
            "core-kernel launch count is not exactly one owned forward + one owned backward"
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
    """Run the local HGDN core-kernel validation suite."""
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
        default=3,
        help="Number of measured timing iterations per case.",
    )
    args = parser.parse_args()

    status = extension_status()
    print(f"hgdn_megakernel_extension:{json.dumps(status, sort_keys=True)}")
    print(device_report())

    case_specs = [
        ("B1_T8", 1, 8, 1337),
        ("B1_T32", 1, 32, 1338),
        ("B1_T128", 1, 128, 1339),
        ("B1_T512", 1, 512, 1340),
    ]
    if args.include_b2_t512:
        case_specs.append(("B2_T512", 2, 512, 1341))
    if args.include_b1_t2048:
        case_specs.append(("B1_T2048", 1, 2048, 1342))

    timings: dict[str, dict[str, float | int | object]] = {}
    for label, batch, seq, seed in case_specs:
        path = args.case_dir / f"case_b{batch}_t{seq}.pt"
        ensure_case(path, batch=batch, seq=seq, seed=seed)
        print(f"running_case:{label}")
        timings[label] = forward_backward_case(
            path,
            atol=args.atol,
            rtol=args.rtol,
            grad_scale=args.grad_scale,
            timing_warmups=args.timing_warmups,
            timing_repeats=args.timing_repeats,
        )

    launch_counts = count_launches(args.case_dir / "case_b1_t8.pt")
    timings["launch_counts"] = launch_counts
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
