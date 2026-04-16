#!/usr/bin/env python3
"""Compare eager and FLA HGDN recurrence outputs at the recurrence boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hgdn_cuda import fla_chunk_gated_delta_rule_compile_visible  # noqa: E402
from hgdn_runtime_utils import restore_low_dim_params_to_fp32  # noqa: E402
from model import GatedDeltaNet, _HAS_FLA, gdn_recurrent_naive  # noqa: E402


def diff_stats(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float | int]:
    """Return compact error statistics for one tensor pair.

    :param torch.Tensor got: Candidate tensor.
    :param torch.Tensor ref: Reference tensor.
    :return dict[str, float | int]: Max-abs, max-rel, RMSE, norm-relative, and index.
    """
    got_f = got.detach().float().cpu().flatten()
    ref_f = ref.detach().float().cpu().flatten()
    delta = got_f - ref_f
    abs_diff = delta.abs()
    rel_diff = abs_diff / ref_f.abs().clamp_min(1e-6)
    return {
        "max_abs": float(abs_diff.max().item()),
        "max_rel": float(rel_diff.max().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "norm_rel": float(delta.norm().item() / ref_f.norm().clamp_min(1e-6).item()),
        "flat_index": int(abs_diff.argmax().item()),
    }


def make_module(*, allow_neg_eigval: bool) -> GatedDeltaNet:
    """Build the live packed HGDN contract used by the recurrence diagnostic.

    :param bool allow_neg_eigval: Whether to enable beta scaling by `2.0`.
    :return GatedDeltaNet: Prepared HGDN module.
    """
    module = (
        GatedDeltaNet(
            d_model=384,
            n_heads=8,
            head_k_dim=48,
            expand_v=1.0,
            allow_neg_eigval=allow_neg_eigval,
            conv_size=4,
            use_fla=False,
            use_packed_qkv_conv=True,
            use_packed_qkv_proj=True,
            conv_output_contiguous=True,
            gates_fp32=True,
            output_norm_fp32=True,
        )
        .cuda()
        .bfloat16()
        .eval()
    )
    restore_low_dim_params_to_fp32(module, gdn_control_proj_fp32=False)
    return module


def sequence_lengths(max_small_t: int, extra_ts: list[int]) -> list[int]:
    """Build the sorted set of sequence lengths to evaluate.

    :param int max_small_t: Inclusive small-length sweep upper bound.
    :param list[int] extra_ts: Additional lengths to append.
    :return list[int]: Sorted unique sequence lengths.
    """
    return sorted({*range(1, max_small_t + 1), *extra_ts})


def run_case(
    *,
    allow_neg_eigval: bool,
    seq_lengths: list[int],
    seed: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Run one eager-vs-FLA recurrence sweep.

    :param bool allow_neg_eigval: Whether to enable beta scaling by `2.0`.
    :param list[int] seq_lengths: Sequence lengths to check.
    :param int seed: Base RNG seed.
    :param float atol: Absolute tolerance for the `ok` flag.
    :param float rtol: Relative tolerance for the `ok` flag.
    :return dict[str, object]: Sweep summary.
    """
    torch.manual_seed(seed + (10_000 if allow_neg_eigval else 0))
    module = make_module(allow_neg_eigval=allow_neg_eigval)
    max_t = max(seq_lengths)
    x_full = torch.randn(1, max_t, 384, device="cuda", dtype=torch.bfloat16)
    rows: list[dict[str, object]] = []
    first_fail_t: int | None = None

    with torch.no_grad():
        for seq_len in seq_lengths:
            x = x_full[:, :seq_len]
            q, k, v, g, beta = module._project_recurrence_inputs(x)
            eager_out, _ = gdn_recurrent_naive(q, k, v, g.exp(), beta)
            fla_out = fla_chunk_gated_delta_rule_compile_visible(q, k, v, g, beta)
            stats = diff_stats(fla_out, eager_out)
            ok = torch.allclose(
                fla_out.detach().float().cpu(),
                eager_out.detach().float().cpu(),
                atol=atol,
                rtol=rtol,
            )
            row = {
                "T": seq_len,
                "ok": bool(ok),
                **stats,
            }
            rows.append(row)
            if not ok and first_fail_t is None:
                first_fail_t = seq_len

    return {
        "allow_neg_eigval": allow_neg_eigval,
        "first_fail_t": first_fail_t,
        "rows": rows,
    }


def main() -> None:
    """Parse CLI args and run the recurrence mismatch sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-small-t", type=int, default=80)
    parser.add_argument("--extra-ts", type=int, nargs="*", default=[128, 512])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to store the full sweep payload as JSON.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the HGDN recurrence diagnostic")
    if not _HAS_FLA:
        raise RuntimeError("FLA is not available in this environment")

    seqs = sequence_lengths(args.max_small_t, args.extra_ts)
    payload = {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "atol": args.atol,
        "rtol": args.rtol,
        "seq_lengths": seqs,
        "cases": [
            run_case(
                allow_neg_eigval=False,
                seq_lengths=seqs,
                seed=args.seed,
                atol=args.atol,
                rtol=args.rtol,
            ),
            run_case(
                allow_neg_eigval=True,
                seq_lengths=seqs,
                seed=args.seed,
                atol=args.atol,
                rtol=args.rtol,
            ),
        ],
    }

    for case in payload["cases"]:
        rows = case["rows"]
        worst = max(rows, key=lambda row: float(row["norm_rel"]))
        print(
            "case:"
            f"allow_neg_eigval={int(case['allow_neg_eigval'])} "
            f"first_fail_t={case['first_fail_t']} "
            f"worst_T={worst['T']} "
            f"worst_max_abs={worst['max_abs']:.6g} "
            f"worst_norm_rel={worst['norm_rel']:.6g}"
        )
        for seq_len in [1, 4, 8, 16, 32, 64, 80, 128, 512]:
            row = next((row for row in rows if row["T"] == seq_len), None)
            if row is None:
                continue
            print(
                f"  T={row['T']:>4d} ok={int(row['ok'])} "
                f"max_abs={row['max_abs']:.6g} "
                f"max_rel={row['max_rel']:.6g} "
                f"rmse={row['rmse']:.6g} "
                f"norm_rel={row['norm_rel']:.6g}"
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
