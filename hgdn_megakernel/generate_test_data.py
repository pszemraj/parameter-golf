"""Generate real-module HGDN megakernel reference cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hgdn_runtime_utils import restore_low_dim_params_to_fp32  # noqa: E402
from model import GatedDeltaNet, _HAS_FLA  # noqa: E402

CASE_FORMAT_VERSION = 2


def make_module(
    *,
    d_model: int,
    n_heads: int,
    head_k_dim: int,
    expand_v: float,
    conv_size: int,
    use_fla: bool,
) -> GatedDeltaNet:
    """Build the live packed HGDN contract used by the megakernel harness.

    :param int d_model: Model width.
    :param int n_heads: HGDN head count.
    :param int head_k_dim: Per-head key width.
    :param float expand_v: Value expansion factor.
    :param int conv_size: Packed depthwise conv width.
    :param bool use_fla: Whether to route the recurrence through the live FLA path.
    :return GatedDeltaNet: Prepared eager reference module.
    """
    module = (
        GatedDeltaNet(
            d_model=d_model,
            n_heads=n_heads,
            head_k_dim=head_k_dim,
            expand_v=expand_v,
            allow_neg_eigval=True,
            conv_size=conv_size,
            use_fla=use_fla,
            use_packed_qkv_conv=True,
            use_packed_qkv_proj=True,
            conv_output_contiguous=True,
            gates_fp32=True,
            output_norm_fp32=True,
        )
        .cuda()
        .bfloat16()
    )
    restore_low_dim_params_to_fp32(module, gdn_control_proj_fp32=False)
    return module


def _capture_reference(
    module: GatedDeltaNet, x_seed: torch.Tensor
) -> dict[str, object]:
    """Run one live-module reference path and capture outputs and gradients.

    :param GatedDeltaNet module: Reference HGDN module.
    :param torch.Tensor x_seed: Shared input activations without gradients.
    :return dict[str, object]: Reference tensors saved on CPU.
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
        "y": y.detach().cpu(),
        "loss": loss.detach().cpu(),
        "grad_x": x.grad.detach().cpu(),
        "grad_w_qkv": module.w_qkv.weight.grad.detach().cpu(),
        "grad_w_a": module.w_a.weight.grad.detach().cpu(),
        "grad_w_b": module.w_b.weight.grad.detach().cpu(),
        "grad_w_g": module.w_g.weight.grad.detach().cpu(),
        "grad_w_out": module.w_out.weight.grad.detach().cpu(),
        "grad_conv_w": conv_grad.cpu(),
        "grad_A_log": module.A_log.grad.detach().cpu(),
        "grad_dt_bias": module.dt_bias.grad.detach().cpu(),
    }


def generate_case(
    *,
    out: Path,
    batch: int,
    seq: int,
    d_model: int,
    n_heads: int,
    head_k_dim: int,
    expand_v: float,
    conv_size: int,
    seed: int,
) -> None:
    """Generate one serialized HGDN reference case.

    :param Path out: Output `.pt` file path.
    :param int batch: Batch size.
    :param int seq: Sequence length.
    :param int d_model: Model width.
    :param int n_heads: HGDN head count.
    :param int head_k_dim: Per-head key width.
    :param float expand_v: Value expansion factor.
    :param int conv_size: Packed depthwise conv width.
    :param int seed: RNG seed.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to generate HGDN megakernel reference data"
        )
    torch.manual_seed(seed)
    module = make_module(
        d_model=d_model,
        n_heads=n_heads,
        head_k_dim=head_k_dim,
        expand_v=expand_v,
        conv_size=conv_size,
        use_fla=False,
    )
    x_seed = torch.randn(
        batch,
        seq,
        d_model,
        device="cuda",
        dtype=torch.bfloat16,
    )
    eager_reference = _capture_reference(module, x_seed)
    state_dict = {
        key: value.detach().clone() for key, value in module.state_dict().items()
    }
    conv_w = module.qkv_conv.conv.weight.detach().view(
        module.qkv_conv.conv.weight.shape[0],
        module.qkv_conv.conv.weight.shape[-1],
    )
    references: dict[str, dict[str, object]] = {"eager": eager_reference}
    if _HAS_FLA:
        module_fla = make_module(
            d_model=d_model,
            n_heads=n_heads,
            head_k_dim=head_k_dim,
            expand_v=expand_v,
            conv_size=conv_size,
            use_fla=True,
        )
        module_fla.load_state_dict(state_dict)
        references["fla"] = _capture_reference(module_fla, x_seed)
    payload = {
        "format_version": CASE_FORMAT_VERSION,
        "meta": {
            "B": batch,
            "T": seq,
            "D": d_model,
            "H": n_heads,
            "Dk": head_k_dim,
            "Dv": module.head_v_dim,
            "K": conv_size,
            "allow_neg_eigval": True,
            "expand_v": expand_v,
            "has_fla_reference": "fla" in references,
        },
        "inputs": {
            "x": x_seed.detach().cpu(),
            "w_qkv": module.w_qkv.weight.detach().cpu(),
            "w_a": module.w_a.weight.detach().cpu(),
            "w_b": module.w_b.weight.detach().cpu(),
            "w_g": module.w_g.weight.detach().cpu(),
            "w_out": module.w_out.weight.detach().cpu(),
            "conv_w": conv_w.cpu(),
            "A_log": module.A_log.detach().float().cpu(),
            "dt_bias": module.dt_bias.detach().float().cpu(),
        },
        "references": references,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"wrote {out} with meta={payload['meta']}")


def main() -> None:
    """Parse CLI args and emit a serialized HGDN case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--D", type=int, default=384)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--Dk", type=int, default=48)
    parser.add_argument("--expand-v", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    generate_case(
        out=args.out,
        batch=args.B,
        seq=args.T,
        d_model=args.D,
        n_heads=args.H,
        head_k_dim=args.Dk,
        expand_v=args.expand_v,
        conv_size=args.K,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
