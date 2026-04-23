"""Shared runtime helpers for HGDN trainer, scripts, and tests."""

from __future__ import annotations

import io
import zlib
from typing import Any

import torch
from torch import Tensor, nn

from model import scalar_param_patterns, uses_scalar_optimizer

INT8_CLIP_Q = 99.99984 / 100.0
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536


def normalize_wandb_watch_mode(mode: str) -> str:
    """Normalize the requested W&B watch mode.

    :param str mode: Raw watch mode string.
    :raises ValueError: If the mode is unsupported.
    :return str: Normalized watch mode.
    """
    normalized = mode.lower()
    if normalized in {"0", "false", "off", "none"}:
        return "none"
    if normalized in {"gradients", "all"}:
        return normalized
    raise ValueError(f"WANDB_WATCH must be one of: none, gradients, all (got {mode!r})")


def quantize_state_dict_int8(
    state_dict: dict[str, Tensor],
    *,
    gdn_w_g_optimizer: str = "scalar",
) -> tuple[dict[str, Any], dict[str, int]]:
    """Quantize a state dict with per-row int8 weights where possible.

    :param dict[str, Tensor] state_dict: Floating-point state dict.
    :param str gdn_w_g_optimizer: Whether `w_g` should ride Adam or Muon,
        defaults to `"scalar"`.
    :return tuple[dict[str, Any], dict[str, int]]: Quantized payload object and serialization stats.
    """
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0
    )
    active_scalar_patterns = scalar_param_patterns(gdn_w_g_optimizer=gdn_w_g_optimizer)
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += t.numel() * t.element_size()
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if any(pattern in name for pattern in active_scalar_patterns):
                kept = t.float().contiguous()
            else:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(torch.float16).contiguous()
            passthrough[name] = kept
            stats["int8_payload_bytes"] += kept.numel() * kept.element_size()
            continue
        t32 = t.float()
        if t32.ndim == 2:
            ca = (
                torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                if t32.numel()
                else torch.empty(t32.shape[0])
            )
            clipped = torch.clamp(t32, -ca[:, None], ca[:, None])
            scale = (ca / 127.0).clamp_min(1 / 127.0)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(
                torch.int8
            )
            scales[name] = scale.to(torch.float16).contiguous()
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        else:
            ca = (
                float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
                if t32.numel()
                else 0.0
            )
            scale = torch.tensor(ca / 127.0 if ca > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(
                torch.round(torch.clamp(t32, -ca, ca) / scale), -127, 127
            ).to(torch.int8)
            scales[name] = scale
        quantized[name] = q.contiguous()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += (
            q.numel() + scales[name].numel() * scales[name].element_size()
        )
    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, Any]) -> dict[str, Tensor]:
    """Restore a quantized state dict payload to dense tensors.

    :param dict[str, Any] obj: Serialized quantized payload.
    :return dict[str, Tensor]: Dequantized state dict.
    """
    out = {}
    qmeta = obj.get("qmeta", {})
    pod = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (
                q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
            ).to(dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(dtype)
    for name, t in obj["passthrough"].items():
        ot = t.detach().cpu().contiguous()
        orig = pod.get(name)
        if isinstance(orig, str):
            ot = ot.to(getattr(torch, orig))
        out[name] = ot
    return out


def serialize_quantized_state_dict_int8(
    state_dict: dict[str, Tensor],
    zlib_level: int = 9,
    *,
    gdn_w_g_optimizer: str = "scalar",
) -> tuple[dict[str, Any], bytes, dict[str, int | float]]:
    """Quantize, serialize, and compress a state dict with byte-level audit stats.

    :param dict[str, Tensor] state_dict: Floating-point state dict to pack.
    :param int zlib_level: Compression level for the final zlib payload, defaults to 9.
    :param str gdn_w_g_optimizer: Whether `w_g` should ride Adam or Muon,
        defaults to `"scalar"`.
    :return tuple[dict[str, Any], bytes, dict[str, int | float]]: Quantized object, compressed blob, and byte audit metrics.
    """
    quant_obj, quant_stats = quantize_state_dict_int8(
        state_dict, gdn_w_g_optimizer=gdn_w_g_optimizer
    )
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=zlib_level)
    quant_raw_bytes = len(quant_raw)
    quant_zlib_bytes = len(quant_blob)
    int8_payload_bytes = quant_stats["int8_payload_bytes"]
    baseline_tensor_bytes = quant_stats["baseline_tensor_bytes"]
    audit = {
        **quant_stats,
        "quant_raw_torch_bytes": quant_raw_bytes,
        "quant_raw_overhead_bytes": quant_raw_bytes - int8_payload_bytes,
        "quant_zlib_bytes": quant_zlib_bytes,
        "baseline_to_payload_ratio": baseline_tensor_bytes / max(int8_payload_bytes, 1),
        "payload_to_zlib_ratio": int8_payload_bytes / max(quant_zlib_bytes, 1),
        "raw_quant_to_zlib_ratio": quant_raw_bytes / max(quant_zlib_bytes, 1),
    }
    return quant_obj, quant_blob, audit


def restore_low_dim_params_to_fp32(
    module: nn.Module,
    *,
    gdn_control_proj_fp32: bool = True,
    gdn_w_g_optimizer: str = "scalar",
) -> None:
    """Restore low-dimensional or scalar parameters to fp32.

    :param nn.Module module: Module whose parameters should be restored in place.
    :param bool gdn_control_proj_fp32: Whether to also restore HGDN control
        projections `w_a/w_b/w_g` to fp32, defaults to True.
    :param str gdn_w_g_optimizer: Whether `w_g` should ride Adam or Muon,
        defaults to `"scalar"`.
    """
    control_patterns = scalar_param_patterns(gdn_w_g_optimizer=gdn_w_g_optimizer)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if not gdn_control_proj_fp32 and any(
                pattern in name for pattern in control_patterns
            ):
                continue
            if (
                uses_scalar_optimizer(
                    name,
                    param_ndim=param.ndim,
                    gdn_w_g_optimizer=gdn_w_g_optimizer,
                )
                and param.dtype != torch.float32
            ):
                param.data = param.data.float()


def maybe_freeze_gdn_conv_weights(module: nn.Module, *, enabled: bool) -> None:
    """Optionally freeze wrapped GDN depthwise-conv weights.

    :param nn.Module module: Module tree to mutate in place.
    :param bool enabled: Whether the profiling-only freeze knob is active.
    """
    if not enabled:
        return
    for submodule in module.modules():
        conv = getattr(submodule, "conv", None)
        if isinstance(conv, nn.Conv1d) and submodule.__class__.__name__ in {
            "CausalConv1d",
            "PackedCausalConv1d",
        }:
            conv.weight.requires_grad_(False)


def prepare_cuda_module(
    module: nn.Module,
    *,
    restore_low_dim_params_to_fp32: Any,
    freeze_conv_weights: bool = False,
    gdn_control_proj_fp32: bool | None = None,
    gdn_w_g_optimizer: str = "scalar",
) -> nn.Module:
    """Apply the shared mixed-precision policy used by HGDN helper scripts.

    :param nn.Module module: Module to prepare in place.
    :param Any restore_low_dim_params_to_fp32: Helper that restores
        scalar/low-dimensional parameters to fp32.
    :param bool freeze_conv_weights: Whether to freeze wrapped GDN conv weights,
        defaults to False.
    :param bool | None gdn_control_proj_fp32: Optional override for the HGDN
        control-projection fp32 policy. When omitted, uses the restore helper's
        default behavior.
    :param str gdn_w_g_optimizer: Whether `w_g` should ride Adam or Muon,
        defaults to `"scalar"`.
    :return nn.Module: Prepared CUDA bf16 module.
    """
    module = module.cuda().bfloat16()
    restore_kwargs: dict[str, Any] = {"gdn_w_g_optimizer": gdn_w_g_optimizer}
    if gdn_control_proj_fp32 is not None:
        restore_kwargs["gdn_control_proj_fp32"] = gdn_control_proj_fp32
    restore_low_dim_params_to_fp32(module, **restore_kwargs)
    maybe_freeze_gdn_conv_weights(module, enabled=freeze_conv_weights)
    return module


def maybe_compile(
    obj: Any, *, enabled: bool, dynamic: bool = False, fullgraph: bool = False
) -> Any:
    """Conditionally wrap an object with `torch.compile`.

    :param Any obj: Callable or module to compile.
    :param bool enabled: Whether compilation is enabled.
    :param bool dynamic: Whether to enable dynamic shapes, defaults to False.
    :param bool fullgraph: Whether to require full-graph capture, defaults to False.
    :return Any: Compiled object when enabled, otherwise the original object.
    """
    if not enabled:
        return obj
    if isinstance(obj, nn.Module):
        obj.compile(dynamic=dynamic, fullgraph=fullgraph)
        return obj
    return torch.compile(obj, dynamic=dynamic, fullgraph=fullgraph)


def maybe_disable_compile(obj: Any, *, enabled: bool, reason: str | None = None) -> Any:
    """Conditionally mark an object as eager-only for `torch.compile`.

    :param Any obj: Callable or module to exclude from Dynamo tracing.
    :param bool enabled: Whether compilation is enabled.
    :param str | None reason: Optional reason string for debugging, defaults to None.
    :return Any: Wrapped object when enabled, otherwise the original object.
    """
    if not enabled:
        return obj
    return torch.compiler.disable(obj, reason=reason)


def prepare_hybrid_compile(
    model: nn.Module,
    *,
    enabled: bool,
    strategy: str,
    compile_top_level: bool = True,
) -> tuple[nn.Module, dict[str, int | str]]:
    """Apply selective compile boundaries for the hybrid stack.

    :param nn.Module model: Hybrid language model to prepare.
    :param bool enabled: Whether compilation is enabled.
    :param str strategy: One of `model`, `selective`, or `hybrid`.
    :param bool compile_top_level: Whether to compile the top-level model object
        in addition to selective submodules, defaults to True.
    :raises ValueError: If `strategy` is not recognized.
    :return tuple[nn.Module, dict[str, int | str]]: Prepared model and compile stats.
    """
    compile_stats: dict[str, int | str] = {
        "strategy": "off" if not enabled else strategy,
        "gdn_disabled": 0,
        "gdn_fla_blocks_compiled": 0,
        "gdn_corekernel_left_enabled": 0,
        "gdn_megakernel_left_enabled": 0,
        "gdn_blocks_compiled": 0,
        "gdn_mlps_compiled": 0,
        "attn_blocks_compiled": 0,
        "model_compiled": 0,
    }
    if not enabled:
        return model, compile_stats

    if strategy not in {"model", "selective", "hybrid"}:
        raise ValueError(
            f"Unsupported COMPILE_STRATEGY={strategy!r}; expected model, selective, or hybrid"
        )

    if strategy in {"selective", "hybrid"}:
        for idx, block_type in enumerate(model.block_types):
            block = model.blocks[idx]
            if block_type == "gdn" and hasattr(block, "gdn") and hasattr(block, "mlp"):
                use_corekernel = bool(getattr(block.gdn, "use_cuda_corekernel", False))
                use_megakernel = bool(getattr(block.gdn, "use_cuda_megakernel", False))
                use_fla = bool(getattr(block.gdn, "use_fla", False))
                if not use_corekernel and not use_megakernel and not use_fla:
                    block.gdn = maybe_disable_compile(
                        block.gdn,
                        enabled=True,
                        reason="Naive GDN fallback stays eager-only",
                    )
                    compile_stats["gdn_disabled"] += 1
                else:
                    block.gdn = maybe_compile(
                        block.gdn,
                        enabled=True,
                        dynamic=False,
                        fullgraph=False,
                    )
                    compile_stats["gdn_blocks_compiled"] += 1
                    if use_fla and not use_corekernel and not use_megakernel:
                        compile_stats["gdn_fla_blocks_compiled"] += 1
                    if use_corekernel:
                        compile_stats["gdn_corekernel_left_enabled"] += 1
                    if use_megakernel:
                        compile_stats["gdn_megakernel_left_enabled"] += 1
                block.mlp = maybe_compile(
                    block.mlp, enabled=True, dynamic=False, fullgraph=True
                )
                compile_stats["gdn_mlps_compiled"] += 1
            elif block_type == "attn":
                model.blocks[idx] = maybe_compile(
                    block, enabled=True, dynamic=False, fullgraph=True
                )
                compile_stats["attn_blocks_compiled"] += 1

    if strategy in {"model", "hybrid"} and compile_top_level:
        model = maybe_compile(model, enabled=True, dynamic=False, fullgraph=False)
        compile_stats["model_compiled"] = 1

    return model, compile_stats
