"""Shared helpers for HGDN maintenance scripts.

These helpers are intentionally small and script-focused. They avoid pulling
trainer/runtime code into a separate package while still eliminating the most
repeated script-side CUDA preparation logic.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def maybe_freeze_gdn_conv_weights(module: torch.nn.Module, *, enabled: bool) -> None:
    """Optionally freeze wrapped GDN depthwise-conv weights.

    :param torch.nn.Module module: Module tree to mutate in place.
    :param bool enabled: Whether the profiling-only freeze knob is active.
    """
    if not enabled:
        return
    for submodule in module.modules():
        conv = getattr(submodule, "conv", None)
        if isinstance(conv, torch.nn.Conv1d) and submodule.__class__.__name__ in {
            "CausalConv1d",
            "PackedCausalConv1d",
        }:
            conv.weight.requires_grad_(False)


def prepare_cuda_module(
    module: torch.nn.Module,
    *,
    restore_low_dim_params_to_fp32: Callable[..., None],
    freeze_conv_weights: bool = False,
    gdn_control_proj_fp32: bool | None = None,
) -> torch.nn.Module:
    """Apply the shared mixed-precision policy used by HGDN helper scripts.

    :param torch.nn.Module module: Module to prepare in place.
    :param Callable[..., None] restore_low_dim_params_to_fp32: Trainer helper that
        restores scalar/low-dimensional parameters to fp32.
    :param bool freeze_conv_weights: Whether to freeze wrapped GDN conv weights,
        defaults to False.
    :param bool | None gdn_control_proj_fp32: Optional override for the HGDN
        control-projection fp32 policy. When omitted, uses the restore helper's
        default behavior.
    :return torch.nn.Module: Prepared CUDA bf16 module.
    """
    module = module.cuda().bfloat16()
    if gdn_control_proj_fp32 is None:
        restore_low_dim_params_to_fp32(module)
    else:
        restore_low_dim_params_to_fp32(
            module, gdn_control_proj_fp32=gdn_control_proj_fp32
        )
    maybe_freeze_gdn_conv_weights(module, enabled=freeze_conv_weights)
    return module
