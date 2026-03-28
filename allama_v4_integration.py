"""Integration helpers for wiring the v4 shared ALlama module into a trainer."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from allama_v4_shared_model import (
    CastedLinear,
    FakeQuantLinear,
    HyperSharedALlama,
    HyperSharedConfig,
    OptimizerBundle,
    build_allama_optimizers,
    restore_low_dim_params_to_fp32,
)


def configure_cuda_fastpath(sdpa_backend: str = "flash") -> dict[str, bool | str]:
    """Set fast CUDA defaults and an explicit SDPA backend.

    :param str sdpa_backend: Requested SDPA backend name.
    :return dict[str, bool | str]: Effective CUDA SDPA capability flags.
    """
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    backend = sdpa_backend.strip().lower()
    if backend == "auto":
        pass
    elif backend == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    elif backend == "efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
    elif backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    elif backend == "cudnn":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(True)
    else:
        raise ValueError(f"unsupported sdpa backend {sdpa_backend!r}")

    return {
        "requested": backend,
        "flash_available": bool(torch.backends.cuda.is_flash_attention_available()),
        "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "cudnn_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
    }


def compile_model_for_train(
    model: nn.Module,
    enabled: bool,
    *,
    fullgraph: bool = True,
) -> nn.Module:
    """Compile the training graph with static shapes.

    :param nn.Module model: Model to compile.
    :param bool enabled: Whether compilation is enabled.
    :param bool fullgraph: Whether to require fullgraph capture.
    :return nn.Module: Compiled model when enabled, else the original model.
    """
    if not enabled or not hasattr(torch, "compile"):
        return model
    return torch.compile(model, dynamic=False, fullgraph=fullgraph)


def build_hyper_shared_model_from_args(
    args: Any,
    vocab_size: int,
    device: torch.device,
) -> HyperSharedALlama:
    """Map an existing trainer args object into the v4 shared-model config.

    :param Any args: Trainer argument object exposing the shared-model fields.
    :param int vocab_size: Token vocabulary size.
    :param torch.device device: Target device for the initialized model.
    :return HyperSharedALlama: Initialized shared ALlama model.
    """
    cfg = HyperSharedConfig(
        vocab_size=vocab_size,
        model_dim=int(args.model_dim),
        embed_dim=int(args.embed_dim),
        num_layers=int(args.num_layers),
        num_shared_blocks=int(args.num_shared_blocks),
        share_pattern=str(args.share_pattern),
        num_heads=int(args.num_heads),
        num_kv_heads=int(args.num_kv_heads),
        mlp_mult=float(args.mlp_mult),
        mlp_multiple_of=int(getattr(args, "mlp_multiple_of", 64)),
        rope_base=float(getattr(args, "rope_base", 10000.0)),
        norm_eps=float(getattr(args, "norm_eps", 1e-5)),
        norm_kind=str(getattr(args, "norm_kind", "rmsnorm")),
        norm_layout=str(getattr(args, "norm_layout", "prenorm")),
        qk_norm=bool(getattr(args, "qk_norm", True)),
        tie_embeddings=bool(getattr(args, "tie_embeddings", True)),
        tied_embed_init_std=float(getattr(args, "tied_embed_init_std", 0.005)),
        logit_softcap=float(getattr(args, "logit_softcap", 30.0)),
        q_gain_init=float(getattr(args, "q_gain_init", 1.5)),
        x0_gate_init=float(getattr(args, "x0_gate_init", -6.0)),
        use_x0_shortcut=bool(getattr(args, "use_x0_shortcut", True)),
        use_final_norm=bool(getattr(args, "use_final_norm", True)),
        zero_init_residual=bool(getattr(args, "zero_init_residual", True)),
        attn_dropout=float(getattr(args, "attn_dropout", 0.0)),
        resid_dropout=float(getattr(args, "resid_dropout", 0.0)),
        use_bias=bool(getattr(args, "use_bias", False)),
        cast_linears=bool(getattr(args, "cast_linears", True)),
    )

    model = HyperSharedALlama(cfg).to(device)
    dtype_name = str(getattr(args, "dtype", "auto")).lower()
    if device.type == "cuda" and dtype_name in {"bf16", "bfloat16", "auto"}:
        model = model.bfloat16()
        restore_low_dim_params_to_fp32(model)
    elif device.type == "cuda" and dtype_name in {"fp16", "float16"}:
        model = model.half()
        restore_low_dim_params_to_fp32(model)
    return model


def swap_selected_linears_to_qat(
    model: nn.Module,
    *,
    qat_bits: int = 6,
    include_patterns: tuple[str, ...] = (
        "qkv",
        "proj",
        "gate_up",
        "down",
        "embed_to_model",
        "model_to_embed",
    ),
) -> nn.Module:
    """Recursively replace selected linears with QAT variants.

    :param nn.Module model: Model to rewrite in-place.
    :param int qat_bits: Target fake-quant bit width.
    :param tuple[str, ...] include_patterns: Name patterns identifying eligible layers.
    :return nn.Module: The rewritten model.
    """

    def replace(parent: nn.Module, prefix: str = "") -> None:
        """Walk the module tree and replace matching linears in-place.

        :param nn.Module parent: Current parent module.
        :param str prefix: Qualified module-name prefix.
        :return None: Applies replacements in-place.
        """
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (CastedLinear, nn.Linear)) and any(
                pat in full_name for pat in include_patterns
            ):
                qat = FakeQuantLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    qat_bits=qat_bits,
                )
                qat.weight.data.copy_(child.weight.data)
                if child.bias is not None and qat.bias is not None:
                    qat.bias.data.copy_(child.bias.data)
                setattr(parent, name, qat)
            else:
                replace(child, full_name)

    replace(model)
    return model


def build_hyper_shared_optimizers_from_args(
    model: HyperSharedALlama, args: Any
) -> OptimizerBundle:
    """Build the Muon + Adam optimizer stack from an existing trainer args object.

    :param HyperSharedALlama model: Shared model whose parameters will be optimized.
    :param Any args: Trainer argument object exposing optimizer hyperparameters.
    :return OptimizerBundle: Multi-optimizer bundle for shared-model training.
    """
    return build_allama_optimizers(
        model,
        tied_embed_lr=float(
            getattr(args, "tied_embed_lr", getattr(args, "embed_lr", 0.03))
        ),
        head_lr=float(getattr(args, "head_lr", 0.01)),
        matrix_lr=float(
            getattr(args, "matrix_lr", getattr(args, "learning_rate", 0.02))
        ),
        scalar_lr=float(getattr(args, "scalar_lr", 0.04)),
        beta1=float(getattr(args, "beta1", 0.9)),
        beta2=float(getattr(args, "beta2", 0.95)),
        adam_eps=float(getattr(args, "adam_eps", 1e-8)),
        muon_momentum=float(getattr(args, "muon_momentum", 0.95)),
        muon_backend_steps=int(getattr(args, "muon_backend_steps", 5)),
    )
