#!/usr/bin/env python3
"""Build and inspect a core/amplifier model.

python inspect_model.py init my_model/ --data /path/to/data
python inspect_model.py init my_model/ --data /path/to/data --spec-strategy preload --spec-workers 12
python inspect_model.py my_model/
python inspect_model.py my_model/ --test-generate 100
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import torch


def main() -> None:
    """Inspect or initialize a core/amplifier model directory."""
    p = argparse.ArgumentParser(description="Build and inspect a core/amplifier model")
    p.add_argument("command", help="'init' to create, or path to existing model dir")
    p.add_argument("model_dir", nargs="?", default=None, help="Model directory (after 'init')")
    p.add_argument("--test-generate", type=int, default=0)
    p.add_argument("--tokenizer", type=str, default=None)

    # Init overrides
    g = p.add_argument_group("init options")
    g.add_argument("--data", type=str, default=None)
    g.add_argument("--storage-dtype", type=str, default=None)
    g.add_argument("--vocab-size", type=int, default=None)
    g.add_argument("--core-dim", type=int, default=None)
    g.add_argument("--branch-lags", type=str, default=None)
    g.add_argument(
        "--branch-temporal-mode",
        type=str,
        default=None,
        choices=["current", "lagged", "hybrid", "ema", "ema_hybrid"],
    )
    g.add_argument("--branch-temporal-lag-scale", type=float, default=None)
    g.add_argument(
        "--residual-token-gate-mode",
        type=str,
        default=None,
        choices=["none", "base", "core_base"],
    )
    g.add_argument(
        "--branch-router-mode",
        type=str,
        default=None,
        choices=["none", "softmax"],
    )
    g.add_argument("--base-bigram-delta", type=str, default=None, choices=["none", "full"])
    g.add_argument("--residual-readout-delta-rank", type=int, default=None)
    g.add_argument("--residual-readout-delta-init-std", type=float, default=None)
    g.add_argument("--num-blocks", type=int, default=None)
    g.add_argument("--smoothing", type=float, default=None)
    g.add_argument("--embedding-init", type=str, default=None, choices=["spectral", "svd"])
    g.add_argument("--spectral-neighbors", type=int, default=None)
    g.add_argument("--lag-identity-base", type=float, default=None)
    g.add_argument("--fixed-dtype", type=str, default=None)
    g.add_argument("--core-layers", type=int, default=None)
    g.add_argument("--core-expansion", type=float, default=None)
    g.add_argument("--residual-core", type=int, default=None, choices=[0, 1])
    g.add_argument("--residual-core-init", type=float, default=None)
    g.add_argument("--readout-rank", type=int, default=None)
    g.add_argument(
        "--scan-backend",
        type=str,
        default=None,
        choices=["auto", "heinsen", "assoc", "assoc_accel", "sequential"],
    )
    g.add_argument("--max-tokens", type=int, default=None)
    g.add_argument("--spec-workers", type=int, default=None)
    g.add_argument(
        "--spec-strategy",
        type=str,
        default=None,
        choices=["auto", "stream", "parallel", "preload", "gpu"],
    )

    args = p.parse_args()

    from core_amplifier_lm import AmplifierSpec, CoreAmplifierLM, ModelConfig, build_spec_optimized

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    # ---- init ----
    if args.command == "init":
        if not args.model_dir:
            p.error("usage: inspect_model.py init <model_dir> --data <path>")
        if not args.data:
            p.error("--data is required for init")

        overrides = {}
        for attr in [
            "data",
            "vocab_size",
            "core_dim",
            "branch_temporal_mode",
            "branch_temporal_lag_scale",
            "residual_token_gate_mode",
            "branch_router_mode",
            "base_bigram_delta",
            "residual_readout_delta_rank",
            "residual_readout_delta_init_std",
            "num_blocks",
            "smoothing",
            "embedding_init",
            "spectral_neighbors",
            "lag_identity_base",
            "storage_dtype",
            "max_tokens",
            "spec_workers",
            "spec_strategy",
            "core_layers",
            "core_expansion",
            "residual_core",
            "residual_core_init",
            "scan_backend",
            "fixed_dtype",
        ]:
            val = getattr(args, attr.replace("-", "_"), None)
            if val is not None:
                if attr == "residual_core":
                    val = bool(val)
                overrides[attr] = val
        if args.readout_rank is not None:
            overrides["readout_rank"] = (
                None if int(args.readout_rank) <= 0 else int(args.readout_rank)
            )
        if args.branch_lags:
            overrides["branch_lags"] = [int(x) for x in args.branch_lags.split(",") if x]

        cfg = ModelConfig.create(args.model_dir, **overrides)
        cfg.save()
        print(f"Created {cfg.config_path}")
        print(cfg.summary())

        fixed_dtype = dtype_map.get(cfg.model.get("fixed_dtype", "bfloat16"), torch.bfloat16)
        spec = build_spec_optimized(
            cfg.data["source"],
            vocab_size=cfg.model["vocab_size"],
            core_dim=cfg.model["core_dim"],
            branch_lags=cfg.branch_lags_tuple,
            num_blocks=cfg.model["num_blocks"],
            smoothing=cfg.model["smoothing"],
            embedding_init=cfg.model.get("embedding_init", "spectral"),
            spectral_neighbors=cfg.model.get("spectral_neighbors", 64),
            lag_identity_base=cfg.model.get("lag_identity_base", 0.15),
            readout_rank=cfg.model.get("readout_rank"),
            fixed_dtype=fixed_dtype,
            storage_dtype=cfg.data.get("storage_dtype", "uint16"),
            max_tokens=cfg.spec.get("max_tokens"),
            num_workers=cfg.spec.get("workers", -1),
            strategy=cfg.spec.get("strategy", "auto"),
            verbose=True,
        )
        spec.save(cfg.spec_path)
        print(f"Spec → {cfg.spec_path}")

        tok = cfg.find_or_copy_tokenizer()
        if tok:
            print(f"Tokenizer → {tok}")
        else:
            print(
                "No tokenizer found (bpb computation requires it — place .model file in data dir)"
            )

        cfg.meta["spec_tokens"] = int(spec.metadata.get("total_tokens", 0))
        cfg.save()
        model_dir = Path(args.model_dir)

    # ---- inspect existing ----
    else:
        model_dir = Path(args.command)
        if not (model_dir / "config.json").exists():
            raise SystemExit(
                f"ERROR: {model_dir}/config.json not found. "
                f"Create with: inspect_model.py init {model_dir} --data <path>"
            )
        cfg = ModelConfig.load(model_dir)
        spec = AmplifierSpec.load(cfg.spec_path)

    # ---- load tokenizer ----
    tokenizer = None
    tok_path = args.tokenizer or (str(cfg.tokenizer_path) if cfg.tokenizer_path else None)
    if tok_path:
        try:
            import sentencepiece as spm

            tokenizer = spm.SentencePieceProcessor(model_file=tok_path)
            print(f"Tokenizer: {tok_path} (vocab={tokenizer.get_piece_size()})")
        except ImportError:
            print("WARNING: sentencepiece not installed")
        except Exception as e:
            print(f"WARNING: tokenizer failed: {e}")

    # ---- display ----
    core_layers = cfg.model.get("core_layers", 3)
    core_type = cfg.model.get("core_type", "mingru")
    core_expansion = cfg.model.get("core_expansion", 2.0)
    residual_core = bool(cfg.model.get("residual_core", True))
    residual_core_init = float(cfg.model.get("residual_core_init", -2.0))
    model = CoreAmplifierLM(
        spec,
        core_layers=core_layers,
        core_type=core_type,
        core_expansion=core_expansion,
        residual_core=residual_core,
        residual_core_init=residual_core_init,
        branch_temporal_mode=cfg.model.get("branch_temporal_mode", "current"),
        branch_temporal_lag_scale=float(cfg.model.get("branch_temporal_lag_scale", 1.0)),
        residual_token_gate_mode=cfg.model.get("residual_token_gate_mode", "none"),
        branch_router_mode=cfg.model.get("branch_router_mode", "none"),
        base_bigram_delta=cfg.model.get("base_bigram_delta", "none"),
        residual_readout_delta_rank=int(cfg.model.get("residual_readout_delta_rank", 0)),
        residual_readout_delta_init_std=float(
            cfg.model.get("residual_readout_delta_init_std", 0.02)
        ),
    )

    print(f"\n{'=' * 60}\nSPEC\n{'=' * 60}")
    print(spec.summary())
    fixed = spec.fixed_nbytes
    print(f"  fixed bytes: {fixed:,} ({fixed / 1e6:.2f} MB)")
    spec_path = cfg.spec_path
    if spec_path.exists():
        spec_file_bytes = spec_path.stat().st_size
        spec_gzip_bytes = len(gzip.compress(spec_path.read_bytes(), compresslevel=9))
        print(f"  spec.pt bytes: {spec_file_bytes:,} | gzip(spec.pt): {spec_gzip_bytes:,}")
    print("\n  Buffers:")
    for name in [
        "token_embed",
        "base_bigram_logits",
        "lag_ops",
        "amp_w1",
        "amp_w2",
        "readout_weight",
        "readout_in_proj",
        "readout_out_proj",
    ]:
        buf = getattr(spec, name, None)
        if buf is None:
            continue
        print(f"    {name:24s} {str(list(buf.shape)):20s} {buf.dtype}")
    if spec.metadata:
        print("\n  Metadata:")
        for k, v in spec.metadata.items():
            print(f"    {k}: {v}")

    print(f"\n{'=' * 60}\nMODEL\n{'=' * 60}")
    tp = model.trainable_parameters
    fb = model.fixed_nbytes
    print(
        f"  trainable: {tp:,} params | fixed: {fb:,} bytes ({fb / 1e6:.2f} MB) | ratio: {fb // max(tp * 4, 1)}x"
    )
    print("\n  Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name:32s} {str(list(param.shape)):20s} {param.numel():>8,}")

    print(f"\n{'=' * 60}\nFORWARD PASS\n{'=' * 60}")
    batch = torch.randint(0, spec.vocab_size, (2, 33))
    inp, tgt = batch[:, :-1], batch[:, 1:]
    with torch.no_grad():
        logits = model(inp)
    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1)
    )
    print(
        f"  loss: {loss.item():.4f}  (expected ~ln({spec.vocab_size})={torch.tensor(float(spec.vocab_size)).log().item():.4f})"
    )
    with torch.no_grad():
        full2 = model(inp)
        st = model.initial_state(inp.size(0))
        parts = []
        for t_idx in range(inp.size(1)):
            lo, st = model.step(inp[:, t_idx], st)
            parts.append(lo[:, None, :])
        step2 = torch.cat(parts, dim=1)
    diff = (full2 - step2).abs().max().item()
    print(f"  step consistency: {diff:.2e} ({'OK' if diff < 1e-4 else 'MISMATCH'})")

    # ---- generate ----
    if args.test_generate > 0:
        print(f"\n{'=' * 60}\nGENERATING {args.test_generate} TOKENS\n{'=' * 60}")
        prompt = torch.randint(0, spec.vocab_size, (1, 4))
        output = model.generate(
            prompt, max_new_tokens=args.test_generate, temperature=1.0, top_k=50
        )
        ids = output[0].tolist()
        print(f"  ids: {ids[:20]}{'...' if len(ids) > 20 else ''}")
        if tokenizer:
            print(f"  text:\n{tokenizer.decode(ids)[:500]}")
        else:
            print("  (no tokenizer — pass --tokenizer or place .model file in model dir)")

    print("\nDone.")


if __name__ == "__main__":
    main()
