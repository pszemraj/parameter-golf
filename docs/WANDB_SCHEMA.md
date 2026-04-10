# HGDN W&B Schema

Last updated: 2026-04-10

HGDN runs log W&B config, history, and summary fields under the schema below.
The relevant trainer is [train_gpt_hybrid.py](../train_gpt_hybrid.py).

## Scope

- Applies to HGDN branch runs only.
- Default project for real HGDN ablations: `pg-hgdn-ablations`.
- Do not query or default to older HGDN runs in other W&B projects; migrate them
  into `pg-hgdn-ablations` instead.
- Treat this schema as experiment protocol.
- Do not use `args.to_dict()` or any other blanket config dump.
- Do not log lowercase Python attribute mirrors alongside the canonical keys.
- If this schema changes, update this file in the same change.

## Config

W&B config for HGDN runs uses canonical env-style uppercase keys only.

Core architecture and size:

- `NUM_LAYERS`
- `MODEL_DIM`
- `MLP_MULT`
- `GDN_RATIO`
- `ATTN_BLOCKS`
- `CONV_BLOCKS`
- `N_PARAMS`
- `NUM_HEADS`
- `NUM_KV_HEADS`
- `GDN_N_HEADS`
- `GDN_EXPAND_V`
- `GDN_HEAD_K_DIM`
- `GDN_CONV_SIZE`
- `NORM_STYLE`
- `RESIDUAL_ALPHA`
- `TIE_EMBEDDINGS`

Training contract and runtime:

- `TRAIN_BATCH_TOKENS`
- `TRAIN_SEQ_LEN`
- `GRAD_ACCUM_STEPS`
- `LOCAL_BATCH_SIZE`
- `ITERATIONS`
- `WARMDOWN_ITERS`
- `WARMUP_STEPS`
- `VAL_BATCH_SIZE`
- `VAL_LOSS_EVERY`
- `TRAIN_LOG_EVERY`
- `MAX_WALLCLOCK_SECONDS`
- `PLANNED_TRAIN_TOKENS`
- `ARTIFACT_LIMIT_BYTES`
- `COMPILE`
- `COMPILE_STRATEGY`
- `COMPILE_GDN_DISABLED`
- `COMPILE_GDN_MLPS_COMPILED`
- `COMPILE_ATTN_BLOCKS_COMPILED`
- `COMPILE_MODEL_COMPILED`
- `SDPA_BACKEND`
- `CUDNN_BENCHMARK`
- `FLA_AVAILABLE`

Optimizer knobs:

- `EMBED_LR`
- `HEAD_LR`
- `TIED_EMBED_LR`
- `MATRIX_LR`
- `SCALAR_LR`
- `MUON_MOMENTUM`
- `MUON_BACKEND_STEPS`
- `MUON_MOMENTUM_WARMUP_START`
- `MUON_MOMENTUM_WARMUP_STEPS`
- `BETA1`
- `BETA2`
- `ADAM_EPS`
- `GRAD_CLIP_NORM`
- `WEIGHT_DECAY`

HGDN kernel and layout knobs:

- `GDN_USE_Q_CONV`
- `GDN_USE_K_CONV`
- `GDN_USE_V_CONV`
- `GDN_USE_PACKED_QKV_CONV`
- `GDN_USE_PACKED_QKV_PROJ`
- `GDN_CONV_OUTPUT_CONTIGUOUS`
- `GDN_Q_CONV_OUTPUT_CONTIGUOUS`
- `GDN_K_CONV_OUTPUT_CONTIGUOUS`
- `GDN_V_CONV_OUTPUT_CONTIGUOUS`
- `GDN_CONTROL_PROJ_FP32`
- `GDN_GATES_FP32`
- `GDN_OUTPUT_NORM_FP32`
- `GDN_USE_CUDA_FRONTEND_NCT`
- `GDN_USE_CUDA_PACKED_CONV`
- `GDN_USE_CUDA_PACKED_CONV_ATEN_BACKWARD`
- `GDN_USE_CUDA_PACKED_CONV_ATEN_WEIGHT_BACKWARD`
- `GDN_USE_CUDA_FUSED_FRONTEND`
- `GDN_USE_CUDA_FUSED_FRONTEND_LIB`
- `GDN_USE_CUDA_FUSED_OUTPUT`
- `GDN_USE_CUDA_SPLIT_NORM`
- `GDN_USE_CUDA_SPLIT_NORM_LIB`
- `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD`
- `GDN_PACKED_QKV_SINGLE_CONTIG`
- `GDN_PACKED_QKV_SPLIT_COPY`
- `WANDB_WATCH`
- `WANDB_WATCH_LOG_FREQ`

## History

Keep only genuinely time-varying metrics in per-step history:

- `train/loss`
- `train/lr`
- `train/lr_scale`
- `train/processed_tokens`
- `train/step_ms`
- `train/tokens_per_s`
- `eval/loss`
- `eval/bpb`
- `watch/*`

Do not log static descriptors to history.

## Summary

One-shot and final outcomes belong in W&B summary:

- `profile_dir`
- `perf_ignore_steps_final`
- `perf_measured_steps_final`
- `perf_step_ms_final`
- `perf_tokens_per_s_final`
- `perf_skip_final_eval`
- `system/peak_mem_alloc_mib`
- `system/peak_mem_reserved_mib`
- `roundtrip_val_loss_final`
- `roundtrip_val_bpb_final`
- `roundtrip_eval_time_ms_final`
- `artifact_status_final`
- `artifact_warning_final`
- `artifact_bytes_final`
- `artifact_headroom_bytes_final`
- `artifact/raw_state_dict_bytes_final`
- `artifact/quant_baseline_tensor_bytes_final`
- `artifact/quant_int8_payload_bytes_final`
- `artifact/quant_raw_torch_bytes_final`
- `artifact/quant_raw_torch_overhead_bytes_final`
- `artifact/quant_baseline_to_payload_ratio_final`
- `artifact/quant_payload_to_zlib_ratio_final`
- `artifact/quant_raw_to_zlib_ratio_final`
- `artifact/code_bytes_final`
- `artifact/int8_payload_zlib_bytes_final`

## Compatibility

- New HGDN runs must emit the canonical uppercase config schema above.
- Analysis tools may keep fallback readers for older lowercase runs, but the
  trainer must not emit duplicate lowercase and uppercase mirrors.
