# Metrics

Last updated: 2026-03-31 03:58 EDT

This file tracks model-quality results from local 5090 runs in [`runs_hconv_quality_5090/`](../runs_hconv_quality_5090/).

## How Metrics Are Recorded

- Timestamps are the local file modification times of each run's `train.log` in `America/New_York`.
- For quality-comparison runs, `val_bpb` is taken from the final scheduled validation line, e.g. `step:750/750 val_loss:... val_bpb:...`.
- `roundtrip_val_bpb` is taken from `final_int8_zlib_roundtrip_exact ... val_bpb:...` after int8+zlib serialization and reload.
- `int8+zlib_bytes` is taken from `Serialized model int8+zlib: ... bytes`.
- The quality-comparison contract for new reruns is the current local sweep-harness setup:
  - `TRAIN_SEQ_LEN=1024`
  - `TRAIN_BATCH_TOKENS=262144`
  - `MAX_STEPS=750`
  - `planned_train_tokens=196,608,000`
  - `EVAL_MODE=sampled`
  - `VAL_BATCH_SIZE=8192`
  - `VAL_BATCHES=8`
  - `VAL_FIRST_STEP=100`
  - `VAL_LOSS_EVERY=250`
  - `TRAIN_LOG_EVERY=25`
  - `MAX_WALLCLOCK_SECONDS=0`
  - `SDPA_BACKEND=auto`
  - `TORCH_BLAS_PREFER_CUBLASLT=1`
- Historical note:
  - The existing quality rows below were collected before the eval-cadence cleanup and therefore used the older `step 0, then every 100, then final` schedule.
  - Canonical reruns after this timestamp use `first eval at step 100`, then every `250`, plus the forced final eval.
- Important discrepancy:
  - The current trainer path still fixes `grad_accum_steps=8` at `WORLD_SIZE=1`.
  - Current `AGENTS.md` guidance for this kind of 1x5090 quality comparison says `GRAD_ACCUM_STEPS=64`, but that guidance came from an earlier parameter-sharing setup that could use more memory than the current hconv family.
  - That means the runs below are internally comparable to each other, while the accumulation mismatch versus the older guidance remains a protocol note rather than automatic evidence that the hconv harness is wrong.

## Quality Runs

| Timestamp | Config | Trainer | Summary | Final val_bpb | Roundtrip val_bpb | int8+zlib_bytes | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-03-31 02:47:40 EDT | `GPT_REF` | `train_gpt.py` | 9-layer GPT reference, tied embeddings, `mlp_mult=2` | 1.3802 | 1.38262331 | 11316588 | [train.log](../runs_hconv_quality_5090/GPT_REF/train.log) |
| 2026-03-31 02:52:13 EDT | `B1` | `train_hconv.py` | Vanilla hybrid: 10 conv + 3 attn, no dilation, no squared gate, no hippo init | 1.3726 | 1.37255835 | 15415397 | [train.log](../runs_hconv_quality_5090/B1/train.log) |
| 2026-03-31 03:00:05 EDT | `C2` | `train_hconv.py` | Pure conv: 15 conv, 0 attn, no dilation, no squared gate, no hippo init | 1.5725 | 1.57226136 | 15235472 | [train.log](../runs_hconv_quality_5090/C2/train.log) |
| 2026-03-31 03:20:00 EDT | `T2` | `train_hconv.py` | Tied-depth main bet: `6` unique conv, `5` unique attn, `18` effective conv, `mlp_mult=2` | 1.3742 | 1.37417743 | 14895138 | [train.log](../runs_hconv_quality_5090/T2/train.log) |
| 2026-03-31 03:27:17 EDT | `T3` | `train_hconv.py` | Tied-depth variant: `4` unique conv, `5` unique attn, `16` effective conv, `mlp_mult=3` | 1.3746 | 1.37488066 | 15295390 | [train.log](../runs_hconv_quality_5090/T3/train.log) |

Current read:

- `B1` beat `GPT_REF` by `0.0076` bpb on this sampled-validation comparison protocol.
- `B1` is currently under the 16,000,000-byte compressed-model limit with `15,415,397` bytes.
- `C2` is much worse than `B1` here by `0.1999` bpb, so the pure-conv variant does not look competitive on this protocol.
- Phase 2 was effectively a near-tie on quality: `T2` finished at `1.3742` and `T3` at `1.3746`, so `T2` keeps a small `0.0004` bpb edge.
- `T2` is also smaller than `T3` by `400,252` compressed bytes, while both remain under the 16 MB cap.
- Relative to `B1`, the tied-depth variants did not win outright on bpb here, but `T2` came within `0.0016` bpb while using fewer compressed bytes.

## Smoke / Bring-Up Runs

These are sanity checks, not comparable quality runs.

| Timestamp | Config | Purpose | Contract | Final val_bpb | Roundtrip val_bpb | int8+zlib_bytes | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-03-31 02:36:03 EDT | `SMOKE_HCONV` | No-compile smoke for the new CLI/output-dir path | `10` steps, `32768` train tokens/step, sampled eval with `1` batch | 3.7166 | 3.71663518 | 15591886 | [train.log](../runs_hconv_quality_5090/SMOKE_HCONV/train.log) |
| 2026-03-31 02:40:02 EDT | `SMOKE_HCONV_COMPILE` | Compile-enabled smoke for the same hconv config | `10` steps, `32768` train tokens/step, sampled eval with `1` batch | 3.7165 | 3.71649684 | 15591744 | [train.log](../runs_hconv_quality_5090/SMOKE_HCONV_COMPILE/train.log) |

Notes:

- `SMOKE_HCONV_COMPILE` was run directly through `torchrun` with the same trainer flags, not through the sweep harness, so it does not have a `launch_summary.txt`.
- The two smoke runs are useful for bring-up and compile-behavior checks, but not for architecture quality comparison.
