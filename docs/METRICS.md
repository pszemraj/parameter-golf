# Metrics

Last updated: 2026-03-31 04:13 EDT

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
  - The earlier `GPT_REF`, `B1`, and `C2` rows below were collected before the eval-cadence cleanup and therefore used the older `step 0, then every 100, then final` schedule.
  - The current canonical reruns use `first eval at step 100`, then every `250`, plus the forced final eval.
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
| 2026-03-31 04:05:45 EDT | `T2` | `train_hconv.py` | Tied-depth main bet: `6` unique conv, `5` unique attn, `18` effective conv, `mlp_mult=2` | 1.3693 | 1.36955833 | 14896112 | [train.log](../runs_hconv_quality_5090/T2/train.log) |
| 2026-03-31 04:12:52 EDT | `T3` | `train_hconv.py` | Tied-depth variant: `4` unique conv, `5` unique attn, `16` effective conv, `mlp_mult=3` | 1.3693 | 1.36937905 | 15288128 | [train.log](../runs_hconv_quality_5090/T3/train.log) |

Current read:

- `B1` beat `GPT_REF` by `0.0076` bpb on this sampled-validation comparison protocol.
- `B1` is currently under the 16,000,000-byte compressed-model limit with `15,415,397` bytes.
- `C2` is much worse than `B1` here by `0.1999` bpb, so the pure-conv variant does not look competitive on this protocol.
- Phase 2 reruns under the cleaned cadence ended in a tie to four decimals on final `val_bpb`: both `T2` and `T3` finished at `1.3693`.
- `T2` is still smaller than `T3` by `392,016` compressed bytes, while both remain under the 16 MB cap.
- `T3` holds the tiny roundtrip edge after export and reload: `1.36937905` vs `1.36955833`.
- Relative to `B1`, both tied-depth variants now improve the final scheduled `val_bpb` by about `0.0033`.

## Smoke / Bring-Up Runs

These are sanity checks, not comparable quality runs.

| Timestamp | Config | Purpose | Contract | Final val_bpb | Roundtrip val_bpb | int8+zlib_bytes | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-03-31 03:48:49 EDT | `SMOKE_HCONV` | Compile-enabled sweep-harness smoke with the cleaned W&B naming | `10` steps, `32768` train tokens/step, no periodic eval, final eval only | 3.7165 | 3.71649312 | 15592083 | [train.log](../runs_hconv_quality_5090/SMOKE_HCONV/train.log) |

Notes:

- The canonical smoke row above is the current sweep-harness rerun that also seeded the cleaned W&B project.
- It remains useful for bring-up and compile-behavior checks, but not for architecture quality comparison.
