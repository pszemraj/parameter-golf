# Metrics

Last updated: 2026-03-31 06:10 EDT

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
- Important discrepancy:
  - The current trainer path still fixes `grad_accum_steps=8` at `WORLD_SIZE=1`.
  - Current `AGENTS.md` guidance for this kind of 1x5090 quality comparison says `GRAD_ACCUM_STEPS=64`, but that guidance came from an earlier parameter-sharing setup that could use more memory than the current hconv family.
  - That means the runs below are internally comparable to each other, while the accumulation mismatch versus the older guidance remains a protocol note rather than automatic evidence that the hconv harness is wrong.

## Quality Runs

| Timestamp | Config | Trainer | Summary | Final val_bpb | Roundtrip val_bpb | int8+zlib_bytes | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-03-31 04:20:52 EDT | `GPT_REF` | `train_gpt.py` | 9-layer GPT reference, tied embeddings, `mlp_mult=2` | 1.3774 | 1.37948790 | 11315671 | [train.log](../runs_hconv_quality_5090/GPT_REF/train.log) |
| 2026-03-31 06:10:27 EDT | `GPT_12L` | `train_gpt.py` | Size-matched GPT gate: 12-layer GPT, tied embeddings, `mlp_mult=2` | 1.3605 | 1.36351803 | 14959016 | [train.log](../runs_hconv_quality_5090/GPT_12L/train.log) |
| 2026-03-31 04:24:45 EDT | `B1` | `train_hconv.py` | Vanilla hybrid: 10 conv + 3 attn, no dilation, no squared gate, no hippo init | 1.3782 | 1.37821319 | 15423044 | [train.log](../runs_hconv_quality_5090/B1/train.log) |
| 2026-03-31 04:28:30 EDT | `C2` | `train_hconv.py` | Pure conv: 15 conv, 0 attn, no dilation, no squared gate, no hippo init | 1.5756 | 1.57559083 | 15233455 | [train.log](../runs_hconv_quality_5090/C2/train.log) |
| 2026-03-31 04:05:45 EDT | `T2` | `train_hconv.py` | Tied-depth main bet: `6` unique conv, `5` unique attn, `18` effective conv, `mlp_mult=2` | 1.3693 | 1.36955833 | 14896112 | [train.log](../runs_hconv_quality_5090/T2/train.log) |
| 2026-03-31 04:12:52 EDT | `T3` | `train_hconv.py` | Tied-depth variant: `4` unique conv, `5` unique attn, `16` effective conv, `mlp_mult=3` | 1.3693 | 1.36937905 | 15288128 | [train.log](../runs_hconv_quality_5090/T3/train.log) |
| 2026-03-31 04:38:41 EDT | `I1` | `train_hconv.py` | T2 + dilated conv | 1.3726 | 1.37247236 | 14898621 | [train.log](../runs_hconv_quality_5090/I1/train.log) |
| 2026-03-31 04:46:04 EDT | `I2` | `train_hconv.py` | T2 + squared gate | 1.3930 | 1.39314138 | 14897734 | [train.log](../runs_hconv_quality_5090/I2/train.log) |
| 2026-03-31 04:54:17 EDT | `I4` | `train_hconv.py` | T2 + dilated conv + squared gate | 1.4023 | 1.40250911 | 14905386 | [train.log](../runs_hconv_quality_5090/I4/train.log) |
| 2026-03-31 05:01:22 EDT | `I4H` | `train_hconv.py` | T2 + dilated conv + squared gate + hippo init | 1.4313 | 1.43114624 | 14896389 | [train.log](../runs_hconv_quality_5090/I4H/train.log) |

Current read:

- `GPT_12L` is the critical size-matched gate, and it beats both `T2` and `T3` on final scheduled `val_bpb`: `1.3605` vs `1.3693`.
- The size gap is effectively closed for that gate. `GPT_12L` finishes at `15,025,592` total submission bytes versus `14,962,841` for `T2`, a difference of only `62,751` bytes.
- Relative to `T2`, the size-matched GPT improves the final scheduled `val_bpb` by about `0.0088`, which is enough to fail the hybrid continuation gate from the current experiment plan.
- Relative to `GPT_REF`, the size-matched GPT improves the final scheduled `val_bpb` by about `0.0169`, so extra bytes are buying quality in the all-attention family without needing the hybrid blocks.
- Under the current decision rule, the hybrid path is not earning its bytes; `T2_eager`, `T5`, and `T6` are therefore tabled rather than promoted to official follow-up runs.
- `GPT_REF` currently edges `B1` by `0.0008` bpb on the cleaned official protocol.
- `B1` is still under the 16,000,000-byte compressed-model limit with `15,423,044` compressed model bytes.
- `C2` is much worse than `B1` here by `0.1999` bpb, so the pure-conv variant does not look competitive on this protocol.
- Phase 2 reruns under the cleaned cadence ended in a tie to four decimals on final `val_bpb`: both `T2` and `T3` finished at `1.3693`.
- `T2` is still smaller than `T3` by `392,016` compressed bytes, while both remain under the 16 MB cap.
- `T3` holds the tiny roundtrip edge after export and reload: `1.36937905` vs `1.36955833`.
- Relative to `GPT_REF`, both tied-depth variants improve the final scheduled `val_bpb` by about `0.0081`.
- Relative to `B1`, both tied-depth variants improve the final scheduled `val_bpb` by about `0.0089`.
- `I1` regresses against `T2` by about `0.0033` bpb, so the dilated variant is not helping so far.
- `I2` regresses much harder, by about `0.0237` bpb versus `T2`, so squared gating looks actively harmful on this setup.
- `I4` is even worse than `I2`, landing about `0.0330` bpb behind `T2`.
- `I4H` is the worst of the tested innovation set, about `0.0620` bpb behind `T2`.
- None of the phase-3 innovations beat the plain tied-depth base; the best quality remains the `T2` / `T3` pair.

## Smoke / Bring-Up Runs

These are sanity checks, not comparable quality runs.

| Timestamp | Config | Purpose | Contract | Final val_bpb | Roundtrip val_bpb | int8+zlib_bytes | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-03-31 03:48:49 EDT | `SMOKE_HCONV` | Compile-enabled sweep-harness smoke | `10` steps, `32768` train tokens/step, no periodic eval, final eval only | 3.7165 | 3.71649312 | 15592083 | [train.log](../runs_hconv_quality_5090/SMOKE_HCONV/train.log) |

Notes:

- The canonical smoke row above is kept only as a local bring-up record; smoke runs are no longer kept in the official W&B project.
- It remains useful for bring-up and compile-behavior checks, but not for architecture quality comparison.
