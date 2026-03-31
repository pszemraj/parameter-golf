# Performance

Last updated: 2026-03-31 03:00 EDT

This file tracks local training-speed measurements from runs in [`runs_hconv_quality_5090/`](../runs_hconv_quality_5090/).

## How Speed Is Calculated

- `step_avg_ms` comes from the final step log line in each run's `train.log`.
- `steps/s = 1000 / step_avg_ms`.
- `tok/s = TRAIN_BATCH_TOKENS * steps/s`.
- For the quality runs below, `warmup_steps=20`, so the reported `step_avg_ms` excludes compile/warmup time and is appropriate for comparing steady-state training speed.
- For `SMOKE_HCONV_COMPILE`, `warmup_steps=0`, so the reported average includes the one-time cold compile cost and should be read as a bring-up measurement, not a steady-state throughput number.

## Quality-Comparison Speed

| Timestamp | Config | Train batch tokens | Final step_avg_ms | steps/s | tok/s | Peak alloc MiB | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-03-31 02:47:40 EDT | `GPT_REF` | 262144 | 367.98 | 2.7175 | 712387 | 5284 | [train.log](../runs_hconv_quality_5090/GPT_REF/train.log) |
| 2026-03-31 02:52:13 EDT | `B1` | 262144 | 305.21 | 3.2764 | 858897 | 4901 | [train.log](../runs_hconv_quality_5090/B1/train.log) |
| 2026-03-31 03:00:05 EDT | `C2` | 262144 | 281.35 | 3.5543 | 931736 | 4715 | [train.log](../runs_hconv_quality_5090/C2/train.log) |

Current read:

- `B1` is about `1.206x` faster than `GPT_REF` by both `steps/s` and `tok/s` under the same fixed-token contract.
- `B1` also used less peak allocated memory in this local run (`4901 MiB` vs `5284 MiB`).
- `C2` is faster again than `B1` locally, but the quality hit is severe enough that this is not a useful trade on the current comparison protocol.

## Smoke / Compile Behavior

| Timestamp | Config | Train batch tokens | Final step_avg_ms | steps/s | tok/s | Notes | Log |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2026-03-31 02:36:03 EDT | `SMOKE_HCONV` | 32768 | 187.83 | 5.3240 | 174456 | No-compile smoke; useful as a bring-up baseline only | [train.log](../runs_hconv_quality_5090/SMOKE_HCONV/train.log) |
| 2026-03-31 02:40:02 EDT | `SMOKE_HCONV_COMPILE` | 32768 | 1038.45 | 0.9630 | 31555 | Includes cold compile overhead because `warmup_steps=0`; first step was `9329.90 ms` | [train.log](../runs_hconv_quality_5090/SMOKE_HCONV_COMPILE/train.log) |

Takeaway:

- Compile can be expensive up front on this machine, but the warmed quality runs are the numbers that matter for architecture speed comparison.
- The smoke compile run is useful to remember that a short no-warmup benchmark can make the compile path look much worse than the actual steady-state quality protocol.
