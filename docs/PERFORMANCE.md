# Performance

Last updated: 2026-03-31 04:46 EDT

This file tracks local training-speed measurements from runs in [`runs_hconv_quality_5090/`](../runs_hconv_quality_5090/).

## How Speed Is Calculated

- `step_avg_ms` comes from the final step log line in each run's `train.log`.
- `steps/s = 1000 / step_avg_ms`.
- `tok/s = TRAIN_BATCH_TOKENS * steps/s`.
- For the quality runs below, `warmup_steps=20`, so the reported `step_avg_ms` excludes compile/warmup time and is appropriate for comparing steady-state training speed.
- The current quality harness now skips random-init validation and schedules sampled eval at `step 100`, then every `250`, plus the forced final eval.
- For `SMOKE_HCONV_COMPILE`, `warmup_steps=0`, so the reported average includes the one-time cold compile cost and should be read as a bring-up measurement, not a steady-state throughput number.
- Important discrepancy:
  - The current trainer path uses `grad_accum_steps=8` on `WORLD_SIZE=1`.
  - Current `AGENTS.md` guidance for 1x5090 quality comparisons says `GRAD_ACCUM_STEPS=64`, but that guidance came from an earlier parameter-sharing setup that could use more memory than the current hconv family.
  - So the quality-run speed numbers below are valid for this current harness, with the accumulation mismatch noted as context rather than treated as an automatic protocol failure.

## Quality-Comparison Speed

| Timestamp | Config | Train batch tokens | Final step_avg_ms | steps/s | tok/s | Peak alloc MiB | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-03-31 04:20:52 EDT | `GPT_REF` | 262144 | 353.91 | 2.8256 | 740708 | 5284 | [train.log](../runs_hconv_quality_5090/GPT_REF/train.log) |
| 2026-03-31 04:24:45 EDT | `B1` | 262144 | 307.84 | 3.2484 | 851559 | 4901 | [train.log](../runs_hconv_quality_5090/B1/train.log) |
| 2026-03-31 04:28:30 EDT | `C2` | 262144 | 298.75 | 3.3473 | 877469 | 4715 | [train.log](../runs_hconv_quality_5090/C2/train.log) |
| 2026-03-31 04:05:45 EDT | `T2` | 262144 | 537.99 | 1.8588 | 487266 | 8189 | [train.log](../runs_hconv_quality_5090/T2/train.log) |
| 2026-03-31 04:12:52 EDT | `T3` | 262144 | 518.02 | 1.9304 | 506050 | 7941 | [train.log](../runs_hconv_quality_5090/T3/train.log) |
| 2026-03-31 04:38:41 EDT | `I1` | 262144 | 539.22 | 1.8545 | 486154 | 8223 | [train.log](../runs_hconv_quality_5090/I1/train.log) |
| 2026-03-31 04:46:04 EDT | `I2` | 262144 | 513.64 | 1.9469 | 510365 | 8189 | [train.log](../runs_hconv_quality_5090/I2/train.log) |

Current read:

- `B1` is about `1.1495x` faster than `GPT_REF` by both `steps/s` and `tok/s` under the same fixed-token contract.
- `B1` also used less peak allocated memory in this local run (`4901 MiB` vs `5284 MiB`).
- `C2` is faster again than `B1` locally, but the quality hit is severe enough that this is not a useful trade on the current comparison protocol.
- `T3` is about `1.0385x` faster than `T2` on both `steps/s` and `tok/s`, and it used slightly less peak allocated memory (`7941 MiB` vs `8189 MiB`).
- The speed trade for phase 2 is now cleaner than before: `T3` is the faster tied-depth variant, while `T2` keeps the compressed-size edge.
- `I1` is essentially the same speed as `T2` and uses slightly more memory, so its quality regression is not buying meaningful throughput.
- `I2` is about `1.0474x` faster than `T2`, but that speed gain comes with a much larger quality hit.

## Smoke / Compile Behavior

| Timestamp | Config | Train batch tokens | Final step_avg_ms | steps/s | tok/s | Notes | Log |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2026-03-31 03:48:49 EDT | `SMOKE_HCONV` | 32768 | 1054.45 | 0.9484 | 31076 | Compile-enabled harness smoke; includes cold compile overhead because `warmup_steps=0`; first step was `9314.41 ms` | [train.log](../runs_hconv_quality_5090/SMOKE_HCONV/train.log) |

Takeaway:

- Compile can be expensive up front on this machine, but the warmed quality runs are the numbers that matter for architecture speed comparison.
- The canonical smoke run is still useful to remember that a short no-warmup benchmark can make the compile path look much worse than the actual steady-state quality protocol.
