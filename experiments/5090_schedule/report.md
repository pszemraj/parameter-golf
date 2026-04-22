# 5090 Schedule Report

## Status

- The original `blocks0` hold sweep, edge follow-up, and `1B` transfer confirmation are complete.
- The tuned-hold `blocks1` family is complete at `1B`.
- The wide `blocks0` / `blocks1` / `blocks2` multi-seed confirmation is now complete.
- Working schedule defaults are still:
  - `lr_hold_steps=3500` for the `4096`-step / `512M` screening contract
  - `lr_hold_steps=7000` for the `8192`-step / `1B` contract
- Those defaults are now proven for the old `blocks0` lane, but they are not yet fully revalidated on the true post-confirmation frontier.

## Training Budget Policy

- Full-dataset frozen spec/statistics build is mandatory.
- `512M` is the default schedule-screening budget.
- `1B` is the confirmation budget for the winning schedule points.
- Anything shorter than `512M` is only for smoke tests or obvious harness checks.

This is the current best speed/rigor compromise:

- `512M` was predictive enough to move the hold default in the earlier `blocks0` lane.
- `1B` confirmation then proved that the gain was real.
- Close post-confirmation rankings are now small enough that the final claims still need the `1B` pass.

## What Was Verified

- Warmup-hold-cosine is the real root schedule path.
- `lr_hold_steps` is wired end to end.
- The trainer records structured train/eval metrics and final run results, so schedule comparisons do not depend on ad hoc log parsing.
- The sweep harness supports a fixed effective-step-token contract, so local batch changes can preserve apples-to-apples optimizer-step semantics with derived `grad_accum`.

## Blocks0 Hold Sweep Result

There is strong evidence that the inherited `1500`-step hold was too short for the radical `blocks0` controllers on the `512M` screen.

| Controller | `h0` | `h500` | `h1500` | `h2500` | `h3500` | `h4096` | Best tested |
|---|---:|---:|---:|---:|---:|---:|---:|
| `blocks0 12x10` | `2.2949891937` | `2.2876574385` | `2.2777913795` | `2.2696659544` | `2.2690508796` | `2.2830053665` | `h3500` |
| `blocks0 10x12` | `2.3017589365` | `2.2910034081` | `2.2794286891` | `2.2715466346` | `2.2669840064` | `2.2834303277` | `h3500` |

Interpretation:

- hold-then-cosine is real signal for this family
- the schedule wants a very late tail, not no decay
- `h4096` was clearly worse, so “never decay” is not the answer here

## 1B Hold Transfer Result

The proportional `h3500 -> h7000` transfer held up on the `1B` budget for the `blocks0` finalists.

| Controller | inherited `h1500` @ `1B` | tuned `h7000` @ `1B` | Delta |
|---|---:|---:|---:|
| `blocks0 12x10` | `2.2113941366` | `2.1954688682` | `-0.0159252684` |
| `blocks0 10x12` | `2.2128156660` | `2.1878016930` | `-0.0250139730` |

This was genuine optimization gain. Throughput stayed effectively unchanged.

## Wide Multi-Seed Confirmation Result

The finished three-seed batch changed the interpretation of the schedule frontier.

| Run | Mean `val_bpb` | Std | Mean steady tok/s | Mean artifact bytes |
|---|---:|---:|---:|---:|
| `blocks1_resid10_e12_h7000_1b` | `2.1865341393` | `0.0052472581` | `372,888` | `4,790,559` |
| `blocks1_resid12_e10_h7000_1b` | `2.1866023565` | `0.0051325073` | `374,271` | `4,791,951` |
| `blocks0_resid12_e10_h7000_1b` | `2.1899359311` | `0.0039254056` | `386,331` | `3,945,168` |
| `blocks0_resid10_e12_h7000_1b` | `2.1947452986` | `0.0061822180` | `384,669` | `3,943,662` |
| `blocks2_resid12_e8_h7000_1b` | `2.2005760974` | `0.0018741094` | `442,062` | `5,326,970` |

Key read:

- inside `blocks1`, `10x12` and `12x10` are effectively tied:
  - mean delta `= -0.0000682172` bpb for `10x12 - 12x10`
- inside `blocks0`, `12x10` is now the better mean representative:
  - mean delta `= -0.0048093676` bpb for `12x10 - 10x12`
- `blocks2 12x8` is still worse on quality, but:
  - fastest of the structural controls
  - lowest seed variance in the batch

Implication:

- the original “`blocks1 12x10` is the clear winner” single-seed story was too strong
- the original “`blocks0 10x12` is the lean default” story is now also stale
- schedule is still the next highest-value lever, because the remaining architecture gaps are small relative to the gains already observed from hold tuning

## Next Schedule Batch

The right immediate next move is a hold-transfer retune screen on the true post-confirmation representatives.

Use:

- `blocks1_resid10_e12`
- `blocks0_resid12_e10`
- `blocks2_resid12_e8`

Contract:

- `512M`
- `seq_len=512`
- `TARGET_EFFECTIVE_STEP_TOKENS=131072`
- `carry_chunks=8`
- `bptt_chunks=1`
- `max_lr=3e-3`
- `min_lr=3e-4`
- `warmup_steps=100`
- `weight_decay=1e-3`
- `COMPILE=0`
- `TORCH_BLAS_PREFER_CUBLASLT=1`
- seeds `1337`, `2027`
- `lr_hold_steps in {2500, 3500, 4096}`

This batch should be interpreted as screening only.
Only the winning hold settings should be promoted to the `1B` contract for stronger claims.

Convenience launcher:

```bash
bash scripts/run_5090_schedule_retune.sh
```

Useful overrides:

```bash
SEEDS="1337 2027" bash scripts/run_5090_schedule_retune.sh
HOLDS="2500 3500 4096" bash scripts/run_5090_schedule_retune.sh
RUN_BLOCKS2=0 bash scripts/run_5090_schedule_retune.sh
DRY_RUN=1 bash scripts/run_5090_schedule_retune.sh
```

## Open Schedule Questions

- Does the inherited `h3500 / h7000` transfer still hold on `blocks1 10x12`, or was that mostly a `blocks0` lane optimum?
- Does `blocks2 12x8` want a different hold because its frozen side is deeper and its controller is smaller?
- After hold is rechecked, is `max_lr=3e-3` still the right peak LR for the one-block winner?
- Once LR is retuned, does one of the tied `blocks1` geometries separate cleanly, or do they remain interchangeable?
- After the schedule lane is settled, does longer context help the tuned one-block family more than the lean zero-block control?
