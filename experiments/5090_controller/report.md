# 5090 Controller Report

## Status

- The corrected full-spec controller replay is complete.
- The `blocks0` and `blocks1` `1B` confirmation queue is mostly complete:
  - `12x6`, `12x10`, `10x12`, and `blocks1 12x6` are finished
  - only checkpointed `blocks0 16x8` remains in flight
- All evidence below uses:
  - the uncapped full local shard set for frozen-spec construction
  - exact `val_bpb`
  - W&B project `pg-core-amp`
  - `COMPILE=0`
  - TF32 enabled
  - `TORCH_BLAS_PREFER_CUBLASLT=1`
- The current local question is no longer whether the old moderate `blocks3` controller wins. It does not.
- The strongest completed local points are now radical parallel-minGRU controllers on smaller frozen specs, especially `blocks0`.

Reproduction artifacts live in:

- [experiments/5090_controller/fullspec_blocks0_controller_v2/summary.tsv](/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_controller/fullspec_blocks0_controller_v2/summary.tsv)
- [experiments/5090_controller/fullspec_blocks0_controller_v3/summary.tsv](/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_controller/fullspec_blocks0_controller_v3/summary.tsv)
- [experiments/5090_controller/fullspec_blocks0_controller_v6/summary.tsv](/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_controller/fullspec_blocks0_controller_v6/summary.tsv)
- [experiments/5090_controller/fullspec_blocks1_radical_v1/summary.tsv](/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_controller/fullspec_blocks1_radical_v1/summary.tsv)
- [experiments/5090_controller/fullspec_blocks2_confirm1b_v1/summary.tsv](/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_controller/fullspec_blocks2_confirm1b_v1/summary.tsv)
- [experiments/5090_controller/fullspec_blocks3_confirm1b_v1/summary.tsv](/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_controller/fullspec_blocks3_confirm1b_v1/summary.tsv)

## Current Frontier

Artifact estimates below use the corrected record-style accounting:

- repo code bytes
- `gzip(spec.pt)`
- trainable int8-zlib payload

| Run | Frozen structure | Trainable params | Planned tokens | Best val_bpb | Steady tok/s | Artifact estimate |
|---|---|---:|---:|---:|---:|---:|
| `blocks0_resid12_e10_c8t1_r3_current_1b` | `blocks0` | `839,129` | `1,073,741,824` | `2.2113941366` | `386,200` | `3,879,987` |
| `blocks0_resid10_e12_c8t1_r3_current_1b` | `blocks0` | `839,031` | `1,073,741,824` | `2.2128156660` | `384,622` | `3,878,440` |
| `blocks1_resid12_e6_c8t1_r3_current_1b` | `blocks1` | `505,049` | `1,073,741,824` | `2.2356768287` | `590,071` | `4,102,717` |
| `blocks0_resid12_e6_c8t1_r3_current_1b` | `blocks0` | `505,049` | `1,073,741,824` | `2.2363421409` | `618,833` | `3,255,888` |
| `blocks0_resid16_e8_c8t1_r3_current_512m_gc1` | `blocks0` | `895,005` | `536,870,912` | `2.2815471392` | `273,637` | `3,962,318` |
| `blocks0_resid12_e8_c8t1_r3_current_512m` | `blocks0` | `672,089` | `536,870,912` | `2.2859021694` | `474,391` | `3,544,631` |
| `blocks0_resid12_e6_c8t1_r3_current_512m` | `blocks0` | `505,049` | `536,870,912` | `2.2979334823` | `616,452` | `3,231,686` |
| `blocks1_resid12_e6_c8t1_r3_current_512m` | `blocks1` | `505,049` | `536,870,912` | `2.2983212585` | `585,746` | `4,093,534` |
| `blocks2_resid6_e25_c8t1_1b` | `blocks2` | smaller residual controller | `1,073,741,824` | `2.3500271326` | `1,858,685` | `4,196,469` |
| `resid4_e25_c8t1_1b` | `blocks3` | smaller residual controller | `1,073,741,824` | `2.3436810117` | `1,913,826` | `4,977,266` |

Current read on that table:

- `blocks0_resid12_e10_c8t1_r3_current_1b` is now the best completed local pure-quality point.
- `blocks0_resid10_e12_c8t1_r3_current_1b` stayed extremely close on the longer budget:
  - it trails `12 x 10.0` by only about `0.00142` bpb
  - it matches controller mass almost exactly
  - that means controller geometry is still real signal, not screening noise
- `blocks1_resid12_e6_c8t1_r3_current_1b` slightly beat `blocks0_resid12_e6_c8t1_r3_current_1b` on the longer budget:
  - `2.2356768287` vs `2.2363421409`
  - that is only about `0.00067` bpb, but it is enough to keep a one-block frozen structure very much alive as a transfer guardrail
- `blocks0_resid16_e8_c8t1_r3_current_512m_gc1` proved larger checkpointed controllers are viable, but it is not the preferred local frontier point yet:
  - slower by about `29%` vs `12 x 10.0`
  - slightly worse by about `0.00376` bpb
- the still-running checkpointed `16x8` `1B` confirmation is promising enough to watch, but not to elevate yet:
  - midpoint `4096 -> 2.3055954707`
  - next eval `4608 -> 2.2713556688`
  - that is real progress, but not enough evidence yet to displace the completed `12x10` / `10x12` pair

## Corrected `blocks3` Evidence

The corrected `blocks3` screens matter mostly as a filter on what not to over-interpret.

### Plain vs residual follow-up

| Run | Best val_bpb | Steady tok/s | Peak alloc MiB |
|---|---:|---:|---:|
| `plain3_e20` | `2.4833587679` | `2,184,121` | `6,381` |
| `resid5_e20` | `2.4865782548` | `1,837,046` | `13,654` |

Result:

- On corrected `blocks3`, `plain3_e20` beat `resid5_e20` by about `0.00322` bpb.
- The earlier capped-spec claim that the residual default clearly won on `blocks3` does not survive.

### Corrected `blocks3` controller neighborhood

| Run | Best val_bpb | Steady tok/s |
|---|---:|---:|
| `plain3_e25_c8t1` | `2.4820524220` | `1,972,792` |
| `resid4_e25_c8t1` | `2.4824969375` | `1,813,716` |
| `resid4_e20_c8t1` | `2.4828974740` | `1,960,262` |
| `plain4_e20_c8t1` | `2.4838029669` | `1,941,708` |

Result:

- `plain3_e25_c8t1` is the corrected single-seed `blocks3` leader.
- Residualization is not a blanket winner on `blocks3`.
- Moderate controller tuning on `blocks3` is now clearly a secondary frontier behind the radical `blocks0/blocks1` family.

### Corrected `bptt` sweep

Plain family:

- `plain4_e20_c8t1 = 2.4838029669`
- `plain4_e20_c8t2 = 2.4857629362`
- `plain4_e20_c8t4 = 2.4881083439`

Residual family:

- `resid4_e25_c8t1 = 2.4824969375`
- `resid4_e25_c8t2 = 2.4867395116`
- `resid4_e25_c8t4 = 2.4870242558`

Result:

- `bptt=1` won clearly in both families.
- Higher `bptt` increased memory sharply without buying quality.
- Semi-TBPTT is not the local win condition here.

### Corrected `carry` sweep

Residual `4 x 2.0`:

- `resid4_e20_c8t1 = 2.4828974740`
- `resid4_e20_c16t1 = 2.4834046464`
- `resid4_e20_c32t1 = 2.4831431581`

Residual `4 x 2.5`:

- `resid4_e25_c8t1 = 2.4824969375`
- `resid4_e25_c16t1 = 2.4828899493`
- `resid4_e25_c32t1 = 2.4834898042`

Result:

- `carry=8` is the corrected winner in both residual families.
- The earlier capped-spec suggestion that `carry=16` was helping does not hold up.

## Temporal Probe Result

The bounded temporal-mode probe finished and stayed inside the same family:

- same `blocks3` frozen structure
- same `resid4_e25_c8t1` controller
- only `branch_temporal_mode` changed

Corrected results at `512M` tokens:

| Mode | Best val_bpb | Delta vs `current` | Steady tok/s |
|---|---:|---:|---:|
| `current` | `2.3723031488` | `0.0000000000` | `1,902,301` |
| `hybrid` | `2.3972226129` | `+0.0249194641` | `1,807,879` |
| `lagged` | `2.4251241108` | `+0.0528209620` | `1,858,847` |

Result:

- `current` won decisively.
- `hybrid` recovered some quality relative to `lagged`, but still lost badly to `current`.
- Pure explicit delayed taps are not the right next move on current evidence.

This is important for the anti-transformer guardrail:

- the negative result does not imply attention is needed
- it implies the current frozen temporal role is weakly designed, and naive lag substitution is not enough

## What The Evidence Says

Does more controller depth help?

- Yes, strongly, once the frozen side is shrunk enough.
- The biggest gains came from controller-up/spec-down reallocations, not from more frozen amplifier depth.

Does more controller width via expansion help?

- Sometimes, but not as a generic knob.
- The near-tie between `12 x 10.0` and `10 x 12.0` says geometry matters.
- The smaller `blocks2` frontier also preferred added depth over a pure width increase.

Is residualization materially improving trainability?

- Yes for the radical deeper controllers.
- No as a blanket rule on the smaller corrected `blocks3` screen.
- Residualization is now a controlled tool, not a universal default.

Is semi-TBPTT helping beyond simple carry?

- No.
- `bptt=1` and `carry=8` remain the working horizon default.

Is the learned amplifier stack earning its bytes?

- Not at the old `blocks9` size. That structure already lost the corrected structure screen.
- `blocks1` remains worth keeping alive as a guardrail contender.
- The present best quality points are still `blocks0`, so extra frozen depth is guilty until proven useful.

## Best Current Calls

Best pure-quality contender:

- `blocks0_resid12_e10_c8t1_r3_current_1b`

Best geometry control / near-tie:

- `blocks0_resid10_e12_c8t1_r3_current_1b`

Best quality-speed tradeoff:

- `blocks0_resid12_e6_c8t1_r3_current_1b`

Best nonzero-amplifier contender:

- `blocks1_resid12_e6_c8t1_r3_current_1b`

Best larger-controller stress point:

- `blocks0_resid16_e8_c8t1_r3_current_512m_gc1`

## Regression-To-Transformer Guardrail

The project is still inside the intended family.

- The winners are parallel minGRU controllers.
- There is still no attention and no token-token mixing.
- The best structural move so far was removing frozen amplifier depth, not reintroducing transformer-style trainable depth.

The real current risk is different:

- the controller can become too dominant if the frozen side stays too static
- that is a recurrent-model design problem, not a transformer regression
- the right response is to make the frozen side earn its keep experimentally, not to backslide into attention

## Next Step

The next rigorous checkpoint is no longer "start confirmations." It is:

- finish the checkpointed `blocks0_resid16_e8_c8t1_r3_current_1b_gc1` stress point
- then run schedule sweeps on the completed `1B` leaders
- keep `blocks1 12x6` in that schedule set as the live nonzero-amplifier guardrail
