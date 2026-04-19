# 5090 Temporal Branch Probe

## Purpose

Test one bounded in-family temporal variation without drifting toward a transformer:

- keep the frozen `blocks3` structure fixed
- keep the `resid4_e25_c8t1` controller fixed
- keep token budget, seed, logging, optimizer, and runtime stack fixed
- change only `branch_temporal_mode`

Modes:

- `current`: frozen branches see the current controller output
- `lagged`: frozen branches see delayed controller outputs at their branch lags
- `hybrid`: frozen branches see a fixed blend of current and lagged controller outputs

## Protocol

- Structure: `num_blocks=3`, `branch_lags=1,2,3,4,6,8,12,16,24,32,48,64`, full readout
- Controller: residual minGRU, `core_layers=4`, `core_expansion=2.5`, `carry_chunks=8`, `bptt_chunks=1`
- Schedule: `lr=3e-3`, `warmup=100`, `hold=1500`, `min_lr=3e-4`, `weight_decay=1e-3`
- Budget: `4096` steps = `536,870,912` train tokens
- Runtime: `1x RTX 5090`, `TORCH_BLAS_PREFER_CUBLASLT=1`, compile off, TF32 on
- Status: screening, 1 seed

Commands are recorded in:

- [experiments/5090_temporal/fullspec_blocks3_resid4e25_current_v1/commands.txt](/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_temporal/fullspec_blocks3_resid4e25_current_v1/commands.txt)
- [experiments/5090_temporal/commands.txt](/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_temporal/commands.txt)

## Corrected Results

| Mode | Best val_bpb | Delta vs `current` | Steady tok/s | Artifact estimate |
|---|---:|---:|---:|---:|
| `current` | `2.3723031488` | `0.0000000000` | `1,902,301` | `4,977,244` |
| `hybrid` | `2.3972226129` | `+0.0249194641` | `1,807,879` | `4,977,187` |
| `lagged` | `2.4251241108` | `+0.0528209620` | `1,858,847` | `4,977,333` |

Key deltas:

- `hybrid` recovered about `0.02790` bpb relative to pure `lagged`.
- `hybrid` still stayed about `0.02492` bpb behind `current`.
- Artifact size is effectively unchanged across all three modes.
- `current` was also the fastest completed point here.

## Interpretation

The first explicit-temporal replacement is a clear loser locally.

What this means:

- The frozen side is currently more useful when branches read the current recurrent state.
- Replacing that with stale controller snapshots hurts quality materially.
- A fixed hybrid blend is better than pure lagged substitution, but still not good enough.
- The cleaner current win is coming from controller-up/spec-down reallocation, not from making branch reads more delay-heavy.

What this does not mean:

- It does not mean multi-timescale temporal structure is a dead idea.
- It does not mean the frozen side is already optimal.
- It does mean that naive explicit lag taps are not the next best use of time.

## Transformer-Regression Guardrail

This probe stayed inside the intended family:

- no attention
- no token-token mixing
- same parallel minGRU controller
- only frozen branch readout timing changed

The negative result is useful because it prevents a bad reflex:

- if explicit-lag substitution loses, the answer is not "just add transformer-style machinery"
- the answer is to design a better frozen temporal role while keeping the trainable core recurrent and parallel

## Current Recommendation

- Keep `branch_temporal_mode=current` as the default.
- Do not spend longer confirmation budget on pure `lagged`.
- Do not spend longer confirmation budget on this first `hybrid` either.
- Prioritize controller confirmation and schedule work before revisiting frozen temporal variants.
