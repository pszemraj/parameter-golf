# 5090 Temporal Branch Probe

## Purpose

Test one bounded in-family extension:
- keep the frozen `blocks3` structure fixed
- keep the `resid4_e25_c8t1` controller fixed
- keep schedule, token budget, seed, data path, tokenizer path, and logging contract fixed
- change only `branch_temporal_mode`

This is explicitly not a transformer move. The probe changes how the frozen side reads recurrent state:
- `current`: each frozen branch projects the current controller output
- `lagged`: each frozen branch projects the controller output from `lag_n` steps ago

## Protocol

- Structure: `num_blocks=3`, `branch_lags=1,2,3,4,6,8,12,16,24,32,48,64`, full readout
- Controller: `core_layers=4`, `core_expansion=2.5`, residual minGRU, `carry_chunks=8`, `bptt_chunks=1`
- Schedule: `lr=3e-3`, `warmup=100`, `hold=1500`, `min_lr=3e-4`, `weight_decay=1e-3`
- Budget: `4096` steps = `536,870,912` train tokens
- Runtime: `1x RTX 5090`, `TORCH_BLAS_PREFER_CUBLASLT=1`, compile off, TF32 on
- Status: screening, 1 seed

Exact launch commands are recorded in [commands.txt](/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_temporal/commands.txt).

## Results

| Mode | Best val_bpb | Best val_loss | Steady tok/s | Elapsed sec | Peak alloc MiB | W&B |
|---|---:|---:|---:|---:|---:|---|
| `current` | `2.4138021379` | `4.0952300429` | `1,904,855` | `281.95` | `7525.17` | [sbtohfwt](https://wandb.ai/pszemraj/pg-core-amp/runs/sbtohfwt) |
| `lagged` | `2.4529551592` | `4.1616566181` | `1,872,626` | `286.76` | `7528.12` | [zd9hhq5b](https://wandb.ai/pszemraj/pg-core-amp/runs/zd9hhq5b) |

Direct delta, `lagged - current`:
- `+0.0391530213` worse `val_bpb`
- `+0.0664265752` worse `val_loss`
- `-32,229` tok/s, about `1.69%` slower
- `+4.81s` wallclock

## Interpretation

The pure explicit-lag replacement is a clear loss locally.

What this means:
- The frozen amplifier is currently more useful when it sees the current recurrent state and lets the lag operators act as fixed projections of that state.
- Replacing that with stale controller snapshots hurts both quality and speed.
- The result is large enough on this `512M`-token screen that `lagged` should not become the new default or immediate H100 candidate.

What this does not mean:
- It does not mean the frozen side should stay exactly as-is forever.
- It does not mean multi-timescale temporal taps are a bad idea in general.
- It does mean that pure substitution of current-state branch projections with explicit delayed taps is not the right next move.

## Transformer-Regression Guardrail

This probe stayed inside the intended family:
- no attention
- no token-token mixing
- no transformer block reintroduced
- the trainable part remained the same parallel minGRU controller

The negative result is still useful. It says that "make it more explicit-temporal" is not automatically an improvement, and it avoids the bad reflex of compensating by just adding more generic trainable depth.

## Next Candidate Variants

If we expand this axis further, the cleaner next probes are:
- a hybrid `current + lagged` frozen branch view rather than pure replacement
- a frozen multiscale filter bank that preserves strong current-state access
- a light EMA-style temporal view only if it can be implemented without chunk-local approximation

Current recommendation after this probe:
- keep `branch_temporal_mode=current` as the default
- do not spend H100 confirmation budget on pure `lagged` mode
