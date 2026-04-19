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
- `hybrid`: each frozen branch reads the current controller output plus a fixed lag-dependent blend of the delayed controller output

## Protocol

- Structure: `num_blocks=3`, `branch_lags=1,2,3,4,6,8,12,16,24,32,48,64`, full readout
- Controller: `core_layers=4`, `core_expansion=2.5`, residual minGRU, `carry_chunks=8`, `bptt_chunks=1`
- Schedule: `lr=3e-3`, `warmup=100`, `hold=1500`, `min_lr=3e-4`, `weight_decay=1e-3`
- Budget: `4096` steps = `536,870,912` train tokens
- Runtime: `1x RTX 5090`, `TORCH_BLAS_PREFER_CUBLASLT=1`, compile off, TF32 on
- Status: screening, 1 seed

Exact launch commands are recorded in [commands.txt](/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_temporal/commands.txt).

## Results

Artifact estimates below use the corrected accounting path:
- `gzip(spec.pt)`
- `int8 zlib(trainable controller payload)`
- repo code bytes

| Mode | Best val_bpb | Delta vs `current` | Steady tok/s | Delta tok/s | Artifact estimate | Trainable int8 zlib | W&B |
|---|---:|---:|---:|---:|---:|---:|---|
| `current` | `2.4138021379` | `0.0000000000` | `1,904,855` | `0` | `4,275,890` | `59,612` | [sbtohfwt](https://wandb.ai/pszemraj/pg-core-amp/runs/sbtohfwt) |
| `hybrid` | `2.4268408799` | `+0.0130387420` | `1,811,334` | `-93,521` | `4,275,714` | `59,436` | [2an0bp8w](https://wandb.ai/pszemraj/pg-core-amp/runs/2an0bp8w) |
| `lagged` | `2.4529551592` | `+0.0391530213` | `1,872,626` | `-32,229` | `4,277,292` | `61,014` | [zd9hhq5b](https://wandb.ai/pszemraj/pg-core-amp/runs/zd9hhq5b) |

Key deltas:
- `hybrid` recovered `0.0261142793` bpb relative to pure `lagged`, but it still stayed `0.0130387420` bpb behind `current`.
- `hybrid` was also the slowest point here, about `4.91%` slower than `current`.
- Artifact size is effectively a wash across all three temporal modes; the decision is quality and speed, not bytes.

## Interpretation

The pure explicit-lag replacement is a clear loss locally, and the simple hybrid still does not beat the default `current` view.

What this means:
- The frozen amplifier is currently more useful when it sees the current recurrent state and lets the lag operators act as fixed projections of that state.
- Replacing that with stale controller snapshots hurts both quality and speed.
- Adding stale taps back in through a fixed hybrid blend helps quality relative to pure `lagged`, but not enough to beat the simpler default.
- The result is large enough on this `512M`-token screen that neither `lagged` nor this first `hybrid` should become the new default or immediate H100 candidate.

What this does not mean:
- It does not mean the frozen side should stay exactly as-is forever.
- It does not mean multi-timescale temporal taps are a bad idea in general.
- It does mean that pure substitution of current-state branch projections with explicit delayed taps is not the right next move.
- It also means the cleaner current win is coming from controller-up/spec-down reallocation, not from making the frozen branch reader more lag-heavy.

## Transformer-Regression Guardrail

This probe stayed inside the intended family:
- no attention
- no token-token mixing
- no transformer block reintroduced
- the trainable part remained the same parallel minGRU controller

The negative result is still useful. It says that "make it more explicit-temporal" is not automatically an improvement, and it avoids the bad reflex of compensating by just adding more generic trainable depth.

## Next Candidate Variants

If we expand this axis further, the cleaner next probes are:
- a stronger multiscale filter bank that preserves strong current-state access
- a lag-dependent hybrid with a different fixed blend schedule only if we have evidence that the slower branch path is still worth it
- a light EMA-style temporal view only if it can be implemented without chunk-local approximation
- controller-up/spec-down reallocations, because those are the first probes that are currently beating the prior anchor cleanly

Current recommendation after this probe:
- keep `branch_temporal_mode=current` as the default
- do not spend H100 confirmation budget on pure `lagged` mode
- do not spend H100 confirmation budget on the first fixed `hybrid` mode either
