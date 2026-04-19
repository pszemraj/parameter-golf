# 5090 Controller Report

## Status
- Completed the first matched-token controller baseline on the original full frozen structure.
- Completed a clean controller follow-up on the new structural front-runner, `blocks3`.
- All completed controller runs in this report used:
  - `compile=0`
  - exact `val_bpb`
  - matched `planned_train_tokens=50,331,648`
  - W&B project `pg-core-amp`

## Current Evidence

### Baseline on full `blocks9` structure

Frozen structure:
- `12` branches
- `9` blocks
- full readout
- `spec_max_tokens=5,000,000`

Matched run results:
- `plain3_e20`
  - `core_layers=3`
  - `core_expansion=2.0`
  - `residual_core=0`
  - `carry_chunks=8`
  - `bptt_chunks=1`
  - `384` steps
  - final `val_bpb = 2.5171747775`
  - steady `tok/s = 1,042,359`
  - peak allocated memory `= 11,576 MiB`
- `resid5_e20`
  - `core_layers=5`
  - `core_expansion=2.0`
  - `residual_core=1`
  - `carry_chunks=16`
  - `bptt_chunks=2`
  - `192` steps
  - final `val_bpb = 2.5178967561`
  - steady `tok/s = 966,227`
  - peak allocated memory `= 24,036 MiB`

Result on `blocks9`:
- `plain3_e20` beat `resid5_e20` by about `0.00072` bpb.
- It was also faster and much lighter on memory.

### Follow-up on `blocks3` structure

Frozen structure:
- `12` branches
- `3` blocks
- full readout
- `spec_max_tokens=5,000,000`

Matched run results:
- `plain3_e20`
  - final `val_bpb = 2.5158438607`
  - steady `tok/s = 2,191,340`
  - peak allocated memory `= 6,381 MiB`
- `resid5_e20`
  - final `val_bpb = 2.5175855416`
  - steady `tok/s = 1,857,335`
  - peak allocated memory `= 13,654 MiB`

Result on `blocks3`:
- `plain3_e20` beat `resid5_e20` by about `0.00174` bpb.
- It ran about `18%` faster.
- It used about `2.14x` less peak allocated memory.

## What This Means

Does more controller depth help?
- Not in the completed local screens.
- The deeper residual controller lost on both the full structure and the `blocks3` structure.

Does more controller width via expansion help?
- Not answered yet in the cleaned `blocks3` regime.
- The current evidence is strong enough that wider residual variants should not be promoted by default.

Is residualization materially improving trainability?
- Not on the current local budget.
- The residual path is stable, but the extra controller depth is not translating into better validation loss.

Is semi-TBPTT helping beyond simple carry?
- The current evidence says “not enough to justify the extra controller” in the tested `resid5_e20` setup.
- That does not prove `bptt_chunks=2` is useless in general, only that the current residual controller package is not winning.

## Best Controller-Only Contender

Current best controller-only contender:
- `plain3_e20` on the `blocks3` structure
- final `val_bpb = 2.5158438607`
- steady `tok/s = 2,191,340`
- artifact estimate `= 4,195,301`

This is now the strongest local contender seen in the completed runs on disk.

## Regression-To-Transformer Guardrail

The best local controller result is not coming from “more controller stack.”
- The winning controller is still a small recurrent policy.
- The winning frozen structure is smaller than the original one.
- That is exactly the opposite of regressing toward a deeper transformer-like default.

The remaining risk is different:
- the frozen amplifier side may still be underpowered or misused
- the controller may be doing most of the real work

That is a better failure mode for this project than accidentally rebuilding a transformer, because it points toward making the frozen temporal side more meaningful rather than just adding generic depth.

## Exact Commands

Full-structure baseline:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/baseline_ab_real \
COMPILE=0 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
RUN_SPECS=$'plain3_e20 3 2.0 8 1 0 -2.0 0.003 1500 0.0003 384 256 512\n\
resid5_e20 5 2.0 16 2 1 -2.0 0.003 1500 0.0003 192 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

`blocks3` controller follow-up:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks3_followup_clean \
NUM_BLOCKS=3 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
TRAIN_FRAC=0.98 \
RUN_SPECS=$'plain3_e20 3 2.0 8 1 0 -2.0 0.003 1500 0.0003 384 256 512\n\
resid5_e20 5 2.0 16 2 1 -2.0 0.003 1500 0.0003 192 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

## Immediate Next Step

The next controller sweep should be narrow and structured around the new winner, not a blind grid.
- Keep the `blocks3` structure fixed.
- Keep `plain3_e20` as the anchor.
- Add only a few nearby controller points:
  - `plain4_e20`
  - `plain3_e25`
  - `resid4_e20`
  - `resid4_e25`

That gives a real answer about whether any extra controller capacity helps once the frozen side is no longer overbuilt, without drifting into “just make the controller deeper.”
