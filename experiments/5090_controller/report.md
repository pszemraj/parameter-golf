# 5090 Controller Report

## Status
- Completed the first matched-token controller baseline on the original full frozen structure.
- Completed a clean controller follow-up on the new structural front-runner, `blocks3`.
- Completed a four-point controller neighborhood screen on `blocks3`.
- Completed a clean matched-token `bptt` sweep on the two best `blocks3` controller families.
- Completed a clean matched-token `carry` sweep on the two best residual `blocks3` controller families.
- All completed controller runs in this report used:
  - `compile=0`
  - exact `val_bpb`
  - matched `planned_train_tokens=50,331,648`
  - W&B project `pg-core-amp`
- A first attempt at the neighborhood screen with `compile=1` was discarded before completion because the shorter runs would not all cross the compile trigger at the same point. That batch is intentionally excluded from the evidence below.

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

### `blocks3` controller neighborhood screen

All runs below used the same frozen structure:
- `12` branches
- `3` blocks
- full readout
- `spec_max_tokens=5,000,000`
- `carry_chunks=8`
- `bptt_chunks=1`

Matched run results:
- `plain3_e25_c8t1`
  - `core_layers=3`
  - `core_expansion=2.5`
  - `residual_core=0`
  - final `val_bpb = 2.5170427183`
  - steady `tok/s = 2,052,454`
  - peak allocated memory `= 6,697 MiB`
- `plain4_e20_c8t1`
  - `core_layers=4`
  - `core_expansion=2.0`
  - `residual_core=0`
  - final `val_bpb = 2.5154361649`
  - steady `tok/s = 2,030,859`
  - peak allocated memory `= 6,814 MiB`
- `resid4_e20_c8t1`
  - `core_layers=4`
  - `core_expansion=2.0`
  - `residual_core=1`
  - final `val_bpb = 2.5133777174`
  - steady `tok/s = 1,970,720`
  - peak allocated memory `= 7,104 MiB`
- `resid4_e25_c8t1`
  - `core_layers=4`
  - `core_expansion=2.5`
  - `residual_core=1`
  - final `val_bpb = 2.5123951758`
  - steady `tok/s = 1,820,432`
  - peak allocated memory `= 7,525 MiB`

Result from the neighborhood screen:
- `plain4_e20` improved on the previous `plain3_e20` anchor by about `0.00041` bpb.
- `plain3_e25` was a regression relative to both `plain3_e20` and `plain4_e20`.
- `resid4_e20` was the first controller point to clearly beat the plain family on the trimmed `blocks3` structure.
- `resid4_e25` is the current single-seed screening leader, beating `plain4_e20` by about `0.00304` bpb and `resid4_e20` by about `0.00098` bpb.
- Residual gates stayed in a narrow non-saturated range, roughly `0.119 -> 0.141`, so the residual path is active without collapsing.

### `blocks3` `bptt` sweep on the leading families

All runs below used:
- `12` branches
- `3` blocks
- full readout
- `spec_max_tokens=5,000,000`
- `carry_chunks=8`
- explicit per-run warmup/hold scaling so schedule stayed matched in token terms

Plain family:
- `plain4_e20_c8t1`
  - `bptt_chunks=1`
  - `warmup_steps=100`
  - `lr_hold_steps=1500`
  - final `val_bpb = 2.5154361649`
  - steady `tok/s = 2,030,714`
  - peak allocated memory `= 6,814 MiB`
- `plain4_e20_c8t2`
  - `bptt_chunks=2`
  - `warmup_steps=50`
  - `lr_hold_steps=750`
  - final `val_bpb = 2.5199320452`
  - steady `tok/s = 2,063,541`
  - peak allocated memory `= 12,064 MiB`
- `plain4_e20_c8t4`
  - `bptt_chunks=4`
  - `warmup_steps=25`
  - `lr_hold_steps=375`
  - final `val_bpb = 2.5207602945`
  - steady `tok/s = 2,078,943`
  - peak allocated memory `= 22,563 MiB`

Residual family:
- `resid4_e25_c8t1`
  - `bptt_chunks=1`
  - `warmup_steps=100`
  - `lr_hold_steps=1500`
  - final `val_bpb = 2.5123951758`
  - steady `tok/s = 1,820,552`
  - peak allocated memory `= 7,525 MiB`
- `resid4_e25_c8t2`
  - `bptt_chunks=2`
  - `warmup_steps=50`
  - `lr_hold_steps=750`
  - final `val_bpb = 2.5175869540`
  - steady `tok/s = 1,854,033`
  - peak allocated memory `= 13,485 MiB`
- `resid4_e25_c8t4`
  - `bptt_chunks=4`
  - `warmup_steps=25`
  - `lr_hold_steps=375`
  - final `val_bpb = 2.5193139021`
  - steady `tok/s = 1,865,468`
  - peak allocated memory `= 25,405 MiB`

Result from the clean `bptt` sweep:
- `bptt=1` won clearly in both families.
- The plain family degraded by about `0.00450` bpb at `bptt=2` and about `0.00532` bpb at `bptt=4`.
- The residual family degraded by about `0.00519` bpb at `bptt=2` and about `0.00692` bpb at `bptt=4`.
- Higher `bptt` increased steady `tok/s` slightly because there were fewer optimizer steps, but that came with sharply worse memory:
  - plain family: `6.8 -> 12.1 -> 22.6 GiB`
  - residual family: `7.5 -> 13.5 -> 25.4 GiB`
- On the current local screening budget, semi-TBPTT is not a quality win.

### `blocks3` `carry` sweep on the best residual families

All runs below used:
- `12` branches
- `3` blocks
- full readout
- `spec_max_tokens=5,000,000`
- `bptt_chunks=1`

`resid4_e20` family:
- `resid4_e20_c8t1`
  - final `val_bpb = 2.5133777174`
  - steady `tok/s = 1,971,311`
- `resid4_e20_c16t1`
  - final `val_bpb = 2.5123756944`
  - steady `tok/s = 1,971,871`
- `resid4_e20_c32t1`
  - final `val_bpb = 2.5137890659`
  - steady `tok/s = 1,967,211`

`resid4_e25` family:
- `resid4_e25_c8t1`
  - final `val_bpb = 2.5123951758`
  - steady `tok/s = 1,820,046`
- `resid4_e25_c16t1`
  - final `val_bpb = 2.5136024590`
  - steady `tok/s = 1,819,138`
- `resid4_e25_c32t1`
  - final `val_bpb = 2.5141047368`
  - steady `tok/s = 1,819,390`

Result from the clean `carry` sweep:
- `carry=16` helped the smaller residual controller by about `0.00100` bpb relative to `carry=8`.
- `carry=32` regressed for that same controller.
- For the larger residual controller, both `carry=16` and `carry=32` regressed from `carry=8`.
- The top two screening points are now effectively tied:
  - `resid4_e20_c16t1 = 2.5123756944`
  - `resid4_e25_c8t1 = 2.5123951758`
- The gap is about `0.00002` bpb, which is far too small to over-interpret at this short budget.
- That means the right next step is longer confirmation, not more screening knobs.

## What This Means

Does more controller depth help?
- Yes, but only after trimming the frozen side.
- On overbuilt `blocks9`, the deeper residual controller package lost.
- On `blocks3`, moving from `plain3_e20` to `plain4_e20` helped modestly, and moving to `resid4_e20` or `resid4_e25` helped more.

Does more controller width via expansion help?
- Not as a generic knob.
- `plain3_e25` lost against `plain4_e20`, so width alone is not rescuing the plain family.
- Inside the residual `4`-layer controller, `expansion=2.5` beat `2.0` by about `0.00098` bpb at the cost of about `8%` throughput.

Is residualization materially improving trainability?
- On `blocks3`, yes.
- The residual path is stable and appears to be earning its keep once the frozen stack is not overbuilt.
- The current best two screening points are both residual controllers.

Is semi-TBPTT helping beyond simple carry?
- No, not on the current local budget.
- A clean matched-token `bptt` sweep showed `bptt=1` beating `2` and `4` in both the plain and residual families.
- Higher `bptt` also caused large memory inflation without offsetting quality gains.

## Best Controller-Only Contender

Current best controller-only contender:
- `resid4_e20_c16t1` on the `blocks3` structure by a hair in the current screen
- final `val_bpb = 2.5123756944`
- steady `tok/s = 1,971,871`
- artifact estimate `= 4,197,310`

The strongest simpler anchor remains:
- `plain4_e20_c8t1` on `blocks3`
- final `val_bpb = 2.5154361649`
- steady `tok/s = 2,030,859`

The real takeaway is stronger than the nominal winner:
- the best two residual points are essentially tied at the screening budget
- the gap to the plain anchor is only a few thousandths of a bpb
- longer confirmation runs are required before making strong ranking claims

## Regression-To-Transformer Guardrail

The latest winner is a somewhat larger controller, so this guardrail has to stay active.
- The winning controller is still only `4` minGRU layers, not a deep generic stack.
- The winning frozen structure is still `blocks3`, not the old `blocks9`.
- Artifact size is unchanged across these controller screens because the frozen side is still doing the byte-heavy work.
- `carry=16` helping `resid4_e20` is a healthier signal than `bptt>1`, because it keeps the controller in the same recurrent family without adding truncated unroll.
- That means the project is still in the intended family, but controller creep is still a real thing to watch.

The remaining risk is different:
- the frozen amplifier side may still be too static or weakly temporal
- the controller may start absorbing too much of the modeling burden if we keep adding capacity without proving horizon gains

That is still a better failure mode than accidentally rebuilding a transformer, because it points toward strengthening the frozen temporal role rather than blindly stacking more trainable depth.

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

`blocks3` neighborhood screen:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks3_neighborhood_v1 \
NUM_BLOCKS=3 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
TRAIN_FRAC=0.98 \
RUN_SPECS=$'plain4_e20_c8t1 4 2.0 8 1 0 -2.0 0.003 1500 0.0003 384 256 512\n\
plain3_e25_c8t1 3 2.5 8 1 0 -2.0 0.003 1500 0.0003 384 256 512\n\
resid4_e20_c8t1 4 2.0 8 1 1 -2.0 0.003 1500 0.0003 384 256 512\n\
resid4_e25_c8t1 4 2.5 8 1 1 -2.0 0.003 1500 0.0003 384 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

Clean `bptt` sweep with per-run warmup scaling:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks3_bptt_v2 \
NUM_BLOCKS=3 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
TRAIN_FRAC=0.98 \
RUN_SPECS=$'plain4_e20_c8t1 4 2.0 8 1 0 -2.0 0.003 100 1500 0.0003 384 256 512\n\
plain4_e20_c8t2 4 2.0 8 2 0 -2.0 0.003 50 750 0.0003 192 256 512\n\
plain4_e20_c8t4 4 2.0 8 4 0 -2.0 0.003 25 375 0.0003 96 256 512\n\
resid4_e25_c8t1 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e25_c8t2 4 2.5 8 2 1 -2.0 0.003 50 750 0.0003 192 256 512\n\
resid4_e25_c8t4 4 2.5 8 4 1 -2.0 0.003 25 375 0.0003 96 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

Clean `carry` sweep at fixed `bptt=1`:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks3_carry_v1 \
NUM_BLOCKS=3 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
TRAIN_FRAC=0.98 \
RUN_SPECS=$'resid4_e20_c8t1 4 2.0 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e20_c16t1 4 2.0 16 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e20_c32t1 4 2.0 32 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e25_c8t1 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e25_c16t1 4 2.5 16 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e25_c32t1 4 2.5 32 1 1 -2.0 0.003 100 1500 0.0003 384 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

## Immediate Next Step

The next step should be longer confirmation, not more broad screening.
- Keep the `blocks3` structure fixed.
- Keep `bptt_chunks=1`.
- Confirm the two near-tied leaders on a much longer token budget:
  - `resid4_e20_c16t1`
  - `resid4_e25_c8t1`
- Only after that should schedule refinement resume.

That is the right move because the short screening budget has already done its job: it found the plausible winners, but it is not long enough to separate them cleanly.
