# 5090 Final Recommendation

## Current Status
- The root Core/Amplifier path now has disciplined local artifacts plus W&B logging to `pg-core-amp`.
- Exact `val_bpb` is working locally through the official tokenizer path.
- A completed structural sweep on `1x RTX 5090` narrowed the frozen side.
- Two completed controller screens on the new structural winner narrowed the controller side.
- The committed `docs/5090_log.md` logbook is now tracking timestamps, code commits, and important W&B runs.

## Top 3 Local Contenders

1. `resid4_e25_c8t1` on `blocks3`
   - final `val_bpb = 2.5123951758`
   - steady `tok/s = 1,820,432`
   - artifact estimate `= 4,197,310`
2. `resid4_e20_c8t1` on `blocks3`
   - final `val_bpb = 2.5133777174`
   - steady `tok/s = 1,970,720`
   - artifact estimate `= 4,197,310`
3. `plain4_e20_c8t1` on `blocks3`
   - final `val_bpb = 2.5154361649`
   - steady `tok/s = 2,030,859`
   - artifact estimate `= 4,197,310`

Carry screening refined the top two points further:
- `resid4_e20_c16t1`: `2.5123756944`
- `resid4_e25_c8t1`: `2.5123951758`

Those are effectively tied at the short screening budget.
The earlier `plain3_e20 + blocks3` point remains a useful simple reference at `val_bpb = 2.5158438607`.

## Exact Reproduction Commands

### Structural sweep that picked `blocks3`

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_structure/wandb_round1 \
PRESET=structure_default \
RUN_SPECS=$'blocks9 1,2,3,4,6,8,12,16,24,32,48,64 9 0\n\
blocks3 1,2,3,4,6,8,12,16,24,32,48,64 3 0\n\
blocks6 1,2,3,4,6,8,12,16,24,32,48,64 6 0\n\
branches8_pow2 1,2,4,8,16,32,64,128 9 0\n\
readout256 1,2,3,4,6,8,12,16,24,32,48,64 9 256\n\
readout128 1,2,3,4,6,8,12,16,24,32,48,64 9 128' \
conda run -s --name train python tools/run_core_amp_sweep.py structure
```

### Controller follow-up on `blocks3`

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

### Controller neighborhood screen on `blocks3`

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

### Clean `bptt` sweep on `blocks3`

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

### Earlier `blocks0` sanity probe

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_structure/blocks0_real \
CORE_LAYERS=5 \
CORE_EXPANSION=2.0 \
RESIDUAL_CORE=1 \
RESIDUAL_CORE_INIT=-2.0 \
SEQ_LEN=512 \
BATCH_SIZE=256 \
NUM_STEPS=192 \
CARRY_CHUNKS=16 \
BPTT_CHUNKS=2 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
SPEC_MAX_TOKENS=5000000 \
RUN_SPECS=$'blocks0 1,2,3,4,6,8,12,16,24,32,48,64 0 0' \
conda run -s --name train python tools/run_core_amp_sweep.py structure
```

## Best Current Calls

Best pure-quality contender:
- `resid4_e20_c16t1` on `blocks3` by a negligible screening margin
- practical interpretation:
  - `resid4_e20_c16t1` and `resid4_e25_c8t1` are tied closely enough that longer confirmation is required

Best quality/speed tradeoff on the 5090:
- `resid4_e20_c16t1` on `blocks3`
- it matches the best current screening quality while running about `8%` faster than `resid4_e25_c8t1`

Most likely to transfer cleanly to `1x H100`:
- `resid4_e20_c16t1` on `blocks3`
- reason:
  - it currently matches the best screening quality without taking the wider residual controller
  - it keeps the stronger `blocks3` frozen structure
  - it looks like the cleaner first long-budget confirmation point while the top two screening candidates remain tied

## Findings Likely To Be 5090-Specific

- Absolute throughput numbers
- Absolute memory headroom numbers
- The exact compile economics and warmup tradeoffs
- Any interaction with `TORCH_BLAS_PREFER_CUBLASLT=1` on this local stack

These are likely not 5090-specific:
- `blocks3` beating `blocks9`
- the old `resid5_e20` package losing on both `blocks9` and `blocks3`
- the `blocks3` controller neighborhood preferring moderate controller growth over re-expanding the frozen stack
- `readout256` being much more acceptable than `readout128`

## Code Improvements Vs Hyperparameter Findings

Code improvements:
- explicit W&B integration in `train_core_amplifier.py` with:
  - static run descriptors in W&B config
  - minimal train/eval history keys
  - final runtime/artifact values in W&B summary
- real-vs-smoke preset separation in `tools/run_core_amp_sweep.py`
- artifact budget headroom/status in structured run outputs and summaries
- step-0 eval removed from the trainer
- sweep-runner defaults aligned to the actual local `5M` spec-build budget

Pure hyperparameter / architecture findings:
- `blocks3` is better than the old `blocks9` default locally
- `plain3_e20` is better than `resid5_e20` on both `blocks9` and `blocks3`
- on `blocks3`, moderate controller growth helps:
  - `plain4_e20` beats `plain3_e20`
  - `resid4_e20` beats the plain family
  - `resid4_e25` is the current single-seed leader
- on the current local screening budget, `bptt=1` beats `bptt=2` and `bptt=4` in both the plain and residual families
- on the current local screening budget, carry helps only in a narrow way:
  - `resid4_e20` prefers `carry=16`
  - `resid4_e25` prefers `carry=8`
- `branches8_pow2` and `readout128` both lose enough quality that they should not become the default
- `readout256` is the only tested compression point that still looks plausibly useful

## Regression-To-Transformer Guardrail

Current evidence says we are not drifting back into a transformer-shaped local optimum.
- The winning frozen structure is shallower, not deeper.
- The winning controllers are still only `4` recurrent layers, not a deep generic stack.
- The clean `bptt` sweep says we do not need more truncated recurrent unroll to get the current best result.
- The whole picture still favors a modest frozen multi-timescale front-end plus a modest recurrent controller.

The remaining architectural risk is different:
- the frozen side may still be too static or too weakly temporal
- the controller may start carrying too much of the real modeling burden if we keep buying gains with trainable depth alone

That still points toward improving the frozen temporal role, not toward adding generic stack depth.

## Unresolved Questions

- Does `plain3_e20` also win on `blocks0`, or does the modest `blocks3` amplifier remain important there?
- How much of the `resid4_e25` edge survives three-seed confirmation?
- Do the current near-tied `bptt=1` leaders stay tied on a much longer confirmation budget?
- Is there a better 8-branch lag set than the power-of-two one tested here?
- Can we design a more meaningful frozen temporal mixer that stays parallelizable without turning into attention?
- How much of the current ranking survives on `1x H100` and then on the final `8x H100` regime?
