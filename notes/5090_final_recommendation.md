# 5090 Final Recommendation

## Current Status
- The root Core/Amplifier path now has disciplined local artifacts plus W&B logging to `pg-core-amp`.
- Exact `val_bpb` is working locally through the official tokenizer path.
- A completed structural sweep on `1x RTX 5090` narrowed the frozen side.
- A completed controller follow-up on the new structural winner narrowed the controller side.

## Top 3 Local Contenders

1. `plain3_e20` on `blocks3`
   - final `val_bpb = 2.5158438607`
   - steady `tok/s = 2,191,340`
   - artifact estimate `= 4,195,301`
2. `plain3_e20` on full `blocks9`
   - final `val_bpb = 2.5171747775`
   - steady `tok/s = 1,042,359`
   - artifact estimate `= 9,229,159`
3. `resid5_e20` on `blocks3`
   - final `val_bpb = 2.5175855416`
   - steady `tok/s = 1,857,335`
   - artifact estimate `= 4,195,301`

If the goal is pure local screening quality, `plain3_e20 + blocks3` is the current winner.

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
- `plain3_e20` on `blocks3`

Best quality/speed tradeoff on the 5090:
- also `plain3_e20` on `blocks3`
- it is better in quality than every completed alternative and faster than the heavier full-structure points

Most likely to transfer cleanly to `1x H100`:
- `plain3_e20` on `blocks3`
- reason:
  - it wins on quality locally
  - it is structurally simpler than the old `blocks9` baseline
  - it still keeps a real frozen amplifier front-end, so it is not collapsing all the way to the `blocks0` edge case

## Findings Likely To Be 5090-Specific

- Absolute throughput numbers
- Absolute memory headroom numbers
- The exact compile economics and warmup tradeoffs
- Any interaction with `TORCH_BLAS_PREFER_CUBLASLT=1` on this local stack

These are likely not 5090-specific:
- `blocks3` beating `blocks9`
- `plain3_e20` beating `resid5_e20` in the completed local screens
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
- `branches8_pow2` and `readout128` both lose enough quality that they should not become the default
- `readout256` is the only tested compression point that still looks plausibly useful

## Regression-To-Transformer Guardrail

Current evidence says we are not drifting back into a transformer-shaped local optimum.
- The winning frozen structure is shallower, not deeper.
- The winning controller is smaller, not more stacked.
- The whole picture favors a modest frozen multi-timescale front-end plus a small recurrent controller.

The remaining architectural risk is different:
- the frozen side may still be too static or too weakly temporal
- the controller may be carrying too much of the real modeling burden

That points toward improving the frozen temporal role, not toward adding generic stack depth.

## Unresolved Questions

- Does `plain3_e20` also win on `blocks0`, or does the modest `blocks3` amplifier remain important there?
- Can a slightly larger plain controller, such as `plain4_e20` or `plain3_e25`, beat the current winner on `blocks3` without blowing up complexity?
- Is there a better 8-branch lag set than the power-of-two one tested here?
- Can we design a more meaningful frozen temporal mixer that stays parallelizable without turning into attention?
- How much of the current ranking survives on `1x H100` and then on the final `8x H100` regime?
