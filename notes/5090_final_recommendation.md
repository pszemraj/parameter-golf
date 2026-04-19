# 5090 Final Recommendation

## Current Status

- The Core/Amplifier path is logging cleanly to W&B project `pg-core-amp`.
- Exact `val_bpb` is working locally through the official tokenizer path.
- Artifact accounting now includes the int8-zlib trainable controller payload, not just repo code plus `gzip(spec.pt)`.
- The old `blocks9` frozen stack has already been beaten locally by a much smaller frozen structure.
- The strongest new signal is now a `blocks2` frontier where larger recurrent controllers are buying real quality gains.

## Top 3 Local Contenders

These are grouped by evidence level rather than pretending every run used the same budget.

1. `blocks2_resid12_e6_c8t1_r3_current_512m`
   - evidence: best current pure-quality screening leader
   - final `val_bpb = 2.3518192702`
   - steady `tok/s = 560,313`
   - trainable params `= 505,099`
   - artifact estimate `= 3,782,481`
   - why it matters:
     - this is the first truly large recurrent controller that produced a clear local jump
     - it beat `blocks2_resid6_e25_current_512m` by about `0.04062` bpb on the same screen

2. `blocks2_resid6_e25_c8t1_1b`
   - evidence: strongest longer-budget confirmed point so far
   - final `val_bpb = 2.3644974368`
   - steady `tok/s = 1,861,968`
   - artifact estimate `= 3,465,503`
   - why it matters:
     - this confirms the new `blocks2` family is genuinely better than the older `blocks3` reference

3. `blocks2_resid5_e25_c8t1_current_512m`
   - evidence: best current quality/speed screening point
   - final `val_bpb = 2.3986403191`
   - steady `tok/s = 2,023,976`
   - artifact estimate `= 3,452,276`
   - why it matters:
     - this remains the efficient point on the new frontier
     - it is the fallback if the larger controllers do not survive longer-budget confirmation

Useful negative result on the temporal axis:
- `resid4_e25_c8t1_current_512m` on `blocks3`: `2.4138021379`
- `resid4_e25_c8t1_hybrid_512m` on `blocks3`: `2.4268408799`
- `resid4_e25_c8t1_lagged_512m` on `blocks3`: `2.4529551592`
- conclusion:
  - `current` is still the right default branch reader
  - neither pure lagged nor first-pass hybrid should get H100 confirmation budget

## Exact Reproduction Commands

### 1. `blocks2_resid12_e6_c8t1_r3_current_512m`

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks2_radical_v1 \
PRESET=controller_default \
NUM_BLOCKS=2 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=256 \
VAL_STEPS=8 \
LOG_EVERY=64 \
LOG_STATE_EVERY=256 \
SAVE_EVERY=2048 \
TRAIN_FRAC=0.98 \
BRANCH_TEMPORAL_MODE=current \
BRANCH_TEMPORAL_LAG_SCALE=1.0 \
RUN_SPECS=$'blocks2_resid12_e6_c8t1_r3_current_512m 12 6.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 2. `blocks2_resid6_e25_c8t1_1b`

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks2_confirm1b_v1 \
PRESET=controller_default \
NUM_BLOCKS=2 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=512 \
VAL_STEPS=8 \
LOG_EVERY=128 \
LOG_STATE_EVERY=512 \
SAVE_EVERY=4096 \
TRAIN_FRAC=0.98 \
BRANCH_TEMPORAL_MODE=current \
BRANCH_TEMPORAL_LAG_SCALE=1.0 \
RUN_SPECS=$'blocks2_resid6_e25_c8t1_1b 6 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 8192 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 3. `blocks2_resid5_e25_c8t1_current_512m`

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks2_resid5e25_current_v1 \
PRESET=controller_default \
NUM_BLOCKS=2 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=256 \
VAL_STEPS=8 \
LOG_EVERY=64 \
LOG_STATE_EVERY=256 \
SAVE_EVERY=2048 \
TRAIN_FRAC=0.98 \
BRANCH_TEMPORAL_MODE=current \
BRANCH_TEMPORAL_LAG_SCALE=1.0 \
RUN_SPECS=$'blocks2_resid5_e25_c8t1_current_512m 5 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 4096 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

## Best Current Calls

Best pure-quality contender:
- screening leader: `blocks2_resid12_e6_c8t1_r3_current_512m`
- strongest confirmed fallback: `blocks2_resid6_e25_c8t1_1b`
- reason:
  - `blocks2_resid12_e6` is the best local short-budget quality point right now
  - `blocks2_resid6_e25_c8t1_1b` is already confirmed on the new family

Best quality/speed tradeoff on the 5090:
- `blocks2_resid5_e25_c8t1_current_512m`
- reason:
  - it is dramatically faster than the radical `12 x 6.0` point
  - it still beats the old frontier cleanly
  - it also reduces corrected artifact size materially
- caveat:
  - this is still a screening result, not yet a longer-budget confirmation

Most likely to transfer cleanly to `1x H100`:
- primary candidate: `blocks2_resid12_e6_c8t1_r3_current_512m`
- conservative fallback: `blocks2_resid6_e25_c8t1_1b`
- reason:
  - the radical `12 x 6.0` point is the strongest current quality signal while still staying fully recurrent
  - the smaller `blocks2_resid6_e25` point is already confirmed and gives a safer fallback

## Findings Likely To Be 5090-Specific

- absolute throughput numbers
- absolute memory headroom numbers
- compile warmup economics
- any interaction with `TORCH_BLAS_PREFER_CUBLASLT=1` on this local stack

These are less likely to be 5090-specific:
- `blocks3` beating the old `blocks9` frozen default
- pure `lagged` temporal branches losing clearly to `current`
- first-pass `hybrid` temporal branches still losing to `current`
- controller-up/spec-down reallocation being more promising than temporal-branch substitutions
- depth beating width on the new `blocks2` frontier
- deeper radical minGRU scaling with a more closed residual init can buy a much larger quality jump than the earlier small-controller sweeps

## Code Improvements Vs Hyperparameter Findings

Code improvements:
- W&B integration in `train_core_amplifier.py` with a cleaner config/history/summary split
- per-run metadata capture for commit, environment, device, and runtime settings
- corrected artifact accounting that includes int8-zlib trainable controller payload
- `branch_temporal_mode=current|lagged|hybrid` plus carried branch-history support
- updated reports and logbook under `docs/` and `experiments/`

Pure hyperparameter / architecture findings:
- `blocks3` is better than the old `blocks9` default locally
- `bptt=1` is currently better than `2` or `4` on the tested local budgets
- carry helps only narrowly inside the `blocks3` family
- pure `lagged` temporal branches are a bad swap for the default `current` branch reader
- first-pass `hybrid` temporal branches are better than pure `lagged`, but still not good enough
- shrinking the frozen amplifier from `blocks3` to `blocks2` and increasing recurrent controller capacity is the strongest new local signal so far
- on that new `blocks2` frontier, `6 x 2.5` beat `5 x 3.0`, so depth is currently buying more than width
- the first half-million-parameter controller (`12 x 6.0`, `rinit=-3.0`) beat the entire earlier local frontier

## Regression-To-Transformer Guardrail

Current evidence still says we are not drifting back into a transformer-shaped local optimum.

- The best new result came from shrinking the frozen amplifier, not from adding more frozen depth.
- The controller is still a parallel minGRU stack.
- There is still no attention and no token-token mixing.
- The clean `bptt` sweep says more truncated recurrent unroll is not the path.
- The negative temporal probe says that "make it more explicit-temporal" is not automatically helpful unless it preserves the strong current-state route.

The real risk is different:
- the controller may start carrying too much of the modeling burden if we keep buying gains with trainable capacity alone
- the frozen side may still be too static or too weakly temporal

That means the guardrail is now:
- keep the controller recurrent and parallel
- keep avoiding attention
- improve the frozen temporal role only when the experiments justify it

## Unresolved Questions

- Does `blocks2_resid12_e6_c8t1_r3_current_512m` hold up at `1B` tokens?
- Does `blocks2_resid5_e25_c8t1_current_512m` stay the better quality/speed tradeoff under a longer budget?
- Do the new `blocks2` points hold up across `3` seeds?
- Do larger radical controllers keep scaling cleanly, or does systems cost dominate before quality does?
- Can we shrink the frozen spec further, or compress readout, while keeping the larger recurrent controller as the main learner?
- Is there a stronger frozen temporal mixer that preserves current-state access without turning into attention?
- How much of the current ranking survives on `1x H100` and then on the final `8x H100` regime?
