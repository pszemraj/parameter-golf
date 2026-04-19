# 5090 Final Recommendation

## Current Status

- The Core/Amplifier path is logging cleanly to W&B project `pg-core-amp`.
- Exact `val_bpb` is working locally through the official tokenizer path.
- Artifact accounting now includes the int8-zlib trainable controller payload, not just repo code plus `gzip(spec.pt)`.
- The old `blocks9` frozen stack has already been beaten locally by a much smaller frozen structure.
- The strongest new signal is a `blocks2` frontier where controller capacity beats further frozen-stack depth.

## Top 3 Local Contenders

These are grouped by evidence level rather than pretending every run used the same budget.

1. `blocks2_resid6_e25_c8t1_current_512m`
   - evidence: best current `512M`-token pure-quality screening leader
   - final `val_bpb = 2.3924393341`
   - steady `tok/s = 1,849,000`
   - artifact estimate `= 3,465,660`
   - why it matters:
     - beat `blocks2_resid5_e25` by another `0.00620` bpb
     - beat `blocks2_resid5_e30`, so depth is currently buying more than width on `blocks2`

2. `blocks2_resid5_e25_c8t1_current_512m`
   - evidence: best current `512M`-token quality/speed screening point
   - final `val_bpb = 2.3986403191`
   - steady `tok/s = 2,023,976`
   - artifact estimate `= 3,452,276`
   - why it matters:
     - beat the previous `blocks3 + resid4_e25 + current` anchor by about `0.01516` bpb
     - also ran about `6.26%` faster
     - also shrank corrected artifact estimate by `823,614` bytes

3. `resid4_e25_c8t1_1b` on `blocks3`
   - evidence: strongest longer-budget confirmed point so far
   - final `val_bpb = 2.3792377281`
   - steady `tok/s = 1,914,754`
   - artifact estimate `= 4,197,310`
   - why it matters:
     - this is still the cleanest conservative reference because it already has `1B` local evidence

Useful negative result on the temporal axis:
- `resid4_e25_c8t1_current_512m` on `blocks3`: `2.4138021379`
- `resid4_e25_c8t1_hybrid_512m` on `blocks3`: `2.4268408799`
- `resid4_e25_c8t1_lagged_512m` on `blocks3`: `2.4529551592`
- conclusion:
  - `current` is still the right default branch reader
  - neither pure lagged nor first-pass hybrid should get H100 confirmation budget

## Exact Reproduction Commands

### 1. `blocks2_resid6_e25_c8t1_current_512m`

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks2_resid6e25_current_v1 \
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
RUN_SPECS=$'blocks2_resid6_e25_c8t1_current_512m 6 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 4096 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 2. `blocks2_resid5_e25_c8t1_current_512m`

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

### 3. `resid4_e25_c8t1_1b`

```bash
/home/pszemraj/miniforge3/envs/train/bin/python \
/home/pszemraj/workspace/projects/parameter-golf/train_core_amplifier.py \
experiments/5090_controller/wandb_blocks3_confirm1b_v1/resid4_e25_c8t1_1b \
--data /home/pszemraj/workspace/projects/parameter-golf/data/datasets/fineweb10B_sp1024 \
--storage-dtype uint16 \
--seq-len 512 \
--batch-size 256 \
--grad-accum 1 \
--carry-chunks 8 \
--bptt-chunks 1 \
--num-steps 8192 \
--learning-rate 0.003 \
--lr-schedule cosine \
--min-lr 0.0003 \
--warmup-steps 100 \
--lr-hold-steps 1500 \
--weight-decay 0.001 \
--hard-loss-gamma 0.5 \
--hard-loss-cap 5.0 \
--grad-clip 1.0 \
--core-layers 4 \
--core-expansion 2.5 \
--residual-core 1 \
--residual-core-init -2.0 \
--val-every 512 \
--val-steps 8 \
--save-every 4096 \
--log-every 128 \
--log-state-every 512 \
--train-frac 0.98 \
--seed 1337 \
--wandb \
--wandb-project pg-core-amp \
--wandb-run-name resid4_e25_c8t1_1b \
--wandb-group wandb_blocks3_confirm1b_v1 \
--wandb-tags core_amp,5090,controller,confirmation,long_budget \
--wandb-watch gradients \
--wandb-watch-log-freq 25
```

## Best Current Calls

Best pure-quality contender:
- screening leader: `blocks2_resid6_e25_c8t1_current_512m`
- strongest confirmed fallback: `resid4_e25_c8t1_1b` on `blocks3`
- reason:
  - `blocks2_resid6_e25` is the best local short-budget quality point right now
  - `resid4_e25_c8t1_1b` is still the safest already-confirmed quality reference

Best quality/speed tradeoff on the 5090:
- `blocks2_resid5_e25_c8t1_current_512m`
- reason:
  - it is materially faster than `blocks2_resid6_e25`
  - it still beats the previous `blocks3` `current`-mode anchor cleanly
  - it also reduces corrected artifact size materially
- caveat:
  - this is still a screening result, not yet a longer-budget confirmation

Most likely to transfer cleanly to `1x H100`:
- primary candidate: `blocks2_resid6_e25_c8t1_current_512m`
- conservative fallback: `resid4_e25_c8t1_1b`
- reason:
  - the `blocks2` depth win is the strongest current in-family quality signal
  - it still keeps the frozen side smaller than the older `blocks3` family
  - the `blocks3` `1B` point still matters because it already has longer-budget local evidence

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

- Does `blocks2_resid6_e25_c8t1_current_512m` hold up at `1B` tokens?
- Does `blocks2_resid5_e25_c8t1_current_512m` stay the better quality/speed tradeoff under a longer budget?
- Do the new `blocks2` points hold up across `3` seeds?
- Do deeper `blocks2` controllers improve further, or do they just overload the top recurrent layers?
- Can we shrink the frozen spec further, or compress readout, while keeping the larger recurrent controller as the main learner?
- Is there a stronger frozen temporal mixer that preserves current-state access without turning into attention?
- How much of the current ranking survives on `1x H100` and then on the final `8x H100` regime?
