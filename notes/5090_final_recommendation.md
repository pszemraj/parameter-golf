# 5090 Final Recommendation

> [!WARNING]
> This note is still provisional.
> The old capped-spec runs are invalidated, and some corrected `blocks3`, temporal, and schedule/context families still need to be replayed.
> The recommendations below reflect the strongest corrected local evidence on the current radical controller axis.

## Current Status

- The Core/Amplifier path is logging cleanly to W&B project `pg-core-amp`.
- Exact `val_bpb` is working locally through the official tokenizer path.
- Artifact accounting includes the int8-zlib trainable controller payload, not just repo code plus `gzip(spec.pt)`.
- The local dataset is complete for this variant:
  - `195` train shards
  - about `19.47B` train tokens available to frozen-spec builds
- The strongest corrected signal is no longer the old `blocks2` setup.
- The current best completed local point is now `blocks0 + 12 x 8.0`, which slightly beats `blocks2 + 12 x 8.0` while using a much smaller frozen spec and running faster.

## Top 3 Local Contenders

These are the current corrected leaders under the full-spec contract. They are all single-seed `512M`-token screening runs.

1. `blocks0_resid12_e8_c8t1_r3_current_512m`
   - evidence: best completed pure-quality screening leader
   - final `val_bpb = 2.2859021694`
   - steady `tok/s = 474,391`
   - trainable params `= 672,089`
   - artifact estimate `= 2,801,887`
   - why it matters:
     - this is the strongest corrected local point so far inside the frozen-statistics + parallel minGRU family
     - it slightly beat the completed `blocks2 + 12 x 8.0` point while also running faster and using the smaller `blocks0` frozen spec

2. `blocks0_resid12_e6_c8t1_r3_current_512m`
   - evidence: best current quality/speed tradeoff
   - final `val_bpb = 2.2979334823`
   - steady `tok/s = 616,452`
   - trainable params `= 505,049`
   - artifact estimate `= 2,673,848`
   - why it matters:
     - this stayed essentially tied with `blocks2 + 12 x 6.0` to the end while running faster and cutting the frozen-spec payload almost in half
     - it is the cleanest current tradeoff point and the strongest evidence that the current amplifier blocks may not be earning their bytes

3. `blocks2_resid12_e8_c8t1_r3_current_512m`
   - evidence: strongest completed `blocks2` pure-quality point
   - final `val_bpb = 2.2867730098`
   - steady `tok/s = 440,135`
   - trainable params `= 672,139`
   - artifact estimate `= 4,498,772`
   - why it matters:
     - this was the first corrected local winner before the `blocks0` version edged it out
     - it remains the clean comparison point if later reruns show the `blocks0` win was specific to this narrow radical setup

## Exact Reproduction Commands

### 1. `blocks0_resid12_e8_c8t1_r3_current_512m`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_structure/fullspec_blocks0_radical_v1/blocks0_resid12_e6_c8t1_r3_current_512m \
  MODEL_ROOT=experiments/5090_controller/fullspec_blocks0_controller_v1 \
  PRESET=controller_default \
  COMPILE=0 \
  VAL_EVERY=256 \
  VAL_STEPS=8 \
  LOG_EVERY=64 \
  LOG_STATE_EVERY=256 \
  SAVE_EVERY=2048 \
  TRAIN_FRAC=0.98 \
  BRANCH_TEMPORAL_MODE=current \
  BRANCH_TEMPORAL_LAG_SCALE=1.0 \
  RUN_SPECS=$'blocks0_resid12_e8_c8t1_r3_current_512m 12 8.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 2. `blocks0_resid12_e6_c8t1_r3_current_512m`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  MODEL_ROOT=experiments/5090_structure/fullspec_blocks0_radical_v1 \
  PRESET=structure_default \
  CORE_LAYERS=12 \
  CORE_EXPANSION=6.0 \
  RESIDUAL_CORE=1 \
  RESIDUAL_CORE_INIT=-3.0 \
  CARRY_CHUNKS=8 \
  BPTT_CHUNKS=1 \
  NUM_STEPS=4096 \
  COMPILE=0 \
  VAL_EVERY=256 \
  VAL_STEPS=8 \
  LOG_EVERY=64 \
  LOG_STATE_EVERY=256 \
  SAVE_EVERY=2048 \
  TRAIN_FRAC=0.98 \
  BRANCH_TEMPORAL_MODE=current \
  BRANCH_TEMPORAL_LAG_SCALE=1.0 \
  RUN_SPECS=$'blocks0_resid12_e6_c8t1_r3_current_512m 1,2,3,4,6,8,12,16,24,32,48,64 0 0' \
  conda run -s --name train python tools/run_core_amp_sweep.py structure
```

### 3. `blocks2_resid12_e8_c8t1_r3_current_512m`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_controller/fullspec_blocks2_frontier_v1/_shared_spec \
  MODEL_ROOT=experiments/5090_controller/fullspec_blocks2_radical_v1 \
  PRESET=controller_default \
  NUM_BLOCKS=2 \
  COMPILE=0 \
  VAL_EVERY=256 \
  VAL_STEPS=8 \
  LOG_EVERY=64 \
  LOG_STATE_EVERY=256 \
  SAVE_EVERY=2048 \
  TRAIN_FRAC=0.98 \
  BRANCH_TEMPORAL_MODE=current \
  BRANCH_TEMPORAL_LAG_SCALE=1.0 \
  RUN_SPECS=$'blocks2_resid12_e8_c8t1_r3_current_512m 12 8.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

## Best Current Calls

Best pure-quality contender:
- `blocks0_resid12_e8_c8t1_r3_current_512m`
- reason:
  - this is the best corrected completed local point right now
  - it beats the completed `blocks2 + 12 x 8.0` point while also being faster and smaller on the frozen side

Best quality/speed tradeoff on the 5090:
- `blocks0_resid12_e6_c8t1_r3_current_512m`
- reason:
  - it is about `30%` faster than `blocks0 + 12 x 8.0` while giving up only about `0.01203` bpb
  - it cuts the frozen-spec gzip size from about `3.52 MB` on `blocks2` to about `1.83 MB`
  - it is the cleanest current result if you want both quality and future systems headroom

Most likely to transfer cleanly to `1x H100`:
- primary candidate: `blocks0_resid12_e6_c8t1_r3_current_512m`
- pure-quality candidate: `blocks0_resid12_e8_c8t1_r3_current_512m`
- reason:
  - `blocks0 + 12 x 6.0` currently looks like the cleanest recurrent controller while matching the old `blocks2 + 12 x 6.0` quality
  - `blocks0 + 12 x 8.0` is the current pure-quality winner, but its top residual gate and top-layer state norms are still hotter

## Findings Likely To Be 5090-Specific

- absolute throughput numbers
- absolute memory headroom numbers
- compile warmup economics
- any interaction with `TORCH_BLAS_PREFER_CUBLASLT=1` on this local stack

These are less likely to be 5090-specific:
- the old capped-spec `blocks2` ranking being wrong once the frozen spec is rebuilt from the full corpus
- larger radical minGRU controllers with a more closed residual init buying much larger gains than the earlier moderate-controller sweeps
- the current learned amplifier stack failing to buy a meaningful edge once the controller is strong enough
- a smaller frozen spec plus a stronger recurrent controller being more promising than the older deeper-frozen setup

## Code Improvements Vs Hyperparameter Findings

Code improvements:
- W&B integration in `train_core_amplifier.py` with a cleaner config/history/summary split
- per-run metadata capture for commit, environment, device, and runtime settings
- corrected artifact accounting that includes int8-zlib trainable controller payload
- `branch_temporal_mode=current|lagged|hybrid` plus carried branch-history support
- rebuilt summaries now carry GPU/runtime metadata including CUDA, driver, TF32, and `TORCH_BLAS_PREFER_CUBLASLT`
- updated reports and logbook under `docs/` and `experiments/`

Pure hyperparameter / architecture findings:
- the old capped-spec conclusion that `6 x 2.5` was the `blocks2` winner did not survive the full-spec rebuild
- the first corrected half-million-parameter controller (`12 x 6.0`, `rinit=-3.0`) beat the entire corrected moderate local frontier
- the wider `12 x 8.0` controller then improved pure quality again, but with a hotter top residual gate and a throughput penalty
- removing the current amplifier blocks entirely (`blocks0`) did not hurt the `12 x 6.0` radical controller in any meaningful way by the end of the screening run
- removing the current amplifier blocks entirely (`blocks0`) slightly improved the completed `12 x 8.0` radical controller result while also shrinking the frozen artifact and improving throughput

## Regression-To-Transformer Guardrail

Current evidence still says we are not drifting back into a transformer-shaped local optimum.

- The best corrected new result now comes from removing the current amplifier blocks and increasing recurrent controller capacity, not from adding frozen depth.
- The controller is still a parallel minGRU stack.
- There is still no attention and no token-token mixing.
- The current winners are recurrent controllers reading a frozen statistical basis, not transformer-style token mixers.

The real risk is different:
- the controller may start carrying too much of the modeling burden if we keep buying gains with trainable capacity alone
- the current results already suggest that the present learned amplifier stack may be optional or misallocated
- an increasingly dominant top residual gate in the widest controller would be a warning sign that we are concentrating too much learning burden into one recurrent layer

That means the guardrail is now:
- keep the controller recurrent and parallel
- keep avoiding attention
- treat `blocks0` as a serious contender and only add frozen structure back if it earns its bytes experimentally
- improve the frozen temporal role only when the experiments justify it

## Unresolved Questions

- Does `blocks0_resid12_e8_c8t1_r3_current_512m` keep its edge under a longer budget, or is it mostly a screening-budget effect?
- Does `blocks0_resid12_e6_c8t1_r3_current_512m` hold up at `1B` tokens?
- Do the new corrected `blocks0` points hold up across `3` seeds?
- Does one more controller-only scaling step on `blocks0` beat `12 x 8.0` again without losing stability?
- Is there any minimal frozen structure that beats `blocks0`, or should the learned amplifier stack be removed from the current local frontier entirely?
- Can the wider `12 x 8.0` point be stabilized further with a more closed residual init or a better schedule, without giving up its quality edge?
- How much of the current ranking survives on `1x H100` and then on the final `8x H100` regime?
