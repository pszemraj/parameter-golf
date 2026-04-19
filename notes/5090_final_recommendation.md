# 5090 Final Recommendation

> [!WARNING]
> This note is still provisional, but the leading `blocks2` entries below now reflect the corrected full-spec contract.
> Remaining `blocks3`, temporal, and structure families are still being rerun after fixing the invalid `SPEC_MAX_TOKENS=5000000` default in commit `57d4261`.
> Treat this as the current best local recommendation, not the final word.

## Current Status

- The Core/Amplifier path is logging cleanly to W&B project `pg-core-amp`.
- Exact `val_bpb` is working locally through the official tokenizer path.
- Artifact accounting now includes the int8-zlib trainable controller payload, not just repo code plus `gzip(spec.pt)`.
- The local dataset is now complete for this variant:
  - `195` train shards
  - about `19.47B` train tokens available to frozen-spec builds
- The old `blocks9` frozen stack has already been beaten locally by a much smaller frozen structure, but all structure claims still need the corrected full-spec rerun matrix completed.
- The strongest corrected signal is now a `blocks2` frontier where much larger parallel minGRU controllers are buying real quality gains on the same `512M`-token screening budget.

## Top 3 Local Contenders

These are the current corrected leaders under the full-spec contract. They are all single-seed screening runs unless otherwise noted.

1. `blocks2_resid12_e8_c8t1_r3_current_512m`
   - evidence: best current pure-quality screening leader
   - final `val_bpb = 2.2867730098`
   - steady `tok/s = 440,135`
   - trainable params `= 672,139`
   - artifact estimate `= 4,498,772`
   - why it matters:
     - this is the strongest corrected local quality point so far inside the frozen-spec + parallel minGRU family
     - it beat the corrected `12 x 6.0` radical point by about `0.01067` bpb and the corrected moderate `blocks2_resid5_e30_current_512m` point by about `0.06919` bpb on the same screening contract

2. `blocks2_resid12_e6_c8t1_r3_current_512m`
   - evidence: best current balance of corrected quality and speed inside the radical family
   - final `val_bpb = 2.2974388997`
   - steady `tok/s = 559,764`
   - trainable params `= 505,099`
   - artifact estimate `= 4,366,958`
   - why it matters:
     - this is still a large quality jump over the corrected moderate frontier while staying about `27%` faster than the wider `12 x 8.0` point
     - it looks cleaner and less stressed than the wider winner, making it the current transfer-favorite until longer-budget confirmation says otherwise

3. `blocks2_resid5_e25_c8t1_current_512m`
   - evidence: best current quality/speed screening point
   - final `val_bpb = 2.3640294706`
   - steady `tok/s = 2,026,086`
   - artifact estimate `= 4,029,702`
   - why it matters:
     - this remains the best corrected speed-efficient point on the new `blocks2` frontier
     - it is the fallback if the larger radical controllers do not survive longer-budget confirmation or structure sanity checks

Important pending rerun:
- the old `blocks3` temporal comparisons were made under the invalid capped-spec contract
- they remain useful as hypotheses, but not as final evidence
- until the corrected temporal rerun lands, keep `branch_temporal_mode=current` as the default because it is the cleanest and best-tested setting in the corrected `blocks2` family

## Exact Reproduction Commands

### 1. `blocks2_resid12_e8_c8t1_r3_current_512m`

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

### 2. `blocks2_resid12_e6_c8t1_r3_current_512m`

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
  RUN_SPECS=$'blocks2_resid12_e6_c8t1_r3_current_512m 12 6.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 3. `blocks2_resid5_e25_c8t1_current_512m`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  MODEL_ROOT=experiments/5090_controller/fullspec_blocks2_frontier_v1 \
  PRESET=controller_default \
  REBUILD_SHARED=1 \
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
  RUN_SPECS=$'blocks2_resid5_e25_c8t1_current_512m 5 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 4096 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

## Best Current Calls

Best pure-quality contender:
- screening leader: `blocks2_resid12_e8_c8t1_r3_current_512m`
- cleaner fallback: `blocks2_resid12_e6_c8t1_r3_current_512m`
- reason:
  - `blocks2_resid12_e8` is the best corrected local short-budget quality point right now
  - `blocks2_resid12_e6` trails by only about `0.01067` bpb while being materially faster and less top-heavy

Best quality/speed tradeoff on the 5090:
- `blocks2_resid12_e6_c8t1_r3_current_512m`
- reason:
  - it is about `27%` faster than the wider `12 x 8.0` winner while giving up only about `0.01067` bpb
  - it still beats the corrected moderate `blocks2` frontier by a large margin
  - it keeps the architecture squarely in the controller-up/spec-down direction without pushing the top recurrent layer as hard
- caveat:
  - this is still a screening result, not yet a longer-budget confirmation
  - the fastest efficient fallback remains `blocks2_resid5_e25_c8t1_current_512m`

Most likely to transfer cleanly to `1x H100`:
- primary candidate: `blocks2_resid12_e6_c8t1_r3_current_512m`
- pure-quality candidate: `blocks2_resid12_e8_c8t1_r3_current_512m`
- reason:
  - `12 x 6.0` currently looks like the cleaner recurrent controller and still posts a very strong corrected score
  - `12 x 8.0` is the current pure-quality winner, but its top residual gate and top-layer state norms are much hotter

## Findings Likely To Be 5090-Specific

- absolute throughput numbers
- absolute memory headroom numbers
- compile warmup economics
- any interaction with `TORCH_BLAS_PREFER_CUBLASLT=1` on this local stack

These are less likely to be 5090-specific:
- controller-up/spec-down reallocation inside the corrected `blocks2` family being more promising than the old deeper-frozen default
- the old capped-spec `blocks2` ranking being wrong once the frozen spec is rebuilt from the full corpus
- deeper radical minGRU scaling with a more closed residual init buying a much larger quality jump than the earlier corrected moderate-controller sweeps
- wider radical scaling from `12 x 6.0` to `12 x 8.0` buying a further but more expensive pure-quality gain

## Code Improvements Vs Hyperparameter Findings

Code improvements:
- W&B integration in `train_core_amplifier.py` with a cleaner config/history/summary split
- per-run metadata capture for commit, environment, device, and runtime settings
- corrected artifact accounting that includes int8-zlib trainable controller payload
- `branch_temporal_mode=current|lagged|hybrid` plus carried branch-history support
- updated reports and logbook under `docs/` and `experiments/`

Pure hyperparameter / architecture findings:
- shrinking the frozen amplifier to `blocks2` and increasing recurrent controller capacity is the strongest corrected local signal so far
- on the corrected moderate `blocks2` frontier, `5 x 3.0` beat both `5 x 2.5` and `6 x 2.5`
- the old capped-spec conclusion that `6 x 2.5` was the `blocks2` winner did not survive the full-spec rebuild
- the first corrected half-million-parameter controller (`12 x 6.0`, `rinit=-3.0`) beat the entire corrected moderate local frontier
- the wider `12 x 8.0` controller then improved pure quality again, but with a much hotter top residual gate and a sizable throughput penalty

## Regression-To-Transformer Guardrail

Current evidence still says we are not drifting back into a transformer-shaped local optimum.

- The best corrected new result came from shrinking the frozen amplifier and increasing recurrent controller capacity, not from adding more frozen depth.
- The controller is still a parallel minGRU stack.
- There is still no attention and no token-token mixing.
- The corrected `blocks2` leaders are not "transformer-ish"; they are larger recurrent controllers with the same frozen current-state branch reader.

The real risk is different:
- the controller may start carrying too much of the modeling burden if we keep buying gains with trainable capacity alone
- the frozen side may still be too static or too weakly temporal
- a strong `blocks0` or `blocks1` result with the same radical controller would be a warning sign that the amplifier is becoming optional
- an increasingly dominant top residual gate in the widest controller would be a warning sign that we are concentrating too much learning burden into one recurrent layer

That means the guardrail is now:
- keep the controller recurrent and parallel
- keep avoiding attention
- rerun the structure sanity checks with the radical controllers
- improve the frozen temporal role only when the experiments justify it

## Unresolved Questions

- Does `blocks2_resid12_e8_c8t1_r3_current_512m` hold its pure-quality win under a longer budget, or is it mostly a screening-budget effect?
- Does `blocks2_resid12_e6_c8t1_r3_current_512m` hold up at `1B` tokens?
- Does `blocks2_resid5_e25_c8t1_current_512m` stay the better quality/speed tradeoff under a longer budget?
- Do the new corrected `blocks2` points hold up across `3` seeds?
- Does a matched radical-controller `blocks0/1/2` sweep show that the frozen amplifier is still earning its bytes?
- Can the wider `12 x 8.0` point be stabilized further with a more closed residual init or a better schedule, without giving up its quality edge?
- Do larger radical controllers keep scaling cleanly, or does systems cost dominate before quality does?
- Is there a stronger frozen temporal mixer that preserves current-state access without turning into attention?
- How much of the current ranking survives on `1x H100` and then on the final `8x H100` regime?
