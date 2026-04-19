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
- The current best completed local point is now `blocks0 + 12 x 10.0`, which beats the previous `blocks0 + 12 x 8.0` leader while staying well inside the artifact limit.
- A fixed-parameter follow-up, `blocks0 + 10 x 12.0`, is now running to separate depth-vs-width effects from simple controller-size scaling.

## Top 3 Local Contenders

These are the current corrected leaders under the full-spec contract. They are all single-seed `512M`-token screening runs.

1. `blocks0_resid12_e10_c8t1_r3_current_512m`
   - evidence: best completed pure-quality screening leader
   - final `val_bpb = 2.2777913795`
   - steady `tok/s = 384,214`
   - trainable params `= 839,129`
   - artifact estimate `= 2,936,419`
   - why it matters:
     - this is the strongest corrected local point so far inside the frozen-statistics + parallel minGRU family
     - it beat the previous `blocks0 + 12 x 8.0` leader by about `0.00811` bpb while keeping the same frozen `blocks0` spec

2. `blocks0_resid12_e8_c8t1_r3_current_512m`
   - evidence: stronger quality/speed compromise than the new winner
   - final `val_bpb = 2.2859021694`
   - steady `tok/s = 474,391`
   - trainable params `= 672,089`
   - artifact estimate `= 2,801,887`
   - why it matters:
     - this was the first corrected `blocks0` quality winner
     - it remains materially faster and lighter than `12 x 10.0` while staying within about `0.00811` bpb of the new leader

3. `blocks0_resid12_e6_c8t1_r3_current_512m`
   - evidence: best current quality/speed tradeoff
   - final `val_bpb = 2.2979334823`
   - steady `tok/s = 616,452`
   - trainable params `= 505,049`
   - artifact estimate `= 2,673,848`
   - why it matters:
     - it is still the cleanest current tradeoff point if the priority is keeping much more throughput and systems headroom
     - it remains the strongest evidence that the current amplifier blocks are not earning their bytes

## Exact Reproduction Commands

### 1. `blocks0_resid12_e10_c8t1_r3_current_512m`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_structure/fullspec_blocks0_radical_v1/blocks0_resid12_e6_c8t1_r3_current_512m \
  MODEL_ROOT=experiments/5090_controller/fullspec_blocks0_controller_v2 \
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
  RUN_SPECS=$'blocks0_resid12_e10_c8t1_r3_current_512m 12 10.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 2. `blocks0_resid12_e8_c8t1_r3_current_512m`

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

### 3. `blocks0_resid12_e6_c8t1_r3_current_512m`

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

## Best Current Calls

Best pure-quality contender:
- `blocks0_resid12_e10_c8t1_r3_current_512m`
- reason:
  - this is the best corrected completed local point right now
  - it beats the previous `blocks0 + 12 x 8.0` leader by about `0.00811` bpb while staying comfortably inside the artifact cap

Best quality/speed tradeoff on the 5090:
- `blocks0_resid12_e6_c8t1_r3_current_512m`
- reason:
  - it is about `60%` faster than `blocks0 + 12 x 10.0` while giving up about `0.02014` bpb
  - it cuts the frozen-spec gzip size from about `3.52 MB` on `blocks2` to about `1.83 MB`
  - it is the cleanest current result if you want both quality and future systems headroom

Most likely to transfer cleanly to `1x H100`:
- primary candidate: `blocks0_resid12_e6_c8t1_r3_current_512m`
- pure-quality candidate: `blocks0_resid12_e10_c8t1_r3_current_512m`
- reason:
  - `blocks0 + 12 x 6.0` currently looks like the cleanest recurrent controller while preserving much more systems slack
  - `blocks0 + 12 x 10.0` is the current pure-quality winner, but it is close enough to the local memory ceiling that it should be treated as a quality-first candidate rather than the default transfer bet

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
- the wider `12 x 10.0` controller improved pure quality again on the same `blocks0` structure, reaching `2.27779` bpb while staying artifact-safe
- removing the current amplifier blocks entirely (`blocks0`) did not hurt the `12 x 6.0` radical controller in any meaningful way by the end of the screening run
- removing the current amplifier blocks entirely (`blocks0`) slightly improved the completed `12 x 8.0` radical controller result while also shrinking the frozen artifact and improving throughput
- removing the current amplifier blocks entirely (`blocks0`) remains compatible with controller scaling up to at least `839k` trainable parameters

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
- the new `12 x 10.0` point still shows top-layer concentration, but it is materially better behaved than a simple "wider is automatically less stable" story

That means the guardrail is now:
- keep the controller recurrent and parallel
- keep avoiding attention
- treat `blocks0` as a serious contender and only add frozen structure back if it earns its bytes experimentally
- improve the frozen temporal role only when the experiments justify it

## Unresolved Questions

- Does `blocks0_resid12_e10_c8t1_r3_current_512m` keep its edge under a longer budget, or is it mostly a screening-budget effect?
- Does `blocks0_resid12_e6_c8t1_r3_current_512m` hold up at `1B` tokens?
- Do the new corrected `blocks0` points hold up across `3` seeds?
- Does the fixed-parameter `blocks0_resid10_e12_c8t1_r3_current_512m` run show that the new gain is really about controller shape, not just more controller mass?
- Is there any minimal frozen structure that beats `blocks0`, or should the learned amplifier stack be removed from the current local frontier entirely?
- Can the wider `12 x 10.0` point be stabilized further with a more closed residual init or a better schedule, without giving up its quality edge?
- How much of the current ranking survives on `1x H100` and then on the final `8x H100` regime?
