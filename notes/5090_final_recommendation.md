# 5090 Final Recommendation

> [!WARNING]
> This note is still provisional.
> The corrected full-spec replay and the main `1B` controller confirmations are complete.
> The next source of movement should come from schedule, context, or a revised frozen-side mechanism, not from pretending the current frontier is still undecided.

## Current Status

- Logging is clean in W&B project `pg-core-amp`.
- Exact `val_bpb` is active through the official tokenizer path.
- Artifact accounting now matches the record-style convention:
  - repo code bytes
  - `gzip(spec.pt)`
  - trainable int8-zlib payload
- The local dataset is complete for this family:
  - `195` train shards
  - about `19.47B` train tokens available to frozen-spec builds
- The strongest corrected local signal is controller-up/spec-down reallocation, not more frozen amplifier depth.
- The `1B` confirmation queue changed the recommendation quality materially:
  - `blocks0 12x10` and `blocks0 10x12` both held up
  - checkpointed `blocks0 16x8` also held up and landed as the third-best pure-quality point
  - `blocks1 12x6` slightly beat `blocks0 12x6` on the longer budget
- The first real schedule result also changed the recommendation:
  - on the `512M` screen, the edge follow-up moved the best tested hold from `2500` to `3500` for both `blocks0 12x10` and `blocks0 10x12`
  - the `1B` confirmation then held up with `h7000`
  - the schedule was decaying too early under the inherited root default, and the tuned hold materially changed the top local ranking
- The regression-to-transformer guardrail is still intact:
  - no attention
  - no token-token mixing
  - winners are still parallel minGRU controllers over a frozen statistical basis

## Top 3 Current Contenders

These are the three most useful completed local contenders right now.

1. `blocks0_resid10_e12_h7000_1b`
   - best completed overall point
   - final `val_bpb = 2.1878016930`
   - steady `tok/s = 384,758`
   - trainable params `= 839,129`
   - artifact estimate `= 3,899,812`
   - why it is in:
     - best corrected completed local quality result after schedule tuning
     - strong gain from the tuned late-tail schedule without a throughput penalty

2. `blocks0_resid12_e10_h7000_1b`
   - second-best completed overall point
   - final `val_bpb = 2.1954688682`
   - steady `tok/s = 386,106`
   - trainable params `= 839,129`
   - artifact estimate `= 3,900,555`
   - why it is in:
     - it stayed strong after the tuned hold transfer
     - it is still a close second and keeps the deeper geometry alive as a real alternate shape

3. `blocks1_resid12_e6_c8t1_r3_current_1b`
   - best completed nonzero-amplifier guardrail
   - final `val_bpb = 2.2356768287`
   - steady `tok/s = 590,071`
   - trainable params `= 505,049`
   - artifact estimate `= 4,102,717`
   - why it is in:
     - it is still the cleanest transfer-control architecture with one frozen amplifier block
     - it is the next architecture family that should inherit the tuned schedule

Important nuance:

- `blocks0_resid16_e8_c8t1_r3_current_1b_gc1` remains the third-best pure-quality point at `2.2177299568`, but it is no longer the most useful next contender because it is slower and still not schedule-tuned.
- `blocks0_resid12_e6_c8t1_r3_current_1b` is still the cleanest quality-speed anchor:
  - `final val_bpb = 2.2363421409`
  - `steady tok/s = 618,833`
- The tuned schedule changed the ranking of the top two `blocks0` shapes:
  - `10x12 h7000` now beats `12x10 h7000` by about `0.00767` bpb

## Exact Reproduction Commands

### 1. `blocks0_resid10_e12_h7000_1b`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_structure/fullspec_blocks0_radical_v1/blocks0_resid12_e6_c8t1_r3_current_512m \
  MODEL_ROOT=experiments/5090_schedule/blocks0_10x12_hold_confirm1b_v1 \
  PRESET=controller_default \
  COMPILE=0 \
  VAL_EVERY=512 \
  VAL_STEPS=8 \
  LOG_EVERY=128 \
  LOG_STATE_EVERY=512 \
  SAVE_EVERY=4096 \
  TRAIN_FRAC=0.98 \
  BRANCH_TEMPORAL_MODE=current \
  RUN_SPECS=$'blocks0_resid10_e12_h7000_1b 10 12.0 8 1 1 -3.0 0.003 100 7000 0.0003 8192 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 2. `blocks0_resid12_e10_h7000_1b`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_structure/fullspec_blocks0_radical_v1/blocks0_resid12_e6_c8t1_r3_current_512m \
  MODEL_ROOT=experiments/5090_schedule/blocks0_12x10_hold_confirm1b_v1 \
  PRESET=controller_default \
  COMPILE=0 \
  VAL_EVERY=512 \
  VAL_STEPS=8 \
  LOG_EVERY=128 \
  LOG_STATE_EVERY=512 \
  SAVE_EVERY=4096 \
  TRAIN_FRAC=0.98 \
  BRANCH_TEMPORAL_MODE=current \
  RUN_SPECS=$'blocks0_resid12_e10_h7000_1b 12 10.0 8 1 1 -3.0 0.003 100 7000 0.0003 8192 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 3. `blocks1_resid12_e6_c8t1_r3_current_1b`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_controller/fullspec_blocks1_radical_v1/blocks1_resid12_e6_c8t1_r3_current_512m \
  MODEL_ROOT=experiments/5090_controller/fullspec_blocks1_confirm1b_v1 \
  PRESET=controller_default \
  COMPILE=0 \
  VAL_EVERY=512 \
  VAL_STEPS=8 \
  LOG_EVERY=128 \
  LOG_STATE_EVERY=512 \
  SAVE_EVERY=4096 \
  TRAIN_FRAC=0.98 \
  NUM_BLOCKS=1 \
  BRANCH_TEMPORAL_MODE=current \
  RUN_SPECS=$'blocks1_resid12_e6_c8t1_r3_current_1b 12 6.0 8 1 1 -3.0 0.003 100 1500 0.0003 8192 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

## Best Current Calls

Best pure-quality contender:

- `blocks0_resid10_e12_h7000_1b`

Best quality-speed tradeoff on the 5090:

- `blocks0_resid12_e6_c8t1_r3_current_1b`
- reason:
  - about `60%` faster than `blocks0_resid12_e10...`
  - only about `0.02495` bpb worse
  - much cleaner systems headroom than the wider radical variants

Most likely to transfer cleanly to `1x H100`:

- `blocks1_resid12_e6_c8t1_r3_current_1b` as the structural guardrail candidate
- `blocks0_resid10_e12_h7000_1b` as the pure-quality candidate

Why that split:

- `blocks1_resid12_e6...` preserves one frozen amplifier block and now has a completed `1B` result that is slightly better than the matching `blocks0 12x6` anchor.
- `blocks0_resid10_e12_h7000...` is now the best local quality point after schedule tuning, so it is the right pure-quality transfer candidate.

## Findings Likely To Be 5090-Specific

- absolute throughput numbers
- absolute memory headroom numbers
- compile warmup economics
- local interactions with `TORCH_BLAS_PREFER_CUBLASLT=1`

## Findings Less Likely To Be 5090-Specific

- the capped frozen-spec default was invalid and materially changed conclusions
- extra frozen amplifier depth did not earn its bytes locally
- controller-up/spec-down reallocation is a stronger direction than the old moderate `blocks3` frontier
- `carry=8` and `bptt=1` are the clean current defaults
- the first lag-heavy frozen temporal variants lost clearly to `current`

## Code Improvements Vs Pure Hyperparameter Findings

Code improvements:

- W&B logging is now structured cleanly for this family
- exact environment and runtime metadata are saved per run
- artifact accounting includes the trainable int8-zlib payload
- gradient checkpointing is available for larger recurrent controllers
- the sweep harness now supports smaller local microbatches plus derived `grad_accum` from a fixed effective-step-token contract
- the full-spec rerun flow is restart-safe and summary-safe

Pure hyperparameter / architecture findings:

- `blocks0` is the best corrected structure so far
- `10 x 12.0` with `h7000` is now the current pure-quality leader at `1B`
- `12 x 10.0` with `h7000` remains the strongest close alternate shape
- `16 x 8.0` is now the strongest larger-controller checkpointed point at `1B`
- `12 x 6.0` remains the quality-speed anchor at `1B`
- `blocks1 12 x 6.0` is the best current nonzero-amplifier guardrail point and slightly beat the matching `blocks0 12x6` run at `1B`
- naive lag-heavy temporal variants are not winning
- first schedule evidence says the current default `lr_hold_steps=1500` is too short for the top radical `blocks0` controllers on the `512M` screen
- current working schedule defaults are `3500` for the `512M` screen and `7000` for the `1B` contract

## Regression-To-Transformer Guardrail

Current evidence still says we are not drifting back into a transformer-shaped local optimum.

- best moves are recurrent-controller scaling and frozen-spec simplification
- no attention was added
- no token-token mixing was added
- the trainable core remains a parallel minGRU stack

The real risk to watch is not "accidentally rebuilding a transformer."
It is:

- letting the controller absorb too much of the modeling burden because the frozen side is too weak
- over-trusting `blocks0` before the checkpointed larger stress point and schedule sweeps finish

That is why the next confirmation round should compare `blocks0` against a live `blocks1` guardrail, not just compare one `blocks0` width setting against another.
That is also why schedule sweeps should start on the two best `blocks0` points, then circle back to `blocks1` as a transfer-control condition.

## Unresolved Questions

- Which of the top contenders responds best to schedule tuning?
- Does the same tuned hold help the `blocks1` guardrail family?
- Does `blocks1 10x12 h7000` become a viable transfer candidate, or does `blocks1` still lag too far behind the tuned `blocks0` winners?
- Does `blocks1 12x6` keep its slight edge over `blocks0 12x6` once both get a tuned schedule instead of the inherited default?
- Does longer context help the `12x10` / `10x12` pair more than the cheaper `12x6` anchors?
