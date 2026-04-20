# 5090 Final Recommendation

> [!WARNING]
> This note is still provisional.
> The corrected full-spec replay is finished, the main `blocks0/blocks1` `1B` confirmations are now complete, and only the checkpointed `16x8` stress point is still running before schedule sweeps.

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
  - `blocks1 12x6` slightly beat `blocks0 12x6` on the longer budget
- The regression-to-transformer guardrail is still intact:
  - no attention
  - no token-token mixing
  - winners are still parallel minGRU controllers over a frozen statistical basis

## Top 3 Current Contenders

These are the three most useful completed local contenders right now.

1. `blocks0_resid12_e10_c8t1_r3_current_1b`
   - best completed pure-quality point
   - final `val_bpb = 2.2113941366`
   - steady `tok/s = 386,200`
   - trainable params `= 839,129`
   - artifact estimate `= 3,879,987`
   - why it is in:
     - best corrected completed local quality result
     - still comfortably inside the artifact cap

2. `blocks0_resid10_e12_c8t1_r3_current_1b`
   - geometry control and near-tie
   - final `val_bpb = 2.2128156660`
   - steady `tok/s = 384,622`
   - trainable params `= 839,031`
   - artifact estimate `= 3,878,440`
   - why it is in:
     - only about `0.00142` bpb behind `12 x 10.0` on the same `1B` budget
     - same controller mass, nearly same speed
     - strongest evidence that controller geometry matters

3. `blocks1_resid12_e6_c8t1_r3_current_1b`
   - best completed nonzero-amplifier contender
   - final `val_bpb = 2.2356768287`
   - steady `tok/s = 590,071`
   - trainable params `= 505,049`
   - artifact estimate `= 4,102,717`
   - why it is in:
     - keeps one frozen amplifier block alive as a transfer guardrail
     - it slightly beat `blocks0_resid12_e6...` on the same `1B` budget

Important nuance:

- `blocks0_resid12_e6_c8t1_r3_current_1b` is still the cleanest quality-speed anchor:
  - `final val_bpb = 2.2363421409`
  - `steady tok/s = 618,833`
- The checkpointed `16 x 8.0` `1B` run is still in flight, so it is not in the top 3 yet.

## Exact Reproduction Commands

### 1. `blocks0_resid12_e10_c8t1_r3_current_1b`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_structure/fullspec_blocks0_radical_v1/blocks0_resid12_e6_c8t1_r3_current_512m \
  MODEL_ROOT=experiments/5090_controller/fullspec_blocks0_confirm1b_v1 \
  PRESET=controller_default \
  COMPILE=0 \
  VAL_EVERY=512 \
  VAL_STEPS=8 \
  LOG_EVERY=128 \
  LOG_STATE_EVERY=512 \
  SAVE_EVERY=4096 \
  TRAIN_FRAC=0.98 \
  BRANCH_TEMPORAL_MODE=current \
  RUN_SPECS=$'blocks0_resid12_e10_c8t1_r3_current_1b 12 10.0 8 1 1 -3.0 0.003 100 1500 0.0003 8192 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 2. `blocks0_resid10_e12_c8t1_r3_current_1b`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_structure/fullspec_blocks0_radical_v1/blocks0_resid12_e6_c8t1_r3_current_512m \
  MODEL_ROOT=experiments/5090_controller/fullspec_blocks0_confirm1b_v1 \
  PRESET=controller_default \
  COMPILE=0 \
  VAL_EVERY=512 \
  VAL_STEPS=8 \
  LOG_EVERY=128 \
  LOG_STATE_EVERY=512 \
  SAVE_EVERY=4096 \
  TRAIN_FRAC=0.98 \
  BRANCH_TEMPORAL_MODE=current \
  RUN_SPECS=$'blocks0_resid10_e12_c8t1_r3_current_1b 10 12.0 8 1 1 -3.0 0.003 100 1500 0.0003 8192 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 3. `blocks1_resid12_e6_c8t1_r3_current_1b`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_controller/fullspec_blocks1_radical_v1/blocks1_resid12_e6_c8t1_r3_current_512m \
  MODEL_ROOT=experiments/5090_controller/fullspec_blocks1_confirm1b_v1 \
  PRESET=controller_default \
  NUM_BLOCKS=1 \
  COMPILE=0 \
  VAL_EVERY=512 \
  VAL_STEPS=8 \
  LOG_EVERY=128 \
  LOG_STATE_EVERY=512 \
  SAVE_EVERY=4096 \
  TRAIN_FRAC=0.98 \
  BRANCH_TEMPORAL_MODE=current \
  RUN_SPECS=$'blocks1_resid12_e6_c8t1_r3_current_1b 12 6.0 8 1 1 -3.0 0.003 100 1500 0.0003 8192 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

## Best Current Calls

Best pure-quality contender:

- `blocks0_resid12_e10_c8t1_r3_current_1b`

Best quality-speed tradeoff on the 5090:

- `blocks0_resid12_e6_c8t1_r3_current_1b`
- reason:
  - about `60%` faster than `blocks0_resid12_e10...`
  - only about `0.02495` bpb worse
  - much cleaner systems headroom than the wider radical variants

Most likely to transfer cleanly to `1x H100`:

- `blocks1_resid12_e6_c8t1_r3_current_1b` as the structural guardrail candidate
- `blocks0_resid12_e10_c8t1_r3_current_1b` as the pure-quality candidate

Why that split:

- `blocks1_resid12_e6...` preserves one frozen amplifier block and now has a completed `1B` result that is slightly better than the matching `blocks0 12x6` anchor.
- `blocks0_resid12_e10...` is still the best local quality point, so it deserves confirmation even if it is the more aggressive candidate.

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
- `12 x 10.0` is the current pure-quality leader at `1B`
- `10 x 12.0` remains the strongest geometry control at `1B`
- `12 x 6.0` remains the quality-speed anchor at `1B`
- `blocks1 12 x 6.0` is the best current nonzero-amplifier guardrail point and slightly beat the matching `blocks0 12x6` run at `1B`
- naive lag-heavy temporal variants are not winning

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

## Unresolved Questions

- Does the checkpointed `blocks0_resid16_e8...` `1B` run close the remaining gap to `12x10` / `10x12`?
- Does the checkpointed `blocks0_resid16_e8...` scale better at `1B` tokens than it did at `512M`?
- Which of the top contenders responds best to schedule tuning?
