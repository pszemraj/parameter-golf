# 5090 Final Recommendation

> [!WARNING]
> This note is still provisional.
> The corrected full-spec replay is finished, but the strongest `blocks0/blocks1` contenders still need longer-budget confirmation and then schedule sweeps.

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
- The regression-to-transformer guardrail is still intact:
  - no attention
  - no token-token mixing
  - winners are still parallel minGRU controllers over a frozen statistical basis

## Top 3 Current Contenders

These are the three most useful local contenders right now for the next confirmation round.

1. `blocks0_resid12_e10_c8t1_r3_current_512m`
   - best completed pure-quality point
   - final `val_bpb = 2.2777913795`
   - steady `tok/s = 384,214`
   - trainable params `= 839,129`
   - artifact estimate `= 3,855,919`
   - why it is in:
     - best corrected completed local quality result
     - still comfortably inside the artifact cap

2. `blocks0_resid10_e12_c8t1_r3_current_512m`
   - geometry control and near-tie
   - final `val_bpb = 2.2794286891`
   - steady `tok/s = 382,789`
   - trainable params `= 839,031`
   - artifact estimate `= 3,854,342`
   - why it is in:
     - only about `0.00164` bpb behind `12 x 10.0`
     - same controller mass, nearly same speed
     - strongest evidence that controller geometry matters

3. `blocks1_resid12_e6_c8t1_r3_current_512m`
   - best completed nonzero-amplifier contender
   - final `val_bpb = 2.2983212585`
   - steady `tok/s = 585,746`
   - trainable params `= 505,049`
   - artifact estimate `= 4,093,534`
   - why it is in:
     - keeps one frozen amplifier block alive as a transfer guardrail
     - only about `0.00039` bpb behind the cheaper `blocks0_resid12_e6...` anchor

Important nuance:

- The pure-quality top 3 among completed screening points is actually:
  - `blocks0_resid12_e10...`
  - `blocks0_resid10_e12...`
  - `blocks0_resid16_e8..._gc1`
- I am not elevating the checkpointed `16 x 8.0` run into the main top-3 recommendation yet because:
  - it is slower by about `29%` vs `12 x 10.0`
  - it still loses on quality by about `0.00376` bpb
  - it is a useful stress point, not the cleanest transfer candidate

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
  RUN_SPECS=$'blocks0_resid12_e10_c8t1_r3_current_512m 12 10.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 2. `blocks0_resid10_e12_c8t1_r3_current_512m`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_structure/fullspec_blocks0_radical_v1/blocks0_resid12_e6_c8t1_r3_current_512m \
  MODEL_ROOT=experiments/5090_controller/fullspec_blocks0_controller_v3 \
  PRESET=controller_default \
  COMPILE=0 \
  VAL_EVERY=256 \
  VAL_STEPS=8 \
  LOG_EVERY=64 \
  LOG_STATE_EVERY=256 \
  SAVE_EVERY=2048 \
  TRAIN_FRAC=0.98 \
  BRANCH_TEMPORAL_MODE=current \
  RUN_SPECS=$'blocks0_resid10_e12_c8t1_r3_current_512m 10 12.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

### 3. `blocks1_resid12_e6_c8t1_r3_current_512m`

```bash
env CUDA_VISIBLE_DEVICES=0 TORCH_BLAS_PREFER_CUBLASLT=1 \
  SHARED_SPEC_DIR=experiments/5090_controller/fullspec_blocks1_radical_v1/blocks1_resid12_e6_c8t1_r3_current_512m \
  MODEL_ROOT=experiments/5090_controller/fullspec_blocks1_radical_v1 \
  PRESET=controller_default \
  NUM_BLOCKS=1 \
  COMPILE=0 \
  VAL_EVERY=256 \
  VAL_STEPS=8 \
  LOG_EVERY=64 \
  LOG_STATE_EVERY=256 \
  SAVE_EVERY=2048 \
  TRAIN_FRAC=0.98 \
  BRANCH_TEMPORAL_MODE=current \
  RUN_SPECS=$'blocks1_resid12_e6_c8t1_r3_current_512m 12 6.0 8 1 1 -3.0 0.003 100 1500 0.0003 4096 256 512' \
  conda run -s --name train python tools/run_core_amp_sweep.py controller
```

## Best Current Calls

Best pure-quality contender:

- `blocks0_resid12_e10_c8t1_r3_current_512m`

Best quality-speed tradeoff on the 5090:

- `blocks0_resid12_e6_c8t1_r3_current_512m`
- reason:
  - about `60%` faster than `blocks0_resid12_e10...`
  - only about `0.02014` bpb worse
  - much cleaner systems headroom than the wider radical variants

Most likely to transfer cleanly to `1x H100`:

- `blocks1_resid12_e6_c8t1_r3_current_512m` as the structural guardrail candidate
- `blocks0_resid12_e10_c8t1_r3_current_512m` as the pure-quality candidate

Why that split:

- `blocks1_resid12_e6...` preserves one frozen amplifier block and stays in a healthier controller regime.
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
- the full-spec rerun flow is restart-safe and summary-safe

Pure hyperparameter / architecture findings:

- `blocks0` is the best corrected structure so far
- `12 x 10.0` is the current pure-quality leader
- `10 x 12.0` is the strongest geometry control
- `12 x 6.0` remains the quality-speed anchor
- `blocks1 12 x 6.0` is the best current nonzero-amplifier guardrail point
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
- over-trusting `blocks0` before longer confirmation

That is why the next confirmation round should compare `blocks0` against a live `blocks1` guardrail, not just compare one `blocks0` width setting against another.

## Unresolved Questions

- Does `blocks0_resid12_e10...` keep its edge at `1B` tokens?
- Does `blocks0_resid10_e12...` overtake it on a longer budget?
- Does `blocks1_resid12_e6...` close the gap when given more tokens?
- Does the checkpointed `blocks0_resid16_e8...` scale better at `1B` tokens than it did at `512M`?
- Which of the top contenders responds best to schedule tuning?
