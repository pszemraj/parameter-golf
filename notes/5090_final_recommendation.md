# 5090 Final Recommendation

> [!WARNING]
> This note is still provisional.
> The corrected full-spec replay, the main `1B` controller confirmations, and the tuned-hold `blocks1` follow-up are complete.
> The current frontier change is still based on single-seed runs, so the next source of certainty should come from seed confirmation and disciplined optimization around the new leader rather than from pretending the ranking is already final.

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
  - the schedule was decaying too early under the inherited root default
- The tuned-hold `blocks1` follow-up changed the frontier again:
  - `blocks1 12x10 h7000` is now the best completed single-seed point overall
  - `blocks1 10x12 h7000` also landed ahead of `blocks0 12x10 h7000`
  - `blocks1 12x6 h7000` became the strongest fast anchor
- The regression-to-transformer guardrail is still intact:
  - no attention
  - no token-token mixing
  - winners are still parallel minGRU controllers over a frozen statistical basis

## Top 3 Current Contenders

These are the three strongest completed local contenders right now by quality.

1. `blocks1_resid12_e10_h7000_1b`
   - best completed single-seed overall point
   - final `val_bpb = 2.1831753851`
   - steady `tok/s = 374,227`
   - artifact estimate `= 4,791,890`
   - why it is in:
     - best completed local quality result after the tuned-hold `blocks1` follow-up
     - strongest evidence so far that one frozen amplifier block still earns its bytes in this family

2. `blocks0_resid10_e12_h7000_1b`
   - best completed `blocks0` point
   - final `val_bpb = 2.1878016930`
   - steady `tok/s = 384,758`
   - artifact estimate `= 3,899,812`
   - why it is in:
     - still the leanest top-quality control condition
     - only about `0.00463` bpb behind the new `blocks1 12x10` leader with materially smaller artifact bytes

3. `blocks1_resid10_e12_h7000_1b`
   - third-best completed overall point
   - final `val_bpb = 2.1935951525`
   - steady `tok/s = 372,809`
   - artifact estimate `= 4,790,848`
   - why it is in:
     - it confirms that the `blocks1` gain is not isolated to a single geometry
     - it keeps the wider/shallower controller shape alive inside the one-block family

Important nuance:

- `blocks0_resid12_e10_h7000_1b` is now fourth overall at `2.1954688682`.
- `blocks1_resid12_e6_h7000_1b` is the current fast anchor:
  - `final val_bpb = 2.2132622271`
  - `steady tok/s = 588,014`
- `blocks1_resid12_e6_h7000_1b` improved over the inherited-hold `blocks1 12x6` point by about `0.02241` bpb.
- The new lead is still single-seed, so the right immediate next move is seed confirmation rather than over-celebrating the architecture switch.

## Exact Reproduction Commands

### 1. `blocks1_resid12_e10_h7000_1b`

```bash
/home/pszemraj/miniforge3/envs/train/bin/python \
  /home/pszemraj/workspace/projects/parameter-golf/train_core_amplifier.py \
  /home/pszemraj/workspace/projects/parameter-golf/experiments/5090_schedule/blocks1_hold_confirm1b_v1/blocks1_resid12_e10_h7000_1b \
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
  --lr-hold-steps 7000 \
  --weight-decay 0.001 \
  --hard-loss-gamma 0.5 \
  --hard-loss-cap 5.0 \
  --grad-clip 1.0 \
  --core-layers 12 \
  --core-expansion 10.0 \
  --residual-core 1 \
  --residual-core-init -3.0 \
  --branch-temporal-mode current \
  --branch-temporal-lag-scale 1.0 \
  --val-every 512 \
  --val-steps 8 \
  --save-every 4096 \
  --log-every 128 \
  --log-state-every 512 \
  --train-frac 0.98 \
  --seed 1337 \
  --wandb \
  --wandb-project pg-core-amp \
  --wandb-run-name blocks1_resid12_e10_h7000_1b \
  --wandb-group blocks1_hold_confirm1b_v1 \
  --wandb-tags core_amp,5090,schedule,screening,hold \
  --wandb-watch gradients \
  --wandb-watch-log-freq 25
```

### 2. `blocks0_resid10_e12_h7000_1b`

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

### 3. `blocks1_resid10_e12_h7000_1b`

```bash
/home/pszemraj/miniforge3/envs/train/bin/python \
  /home/pszemraj/workspace/projects/parameter-golf/train_core_amplifier.py \
  /home/pszemraj/workspace/projects/parameter-golf/experiments/5090_schedule/blocks1_hold_confirm1b_v1/blocks1_resid10_e12_h7000_1b \
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
  --lr-hold-steps 7000 \
  --weight-decay 0.001 \
  --hard-loss-gamma 0.5 \
  --hard-loss-cap 5.0 \
  --grad-clip 1.0 \
  --core-layers 10 \
  --core-expansion 12.0 \
  --residual-core 1 \
  --residual-core-init -3.0 \
  --branch-temporal-mode current \
  --branch-temporal-lag-scale 1.0 \
  --val-every 512 \
  --val-steps 8 \
  --save-every 4096 \
  --log-every 128 \
  --log-state-every 512 \
  --train-frac 0.98 \
  --seed 1337 \
  --wandb \
  --wandb-project pg-core-amp \
  --wandb-run-name blocks1_resid10_e12_h7000_1b \
  --wandb-group blocks1_hold_confirm1b_v1 \
  --wandb-tags core_amp,5090,schedule,screening,hold \
  --wandb-watch gradients \
  --wandb-watch-log-freq 25
```

## Best Current Calls

Best pure-quality contender:

- `blocks1_resid12_e10_h7000_1b`
- caveat:
  - current lead is single-seed and still needs a 3-seed confirmation pass

Best quality-speed tradeoff on the 5090:

- `blocks1_resid12_e6_h7000_1b`
- reason:
  - about `57%` faster than the new `blocks1 12x10` leader
  - only about `0.03009` bpb worse
  - much better quality than the older untuned `12x6` anchors without giving up the fast-controller regime

Most likely to transfer cleanly to `1x H100`:

- `blocks1_resid12_e10_h7000_1b` as the current pure-quality candidate
- `blocks0_resid10_e12_h7000_1b` as the lean control candidate

Why that split:

- `blocks1_resid12_e10_h7000...` is now the best completed local point and keeps one nonzero frozen amplifier block in play.
- `blocks0_resid10_e12_h7000...` remains the leanest close control and is only about `0.00463` bpb behind.

## Findings Likely To Be 5090-Specific

- absolute throughput numbers
- absolute memory headroom numbers
- compile warmup economics
- local interactions with `TORCH_BLAS_PREFER_CUBLASLT=1`

## Findings Less Likely To Be 5090-Specific

- the capped frozen-spec default was invalid and materially changed conclusions
- extra frozen amplifier depth did not earn its bytes locally
- controller-up/spec-down reallocation is a stronger direction than the old moderate `blocks3` frontier
- the tuned late hold helps both `blocks0` and `blocks1`
- one frozen amplifier block can matter again once the schedule is no longer handicapping the controller
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

- `blocks1 12 x 10.0` with `h7000` is now the current single-seed pure-quality leader at `1B`
- `blocks0 10 x 12.0` with `h7000` is the lean best-control point
- `blocks1 10 x 12.0` with `h7000` is the third-best pure-quality point and confirms the one-block family is not a one-off
- `16 x 8.0` is now the strongest larger-controller checkpointed point at `1B`
- `blocks1 12 x 6.0` with `h7000` is the current quality-speed anchor at `1B`
- naive lag-heavy temporal variants are not winning
- the inherited default `lr_hold_steps=1500` is too short for both the top `blocks0` and tuned `blocks1` families
- current working schedule defaults are `3500` for the `512M` screen and `7000` for the `1B` contract

## Regression-To-Transformer Guardrail

Current evidence still says we are not drifting back into a transformer-shaped local optimum.

- best moves are recurrent-controller scaling, tuned late decay, and a minimal nonzero frozen amplifier
- no attention was added
- no token-token mixing was added
- the trainable core remains a parallel minGRU stack

The real risk to watch is not "accidentally rebuilding a transformer."
It is:

- over-trusting a new single-seed `blocks1` lead before seed confirmation lands
- letting the controller absorb too much of the modeling burden if the frozen side is simplified past the point where one block still helps

That is why the next confirmation round should be multi-seed on the top `blocks1` and `blocks0` contenders, not another random architecture hop.
That is also why the next optimization lane should center `blocks1`, with `blocks0 10x12` kept as the lean control instead of being treated as the default winner.

## Next Experiments

1. Multi-seed confirmation on the current frontier.
   Minimum set:
   - `blocks1_resid12_e10_h7000_1b`
   - `blocks0_resid10_e12_h7000_1b`
   Preferred set:
   - add `blocks1_resid10_e12_h7000_1b`
   - add `blocks0_resid12_e10_h7000_1b`
   Use three seeds and keep the full `1B` contract fixed.

2. `max_lr` screening on the one-block family.
   - models: `blocks1 12x10`, `blocks1 10x12`
   - contract: `512M`, `h3500`, same effective step tokens
   - grid: `2.5e-3`, `3.0e-3`, `3.5e-3`, `4.0e-3`

3. `min_lr` follow-up on the best `blocks1` shape after the `max_lr` screen.
   - grid: `1e-4`, `2e-4`, `3e-4`, `5e-4`
   - keep `warmup=100` and `weight_decay=1e-3` fixed until this lands

4. Controller-up neighborhood only after LR settles.
   Primary points:
   - `blocks1 14x10`
   - `blocks1 12x12`
   Optional if memory is acceptable with smaller local microbatch plus derived `grad_accum`:
   - `blocks1 14x12`

5. Longer-context sweep only after schedule and controller shape settle.
   - `seq_len in {768, 1024}`
   - compare the best tuned `blocks1` point against `blocks0 10x12` as the lean control

## Unresolved Questions

- Does `blocks1 12x10 h7000` stay ahead across seeds?
- Is `max_lr=3e-3` still optimal once `blocks1` is the frontier family?
- Is `min_lr=3e-4` too high for the one-block structure?
- Does a larger one-block controller neighborhood like `14x10` or `12x12` materially move quality without breaking the artifact or memory story?
- Does longer context help the one-block winner more than the lean `blocks0` control?
