# 5090 Schedule Report

## Status
- Harness ready
- No evidence-backed schedule winner yet
- Schedule sweep is intentionally deferred until the new `blocks0/blocks1` `1B` confirmations finish

## What Was Verified
- Warmup-hold-cosine is the real root schedule path.
- `lr_hold_steps` is wired end to end.
- The trainer now records structured train/eval metrics and final run results, so schedule comparisons no longer depend on ad hoc log parsing.

## Current Evidence
- No evidence-backed schedule winner yet.
- No claim is made yet about:
  - whether hold-then-cosine helps this controller family
  - whether decay is starting too early
  - whether `min_lr` is too high or too low
- The next schedule sweep should be run on the post-confirmation contenders, not on the invalidated older `blocks3` default.

## Next Planned Screening Contract

Use the confirmed winner from the new `blocks0/blocks1` `1B` queue, then run a labeled screening sweep.

Broad schedule screen for one chosen contender:
```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
MODEL_ROOT=./experiments/5090_schedule/broad_screen \
FORCE_DEVICE=cuda \
COMPILE=0 \
RUN_SPECS=$'sched_lr2_w50_h0_min1e5 5 2.0 16 2 1 -2.0 0.002 0 0.00001 192 256 512\n\
sched_lr3_w100_h500_min1e4 5 2.0 16 2 1 -2.0 0.003 500 0.0001 192 256 512\n\
sched_lr3_w100_h1500_min3e4 5 2.0 16 2 1 -2.0 0.003 1500 0.0003 192 256 512\n\
sched_lr4_w200_h2500_min5e4 5 2.0 16 2 1 -2.0 0.004 2500 0.0005 192 256 512' \
WARMUP_STEPS=100 \
WEIGHT_DECAY=0.001 \
bash scripts/sweep_controller.sh
```

- Confirmation rerun for the best schedule point:
```bash
for seed in 1337 2027 4242; do
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  MODEL_ROOT=./experiments/5090_schedule/confirm_seed_${seed} \
  FORCE_DEVICE=cuda \
  COMPILE=0 \
  SEED=$seed \
  RUN_SPECS=$'winner 5 2.0 16 2 1 -2.0 0.003 1500 0.0003 768 256 512' \
  WARMUP_STEPS=100 \
  WEIGHT_DECAY=0.001 \
  bash scripts/sweep_controller.sh
done
```

## Questions This Sweep Should Answer
- Is hold-then-cosine actually helping this recurrent-controller family?
- Are we decaying too early for the chosen local token budget?
- Is the current `min_lr` floor too conservative?
