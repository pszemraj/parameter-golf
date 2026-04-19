# 5090 Controller Report

## Status
- Completed a stronger corrected full-spec `blocks0` radical controller frontier through `12 x 10.0`.
- Completed a fixed-parameter depth-vs-width follow-up, `blocks0_resid10_e12_c8t1_r3_current_512m`.
- Completed a safer depth-leaning follow-up, `blocks0_resid14_e8_c8t1_r3_current_512m`.
- Completed a checkpointed rerun of the previously OOM `blocks0_resid16_e8_c8t1_r3_current_512m` point.
- Completed the first matched-token controller baseline on the original full frozen structure.
- Completed a clean controller follow-up on the new structural front-runner, `blocks3`.
- Completed a four-point controller neighborhood screen on `blocks3`.
- Completed a clean matched-token `bptt` sweep on the two best `blocks3` controller families.
- Completed a clean matched-token `carry` sweep on the two best residual `blocks3` controller families.
- Completed the first controller-up/spec-down reallocation screen on `blocks2`.
- Completed a `1B` confirmation run on the new `blocks2` family.
- Completed the first radical half-million-parameter recurrent controller screen on `blocks2`.
- All completed controller runs in this report used:
  - `compile=0`
  - exact `val_bpb`
  - W&B project `pg-core-amp`
- The earlier `blocks9` and `blocks3` controller screens used matched `planned_train_tokens=50,331,648`.
- The later `blocks2` reallocation screen used matched `planned_train_tokens=536,870,912` inside its own subfamily.
- A first attempt at the neighborhood screen with `compile=1` was discarded before completion because the shorter runs would not all cross the compile trigger at the same point. That batch is intentionally excluded from the evidence below.

## Current Corrected Source Of Truth

The corrected full-spec `blocks0` radical-controller frontier is now the most relevant controller evidence in this report.

Artifact note:
- the older `run_results.json` files for pre-`26438ae` controller runs used the earlier payload estimator
- the artifact estimates listed in this section have been recomputed with the corrected record-style trainable export path so the local frontier is compared on the same byte-counting convention

Frozen structure:
- `12` branches
- `0` amplifier blocks
- full readout
- full available local train-shard set when the shared frozen spec was built
- `gzip(spec.pt) = 1,830,185`

Matched run results:
- `blocks0_resid12_e6_c8t1_r3_current_512m`
  - `core_layers=12`
  - `core_expansion=6.0`
  - `carry_chunks=8`
  - `bptt_chunks=1`
  - final `val_bpb = 2.2979334823`
  - steady `tok/s = 616,452`
  - trainable params `= 505,049`
  - artifact estimate `= 3,231,686`
- `blocks0_resid12_e8_c8t1_r3_current_512m`
  - `core_layers=12`
  - `core_expansion=8.0`
  - `carry_chunks=8`
  - `bptt_chunks=1`
  - final `val_bpb = 2.2859021694`
  - steady `tok/s = 474,391`
  - trainable params `= 672,089`
  - artifact estimate `= 3,544,631`
- `blocks0_resid12_e10_c8t1_r3_current_512m`
  - `core_layers=12`
  - `core_expansion=10.0`
  - `carry_chunks=8`
  - `bptt_chunks=1`
  - final `val_bpb = 2.2777913795`
  - steady `tok/s = 384,214`
  - trainable params `= 839,129`
  - artifact estimate `= 3,855,919`
- `blocks0_resid10_e12_c8t1_r3_current_512m`
  - `core_layers=10`
  - `core_expansion=12.0`
  - `carry_chunks=8`
  - `bptt_chunks=1`
  - final `val_bpb = 2.2794286891`
  - steady `tok/s = 382,789`
  - trainable params `= 839,031`
  - artifact estimate `= 3,854,342`
- `blocks0_resid16_e8_c8t1_r3_current_512m_gc1`
  - `core_layers=16`
  - `core_expansion=8.0`
  - `carry_chunks=8`
  - `bptt_chunks=1`
  - `gradient_checkpointing=1`
  - final `val_bpb = 2.2815471392`
  - steady `tok/s = 273,637`
  - trainable params `= 895,005`
  - artifact estimate `= 3,962,318`

Current corrected result:
- controller-only scaling on the `blocks0` structure improved quality through `12 x 10.0`, but bigger is not automatically better.
- `12 x 10.0` beat `12 x 8.0` by about `0.00811` bpb on the same `512M`-token screening contract.
- `12 x 10.0` beat `12 x 6.0` by about `0.02014` bpb on the same contract.
- in the fixed-parameter comparison, `12 x 10.0` beat `10 x 12.0` by about `0.00164` bpb.
- the checkpointed `16 x 8.0` rerun shows a larger recurrent controller is genuinely viable on the 5090:
  - it finished at `2.2815471392`
  - it beat `12 x 8.0` by about `0.00436` bpb
  - it beat `14 x 8.0` by about `0.00478` bpb
  - it still trailed `12 x 10.0` by about `0.00376` bpb
  - it still trailed `10 x 12.0` by about `0.00212` bpb
- the quality gain came with real systems cost:
  - `12 x 10.0` is about `19%` slower than `12 x 8.0`
  - `12 x 10.0` is about `38%` slower than `12 x 6.0`
  - `16 x 8.0` with checkpointing is about `29%` slower than `12 x 10.0`
  - `16 x 8.0` with checkpointing is about `56%` slower than `12 x 6.0`
- the two `16 x 8.0` outcomes are a useful systems result:
  - without checkpointing: immediate OOM after the first step on the fixed contract
  - with checkpointing: stable completion at only about `5.24 GiB` peak reserved
- `10 x 12.0` still shows the new frontier is not just a raw-parameter story:
  - it ran at almost the same speed as `12 x 10.0`
  - it used almost the same controller byte budget
  - it still finished slightly worse, so controller geometry matters
- the strongest current architectural conclusion is still structural, not transformer-like:
  - we are scaling a parallel minGRU controller, not reintroducing attention
  - the current learned amplifier blocks look weaker than the frozen lag/readout basis plus the recurrent controller
  - the real risk is controller dominance over a too-static frozen side, not regression toward a transformer
- the main caution flag on the checkpointed `16 x 8.0` run is recurrent-layer concentration:
  - the top residual gate climbed to about `0.463`
  - the top state norm reached roughly `153`
  - that is a parallel-RNN saturation pattern worth tuning, not evidence that attention is needed

## Current Evidence

### Baseline on full `blocks9` structure

Frozen structure:
- `12` branches
- `9` blocks
- full readout
- `spec_max_tokens=5,000,000`

Matched run results:
- `plain3_e20`
  - `core_layers=3`
  - `core_expansion=2.0`
  - `residual_core=0`
  - `carry_chunks=8`
  - `bptt_chunks=1`
  - `384` steps
  - final `val_bpb = 2.5171747775`
  - steady `tok/s = 1,042,359`
  - peak allocated memory `= 11,576 MiB`
- `resid5_e20`
  - `core_layers=5`
  - `core_expansion=2.0`
  - `residual_core=1`
  - `carry_chunks=16`
  - `bptt_chunks=2`
  - `192` steps
  - final `val_bpb = 2.5178967561`
  - steady `tok/s = 966,227`
  - peak allocated memory `= 24,036 MiB`

Result on `blocks9`:
- `plain3_e20` beat `resid5_e20` by about `0.00072` bpb.
- It was also faster and much lighter on memory.

### Follow-up on `blocks3` structure

Frozen structure:
- `12` branches
- `3` blocks
- full readout
- `spec_max_tokens=5,000,000`

Matched run results:
- `plain3_e20`
  - final `val_bpb = 2.5158438607`
  - steady `tok/s = 2,191,340`
  - peak allocated memory `= 6,381 MiB`
- `resid5_e20`
  - final `val_bpb = 2.5175855416`
  - steady `tok/s = 1,857,335`
  - peak allocated memory `= 13,654 MiB`

Result on `blocks3`:
- `plain3_e20` beat `resid5_e20` by about `0.00174` bpb.
- It ran about `18%` faster.
- It used about `2.14x` less peak allocated memory.

### `blocks3` controller neighborhood screen

All runs below used the same frozen structure:
- `12` branches
- `3` blocks
- full readout
- `spec_max_tokens=5,000,000`
- `carry_chunks=8`
- `bptt_chunks=1`

Matched run results:
- `plain3_e25_c8t1`
  - `core_layers=3`
  - `core_expansion=2.5`
  - `residual_core=0`
  - final `val_bpb = 2.5170427183`
  - steady `tok/s = 2,052,454`
  - peak allocated memory `= 6,697 MiB`
- `plain4_e20_c8t1`
  - `core_layers=4`
  - `core_expansion=2.0`
  - `residual_core=0`
  - final `val_bpb = 2.5154361649`
  - steady `tok/s = 2,030,859`
  - peak allocated memory `= 6,814 MiB`
- `resid4_e20_c8t1`
  - `core_layers=4`
  - `core_expansion=2.0`
  - `residual_core=1`
  - final `val_bpb = 2.5133777174`
  - steady `tok/s = 1,970,720`
  - peak allocated memory `= 7,104 MiB`
- `resid4_e25_c8t1`
  - `core_layers=4`
  - `core_expansion=2.5`
  - `residual_core=1`
  - final `val_bpb = 2.5123951758`
  - steady `tok/s = 1,820,432`
  - peak allocated memory `= 7,525 MiB`

Result from the neighborhood screen:
- `plain4_e20` improved on the previous `plain3_e20` anchor by about `0.00041` bpb.
- `plain3_e25` was a regression relative to both `plain3_e20` and `plain4_e20`.
- `resid4_e20` was the first controller point to clearly beat the plain family on the trimmed `blocks3` structure.
- `resid4_e25` is the current single-seed screening leader, beating `plain4_e20` by about `0.00304` bpb and `resid4_e20` by about `0.00098` bpb.
- Residual gates stayed in a narrow non-saturated range, roughly `0.119 -> 0.141`, so the residual path is active without collapsing.

### `blocks3` `bptt` sweep on the leading families

All runs below used:
- `12` branches
- `3` blocks
- full readout
- `spec_max_tokens=5,000,000`
- `carry_chunks=8`
- explicit per-run warmup/hold scaling so schedule stayed matched in token terms

Plain family:
- `plain4_e20_c8t1`
  - `bptt_chunks=1`
  - `warmup_steps=100`
  - `lr_hold_steps=1500`
  - final `val_bpb = 2.5154361649`
  - steady `tok/s = 2,030,714`
  - peak allocated memory `= 6,814 MiB`
- `plain4_e20_c8t2`
  - `bptt_chunks=2`
  - `warmup_steps=50`
  - `lr_hold_steps=750`
  - final `val_bpb = 2.5199320452`
  - steady `tok/s = 2,063,541`
  - peak allocated memory `= 12,064 MiB`
- `plain4_e20_c8t4`
  - `bptt_chunks=4`
  - `warmup_steps=25`
  - `lr_hold_steps=375`
  - final `val_bpb = 2.5207602945`
  - steady `tok/s = 2,078,943`
  - peak allocated memory `= 22,563 MiB`

Residual family:
- `resid4_e25_c8t1`
  - `bptt_chunks=1`
  - `warmup_steps=100`
  - `lr_hold_steps=1500`
  - final `val_bpb = 2.5123951758`
  - steady `tok/s = 1,820,552`
  - peak allocated memory `= 7,525 MiB`
- `resid4_e25_c8t2`
  - `bptt_chunks=2`
  - `warmup_steps=50`
  - `lr_hold_steps=750`
  - final `val_bpb = 2.5175869540`
  - steady `tok/s = 1,854,033`
  - peak allocated memory `= 13,485 MiB`
- `resid4_e25_c8t4`
  - `bptt_chunks=4`
  - `warmup_steps=25`
  - `lr_hold_steps=375`
  - final `val_bpb = 2.5193139021`
  - steady `tok/s = 1,865,468`
  - peak allocated memory `= 25,405 MiB`

Result from the clean `bptt` sweep:
- `bptt=1` won clearly in both families.
- The plain family degraded by about `0.00450` bpb at `bptt=2` and about `0.00532` bpb at `bptt=4`.
- The residual family degraded by about `0.00519` bpb at `bptt=2` and about `0.00692` bpb at `bptt=4`.
- Higher `bptt` increased steady `tok/s` slightly because there were fewer optimizer steps, but that came with sharply worse memory:
  - plain family: `6.8 -> 12.1 -> 22.6 GiB`
  - residual family: `7.5 -> 13.5 -> 25.4 GiB`
- On the current local screening budget, semi-TBPTT is not a quality win.

### `blocks3` `carry` sweep on the best residual families

All runs below used:
- `12` branches
- `3` blocks
- full readout
- `spec_max_tokens=5,000,000`
- `bptt_chunks=1`

`resid4_e20` family:
- `resid4_e20_c8t1`
  - final `val_bpb = 2.5133777174`
  - steady `tok/s = 1,971,311`
- `resid4_e20_c16t1`
  - final `val_bpb = 2.5123756944`
  - steady `tok/s = 1,971,871`
- `resid4_e20_c32t1`
  - final `val_bpb = 2.5137890659`
  - steady `tok/s = 1,967,211`

`resid4_e25` family:
- `resid4_e25_c8t1`
  - final `val_bpb = 2.5123951758`
  - steady `tok/s = 1,820,046`
- `resid4_e25_c16t1`
  - final `val_bpb = 2.5136024590`
  - steady `tok/s = 1,819,138`
- `resid4_e25_c32t1`
  - final `val_bpb = 2.5141047368`
  - steady `tok/s = 1,819,390`

Result from the clean `carry` sweep:
- `carry=16` helped the smaller residual controller by about `0.00100` bpb relative to `carry=8`.
- `carry=32` regressed for that same controller.
- For the larger residual controller, both `carry=16` and `carry=32` regressed from `carry=8`.
- The top two screening points are now effectively tied:
  - `resid4_e20_c16t1 = 2.5123756944`
  - `resid4_e25_c8t1 = 2.5123951758`
- The gap is about `0.00002` bpb, which is far too small to over-interpret at this short budget.
- That means the right next step is longer confirmation, not more screening knobs.

### `blocks2` controller-up / spec-down reallocation

All runs below used:
- `12` branches
- `2` blocks
- full readout
- `spec_max_tokens=5,000,000`
- `branch_temporal_mode=current`
- `carry_chunks=8`
- `bptt_chunks=1`
- `4096` steps = `536,870,912` planned train tokens

Completed results:
- `blocks2_resid5_e25_c8t1_current_512m`
  - `core_layers=5`
  - `core_expansion=2.5`
  - `residual_core=1`
  - final `val_bpb = 2.3986403191`
  - steady `tok/s = 2,023,976`
  - peak allocated memory `= 7,270 MiB`
  - trainable int8 zlib payload `= 75,027`
  - artifact estimate `= 3,452,276`
- `blocks2_resid6_e25_c8t1_current_512m`
  - `core_layers=6`
  - `core_expansion=2.5`
  - `residual_core=1`
  - final `val_bpb = 2.3924393341`
  - steady `tok/s = 1,849,000`
  - peak allocated memory `= 7,881 MiB`
  - trainable int8 zlib payload `= 88,411`
  - artifact estimate `= 3,465,660`
- `blocks2_resid5_e30_c8t1_current_512m`
  - `core_layers=5`
  - `core_expansion=3.0`
  - `residual_core=1`
  - final `val_bpb = 2.3957711310`
  - steady `tok/s = 1,758,943`
  - peak allocated memory `= 7,796 MiB`
  - trainable int8 zlib payload `= 87,556`
  - artifact estimate `= 3,464,805`

Result from the expanded reallocation screen:
- All three `blocks2` points beat the previous `blocks3 + resid4_e25_c8t1 + current` anchor.
- `resid6_e25` is now the best local `512M`-token quality point so far.
- `resid5_e25` is still the best quality/speed tradeoff on the same frontier.
- `resid5_e30` is dominated by `resid6_e25`: it is worse on quality and slower.
- The win came from removing one frozen amplifier block and reinvesting capacity into the recurrent controller, not from adding any attention or token-token mixing.

Diagnostics from the new frontier:
- `resid5_e25` used the fifth recurrent layer more heavily over time, but stayed stable.
- `resid6_e25` improved quality again, but concentrated load more aggressively in the top layer:
  - final top-layer residual gate about `0.631`
  - top-layer state norms briefly above `110`
- `resid5_e30` did not buy its way onto the frontier despite having trainable size comparable to `resid6_e25`.
- That means depth is currently buying more than width on the smaller `blocks2` frozen spec.

### `blocks2` longer-budget confirmation and radical scaling

Confirmed longer-budget result:
- `blocks2_resid6_e25_c8t1_1b`
  - `core_layers=6`
  - `core_expansion=2.5`
  - `residual_core=1`
  - final `val_bpb = 2.3644974368`
  - steady `tok/s = 1,861,968`
  - peak allocated memory `= 7,881 MiB`
  - trainable int8 zlib payload `= 88,254`
  - artifact estimate `= 3,465,503`

First radical scale-up result:
- `blocks2_resid12_e6_c8t1_r3_current_512m`
  - `core_layers=12`
  - `core_expansion=6.0`
  - `residual_core_init=-3.0`
  - trainable params `= 505,099`
  - final `val_bpb = 2.3518192702`
  - steady `tok/s = 560,313`
  - peak allocated memory `= 20,382 MiB`
  - trainable int8 zlib payload `= 405,232`
  - artifact estimate `= 3,782,481`

What changed at the radical scale:
- The first half-million-parameter controller beat every smaller local point on the same `512M` screening budget.
- It beat `blocks2_resid6_e25_c8t1_current_512m` by about `0.04062` bpb.
- It did that without any attention or token-token mixing.
- The deeper controller needed a more closed residual init to stay in a healthier regime:
  - early logged residual gates stayed around `0.05-0.06`
  - the run avoided the immediate top-layer blow-open pattern seen in the smaller `6 x 2.5` controller
- The cost is real:
  - throughput dropped to about `30%` of the smaller `blocks2` frontier
  - peak memory jumped above `20 GiB`

## What This Means

Does more controller depth help?
- Yes, and the strongest new signal comes from trimming the frozen side first.
- On overbuilt `blocks9`, the deeper residual controller package lost.
- On `blocks3`, moving from `plain3_e20` to `plain4_e20` helped modestly, and moving to `resid4_e20` or `resid4_e25` helped more.
- On `blocks2`, depth kept paying:
  - `resid5_e25` beat the old `blocks3` anchor cleanly
  - `resid6_e25` then beat `resid5_e25` by another `0.00620` bpb
  - `resid12_e6` then beat `resid6_e25` by another `0.04062` bpb on the same `512M` screen

Does more controller width via expansion help?
- Not as a generic knob.
- `plain3_e25` lost against `plain4_e20`, so width alone is not rescuing the plain family.
- Inside the residual `4`-layer controller, `expansion=2.5` beat `2.0` by about `0.00098` bpb at the cost of about `8%` throughput.
- On the smaller `blocks2` frozen structure, width alone is not the best use of extra controller parameters:
  - `5 x 3.0` lost to `6 x 2.5`
  - it was also slower than `6 x 2.5`

Is residualization materially improving trainability?
- On `blocks3`, yes.
- The residual path is stable and appears to be earning its keep once the frozen stack is not overbuilt.
- The current best two screening points are both residual controllers.

Is semi-TBPTT helping beyond simple carry?
- No, not on the current local budget.
- A clean matched-token `bptt` sweep showed `bptt=1` beating `2` and `4` in both the plain and residual families.
- Higher `bptt` also caused large memory inflation without offsetting quality gains.

## Best Controller-Only Contender

Current best controller-only contender:
- best pure-quality screening point: `blocks2_resid12_e6_c8t1_r3_current_512m`
- final `val_bpb = 2.3518192702`
- steady `tok/s = 560,313`
- artifact estimate `= 3,782,481`

Best current quality/speed point on the same frontier:
- `blocks2_resid5_e25_c8t1_current_512m`
- final `val_bpb = 2.3986403191`
- steady `tok/s = 2,023,976`
- artifact estimate `= 3,452,276`

Best confirmed longer-budget point on the new family:
- `blocks2_resid6_e25_c8t1_1b`
- final `val_bpb = 2.3644974368`
- steady `tok/s = 1,861,968`
- artifact estimate `= 3,465,503`

Best earlier short-budget `blocks3` controller screen:
- `resid4_e20_c16t1` on `blocks3`
- final `val_bpb = 2.5123756944`
- steady `tok/s = 1,971,871`
- artifact estimate `= 4,197,310`

The strongest simpler anchor remains:
- `plain4_e20_c8t1` on `blocks3`
- final `val_bpb = 2.5154361649`
- steady `tok/s = 2,030,859`

The real takeaway is stronger than the nominal winner:
- inside the earlier `blocks3` screen, the best two residual points are essentially tied
- once we reallocated one frozen block into the recurrent controller, the new `blocks2` frontier separated itself clearly from the old `blocks3` anchor
- within that new frontier, depth is currently a better spend than width
- once we allowed a genuinely large recurrent controller, quality improved by a much larger margin than any of the earlier small-controller sweeps
- longer confirmation runs are still required before making strong transfer claims

## Regression-To-Transformer Guardrail

The latest winner is a larger recurrent controller on a smaller frozen spec, so this guardrail has to stay active.
- The winning controller is still a minGRU stack, not a deep generic stack and not attention.
- The winning move was to remove frozen amplifier depth, not to add more frozen blocks.
- The corrected artifact estimate dropped sharply because the spec got smaller.
- `carry=16` helping `resid4_e20` and the `blocks2` frontier beating the old anchor are both healthier signals than `bptt>1`, because they stay inside the same recurrent family without adding truncated unroll or token-token mixing.
- The half-million-parameter controller is still an RNN, but it raises a different risk: controller compute can now dominate enough that we must watch systems cost carefully.
- That means the project is still in the intended family, but controller creep is still a real thing to watch.

The remaining risk is different:
- the frozen amplifier side may still be too static or weakly temporal
- the controller may start absorbing too much of the modeling burden if we keep adding capacity without proving horizon gains

That is still a better failure mode than accidentally rebuilding a transformer, because it points toward strengthening the frozen temporal role rather than blindly stacking more trainable depth.

## Exact Commands

Full-structure baseline:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/baseline_ab_real \
COMPILE=0 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
RUN_SPECS=$'plain3_e20 3 2.0 8 1 0 -2.0 0.003 1500 0.0003 384 256 512\n\
resid5_e20 5 2.0 16 2 1 -2.0 0.003 1500 0.0003 192 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

`blocks3` controller follow-up:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks3_followup_clean \
NUM_BLOCKS=3 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
TRAIN_FRAC=0.98 \
RUN_SPECS=$'plain3_e20 3 2.0 8 1 0 -2.0 0.003 1500 0.0003 384 256 512\n\
resid5_e20 5 2.0 16 2 1 -2.0 0.003 1500 0.0003 192 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

`blocks3` neighborhood screen:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks3_neighborhood_v1 \
NUM_BLOCKS=3 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
TRAIN_FRAC=0.98 \
RUN_SPECS=$'plain4_e20_c8t1 4 2.0 8 1 0 -2.0 0.003 1500 0.0003 384 256 512\n\
plain3_e25_c8t1 3 2.5 8 1 0 -2.0 0.003 1500 0.0003 384 256 512\n\
resid4_e20_c8t1 4 2.0 8 1 1 -2.0 0.003 1500 0.0003 384 256 512\n\
resid4_e25_c8t1 4 2.5 8 1 1 -2.0 0.003 1500 0.0003 384 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

Clean `bptt` sweep with per-run warmup scaling:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks3_bptt_v2 \
NUM_BLOCKS=3 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
TRAIN_FRAC=0.98 \
RUN_SPECS=$'plain4_e20_c8t1 4 2.0 8 1 0 -2.0 0.003 100 1500 0.0003 384 256 512\n\
plain4_e20_c8t2 4 2.0 8 2 0 -2.0 0.003 50 750 0.0003 192 256 512\n\
plain4_e20_c8t4 4 2.0 8 4 0 -2.0 0.003 25 375 0.0003 96 256 512\n\
resid4_e25_c8t1 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e25_c8t2 4 2.5 8 2 1 -2.0 0.003 50 750 0.0003 192 256 512\n\
resid4_e25_c8t4 4 2.5 8 4 1 -2.0 0.003 25 375 0.0003 96 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

Clean `carry` sweep at fixed `bptt=1`:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_controller/wandb_blocks3_carry_v1 \
NUM_BLOCKS=3 \
SPEC_MAX_TOKENS=5000000 \
COMPILE=0 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
TRAIN_FRAC=0.98 \
RUN_SPECS=$'resid4_e20_c8t1 4 2.0 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e20_c16t1 4 2.0 16 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e20_c32t1 4 2.0 32 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e25_c8t1 4 2.5 8 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e25_c16t1 4 2.5 16 1 1 -2.0 0.003 100 1500 0.0003 384 256 512\n\
resid4_e25_c32t1 4 2.5 32 1 1 -2.0 0.003 100 1500 0.0003 384 256 512' \
conda run -s --name train python tools/run_core_amp_sweep.py controller
```

## Immediate Next Step

The next step should be longer confirmation, not more broad screening.
- Keep the `blocks3` structure fixed.
- Keep `bptt_chunks=1`.
- Confirm the two near-tied leaders on a much longer token budget:
  - `resid4_e20_c16t1`
  - `resid4_e25_c8t1`
- Only after that should schedule refinement resume.

That is the right move because the short screening budget has already done its job: it found the plausible winners, but it is not long enough to separate them cleanly.
