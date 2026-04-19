# 5090 Structure Report

## Status
- Completed a matched-token local structure sweep on `1x RTX 5090` with W&B logging in `pg-core-amp`, group `wandb_round1`.
- Kept the controller fixed to the current residual baseline:
  - `core_layers=5`
  - `core_expansion=2.0`
  - `residual_core=1`
  - `carry_chunks=16`
  - `bptt_chunks=2`
- Kept the training contract fixed across the main sweep:
  - `seq_len=512`
  - `batch_size=256`
  - `num_steps=192`
  - `planned_train_tokens=50,331,648`
  - `val_every=64`
  - `val_steps=8`
  - `compile=0`
  - `spec_max_tokens=5,000,000`

## Completed Sweep

| run | structure change | final val_bpb | steady tok/s | artifact estimate bytes |
| --- | --- | ---: | ---: | ---: |
| `blocks3` | `num_blocks=3` | `2.5175855416` | `1,857,977` | `4,195,301` |
| `blocks9` | `num_blocks=9`, full readout, 12 branches | `2.5178967561` | `973,724` | `9,229,159` |
| `readout256` | `readout_rank=256` | `2.5183868337` | `976,255` | `8,997,256` |
| `blocks6` | `num_blocks=6` | `2.5189191614` | `1,276,859` | `6,712,589` |
| `branches8_pow2` | branches `1,2,4,8,16,32,64,128` | `2.5196091662` | `1,342,207` | `4,759,026` |
| `readout128` | `readout_rank=128` | `2.5196172754` | `978,171` | `8,689,070` |

Main sweep summary:
- `blocks3` was the best quality point in the completed sweep.
- `blocks3` beat the full `blocks9` baseline by about `0.00031` bpb while running about `1.91x` faster.
- `blocks3` also cut the local artifact estimate by about `5.03 MB`.
- `blocks6` was worse than both `blocks3` and `blocks9`, which argues against a monotonic “more frozen depth helps” story.

## Earlier Probe

Separate earlier matched screen:
- `blocks0_real`
  - same controller family and train-token budget
  - final `val_bpb = 2.5187936285`
  - steady `tok/s = 3,458,917`
  - artifact estimate `= 1,659,531`

Interpretation of the earlier probe:
- Moving from `blocks0` to `blocks3` improved quality by about `0.00121` bpb.
- That gain is real, so the frozen amplifier is not entirely decorative.
- But the gain from `blocks3` to `blocks9` vanished locally, and the extra blocks were pure cost.

## What The Knobs Are Worth

Amplifier depth:
- `num_blocks=3` is currently the best local structural setting.
- `num_blocks=6` and `num_blocks=9` do not justify their extra compute or bytes on this 5090 screen.
- This is the strongest current evidence that the present frozen stack is overbuilt.

Branch count:
- The curated 8-lag power-of-two set lost about `0.00171` bpb relative to the 12-branch `blocks9` baseline.
- It did save about `4.47 MB` of local artifact estimate and improved throughput by about `38%`.
- Branch count is not free to cut, but it is a far healthier tradeoff than adding more blocks.

Readout compression:
- `readout_rank=256` lost about `0.00049` bpb relative to `blocks9`.
- `readout_rank=128` lost about `0.00172` bpb.
- `rank=256` is the only readout compression point here that still looks plausible.
- The byte savings from `rank=256` were modest because the full model is still dominated by the rest of the frozen stack.

## Answering The Main Questions

Which structural knobs are worth their bytes?
- `num_blocks=3` looks worth keeping.
- A full `9`-block stack does not.
- `readout_rank=256` is plausible if we later need a little more artifact headroom, but it is not the main structural win.
- The 8-branch power-of-two set is a possible quality/size tradeoff, not a new default.

Which are worth their compute?
- `num_blocks=3` is currently the best compute use among the tested frozen depths.
- `num_blocks=9` is not worth its compute locally.
- The 8-branch set saves compute, but the quality hit is larger than the `blocks3` vs `blocks9` delta.

Is the current `12`-branch / `9`-block / full-readout structure justified locally?
- No.
- The completed local sweep does not justify keeping all `9` blocks.
- It also does not justify treating the full structure as the default screening shape going forward.

Which structure should become the default for subsequent controller sweeps?
- Default next structure: `12` branches, `3` blocks, full readout.
- Backup low-byte variant: `12` branches, `3` blocks, `readout_rank=256` only if later artifact pressure matters more than the small quality hit.

## Regression-To-Transformer Guardrail

This sweep was useful partly because it argues against a transformer-like instinct of “just keep more stack.”
- The best result did not come from the deepest frozen path.
- The frozen side seems to benefit from a small amount of amplification, but not from a large stack of repeated blocks.
- That pushes the next search toward using the recurrent controller better, or making the frozen temporal taps more meaningful, rather than recreating depth-heavy baseline behavior.

## Exact Commands

Completed structure sweep:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_structure/wandb_round1 \
PRESET=structure_default \
RUN_SPECS=$'blocks9 1,2,3,4,6,8,12,16,24,32,48,64 9 0\n\
blocks3 1,2,3,4,6,8,12,16,24,32,48,64 3 0\n\
blocks6 1,2,3,4,6,8,12,16,24,32,48,64 6 0\n\
branches8_pow2 1,2,4,8,16,32,64,128 9 0\n\
readout256 1,2,3,4,6,8,12,16,24,32,48,64 9 256\n\
readout128 1,2,3,4,6,8,12,16,24,32,48,64 9 128' \
conda run -s --name train python tools/run_core_amp_sweep.py structure
```

Earlier `blocks0` probe:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
MODEL_ROOT=experiments/5090_structure/blocks0_real \
CORE_LAYERS=5 \
CORE_EXPANSION=2.0 \
RESIDUAL_CORE=1 \
RESIDUAL_CORE_INIT=-2.0 \
SEQ_LEN=512 \
BATCH_SIZE=256 \
NUM_STEPS=192 \
CARRY_CHUNKS=16 \
BPTT_CHUNKS=2 \
VAL_EVERY=64 \
VAL_STEPS=8 \
LOG_EVERY=16 \
LOG_STATE_EVERY=64 \
SPEC_MAX_TOKENS=5000000 \
RUN_SPECS=$'blocks0 1,2,3,4,6,8,12,16,24,32,48,64 0 0' \
conda run -s --name train python tools/run_core_amp_sweep.py structure
```

## Immediate Next Step

The next highest-value follow-up after this report is not “more blocks.”
- It is controller comparison on top of the `blocks3` structure.
- If the plain 3-layer controller still wins there, then the current strongest local contender becomes a small recurrent controller steering a modest frozen multi-timescale front-end, which is much closer to the intended alternative path than the old full `9`-block stack.
