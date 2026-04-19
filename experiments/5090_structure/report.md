# 5090 Structure Report

## Status
- Completed the corrected full-spec structure rerun on `1x RTX 5090`.
- W&B project: `pg-core-amp`
- W&B group: `fullspec_round1`
- Fixed controller contract for every point:
  - `core_layers=5`
  - `core_expansion=2.0`
  - `residual_core=1`
  - `carry_chunks=16`
  - `bptt_chunks=2`
- Fixed training contract:
  - `seq_len=512`
  - `batch_size=256`
  - `num_steps=192`
  - `planned_train_tokens=50,331,648`
  - `val_every=64`
  - `val_steps=8`
  - `compile=0`
- Corrected frozen-spec contract:
  - no `SPEC_MAX_TOKENS` cap
  - full available local `fineweb_train_*.bin` shard set
  - exact `val_bpb`

## Corrected Full-Spec Results

| run | structure change | best val_bpb | steady tok/s | gzip(spec.pt) | artifact estimate bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| `blocks0` | `num_blocks=0`, 12 branches, full readout | `2.4859076582` | `3,336,299` | `1,830,185` | `2,435,639` |
| `blocks3` | `num_blocks=3`, 12 branches, full readout | `2.4865782548` | `1,837,963` | `4,370,119` | `4,975,876` |
| `readout256` | `num_blocks=9`, 12 branches, `readout_rank=256` | `2.4867157200` | `959,199` | `9,219,818` | `9,825,985` |
| `blocks6` | `num_blocks=6`, 12 branches, full readout | `2.4867944734` | `1,267,624` | `6,909,998` | `7,516,019` |
| `readout128` | `num_blocks=9`, 12 branches, `readout_rank=128` | `2.4875669087` | `964,897` | `8,907,930` | `9,514,139` |
| `branches8_pow2` | `num_blocks=9`, branches `1,2,4,8,16,32,64,128` | `2.4885888269` | `1,319,498` | `4,946,012` | `5,551,928` |
| `blocks9` | `num_blocks=9`, 12 branches, full readout | `2.4891548818` | `966,778` | `9,449,486` | `10,055,452` |

Ranking summary:
- `blocks0` is the best quality point and the best throughput point.
- `blocks3` is the best non-zero-amplifier point.
- Every heavier `9`-block variant lost to both `blocks0` and `blocks3`.

## What The Knobs Actually Bought

Amplifier depth:
- `num_blocks=0` beat `num_blocks=3` by about `0.00067` bpb while running about `1.82x` faster.
- `num_blocks=6` and `num_blocks=9` were strictly worse than `blocks3` on both quality and throughput.
- The current frozen amplifier stack is not carrying its weight locally.

Branch count:
- Comparing only the `9`-block family, the curated 8-lag power-of-two set beat the old 12-branch full-readout baseline by about `0.00057` bpb.
- It also ran about `36%` faster and cut artifact estimate by about `4.50 MB`.
- That is a useful finding, but it still did not beat the much simpler `blocks0` or `blocks3` points.

Readout compression:
- Within the `9`-block family, `readout_rank=256` beat the old full-readout baseline by about `0.00244` bpb.
- `readout_rank=128` also beat full readout, but trailed `rank=256` by about `0.00085` bpb.
- Readout compression looks like a genuine structural improvement for the overbuilt `9`-block path, but it does not rescue that path enough to become the new default.

Artifact pressure:
- Even the heaviest corrected point stayed under the 16 MB artifact cap.
- The best quality point, `blocks0`, also left the most headroom: about `13.56 MB`.
- The main local cost of extra frozen structure is wasted quality and throughput, not artifact disqualification.

## Direct Answers

Which structural knobs are worth their bytes?
- On current local evidence, none of the extra amplifier depth is worth its bytes.
- The only structural compression knob that looked genuinely good was `readout_rank=256` inside the `9`-block family.
- The 8-branch power-of-two set is a decent quality/size tradeoff if the user specifically wants a non-zero frozen stack and lower frozen bytes.

Which are worth their compute?
- `blocks0` is the best compute use.
- If the user insists on keeping non-zero amplifier depth alive for later exploration, `blocks3` is the only plausible depth point.
- `blocks6` and `blocks9` are not worth their compute on this 5090 screen.

Is the old `12`-branch / `9`-block / full-readout structure justified locally?
- No.
- It is the worst quality point in the corrected structure screen and one of the slowest.

Which structure should become the default for subsequent controller sweeps?
- Best empirical default: `blocks0`, `12` branches, full readout.
- Best non-zero-amplifier fallback: `blocks3`, `12` branches, full readout.
- Best compressed heavy-stack fallback: `blocks9`, `12` branches, `readout_rank=256`.

## Regression-To-Transformer Guardrail

This corrected sweep pushes away from transformer-like reflexes rather than toward them.
- More frozen stack depth did not help.
- The best point was the leanest frozen structure.
- The next search should keep investing in the parallel recurrent controller or in more meaningful frozen temporal taps, not in piling up more repeated frozen blocks.

## Exact Commands

Corrected full-spec structure rerun:

```bash
conda run -s --name train python tools/run_core_amp_fullspec_reruns.py --family structure_round1
```

Direct underlying structure sweep root:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_BLAS_PREFER_CUBLASLT=1 \
WANDB=1 \
WANDB_PROJECT=pg-core-amp \
SKIP_DONE=1 \
COMPILE=0 \
TRAIN_FRAC=0.98 \
DATA_PATH=/home/pszemraj/workspace/projects/parameter-golf/data/datasets/fineweb10B_sp1024 \
MODEL_ROOT=/home/pszemraj/workspace/projects/parameter-golf/experiments/5090_structure/fullspec_round1 \
PRESET=structure_default \
RUN_SPECS=$'blocks0 1,2,3,4,6,8,12,16,24,32,48,64 0 0\n\
blocks3 1,2,3,4,6,8,12,16,24,32,48,64 3 0\n\
blocks6 1,2,3,4,6,8,12,16,24,32,48,64 6 0\n\
blocks9 1,2,3,4,6,8,12,16,24,32,48,64 9 0\n\
branches8_pow2 1,2,4,8,16,32,64,128 9 0\n\
readout256 1,2,3,4,6,8,12,16,24,32,48,64 9 256\n\
readout128 1,2,3,4,6,8,12,16,24,32,48,64 9 128' \
conda run -s --name train python tools/run_core_amp_sweep.py structure
```

## Immediate Next Step

The highest-value follow-up is not more frozen stack.
- Let the corrected controller queue finish on top of the better structure evidence.
- Use `blocks0` as the empirical best local default.
- Keep `blocks3` alive only as the best non-zero-amplifier checkpoint for later transfer testing.
