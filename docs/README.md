# HGDN Branch Status

Last updated: 2026-04-02 21:34:28 EDT

Branch: `exp/hgdn`

This file is the branch-local status writeup for the hybrid Gated DeltaNet (HGDN) integration work. It summarizes what has been implemented, what has been measured, what is currently believed, and what should happen next.

## Scope

This branch is not redesigning the model family. The architecture remains an interleaved hybrid of:

- GDN residual blocks
- causal attention residual blocks
- a shared trainer derived from the repo baseline

The work completed here is integration, compile/runtime stabilization, GPU validation, and empirical screening.

## Current Status

The HGDN implementation is integrated, documented, tested, and locally validated on the RTX 4070 Laptop GPU.

What is now true:

- CPU correctness tests pass.
- The FLA GDN kernel path works on CUDA.
- `torch.compile` works on the hybrid path when graph breaks are allowed.
- The default compile strategy is now `COMPILE_STRATEGY=model`.
- A dedicated perf harness exists for fair throughput screens.
- The initial quality comparison at `TRAIN_SEQ_LEN=2048` favors the hybrid over the matched depth-control baseline.

## Important Files

- `model.py`: hybrid HGDN architecture and presets
- `train_gpt_hybrid.py`: hybrid trainer
- `scripts/sweep.sh`: launch helper and perf-harness env contract
- `docs/REFERENCE.md`: architecture/reference notes
- `docs/TODO.md`: deferred follow-ups and break-glass items

## Key Implementation Changes

The branch now includes the following major changes:

- Launch/path fixes so the hybrid helpers work from repo root and from `scripts/`.
- Better tokenizer/data-path errors for fresh clones.
- W&B metric cleanup:
  - history uses `train/*` and `eval/*`
  - final artifact/roundtrip metrics go to summary
- No step-0 eval by default for quality runs.
- `COMPILE=0` support for eager fallback.
- `nn.Module.compile()` for module compilation on torch 2.11.
- `fullgraph=False` on the top-level hybrid compile path so FLA graph breaks do not hard-fail.
- A fair perf harness:
  - `PERF_TIMING=1`
  - `PERF_IGNORE_STEPS=N`
  - `PERF_ISOLATE_COMPILE_CACHE=1`
  - `PERF_SKIP_FINAL_EVAL=1`
- Peak memory stats are reset after compile warmup so warmup does not pollute runtime measurements.
- The default compile strategy was experimentally changed back from selective hybrid compilation to `COMPILE_STRATEGY=model` because the selective path was materially slower on this GPU.

## Compile Findings

### FLA / `torch.compile`

- FLA internally uses `torch.compiler.disable()` on the GDN kernel dispatch path.
- `fullgraph=True` is therefore not viable for the whole hybrid model.
- The correct stable top-level setting is `fullgraph=False`.

### Compile strategy comparison

Measured on the 16-layer hybrid preset:

| Config | Seq | Compile strategy | Step ms | Tokens/s |
|---|---:|---|---:|---:|
| Hybrid tight | 1024 | `model` | `1123.78` | `58,317` |
| Hybrid tight | 1024 | `hybrid` | `1460.98` | `44,857` |
| Hybrid tight | 2048 | `model` | `1159.84` | `56,504` |
| Hybrid tight | 2048 | `hybrid` | `1501.94` | `43,634` |

Conclusion:

- The selective `COMPILE_STRATEGY=hybrid` path is roughly `30%` slower than plain top-level model compilation on the measured HGDN preset.
- It remains available as an experimental knob, but it is not the default.

## Throughput Screening Results

All throughput screens below were run with:

- `NGPU=1`
- `ITERATIONS=50`
- `MAX_WALLCLOCK_SECONDS=0`
- `VAL_LOSS_EVERY=0`
- `TRAIN_LOG_EVERY=10`
- `PERF_TIMING=1`
- `PERF_IGNORE_STEPS=10`
- `PERF_ISOLATE_COMPILE_CACHE=1`
- `PERF_SKIP_FINAL_EVAL=1`
- `COMPILE_STRATEGY=model`

### Baseline depth-control vs hybrid

| Config | Seq | Blocks | Step ms | Tokens/s | Ratio vs depth |
|---|---:|---|---:|---:|---:|
| Depth control | 1024 | `0G+16A` | `812.40` | `80,670` | `1.00x` |
| Hybrid `GDN_RATIO=3` | 1024 | `12G+4A` | `1123.78` | `58,317` | `1.38x` |
| Depth control | 2048 | `0G+16A` | `941.98` | `69,573` | `1.00x` |
| Hybrid `GDN_RATIO=3` | 2048 | `12G+4A` | `1159.84` | `56,504` | `1.23x` |

Interpretation:

- The hybrid penalty shrinks at `2048`, but the pure-attention depth control is still faster on raw throughput.
- The original throughput question is answered: HGDN is viable, but not free.

### Reduced GDN density

`GDN_RATIO=1` means `8G+8A` at 16 layers.

| Config | Seq | Blocks | Step ms | Tokens/s | Ratio vs depth |
|---|---:|---|---:|---:|---:|
| Hybrid `GDN_RATIO=1` | 1024 | `8G+8A` | `1025.46` | `63,909` | `1.26x` |
| Hybrid `GDN_RATIO=1` | 2048 | `8G+8A` | `1094.65` | `59,870` | `1.16x` |

Interpretation:

- `GDN_RATIO=1` is the current operating point.
- It preserves the hybrid architecture while cutting the throughput penalty materially.
- On this GPU/kernel stack, it is the best measured compromise so far.

## Budget Check

The `GDN_RATIO=1` hybrid needed an MLP width bump to use more of the artifact budget.

Measured estimate sweep:

| `MLP_MULT` | Params | Estimated artifact | Headroom |
|---|---:|---:|---:|
| `3.0` | `24,100,032` | `14.98 MB` | `6.2%` |
| `3.125` | `24,689,856` | `15.34 MB` | `3.9%` |
| `3.25` | `25,279,680` | `15.70 MB` | `1.6%` |
| `3.375` | `25,869,504` | `16.07 MB` | `-0.6%` |

Current conclusion:

- `GDN_RATIO=1, MLP_MULT=3.25` is the best local budget-filling hybrid variant found so far.

## Quality Comparison

Both quality runs below used:

- `NGPU=1`
- `ITERATIONS=2000`
- `TRAIN_BATCH_TOKENS=65536`
- `TRAIN_SEQ_LEN=2048`
- `VAL_LOSS_EVERY=500`
- `COMPILE_STRATEGY=model`
- `USE_WANDB=0`

Hybrid run:

- `RUN_ID=quality_hybrid_r1_mlp325_seq2k`
- `GDN_RATIO=1`
- `MLP_MULT=3.25`

Depth run:

- `RUN_ID=quality_depth_seq2k`

### Validation BPB

| Config | 500 | 1000 | 1500 | 2000 | Final int8 roundtrip |
|---|---:|---:|---:|---:|---:|
| Hybrid `GDN_RATIO=1, MLP_MULT=3.25` | `2.9000` | `2.7492` | `2.6148` | `2.5138` | `2.5209` |
| Depth control | `3.0342` | `2.8792` | `2.7660` | `2.6604` | `2.6778` |

### Delta

| Checkpoint | Hybrid improvement over depth |
|---|---:|
| `500` | `0.1342 BPB` |
| `1000` | `0.1300 BPB` |
| `1500` | `0.1512 BPB` |
| `2000` | `0.1466 BPB` |
| Final roundtrip | `0.1569 BPB` |

### Quantization sensitivity

| Config | Pre-roundtrip | Roundtrip | Degradation |
|---|---:|---:|---:|
| Hybrid | `2.5138` | `2.5209` | `+0.0071` |
| Depth | `2.6604` | `2.6778` | `+0.0174` |

Interpretation:

- The hybrid wins cleanly at every measured checkpoint.
- The hybrid also degrades less after the int8 roundtrip than the depth-control baseline.
- The architecture question is currently answered in favor of the hybrid on this local quality comparison.

## Caveat: Actual Artifact Bytes Are Still Underfilled

Even with the stronger `GDN_RATIO=1, MLP_MULT=3.25` hybrid:

- Hybrid actual total artifact bytes: `10,962,876`
- Depth actual total artifact bytes: `9,454,996`

So both families are still substantially under the `16,000,000` byte limit in actual `code + int8_zlib` bytes.

This means:

- The quality result is already useful and meaningful.
- But the branch is not yet at a final submission-grade budget-filled comparison.
- The next work should focus on capacity filling and matched-budget refinement, not re-asking whether HGDN works at all.

## Current Recommendation

If continuing this branch, the current best path is:

1. Keep `COMPILE_STRATEGY=model`.
2. Treat `GDN_RATIO=1, MLP_MULT=3.25, TRAIN_SEQ_LEN=2048` as the primary HGDN operating point.
3. Stop spending time on selective compile unless an H100 result contradicts the 4070 measurements.
4. Use the hybrid as the current winner over the depth-control baseline at this local scale.
5. Spend the next iteration on filling the actual artifact budget while preserving the matched comparison contract.

## Likely Next Work

- Build a more budget-filled hybrid variant using the real `int8+zlib` artifact bytes, not only the proxy estimate.
- Build a correspondingly budget-filled depth-control comparison so artifact size is better matched.
- Re-run the same `2048` quality comparison on those fuller configs.
- If possible later, re-check throughput and compile strategy on the target H100 environment.

## Recent Branch Checkpoints

- `88fc8c6` `perf: default hgdn compile strategy to model`
- `d3bea59` `perf: add fair throughput timing harness`
- `135cd91` `perf: selectively compile hgdn attention path`
- `f5b3733` `perf: use module.compile for hgdn model`
- `5cf3b8f` `fix: allow hgdn compile graph breaks`
