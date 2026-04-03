# HGDN Branch Status

Last updated: 2026-04-02 21:54:53 EDT

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

## Artifact Audit

The old branch-local artifact estimate was wrong. It used a hand proxy based on large-tensor int8 packing plus a fixed compression factor, and it overstated the final submission size by roughly `30-40%`.

The correct byte story has two stages:

- shape-driven int8 packing, which is nearly deterministic for a given state dict layout
- value-driven zlib compression on the serialized quantized payload, which changes materially after training

### Actual trained-run endpoints

From the logged 2k quality runs:

| Config | Raw `state_dict` bytes | `int8+zlib` bytes | Code bytes | Total artifact |
|---|---:|---:|---:|---:|
| Hybrid `GDN_RATIO=1, MLP_MULT=3.25` | `100,338,699` | `10,909,417` | `53,459` | `10,962,876` |
| Depth control `MLP_MULT=3.75` | `100,047,451` | `9,401,537` | `53,459` | `9,454,996` |

### Exact init-state quantization audit

These numbers use the real `quantize_state_dict_int8 -> torch.save -> zlib.compress` path on the same model shapes:

| Config | Params | Int8 payload | Raw quant `torch.save` | Init `int8+zlib` |
|---|---:|---:|---:|---:|
| Hybrid `GDN_RATIO=1, MLP_MULT=3.25` | `25,279,680` | `25,650,944` | `25,750,713` | `8,140,843` |
| Depth control `MLP_MULT=3.75` | `25,193,600` | `25,374,208` | `25,453,817` | `7,271,843` |

Interpretation:

- The int8 payload itself is only about `3.95x` smaller than raw fp32 tensor bytes.
- The large extra shrink happens in zlib on the serialized quantized object.
- Training increases payload entropy, so trained checkpoints compress noticeably worse than init-state ones:
  - hybrid payload-to-zlib: about `3.15x` at init, `2.35x` after training
  - depth payload-to-zlib: about `3.49x` at init, `2.70x` after training

Current conclusion:

- The old proxy table should not be used for budget decisions.
- Future size decisions should use the trainer's real artifact audit fields, not a static heuristic.
- The hybrid is genuinely less compressible than the depth control, so size matching needs to use actual bytes, not only parameter counts.

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

## Time-Matched Reindex

The existing quality comparison is step-matched. The hybrid is slower, so the fair wall-clock question is whether the depth control would erase the gap if given its extra steps.

Using the logged depth checkpoints and a log-linear fit of `eval/bpb` against `train_time_ms`, the predicted depth-control BPB at the hybrid's wall times is:

| Hybrid checkpoint | Hybrid train time | Hybrid BPB | Predicted depth BPB at same wall time | Hybrid advantage |
|---|---:|---:|---:|---:|
| `500` | `567,508 ms` | `2.9000` | `2.9996` | `0.0996` |
| `1000` | `1,134,115 ms` | `2.7492` | `2.8163` | `0.0671` |
| `1500` | `1,701,439 ms` | `2.6148` | `2.7089` | `0.0941` |
| `2000` | `2,272,172 ms` | `2.5138` | `2.6324` | `0.1186` |

Interpretation:

- The hybrid still wins under equal wall time.
- The gap narrows at the `1000` checkpoint, then widens again.
- Even before size matching, the branch is not relying on a misleading equal-step-only win.

## Caveat: Current Quality Win Is Not Yet Size-Matched

Even with the stronger `GDN_RATIO=1, MLP_MULT=3.25` hybrid:

- Hybrid actual total artifact bytes: `10,962,876`
- Depth actual total artifact bytes: `9,454,996`

That means the hybrid had about a `16%` artifact-size advantage in the current comparison. Some unknown fraction of the BPB gap is therefore still "more compressed params."

The next matched depth-control candidate is:

- `depth` with `MLP_MULT=4.7`

Why this setting:

- exact init-state quant audit gives `8,439,594` compressed payload bytes
- applying the observed depth train/init compression factor from `quality_depth_seq2k` projects to about `10,964,745` total bytes including code
- that is effectively matched to the hybrid's `10,962,876` total bytes

This means:

- The quality result is already useful and meaningful.
- But the branch is not yet at a final submission-grade matched-budget comparison.
- The next work should focus on size matching and then wall-clock-aware scaling, not re-asking whether HGDN works at all.

## Current Recommendation

If continuing this branch, the current best path is:

1. Keep `COMPILE_STRATEGY=model`.
2. Treat `GDN_RATIO=1, MLP_MULT=3.25, TRAIN_SEQ_LEN=2048` as the primary HGDN operating point.
3. Stop spending time on selective compile unless an H100 result contradicts the 4070 measurements.
4. Use the hybrid as the current quality winner over the depth-control baseline at this local scale, including after wall-time reindexing.
5. Run the size-matched depth-control follow-up next with `MLP_MULT=4.7`.
6. After that rerun, move to compute-optimal scaling instead of blindly filling the full 16MB.

## Likely Next Work

- Re-run the same `2048` quality comparison with the size-matched depth control (`MLP_MULT=4.7`).
- Add the trainer's new byte-audit fields to any future quality-run summaries when comparing branches.
- After the size-matched rerun, run a small wall-clock-capped HGDN scaling sweep to find the real compute-optimal size.
- If possible later, re-check throughput and compile strategy on the target H100 environment.

## Recent Branch Checkpoints

- `88fc8c6` `perf: default hgdn compile strategy to model`
- `d3bea59` `perf: add fair throughput timing harness`
- `135cd91` `perf: selectively compile hgdn attention path`
- `f5b3733` `perf: use module.compile for hgdn model`
- `5cf3b8f` `fix: allow hgdn compile graph breaks`
