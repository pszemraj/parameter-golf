# Profiling Log

Last updated: 2026-04-04 17:35 EDT

This file records profiler-driven checkpoints that should survive beyond the raw
artifacts under `profiles/`.

## 2026-04-04 — Local HGDN phase-1 attribution (`rtx4070_phase1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1/`
- synthesized analysis: `profiles/rtx4070_phase1/analysis/analysis.md`
- commit at run time: `6992e85`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- hybrid operating point: `16L x 384d`, `GDN_RATIO=1`, `MLP_MULT=3.25`, `seq=2048`
- sequential workflow:
  - CUDA preflight
  - bare GDN hotpath
  - HybridGPT forward/backward hotpath
  - optimizer-only hotpath
  - full trainer eager profile
  - bucket attribution + boundary audit

### Main findings

The first useful result is that the major step-level `aten::copy_` cost is not a
bare-GDN kernel issue.

Observed `aten::copy_`:

- bare GDN: `1.15 ms`
- hybrid forward/backward: `11.84 ms`
- optimizer only: `0.26 ms`
- full trainer eager step: `795.81 ms`

Observed `aten::mul`:

- bare GDN: `0.36 ms`
- hybrid forward/backward: `13.40 ms`
- optimizer only: `2.74 ms`
- full trainer eager step: `1047.57 ms`

Interpretation:

- `copy_` is mostly a full-step integration/training-shell problem, not a bare
  recurrence problem.
- `mul` is also mostly a hybrid-integration plus training-shell issue.
- the genuine HGDN model-path hotspots are still:
  - `gdn.recurrence`
  - `gdn.norm_qkv`
  - `gdn.conv_qkv`
  - `gdn.output_gate`

### Boundary audit

The boundary audit is the most actionable part of the run.

What it showed:

- after `w_q/w_k/w_v`, q/k/v are contiguous `(B, T, D)`
- after `q_conv/k_conv/v_conv`, q/k/v become non-contiguous with stride
  `(786432, 1, 2048)`
- that non-contiguous layout persists through:
  - `norm_qkv`
  - `recurrence_inputs`
- recurrence output `o` becomes contiguous again before the output-gate path

Interpretation:

- the most concrete next target is the **conv-to-recurrence layout path**
- this is stronger evidence than generic “copy is high” reasoning because it
  identifies an exact boundary where layout quality degrades and then persists
  into FLA inputs

### Decision

The next semantics-preserving optimization pass should target:

1. post-conv q/k/v layout cleanup before recurrence
2. only after that, adjacent HGDN glue around `norm_qkv` and `output_gate`

What should not be the first target from this run:

- generic trainer-wide `copy_` chasing without boundary evidence
- optimizer work
- GEMM-focused tuning

### Notes

- this run was intentionally local-only and eager for attribution
- H100 confirmation should happen only after a local conv-to-recurrence layout
  intervention shows a measurable win

## 2026-04-04 — Local conv-output layout variant (`rtx4070_phase1_convcontig`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_convcontig/`
- structured comparison: `profiles/rtx4070_phase1_convcontig/compare_vs_rtx4070_phase1/comparison.md`
- commit at run time: `82be69c`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- one change enabled:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- everything else matched the earlier `rtx4070_phase1` bundle

### Main findings

This is the first boundary variant that produced a clear local win rather than
just moving time around.

Boundary result:

- q/k/v now stay contiguous from `conv_qkv` through `norm_qkv` into
  `recurrence_inputs`
- the old non-contiguous stride `(786432, 1, 2048)` becomes contiguous
  `(786432, 384, 1)` at the conv output

Trainer-eager profiler result:

- total trainer self-device time:
  - baseline: `25,990.59 ms`
  - candidate: `25,258.00 ms`
  - delta: `-732.59 ms` (`-2.82%`)
- targeted trainer buckets:
  - `aten::copy_`: `795.81 -> 776.98 ms`
  - `aten::mul`: `1047.57 -> 1000.68 ms`
  - `gdn.recurrence`: `268.79 -> 194.16 ms`
  - `gdn.norm_qkv`: `113.18 -> 96.47 ms`
  - `gdn.output_gate`: `145.90 -> 141.58 ms`
  - `gdn.conv_qkv`: `236.67 -> 322.01 ms`

Interpretation:

- the explicit contiguous materialization is paying for itself overall
- the improvement is coming from a cheaper recurrence/norm path and slightly
  lower step-level `copy_` / `mul`
- the tradeoff is that `gdn.conv_qkv` itself becomes more expensive because the
  conv path is now doing the explicit layout materialization once up front

### Decision

`GDN_CONV_OUTPUT_CONTIGUOUS=1` is a real local candidate and should stay in the
experiment surface.

Current judgment:

- promising enough for an eventual 1xH100 confirmation
- not ready to become the default path yet
- next local target should move to the gate/output side, because after the
  layout fix the largest remaining bare-GDN bucket is now `gdn.gates`

### Next target

The next local-first investigation should focus on the remaining HGDN fp32
islands and elementwise glue:

1. `gdn.gates`
2. `gdn.output_gate`
3. any unnecessary dtype promotion around `softplus`, `sigmoid`, and
   `rms_norm(o.float()).to(x.dtype)`

## 2026-04-04 — Follow-up local HGDN experiments against `rtx4070_phase1_convcontig`

Bundles:

- gate precision candidate: `profiles/rtx4070_phase1_gatebf16/`
- output-norm precision candidate: `profiles/rtx4070_phase1_outputnormbf16/`
- q/k-only contiguous candidate: `profiles/rtx4070_phase1_qkcontig/`
- `v_conv` ablation candidate: `profiles/rtx4070_phase1_vconv0/`
- commit at run time: `318bc22`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- all comparisons used `rtx4070_phase1_convcontig` as the baseline
- runs were executed sequentially:
  - CUDA preflight
  - hotpath GDN forward/backward
  - hotpath hybrid forward/backward
  - compare against the structured conv-contiguous baseline

### Candidate matrix

1. `GDN_GATES_FP32=0`
2. `GDN_OUTPUT_NORM_FP32=0`
3. `GDN_Q_CONV_OUTPUT_CONTIGUOUS=1`, `GDN_K_CONV_OUTPUT_CONTIGUOUS=1`, `GDN_V_CONV_OUTPUT_CONTIGUOUS=0`
4. `GDN_USE_V_CONV=0`

### Main findings

None of the four follow-up candidates beat the blanket contiguous baseline.

#### `GDN_GATES_FP32=0`

This was the clearest negative result.

- bare GDN total self-device time:
  - baseline: `177.16 ms`
  - candidate: `255.20 ms`
  - delta: `+78.04 ms` (`+44.05%`)
- main regressions:
  - `gdn.gates`: `52.45 -> 107.01 ms`
  - `gdn.norm_qkv`: `30.05 -> 48.48 ms`
  - `gdn.recurrence`: `31.57 -> 34.77 ms`

Decision:

- keeping the gate softplus path in fp32 is currently the correct choice

#### `GDN_OUTPUT_NORM_FP32=0`

This improved one local bucket but still lost overall.

- bare GDN total self-device time:
  - baseline: `177.16 ms`
  - candidate: `218.98 ms`
  - delta: `+41.82 ms` (`+23.61%`)
- hybrid forward/backward total self-device time:
  - baseline: `1232.48 ms`
  - candidate: `1266.42 ms`
  - delta: `+33.94 ms` (`+2.75%`)
- targeted win:
  - `gdn.output_gate`: `8.29 -> 4.11 ms`
- but offsetting losses:
  - `gdn.norm_qkv`: `34.04 -> 38.57 ms`
  - `gdn.gates`: `39.31 -> 41.70 ms`
  - `aten::copy_`: `11.71 -> 13.74 ms`

Decision:

- the output-norm fp32 island is not the first thing to remove

#### q/k-only contiguous outputs

This was catastrophically worse than making all three conv outputs contiguous.

- bare GDN total self-device time:
  - baseline: `177.16 ms`
  - candidate: `511.72 ms`
  - delta: `+334.56 ms` (`+188.84%`)
- hybrid forward/backward total self-device time:
  - baseline: `1232.48 ms`
  - candidate: `1691.34 ms`
  - delta: `+458.86 ms` (`+37.23%`)
- dominant regression:
  - bare `gdn.recurrence`: `31.57 -> 314.31 ms`
  - hybrid `gdn.recurrence`: `43.99 -> 332.35 ms`

Decision:

- q/k-only contiguity is not a “cheaper version” of the contiguous fix
- if contiguity is forced, it needs to be coherent across q/k/v

#### `GDN_USE_V_CONV=0`

This reduced some conv-related work but still lost overall.

- bare GDN total self-device time:
  - baseline: `177.16 ms`
  - candidate: `199.40 ms`
  - delta: `+22.24 ms` (`+12.55%`)
- hybrid forward/backward total self-device time:
  - baseline: `1232.48 ms`
  - candidate: `1385.00 ms`
  - delta: `+152.53 ms` (`+12.38%`)
- small local wins:
  - hybrid `aten::copy_`: `11.71 -> 9.25 ms`
  - hybrid `aten::convolution_backward`: `3.17 -> 2.12 ms`
  - hybrid `aten::_conv_depthwise2d`: `2.52 -> 1.68 ms`
- larger regressions:
  - hybrid `gdn.norm_qkv`: `34.04 -> 50.56 ms`
  - hybrid `gdn.output_gate`: `8.29 -> 11.31 ms`
  - hybrid `gdn.recurrence`: `43.99 -> 48.43 ms`

Decision:

- removing `v_conv` is not a free speed win under the current HGDN path

### Decision

The winning local variant is still:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_GATES_FP32=1`
- `GDN_OUTPUT_NORM_FP32=1`
- `GDN_USE_Q_CONV=1`
- `GDN_USE_K_CONV=1`
- `GDN_USE_V_CONV=1`

What these follow-up runs changed in the plan:

- the next target should **not** be another coarse precision toggle
- the next target should **not** be a blind conv ablation
- the next step should be finer subrange attribution inside:
  - `gdn.gates`
  - `gdn.norm_qkv`
  - `gdn.output_gate`
  - optionally q/k/v conv paths separately

That is the missing measurement needed before the next real optimization pass.
