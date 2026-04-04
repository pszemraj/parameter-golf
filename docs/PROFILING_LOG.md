# Profiling Log

Last updated: 2026-04-04 18:29 EDT

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

## 2026-04-04 — Fine HGDN subrange attribution on the winning local variant (`rtx4070_phase1_convcontig_subranges`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_convcontig_subranges/`
- synthesized analysis: `profiles/rtx4070_phase1_convcontig_subranges/analysis/analysis.md`
- commit at run time: `c0ed0ae`
- analysis/tooling normalization commit: `8f0c626`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed winning config:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_GATES_FP32=1`
  - `GDN_OUTPUT_NORM_FP32=1`
  - q/k/v convs all enabled
- purpose:
  - replace coarse HGDN ranges with finer subranges so the next optimization
    target is based on specific code paths rather than whole-block labels

### Main findings

The subrange pass changed the next-step decision materially.

Under the current winning local variant, the most useful trainer-eager HGDN
sub-buckets are:

- `gdn.recurrence`: `180.70 ms`
- `gdn.q_conv`: `106.75 ms`
- `gdn.k_conv`: `109.86 ms`
- `gdn.v_conv`: `106.40 ms`
- `gdn.output_norm`: `93.83 ms`
- `gdn.q_norm`: `50.53 ms`
- `gdn.k_norm`: `49.46 ms`
- `gdn.output_gate_mul`: `25.32 ms`
- `gdn.output_gate_proj`: `23.32 ms`
- `gdn.g_proj`: `13.23 ms`
- `gdn.g_pointwise`: `6.66 ms`
- `gdn.beta_proj`: `6.28 ms`

Interpretation:

- the remaining cost is **not** primarily the old coarse `gdn.gates` bucket
- once first-call artifacts are separated out, the gate pointwise path is much
  smaller in the real training step than the earlier bare-GDN view suggested
- the major remaining HGDN costs are now:
  - recurrence
  - the three conv paths together
  - q/k normalization
  - output normalization

### Important caution

The fine-subrange pass also showed that bare-GDN microprofiles can overstate
first-call effects.

Examples:

- bare `gdn.g_pointwise`: `44.69 ms`
- trainer-eager `gdn.g_pointwise`: `6.66 ms`
- bare `gdn.q_conv`: `11.92 ms`
- trainer-eager `gdn.q_conv`: `106.75 ms` across `128` calls, with `k/v` now in
  the same ballpark

Decision:

- use bare-GDN hotpaths to find code-path ownership
- use trainer-eager totals to rank remaining payoff on the real training step

### Decision

The next local optimization pass should target **conv-path consolidation**, not
another precision toggle.

Reason:

- recurrence is still the single largest HGDN sub-bucket
- but the three conv paths together are materially larger than recurrence
- they are followed by depthwise-conv backward and `_conv_depthwise2d`
- that makes the highest-value next semantics-preserving experiment:
  - pack q/k/v depthwise causal conv into one shared conv path
  - preserve the current all-path contiguous layout
  - then remeasure recurrence, conv, and copy/mul spillover

What moves down the list after this run:

- more coarse fp32 on/off gate experiments
- q/k-only contiguity experiments
- `v_conv` ablation as the next default step

## 2026-04-04 — Corrected packed qkv conv follow-up (`rtx4070_phase1_packedqkv_fix1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_packedqkv_fix1/`
- structured comparison:
  `profiles/rtx4070_phase1_packedqkv_fix1/compare_vs_rtx4070_phase1_convcontig_subranges/comparison.md`
- packed-fix commit at run time: `b17b80d`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- candidate:
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- purpose:
  - re-run the packed q/k/v depthwise-conv idea after fixing the earlier
    contiguity bug so the candidate is judged on real trainer data rather than
    on a broken layout path

### Main findings

The corrected packed path fixed the original structural flaw.

Boundary result:

- q/k/v are now contiguous after `conv_qkv`
- contiguity is preserved through:
  - `norm_qkv`
  - `recurrence_inputs`
  - `output_proj_input`

That repaired the earlier recurrence/layout regression. The isolated hotpaths
did improve:

- bare GDN total self-device time:
  - baseline: `171.05 ms`
  - candidate: `166.76 ms`
  - delta: `-4.29 ms` (`-2.51%`)
- hybrid forward/backward total self-device time:
  - baseline: `1273.39 ms`
  - candidate: `1162.71 ms`
  - delta: `-110.69 ms` (`-8.69%`)
- hybrid optimizer total self-device time:
  - baseline: `203.39 ms`
  - candidate: `176.65 ms`
  - delta: `-26.74 ms` (`-13.15%`)

But the full trainer-eager step still lost:

- trainer total self-device time:
  - baseline: `25797.53 ms`
  - candidate: `26082.79 ms`
  - delta: `+285.26 ms` (`+1.11%`)

### Why it still lost

The packed candidate bought the expected HGDN-path wins:

- trainer `gdn.recurrence`: `180.70 -> 173.34 ms`
- trainer `gdn.q_norm`: `50.53 -> 48.72 ms`
- trainer `gdn.k_norm`: `49.46 -> 48.99 ms`
- hybrid-fwd-bwd `aten::copy_`: `14.25 -> 11.99 ms`

But those gains were offset by new packing overhead and a slightly worse conv
backward path:

- trainer `gdn.qkv_conv_packed`: `+405.74 ms`
- trainer `aten::cat`: `16.45 -> 99.91 ms`
- trainer `aten::convolution_backward`: `159.30 -> 173.79 ms`
- trainer `aten::_conv_depthwise2d`: `125.02 -> 141.34 ms`
- trainer `aten::copy_`: `776.28 -> 788.81 ms`
- trainer `aten::mul`: `1003.44 -> 1011.34 ms`

Interpretation:

- the packed-conv idea itself is not dead
- but packing only at the conv stage is not enough
- the current implementation still pays too much to build the packed q/k/v
  tensor before the conv, and that cost comes back at trainer scope

### Decision

Do not promote the current packed qkv conv path to H100.

Current judgment:

- fixed packed qkv conv is a **hotpath win but trainer loss**
- keep the contiguity repair in the code because it makes the experiment valid
- reject the current packed-conv-only variant as a local winner

### Next target

If packing is revisited, it needs the next structural step:

1. remove the `aten::cat` tax by emitting packed q/k/v directly from projection
2. then remeasure packed projection + packed conv as one experiment

Absent that, the active default local winner remains:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
