# Profiling Log

Last updated: 2026-04-04 19:25 EDT

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

## 2026-04-04 — Packed qkv projection + packed conv (`rtx4070_phase1_packedqkvproj_fix1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_packedqkvproj_fix1/`
- structured comparison:
  `profiles/rtx4070_phase1_packedqkvproj_fix1/compare_vs_rtx4070_phase1_convcontig_subranges/comparison.md`
- experiment-surface commit at run time: `21f4d73`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- candidate:
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- purpose:
  - follow the packed-conv loss with the missing structural patch
  - remove the q/k/v packing tax at projection time instead of building the
    packed tensor with a late `cat` before the conv

### Main findings

This is the first packed-path candidate that wins on the real local training
step.

The microprofiles alone give a mixed story:

- bare GDN total self-device time:
  - baseline: `171.05 ms`
  - candidate: `208.44 ms`
  - delta: `+37.39 ms` (`+21.86%`)
- hybrid forward/backward total self-device time:
  - baseline: `1273.39 ms`
  - candidate: `1319.72 ms`
  - delta: `+46.33 ms` (`+3.64%`)
- hybrid optimizer total self-device time:
  - baseline: `203.39 ms`
  - candidate: `158.05 ms`
  - delta: `-45.34 ms` (`-22.29%`)

But the trainer-eager result is decisive:

- trainer total self-device time:
  - baseline: `25797.53 ms`
  - candidate: `18282.05 ms`
  - delta: `-7515.47 ms` (`-29.13%`)

Interpretation:

- this candidate is a reminder not to over-rank bare-GDN microprofiles
- the packed projection path helps where the real training step was paying,
  which was not visible from the bare recurrence view alone

### Why it won

Compared with the current conv-contiguous baseline, the trainer-eager trace
shows broad HGDN-path reductions:

- `aten::copy_`: `776.28 -> 560.86 ms`
- `aten::mul`: `1003.44 -> 719.87 ms`
- `gdn.recurrence`: `180.70 -> 123.78 ms`
- `aten::convolution_backward`: `159.30 -> 124.35 ms`
- `aten::_conv_depthwise2d`: `125.02 -> 100.19 ms`
- `gdn.q_norm`: `50.53 -> 34.57 ms`
- `gdn.k_norm`: `49.46 -> 34.82 ms`
- `gdn.output_norm`: `93.83 -> 67.79 ms`
- combined separate trainer conv buckets:
  - baseline `gdn.q_conv + gdn.k_conv + gdn.v_conv`: `323.01 ms`
  - candidate `gdn.qkv_conv_packed`: `256.35 ms`

The specific packed-conv-only failure mode was substantially reduced:

- packed-conv-only trainer `aten::cat`: `99.91 ms`
- packed-projection trainer `aten::cat`: `39.25 ms`

So the original hypothesis was correct:

- packed conv alone was not enough
- packed projection was the missing piece that made packing viable at trainer
  scope

### Boundary read

The candidate boundary audit is also coherent:

- `project_qkv` q/k/v are split views over one packed tensor and are not
  contiguous there
- `conv_qkv` restores contiguous q/k/v outputs
- contiguity is preserved through:
  - `norm_qkv`
  - `recurrence_inputs`
  - `output_proj_input`

This is acceptable because the expensive recurrence-facing path is still seeing
the same good layout as the earlier contiguous baseline.

### Decision

Promote this as the new local winner.

Current local winning HGDN speed candidate:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`

### Next target

This candidate has earned target-hardware confirmation.

Next step:

1. re-check the candidate on 1xH100
2. compare against the current H100 hybrid baseline
3. only then decide whether this becomes the default HGDN perf path

## 2026-04-04 — Keep HGDN control projections in bf16 (`rtx4070_phase1_ctrlbf16_fix1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_ctrlbf16_fix1/`
- structured comparisons:
  - `profiles/rtx4070_phase1_ctrlbf16_fix1/compare_vs_refresh1/comparison.md`
  - `profiles/rtx4070_phase1_ctrlbf16_fix1/compare_vs_fix1/comparison.md`
- experiment-surface commit at run time: working tree after `eb08a00`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
- candidate:
  - fixed baseline plus `GDN_CONTROL_PROJ_FP32=0`
- purpose:
  - test whether the remaining trainer `copy_` and `mul` tax was still coming
    from repeated fp32 → bf16 casts on the tiny HGDN control projections
    `w_a/w_b/w_g`

### Main findings

This is a real local win, and it is stronger than the current packed-path
winner.

Compared with the refreshed packed-path baseline
`rtx4070_phase1_packedqkvproj_refresh1`, the full trainer-eager step improved
substantially:

- trainer total self-device time:
  - baseline: `25957.29 ms`
  - candidate: `16588.06 ms`
  - delta: `-9369.23 ms` (`-36.09%`)

Compared with the original packed-path winner
`rtx4070_phase1_packedqkvproj_fix1`, it still wins materially:

- trainer total self-device time:
  - baseline: `18282.05 ms`
  - candidate: `16588.06 ms`
  - delta: `-1693.99 ms` (`-9.27%`)

This is not just one noisy bucket moving around. The step-level HGDN path got
cheaper in multiple places while preserving the same good recurrence-facing
layout.

### Why it won

Against the refreshed baseline, the trainer-eager trace improved in the exact
family of buckets this experiment targeted:

- `aten::copy_`: `800.91 -> 510.77 ms`
- `aten::mul`: `1027.72 -> 659.15 ms`
- `gdn.qkv_conv_packed`: `366.43 -> 234.58 ms`
- `gdn.recurrence`: `176.97 -> 114.97 ms`
- `aten::convolution_backward`: `177.24 -> 113.86 ms`
- `aten::_conv_depthwise2d`: `143.16 -> 91.58 ms`
- `gdn.q_norm`: `49.55 -> 31.74 ms`
- `gdn.k_norm`: `49.84 -> 31.84 ms`
- `gdn.output_norm`: `97.06 -> 62.05 ms`
- `gdn.output_gate_proj`: `21.70 -> 13.45 ms`

The hotpath views are mixed but still consistent with the interpretation:

- bare GDN:
  - `gdn.g_pointwise`: `66.44 -> 49.85 ms`
  - `gdn.recurrence`: `37.30 -> 34.71 ms`
  - `gdn.output_norm`: `14.63 -> 9.29 ms`
- hybrid forward/backward:
  - `gdn.q_norm`: `57.32 -> 51.48 ms`
  - `gdn.g_pointwise`: `31.76 -> 29.34 ms`
  - `gdn.recurrence`: `46.08 -> 48.46 ms` (slightly worse)
  - `gdn.qkv_conv_packed`: `22.14 -> 25.93 ms` (slightly worse)

Interpretation:

- the local winner is still defined by the *real training step*, not the bare
  recurrence microprofile
- keeping `w_a/w_b/w_g` in bf16 appears to remove a large amount of recast
  overhead that was being paid repeatedly across the training shell

### Boundary read

The important part is that the winning packed-path boundary layout stayed
unchanged:

- `conv_qkv` outputs are still contiguous
- `norm_qkv` outputs are still contiguous
- `recurrence_inputs` remain contiguous bf16 tensors
- recurrence/output-gate/output-proj boundaries are unchanged

So this candidate did not win by giving up the packed-path layout cleanup.
It stacked on top of it.

### Decision

Promote this as the new local winner.

Current local winning HGDN speed candidate:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

### Next target

This candidate has earned target-hardware confirmation, but local work can
still continue first.

The next remaining HGDN-native hotspots on the local winner are:

- `gdn.output_norm`
- `gdn.q_norm`
- `gdn.k_norm`

The first normalization-path follow-up was a manual q/k norm experiment, and it
lost on the real training step. That means the next local follow-up should
shift away from hand-rolled q/k normalization and toward the remaining
output-side path before we spend H100 time.

## 2026-04-04 — Manual q/k norm regression (`rtx4070_phase1_qknorm_fix1`)

Bundle:

- raw artifacts: `profiles/rtx4070_phase1_qknorm_fix1/`
- structured comparison:
  - `profiles/rtx4070_phase1_qknorm_fix1/compare_vs_ctrlbf16/comparison.md`
- experiment-surface commit at run time: working tree after `9cb9f25`

Contract:

- GPU: local RTX 4070 Laptop
- env: `pg`
- fixed baseline:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
- candidate:
  - fixed baseline plus a manual bf16 `q/k` L2 normalization path
- purpose:
  - test whether replacing `F.normalize` with an explicit bf16 pointwise
    normalization path would reduce the remaining `gdn.q_norm` / `gdn.k_norm`
    cost on the local winner

### Main findings

This candidate improved the isolated `q_norm` bucket but lost on the real
trainer step, so it should not remain in the branch.

Bare-GDN and hybrid-forward/backward wins:

- bare `gdn.q_norm`: `54.68 -> 47.86 ms`
- hybrid `gdn.q_norm`: `51.48 -> 46.80 ms`
- hybrid `gdn.recurrence`: `48.46 -> 44.96 ms`

But the full trainer-eager step regressed materially:

- trainer `aten::copy_`: `510.77 -> 791.84 ms`
- trainer `aten::mul`: `659.15 -> 1176.39 ms`
- trainer `gdn.qkv_conv_packed`: `234.58 -> 374.79 ms`
- trainer `gdn.recurrence`: `114.97 -> 183.09 ms`
- trainer `gdn.output_norm`: `62.05 -> 98.79 ms`
- trainer `gdn.q_norm`: `31.74 -> 63.04 ms`
- trainer `gdn.k_norm`: `31.84 -> 63.33 ms`

### Boundary read

The recurrence-facing layout did not improve at all:

- `conv_qkv` stayed contiguous bf16
- `norm_qkv` stayed contiguous bf16
- `recurrence_inputs` stayed contiguous bf16

So the manual q/k path did not buy anything at the boundary level. It only
changed kernel selection inside the normalization path, and the full-step side
effects were strongly negative.

### Decision

Reject this candidate and revert it.

What this means:

- `gdn.q_norm` and `gdn.k_norm` are still important buckets
- but a hand-written bf16 replacement for `F.normalize` is not the right fix
- future work on the norm path should preserve the current operator family
  unless a new experiment shows a step-level win

### Next target

Stay on the current winner:

- `GDN_CONV_OUTPUT_CONTIGUOUS=1`
- `GDN_USE_PACKED_QKV_CONV=1`
- `GDN_USE_PACKED_QKV_PROJ=1`
- `GDN_CONTROL_PROJ_FP32=0`

And move the next local-first investigation to the output-side path:

1. re-screen `GDN_OUTPUT_NORM_FP32=0` on the **current** packed-path winner,
   not the older conv-only baseline
2. if it still loses, stop spending time on dtype-only output-norm changes and
   move to a more structural output-gate consolidation
