# HGDN Compile / Perf TODO

This file tracks follow-up work that is intentionally not enabled by default in the current hybrid trainer.

## Active next steps

### Strategic path after the current front-end seam stalled

- Main objective:
  - beat or match the attention-only baseline on final wall-clock outcome, not
    defend any one HGDN kernel family
- Current evidence says the architecture is still worth pushing:
  - on the fixed-step 1xH100 quality check, the hybrid reached:
    - `eval/bpb = 2.3587`
    - `final roundtrip = 2.3719`
  - the matched attention-only baseline reached:
    - `eval/bpb = 2.5457`
    - `final roundtrip = 2.5950`
  - that is a real learning-per-step advantage even though the current HGDN
    systems winner is still materially slower on H100
- Practical implication:
  - `h100k20` closed the current post-conv front-end seam as an
    `INTEGRATION_BOTTLENECK`
  - the branch still has several live ways to improve the overall
    HGDN-vs-GPT result
  - the next tranche should stop trying to rescue this exact front-end boundary
    and move to the higher-payoff levers below
- Priority order after the current seam:
  1. compute-optimal HGDN resize on H100
     - best immediate candidates already prepared:
       - `configs/hgdn/retune_trim_layers_14.toml`
       - `configs/hgdn/retune_trim_layers_14_mlp3p5.toml`
       - `configs/hgdn/retune_trim_width_320.toml`
       - `configs/hgdn/retune_balanced_14l_mlp3.toml`
     - rationale:
       - the hybrid already learns faster per step on H100
       - a slightly smaller or better-balanced HGDN may keep most of that
         quality edge while buying back enough throughput to win on wall clock
     - first H100 resize read:
       - `14L x 384d x mlp3.25` is live:
         - `2.4438 -> 2.4243` roundtrip vs `h100k6`
         - `915.10 -> 897.96 ms` last step time vs `h100k6`
         - `OVER_LIMIT -> UNDER_LIMIT`
       - `16L x 320d x mlp3.25` is rejected:
         - worse quality and slower than the reference
       - the original `14L x 384d x mlp3.0` run was invalid because the H100
         helper ignored plain `MLP_MULT`; rerun it after the shell fallback fix
     - immediate next resize step:
       - rerun corrected `14L x 384d x mlp3.0`
       - replicate the live `14L x 384d x mlp3.25` winner once
       - bracket the live winner with nearby MLP variants:
         - `14L x 384d x mlp3.125`
         - `14L x 384d x mlp3.375`
       - include one orthogonal deeper candidate:
         - `15L x 384d x mlp2.75`
       - keep the width-trim branch parked unless later evidence changes
     - batching rule:
       - kernel-seam work should stay narrow
       - resize work should use a wider simultaneous H100 batch when the
         candidates are simple architecture variants around a live winner
  2. norm placement screen
     - compare `pre`, `post`, and `keel`
     - rationale:
       - this is a real learning-dynamics lever, not just a systems trick
       - it can improve optimization depth utilization even if raw step time is
         unchanged
  3. remaining non-seam HGDN systems hotspots
     - gate/output projection work
     - residual shell and compiled `add + mul + unsqueeze` glue
     - recurrence-adjacent work only if the compiled profile says recurrence is
       back to dominating
     - rationale:
       - these targets matter even if the current post-conv split/norm seam is
         closed
  4. finalist-only compile/backend work
     - compile-mode shootout
     - Nsight-guided kernel pass
     - possible Hopper-only backend experiments such as cuLA, but only if the
       profiles say the recurrence backend is the main remaining loss
- Stop rule:
  - do not stop because the branch has reached some arbitrary `kXX`
  - stop a seam when the scoreboard says it is flat, saturated, or still an
    integration bottleneck after one genuinely narrower follow-up
  - then move to the next higher-value HGDN lever rather than relitigating the
    same boundary

### Immediate kernel gate

- Status: promoted.
- Active H100-confirmed winner:
  - `winner-20260405-19`
  - equivalent to:
    - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
    - `GDN_USE_PACKED_QKV_CONV=1`
    - `GDN_USE_PACKED_QKV_PROJ=1`
    - `GDN_CONTROL_PROJ_FP32=0`
    - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
- Why it was promoted:
  - the local 4070 signal was borderline, so it was forced through repeated H100 controls plus repeated candidate perf
  - same-day H100 controls were tight:
    - `904.80 ms`
    - `904.12 ms`
  - repeated H100 custom-backward candidate runs agreed exactly on direction:
    - `853.23 ms`
    - `853.20 ms`
  - confirmed H100 compiled perf delta:
    - `904.46 -> 853.21 ms` (`-5.67%`)
  - H100 eager profile also improved:
    - `1670.55 -> 1643.53 ms` (`-1.62%`)
  - H100 compiled profile improved:
    - `922.04 -> 874.83 ms` (`-5.12%`)
- Laptop-noise caveat:
  - this branch is being screened on a laptop RTX 4070
  - treat small local deltas as screening only; the H100 result is the promotion source of truth
- Next exact step:
  1. use `winner-20260405-19` as the active non-extension HGDN baseline
  2. continue profiler-driven kernel work from this promoted baseline, not from `winner-20260405-11`
  3. do not spend more H100 time on `winner-20260405-11-cuda-output-only` unless its implementation changes materially
- Kernel-work guardrail:
  - the last few results make the practical point clear
  - the remaining wins are not going to come from rearranging Python-side views
    or `.contiguous()` calls
  - the H100 profiles are telling us to go after the actual generated/kernel
    path
  - practical consequence:
    - stop treating Python-side layout reshuffles as primary kernel work on the
      compiled HGDN path
    - target lower-level ATen, Triton, CUDA, or other generated-path changes
      instead
  - use `scripts/hgdn_kernel_scoreboard.py` to make family decisions explicit:
    - meaningful win:
      - `max(5 ms, 0.5% of control_step_ms, 3 * control_noise_ms)`
    - flat band:
      - `max(2.5 ms, 0.25% of control_step_ms, 2 * control_noise_ms)`
    - derived integration signals:
      - `compiled_copy_tax = aten::copy_ + direct_copy_kernel_cuda`
      - `compile_specific_penalty = compiled_hotpath_delta - eager_hotpath_delta`
    - family stop rule:
      - `SATURATED` if the control compiled upper bound is already below the
        meaningful-win threshold
      - stop after repeated `FLAT` or repeated `INTEGRATION_BOTTLENECK` results
  - H100 batching policy:
    - when only one family is genuinely live, prefer one larger same-day H100
      batch instead of multiple tiny consecutive rounds
    - default shape:
      - 2 control perf runs
      - control eager and compiled profiles when attribution still matters
      - candidate preflight
      - candidate eager profile
      - 2 candidate perf runs, or 3 if tighter confidence is worth more than a
        second family
      - candidate compiled profile
    - do not batch a second family unless it is genuinely orthogonal and a
      loss would still teach something independent
  - timing reminder:
    - trainer warmup and compile priming are outside the `MAX_WALLCLOCK_SECONDS`
      training timer
    - validation time is also outside the training timer
    - serialization and final roundtrip eval are outside the training timer
    - the challenge still has a separate external 10-minute evaluation cap
    - use compile-heavy kernels when they help steady-state runtime, but do not
      treat compile time as a reason to ignore runtime or eval regressions
- Latest screened candidate:
  - `winner-20260405-19-cuda-fused-frontend`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_USE_CUDA_FUSED_FRONTEND=1`
  - purpose:
    - keep the promoted packed qkv front-end and custom depthwise backward
    - fuse only the post-conv frontend shell that still shows up after the
      packed-conv weight-backward rewrite
    - preserve the recurrence-facing contiguous contract
  - local result:
    - same-day local baseline:
      - `profiles/rtx5090_phase1_winner20260405_19_r8/`
    - candidate:
      - `profiles/rtx5090_phase1_cuda_fusedfrontend_fix1/`
    - direct comparison:
      - `profiles/rtx5090_phase1_cuda_fusedfrontend_fix1/compare_vs_rtx5090_phase1_winner20260405_19_r8/comparison.md`
    - trainer `ProfilerStep*`:
      - `6154.71 -> 4994.66 ms` (`-18.85%`)
    - trainer buckets moved in the right direction:
      - `DistributedDataParallel.forward`: `1792.51 -> 1520.00 ms`
      - `aten::copy_`: `646.49 -> 197.54 ms`
      - `aten::mul`: `1165.79 -> 834.38 ms`
      - `_PackedQKVFrontendFunctionBackward`: `342.39 ms`
      - `causal_dwconv_weight_backward_kernel_k4`: `190.19 ms`
    - boundary audit stayed clean through `conv_qkv`, `norm_qkv`, and
      `recurrence_inputs`
  - H100 result:
    - reject
    - same-day controls:
      - `849.98 ms`
      - `850.04 ms`
    - candidate:
      - `929.75 ms`
      - `928.82 ms`
      - `931.50 ms`
    - mean delta:
      - `850.01 -> 930.02 ms` (`+9.41%`)
    - scoreboard status:
      - `INTEGRATION_BOTTLENECK`
  - mechanism read:
    - eager got much better, but compiled got worse
    - compiled copy tax improved:
      - `compiled_copy_tax = -46.61 ms`
    - compiled shell rows dominated:
      - `DistributedDataParallel.forward = 1147.61 ms`
      - `_PackedQKVFrontendFunctionBackward = 756.54 ms`
      - `## Call CompiledFxGraph ... = 694.59 ms`
      - `aten::mm = 616.68 ms`
      - `build_grad_preact_kernel = 224.03 ms`
  - current decision:
    - reject the old fused-frontend custom-op boundary on H100
    - keep `winner-20260405-19` active
    - do not send this exact family back to H100 without a material boundary
      change
- Latest screened candidate:
  - `winner-20260405-19-cuda-fused-frontend-lib`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_USE_CUDA_FUSED_FRONTEND_LIB=1`
  - H100 result:
    - reject
    - same-day controls:
      - `853.15 ms`
      - `855.07 ms`
    - candidate:
      - `931.88 ms`
      - `931.62 ms`
      - `929.84 ms`
    - mean delta:
      - `854.11 -> 931.11 ms` (`+9.02%`)
    - scoreboard status:
      - `INTEGRATION_BOTTLENECK`
  - mechanism read:
    - eager `ProfilerStep*` improved:
      - `-1039.88 ms`
    - compiled `ProfilerStep*` worsened:
      - `+308.71 ms`
    - compile-specific penalty:
      - `+1348.59 ms`
    - compiled copy tax improved:
      - `-46.50 ms`
    - main compiled custom rows:
      - `hgdn_cuda_v3::packed_qkv_frontend_backward = 755.78 ms`
      - `causal_dwconv_weight_backward_kernel_k4 = 398.00 ms`
      - `build_grad_preact_kernel = 224.08 ms`
      - `split_norm_from_preact_kernel = 153.26 ms`
      - `hgdn_cuda_v3::packed_qkv_frontend = 292.29 ms`
    - graph-shell tax stayed large:
      - `CompiledFxGraph delta = +316.07 ms`
      - `DDP.forward delta = -2.19 ms`
  - current decision:
    - reject the full packed frontend library family on H100
    - keep the packed frontend math idea only if backward ownership changes
      materially
    - do not send this exact family back to H100 again
- Last screened candidate:
  - `winner-20260405-19-cuda-split-norm-lib`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_USE_CUDA_SPLIT_NORM_LIB=1`
  - H100 result:
    - reject
    - same-day controls:
      - `881.35 ms`
      - `880.18 ms`
    - candidate:
      - `961.05 ms`
      - `959.47 ms`
      - `959.11 ms`
    - mean delta:
      - `880.77 -> 959.88 ms` (`+8.98%`)
    - scoreboard status:
      - `INTEGRATION_BOTTLENECK`
  - mechanism read:
    - eager improved:
      - `ProfilerStep* delta = -392.33 ms`
    - compiled worsened:
      - `ProfilerStep* delta = +306.66 ms`
      - `compile_specific_penalty = +698.99 ms`
    - graph-shell tax reopened:
      - `CompiledFxGraph delta = +307.55 ms`
      - `DDP.forward delta = +240.12 ms`
    - copy tax reopened materially:
      - `compiled_copy_tax = +306.55 ms`
    - the split/norm kernels themselves were visible, but not the dominant
      explanation for the total loss:
      - `hgdn_cuda_v4::packed_qkv_split_l2norm_backward = 194.72 ms`
      - `hgdn_cuda_v4::packed_qkv_split_l2norm = 145.95 ms`
  - current decision:
    - close the current post-conv front-end seam
    - keep `winner-20260405-19` as the active HGDN kernel baseline
    - do not send `cuda-fused-frontend`, `cuda-fused-frontend-lib`, or
      `cuda-split-norm-lib` back to H100 at this abstraction level
    - do not spend a forward-only split/norm follow-up on current evidence
    - move the next tranche to compute-optimal HGDN resize on H100
- Older screened candidate:
  - `winner-20260405-19-cuda-packed-conv-aten-bwd`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_USE_CUDA_PACKED_CONV_ATEN_BACKWARD=1`
  - H100 result:
    - reject
    - same-day controls:
      - `880.50 ms`
      - `883.49 ms`
    - candidate:
      - `994.59 ms`
      - `993.22 ms`
    - mean delta:
      - `881.99 -> 993.91 ms` (`+12.69%`)
    - scoreboard status:
      - `INTEGRATION_BOTTLENECK`
  - mechanism read:
    - copy tax reopened badly:
      - `compiled_copy_tax = +868.54 ms`
    - compiled shell rows dominated:
      - `aten::copy_ = 491.69 ms`
      - `direct_copy_kernel_cuda = 478.43 ms`
      - `gdn.qkv_conv_cuda_aten_bwd_left_pad = 297.25 ms`
      - `gdn.qkv_conv_cuda_aten_bwd_input_grad = 201.69 ms`
  - current decision:
    - reject this ATen-backward ownership on H100
    - do not spend more H100 time on this ownership split
- Older screened candidate:
  - `winner-20260405-19-cuda-packed-conv-aten-weight-bwd`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_USE_CUDA_PACKED_CONV_ATEN_WEIGHT_BACKWARD=1`
  - local result:
    - reject
    - direct hotpath result:
      - `gdn_fwd_bwd`: `199.41 -> 249.66 ms`
      - `hybrid_fwd_bwd`: `351.10 -> 362.01 ms`
  - current decision:
    - keep this only as a discarded ownership waypoint
- Older screened candidate:
  - `winner-20260405-19-cuda-packed-conv`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_USE_CUDA_PACKED_CONV=1`
  - latest H100 result:
    - reject
    - same-day controls:
      - `880.62 ms`
      - `881.28 ms`
    - candidate:
      - `904.21 ms`
      - `902.45 ms`
    - mean delta:
      - `880.95 -> 903.33 ms` (`+2.54%`)
    - scoreboard status:
      - `INTEGRATION_BOTTLENECK`
  - mechanism read:
    - the rewritten custom weight-backward kernel is no longer the disaster
      from `h100k15`
    - main compiled custom rows:
      - `_PackedQKVConvFunctionBackward`: `585.07 ms`
      - `causal_dwconv_weight_backward_kernel_k4`: `396.58 ms`
      - `_PackedQKVConvFunction`: `184.90 ms`
    - the remaining shell rows now matter more:
      - `triton_poi_fused_add_cat_0`: `76.09 ms`
      - `silu_grad_from_preact_kernel`: `54.68 ms`
      - `silu_from_preact_kernel`: `45.79 ms`
    - copy tax actually improved:
      - `compiled_copy_tax = -46.95 ms`
  - current decision:
    - reject this packed-conv composition on H100
    - keep the weight-backward rewrite as a useful building block
    - move the next sidecar one boundary later to the fused frontend shell
- Older screened candidate:
  - `winner-20260405-19-cuda-frontend-nct-custom-bwd`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
    - `GDN_USE_CUDA_FRONTEND_NCT=1`
  - H100 result:
    - reject
    - same-day controls:
      - `878.10 ms`
      - `882.01 ms`
    - candidate:
      - `996.66 ms`
      - `996.98 ms`
    - mean delta:
      - `880.06 -> 996.82 ms` (`+13.27%`)
    - scoreboard status:
      - `INTEGRATION_BOTTLENECK`
  - mechanism read:
    - eager `ProfilerStep*` improved:
      - `-577.25 ms`
    - compiled `ProfilerStep*` worsened:
      - `+554.10 ms`
    - compile-specific penalty:
      - `+1131.36 ms`
    - compiled copy tax stayed flat:
      - `-0.54 ms`
    - compiled external-kernel self time reopened:
      - `726.23 ms`
  - current decision:
    - keep `winner-20260405-19` active
    - park the current NCT-frontend family until the compile boundary changes
      materially
- Older screened candidate:
  - `winner-20260405-19-cuda-frontend-nct`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_USE_CUDA_FRONTEND_NCT=1`
  - H100 result:
    - hard reject
    - same-day controls:
      - `883.57 ms`
      - `883.29 ms`
    - candidate:
      - `1050.79 ms`
      - `1051.40 ms`
    - mean delta:
      - `883.43 -> 1051.10 ms` (`+18.98%`)
  - reason:
    - it dropped `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`, so it was not a
      true `k10 + k13` composition
    - compiled copy tax and custom-op overhead reopened badly on H100
  - decision:
    - reject this standalone sidecar on H100
    - keep only the compile-visible NCT frontend idea as a building block
- Older screened candidate:
  - `winner-20260405-19-cuda-split-norm`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_USE_CUDA_SPLIT_NORM=1`
  - purpose:
    - keep the promoted packed qkv front-end and custom depthwise backward
    - replace only the post-conv `split + q/k l2 norm + v materialization`
      stage with a narrow CUDA op
    - keep recurrence math unchanged and preserve the recurrence-facing
      contract
  - local result:
    - directionally positive against `profiles/rtx4070_cuda_base/`
    - console step average:
      - `3320.37 -> 3192.68 ms` (`-3.85%`)
    - `ProfilerStep*` self-device total:
      - `6610.92 -> 6384.67 ms` (`-3.42%`)
    - peak allocated memory:
      - `6184 -> 5984 MiB`
  - H100 result:
    - hard reject with same-day controls
    - controls:
      - `879.46 ms`
      - `878.20 ms`
    - candidate:
      - `959.13 ms`
      - `961.33 ms`
    - mean delta:
      - `878.83 -> 960.23 ms` (`+9.26%`)
    - compiled profile failure mode:
      - `aten::copy_`: `57.72 -> 210.65 ms`
      - `_PackedQKVSplitL2NormFunction: 145.83 ms`
      - `_PackedQKVSplitL2NormFunctionBackward: 194.82 ms`
      - while the real depthwise conv buckets barely moved
  - decision:
    - keep `winner-20260405-19` active
    - reject this standalone sidecar on H100
    - keep it bookmarked only as a possible ingredient for a larger packed
      front-end kernel pipeline
    - next kernel tranche should target the depthwise-conv family directly,
      not another extension island layered on top of the compiled front-end
- Older screened candidate:
  - `winner-20260405-19-split-copy`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_PACKED_QKV_SPLIT_COPY=1`
  - purpose:
    - replace the packed split-plus-three-contiguous path with
      `aten.split_with_sizes_copy`
    - keep the packed recurrence-facing q/k/v contract contiguous
    - keep recurrence math unchanged
  - result:
    - local reject against `profiles/rtx4070_cuda_base/`
    - console step average:
      - `3320.37 -> 3752.76 ms` (`+13.02%`)
    - `ProfilerStep*` self-device total:
      - `6610.92 -> 7546.44 ms` (`+14.15%`)
    - boundary audit stayed clean, but trainer buckets still got worse:
      - `aten::mul`: `1012.30 -> 1276.17 ms`
      - `aten::copy_`: `785.65 -> 798.02 ms`
      - `gdn.recurrence`: `177.23 -> 191.34 ms`
      - `aten::convolution_backward`: `174.81 -> 206.86 ms`
      - `aten::_conv_depthwise2d`: `141.31 -> 159.04 ms`
  - decision:
    - do not spend H100 time on this variant
    - keep `winner-20260405-19` active
    - next front-end pass should go below this ATen-only output-path change
      and target a genuinely lower-level packed output path
- Older screened candidate:
  - `winner-20260405-19-single-contig`
  - equivalent to:
    - `winner-20260405-19`
    - `GDN_PACKED_QKV_SINGLE_CONTIG=1`
  - purpose:
    - keep the promoted packed qkv front-end and custom backward
    - replace three post-conv q/k/v contiguous materializations with one packed contiguous materialization before split
    - keep the q/k normalization operator family unchanged
  - result:
    - local reject against `profiles/rtx4070_cuda_base/`
    - trainer eager self-device time:
      - `25561.13 -> 26793.74 ms` (`+4.82%`)
    - `aten::copy_` improved:
      - `785.65 -> 727.70 ms`
    - but that was outweighed by:
      - `aten::mul`: `1012.30 -> 1219.95 ms`
      - `gdn.recurrence`: `177.23 -> 179.54 ms`
      - `aten::convolution_backward`: `174.81 -> 177.59 ms`
    - optional H100 sidecar also failed to justify promotion:
      - same-day controls:
        - `882.37 ms`
        - `877.13 ms`
      - candidate:
        - `876.36 ms`
        - `878.32 ms`
      - mean delta:
        - `879.75 -> 877.34 ms` (`-0.27%`)
      - eager profile also drifted the wrong way:
        - `gdn.qkv_conv_output_contiguous_packed: 153.34 ms`
        - `gdn.v_contiguous: 16.76 ms`
        - `gdn.q_norm: 77.58 -> 82.71 ms`
        - `gdn.k_norm: 77.74 -> 82.78 ms`
  - decision:
    - H100 sidecar already run; no promotion
    - keep `winner-20260405-19` active
    - next front-end pass should target a lower-level packed output path, not another Python-side materialization reshuffle

### 0. Interim cleanup checkpoint from the redundancy audit

- Status: first low-risk consolidation pass done.
- Reference: `docs/REDUNDANCY_AUDIT.md`
- Purpose: shrink obvious maintenance debt before the next round of HGDN kernel work so profiler and experiment tooling do not keep multiplying parallel copies.
- Completed in the first pass:
  1. removed the superseded `scripts/compare_profiler_reports.py`
  2. extracted shared profiler row/CSV helpers and the canonical HGDN transfer-bucket list into `profiler_report.py`
  3. deduplicated the repeated `GDN_*` env-contract plumbing in `scripts/run_hgdn_local_phase1.sh`
  4. consolidated the duplicated `env_flag` helper into `local_env.py`
  5. reduced repeated bf16 fixture and packed-state-copy setup in `test_model.py`
- Remaining follow-ups:
  1. decide whether `scripts/export_wandb_hgdn_runs.py` is canonical or archival
  2. revisit extraction of shared baseline/hybrid tokenizer-data helpers
  3. revisit extraction of shared quantization core only with artifact-byte regression checks
- Explicit defer:
  - do not blindly merge `train_gpt.py` and `model.py` transformer utilities
  - do not touch baseline/hybrid quantization helpers without artifact-byte regression checks

### 1. H100 profiling-driven HGDN kernel pass

- Status: active top priority.
- Why: the H100 runs confirmed that the architecture is worth optimizing, and the profiler says the current throughput tax is not just "FLA recurrence is slow." A large fraction of the overhead is HGDN-side glue code around the recurrence.
- Current H100 facts:
  - throughput: hybrid is about `1.40x` slower than the attention-only baseline at `seq=2048`
  - quality: hybrid still wins strongly on H100 fixed-2k quality and roundtrip BPB
- Profiling helper:
  - preferred launch surface:
    - `python scripts/hgdn.py ...`
  - `scripts/run_h100_single_gpu_hgdn_profile.sh {hybrid|depth|both|both-eager}`
  - default is now `USE_WANDB=0`
  - eager modes force `COMPILE=0` for attribution-only traces when compiled graphs hide the `record_function` labels
  - local phase-1 workflow is now scripted via:
    - `scripts/run_hgdn_local_phase1.sh`
    - `scripts/analyze_hgdn_phase1.py`
    - `scripts/profile_hgdn_local_hotpath.py`
- Measured hotspot read on the 4 active profiled H100 steps:
  - hybrid flash-attention self CUDA: about `1.15s`
  - depth flash-attention self CUDA: about `2.38s`
  - hybrid GDN recurrence-family kernels: about `0.71s`
  - hybrid depthwise conv stack: about `1.01s`
  - hybrid extra elementwise/Triton glue: about `1.26s`
  - hybrid `aten::copy_`: about `0.51s`
  - depth `aten::copy_`: about `0.04s`
- Takeaway:
  - the recurrence kernel matters, but it is not the whole problem
  - the immediate optimization target is the HGDN scaffolding around the recurrence
  - attention is already on a strong PyTorch flash-SDPA path, so it is not the first place to spend engineering time

Ranked optimization checklist:

1. Eliminate dtype/layout churn in the HGDN path.
   - Why: `aten::copy_` is materially larger in hybrid than depth on H100.
   - What to inspect:
     - casts around `q/k` normalization
     - casts around `g`, `beta`, and `g_out`
     - any view/contiguous path forcing copies before or after the FLA call
     - blanket `fp32 -> bf16` recasts in `CastedLinear` on large feature-map weights
   - Current branch support:
     - `GDN_LOG_LAYOUTS=1` prints one-shot tensor dtype/shape/stride summaries at the FLA boundary
     - `GDN_AUDIT_BOUNDARIES=1` now records structured boundary layouts through `project_qkv -> conv_qkv -> norm_qkv -> recurrence_inputs -> recurrence_output -> output_gate_inputs -> output_proj_input`
     - large feature-map `CastedLinear` weights now stay `bf16`; only low-dimensional and explicit control parameters are restored to `fp32`
   - Expected upside: high, because this is pure overhead and not model math.
2. Fuse or rewrite the `q_conv/k_conv/v_conv` preprocessing path.
   - Why: depthwise conv plus its backward path is one of the largest HGDN-only buckets.
   - What to try:
     - a custom Triton kernel or a more direct fused implementation for depthwise causal conv + SiLU
     - at minimum, benchmark whether `v_conv` can be removed again on H100 without hurting quality
   - Current branch support:
     - `GDN_USE_Q_CONV`, `GDN_USE_K_CONV`, `GDN_USE_V_CONV` now allow direct q/k/v ablations without code edits
   - Expected upside: high.
3. Fuse HGDN elementwise glue.
   - Why: the profiler shows a broad pile of Triton/elementwise kernels beyond the recurrence itself.
   - What to target:
     - q/k normalization path
     - output `rms_norm(o) * silu(g_out)`
     - simple gate/decay prep if it currently spills through multiple kernels
   - Expected upside: medium to high.
4. Only after 1-3, revisit the FLA recurrence kernel itself.
   - Why: the recurrence is important, but the trace says the branch is currently losing plenty of time outside it.
   - What to do:
     - compare current FLA kernel behavior against the surrounding glue cost
     - only invest in recurrence-kernel changes if it remains one of the top HGDN buckets after the glue cleanup
   - Expected upside: medium, but expensive engineering.
5. Run one attribution-only pass with `COMPILE=0`.
   - Why: compiled mode swallows some `record_function` labels, so a short eager profile will be easier to read.
   - Important: use it for diagnosis only, not throughput conclusions.
6. Consider attention-side alternatives only later.
   - Why: the attention stack is already using PyTorch flash SDPA, not a naive slow path.
   - Flex attention or external `flash-attn` should only be tested if HGDN-side kernels stop being the dominant issue.
   - Reminder for later:
     - once the HGDN kernel family is genuinely converged, full-attention work is a plausible next frontier
     - likely candidates are:
       - direct Hopper-oriented attention kernels
       - FlexAttention
       - explicit FA3-style paths if they beat PyTorch SDPA on the target image
     - do not start that work yet; it is intentionally deferred until HGDN-side kernels stop dominating the hybrid gap
     - important current-scope note:
       - this HGDN branch uses PyTorch `scaled_dot_product_attention`, not direct `flash_attn` / FA3 calls
       - so merely having FA2/FA3 installed in the environment should not change the current HGDN profiling path by itself

Suggested immediate sequence:

```bash
scripts/run_hgdn_cuda_preflight.sh
```

Then:

```bash
RUN_PREFIX=h100prof scripts/run_h100_single_gpu_hgdn_profile.sh both
```

Then:

- inspect `profiles/<run_id>/key_averages.json`
- inspect the Chrome/Perfetto trace in `profiles/<run_id>/traces/`
- do one short eager attribution rerun with `COMPILE=0` only if compiled traces remain too anonymous

Latest local attribution checkpoint:

- `rtx4070_phase1` shows that the most concrete next target is the post-conv q/k/v layout path.
- The boundary audit shows q/k/v start contiguous after projection, become non-contiguous after `q_conv/k_conv/v_conv`, and stay non-contiguous into `recurrence_inputs`.
- That makes the next semantics-preserving kernel pass:
  - layout cleanup between `conv_qkv` and recurrence
  - then `norm_qkv` / `output_gate` glue
  - not generic trainer-wide `copy_` chasing
- First boundary candidate now exists and is worth keeping in the experiment surface:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - it fixes q/k/v contiguity all the way from `conv_qkv` to `recurrence_inputs`
  - local trainer self-device time improved from `25,990.59 ms` to `25,258.00 ms` (`-2.82%`)
  - the tradeoff is a more expensive `gdn.conv_qkv` bucket (`236.67 -> 322.01 ms`) in exchange for a cheaper recurrence/norm path
- Negative local follow-ups already ruled out:
  - `GDN_GATES_FP32=0` regressed badly
  - `GDN_OUTPUT_NORM_FP32=0` regressed overall even though `gdn.output_gate` itself got cheaper
  - q/k-only contiguous outputs caused a catastrophic recurrence regression
  - `GDN_USE_V_CONV=0` reduced some conv work but lost overall
- Immediate next local target after this candidate:
  - finer subrange attribution inside `gdn.gates`, `gdn.norm_qkv`, and `gdn.output_gate`
  - then a semantics-preserving cleanup inside the winning all-path contiguous layout
- Fine-subrange attribution is now done on `rtx4070_phase1_convcontig_subranges`.
- Current trainer-eager HGDN sub-bucket ordering on the local winner:
  - `gdn.recurrence`: `180.70 ms`
  - `gdn.q_conv`: `106.75 ms`
  - `gdn.k_conv`: `109.86 ms`
  - `gdn.v_conv`: `106.40 ms`
  - `gdn.output_norm`: `93.83 ms`
  - `gdn.q_norm`: `50.53 ms`
  - `gdn.k_norm`: `49.46 ms`
- Packed qkv conv follow-up is now done on `rtx4070_phase1_packedqkv_fix1`.
- Result:
  - the contiguity bug is fixed and the isolated hotpaths improved
  - but the full trainer-eager step still regressed by `+1.11%`
  - the main new tax is `aten::cat` plus a slightly worse conv backward path
- Updated next step:
  - packed qkv projection + packed conv is now implemented and screened on
    `rtx4070_phase1_packedqkvproj_fix1`
  - that candidate is the first packed-path local winner on the real training
    step:
    - trainer self-device time: `25797.53 -> 18282.05 ms` (`-29.13%`)
    - `aten::copy_`: `776.28 -> 560.86 ms`
    - `aten::mul`: `1003.44 -> 719.87 ms`
    - `gdn.recurrence`: `180.70 -> 123.78 ms`
  - new local winning config:
    - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
    - `GDN_USE_PACKED_QKV_CONV=1`
    - `GDN_USE_PACKED_QKV_PROJ=1`
  - follow-up on `rtx4070_phase1_ctrlbf16_fix1` is now done:
    - keeping `w_a/w_b/w_g` in bf16 via `GDN_CONTROL_PROJ_FP32=0` is a second
      real local win on top of the packed-path baseline
    - vs `rtx4070_phase1_packedqkvproj_refresh1`, trainer self-device time fell
      from `25957.29 ms` to `16588.06 ms` (`-36.09%`)
    - vs the original packed-path winner `rtx4070_phase1_packedqkvproj_fix1`,
      it still improved by `-9.27%`
    - the recurrence-facing contiguous bf16 layout is unchanged
  - current local winning config:
    - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
    - `GDN_USE_PACKED_QKV_CONV=1`
    - `GDN_USE_PACKED_QKV_PROJ=1`
    - `GDN_CONTROL_PROJ_FP32=0`
  - q/k normalization follow-up on the winner is now screened and rejected:
    - manual bf16 q/k norm improved the isolated `q_norm` bucket
    - but it materially regressed the full trainer-eager step
    - result is logged at `profiles/rtx4070_phase1_qknorm_fix1/`
  - immediate next local step:
    - `GDN_OUTPUT_NORM_FP32=0` has now also been re-screened on the current
      packed-path winner and still loses overall
    - result is logged at
      `profiles/rtx4070_phase1_outputnormbf16_packed_fix1/`
  - next local target:
    - structural output-side or packed-conv cleanup, such as in-place SiLU /
      output-gate consolidation, is now also screened and rejected
    - result is logged at `profiles/rtx4070_phase1_inplacesilu_fix1/`
  - next step:
    - stop local output-side tuning for now
    - move to 1xH100 confirmation of the current local winner:
      - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
      - `GDN_USE_PACKED_QKV_CONV=1`
      - `GDN_USE_PACKED_QKV_PROJ=1`
      - `GDN_CONTROL_PROJ_FP32=0`
  - update after `h100k5`:
    - the current local winner now **does** transfer on the compiled H100 perf
      harness
    - hybrid perf improved from `997.66 ms` to `901.05 ms` (`-9.68%`)
    - depth remained effectively flat (`708.95 -> 710.76 ms`)
    - hybrid-vs-depth ratio improved from `1.41x` slower to `1.27x` slower
    - eager H100 trace confirms a large recurrence win (`336.60 -> 188.31 ms`)
      with a more expensive packed qkv front-end
    - missing follow-up:
      - done in `h100k5v2`
      - compiled H100 trace now confirms the same result
      - strongest named compiled-bucket win:
        - `aten::copy_`: `507.45 -> 174.42 ms`
      - other named compiled buckets are mostly flat to mildly better:
        - `aten::mm`: `648.90 -> 618.45 ms`
        - `aten::convolution_backward`: `302.28 -> 288.75 ms`
        - recurrence forward/backward are roughly flat in compiled mode
  - current decision:
    - keep the current winner as the HGDN perf reference path
    - `h100k6` now answers the fixed-step quality question too:
      - hybrid final sampled eval: `2.4201` bpb
      - depth final sampled eval: `2.5373` bpb
      - hybrid roundtrip eval: `2.44379288` bpb
      - depth roundtrip eval: `2.54975979` bpb
  - latest rejected local follow-up:
    - `GDN_USE_PACKED_AB_PROJ=1` on top of the current winner
    - hotpath looked promising enough to screen, but the real local trainer
      eager step regressed from `16588.06 ms` to `21463.22 ms` (`+29.39%`)
    - the main regressions landed exactly in the transfer buckets that matter:
      `aten::copy_`, `aten::mul`, `gdn.qkv_conv_packed`, `gdn.recurrence`,
      `aten::convolution_backward`, and `aten::_conv_depthwise2d`
    - do not send this candidate to H100
  - latest rejected local full-bundle screen:
    - packed-conv `split_with_sizes -> narrow` cleanup on top of the current
      winner
    - preflight looked directionally interesting, but the real local
      trainer-eager bundle regressed:
      - `aten::copy_`: `510.77 -> 653.08 ms`
      - `aten::mul`: `659.15 -> 785.87 ms`
      - `gdn.qkv_conv_packed`: `234.58 -> 280.92 ms`
      - `gdn.recurrence`: `114.97 -> 136.84 ms`
    - take-away:
      - the remaining packed-front-end cost is not going to be fixed by a tiny
        post-conv split primitive swap
  - latest rejected local full-bundle screen:
    - FLA in-kernel q/k l2 normalization on top of the current winner
      (`GDN_QK_L2NORM_IN_KERNEL=1`)
    - steady-state preflight looked interesting enough to promote, but the real
      local trainer-eager bundle regressed hard:
      - `aten::copy_`: `510.77 -> 662.86 ms`
      - `aten::mul`: `659.15 -> 888.74 ms`
      - `gdn.qkv_conv_packed`: `234.58 -> 365.93 ms`
      - `gdn.recurrence`: `114.97 -> 204.03 ms`
    - take-away:
      - moving q/k l2 normalization into the FLA kernel is not a valid free
        win for this HGDN training path
      - do not send this candidate to H100
  - next kernel focus remains:
    - packed qkv front-end cost on H100
    - gate/output glue
    - remaining copy/layout churn after the packed-path win
      - do not revisit `split -> narrow` or similar post-conv split
        micro-optimizations unless a later trace gives a much stronger reason
  - packed-front-end subrange attribution on the current winner is now logged at
    `profiles/current_winner_qkvbreakdown/`
    - real remaining front-end costs:
      - `gdn.qkv_conv_depthwise`
      - `gdn.qkv_conv_output_contiguous`
    - near-zero bookkeeping ranges:
      - `gdn.qkv_conv_input_transpose`
      - `gdn.qkv_conv_trim`
      - `gdn.qkv_conv_output_transpose`
      - `gdn.qkv_conv_split`
    - implication:
      - do not spend more time on tiny packed split/trim cleanup unless a later
        H100 trace contradicts this
  - latest rejected local full-bundle screen:
    - explicit packed-conv input materialization on top of the current winner
    - idea:
      - force `x.transpose(1, 2).contiguous()` before the packed depthwise
        `Conv1d` to attack the remaining compiled `clone + transpose +
        convolution` style kernels seen on H100
    - result:
      - preflight improved, but the real local trainer-eager bundle regressed
        across the transfer buckets that actually matter:
        - `aten::copy_`: `510.77 -> 689.51 ms`
        - `aten::mul`: `659.15 -> 1000.15 ms`
        - `gdn.qkv_conv_packed`: `234.58 -> 356.32 ms`
        - `gdn.recurrence`: `114.97 -> 173.27 ms`
        - `aten::convolution_backward`: `113.86 -> 172.55 ms`
        - `aten::_conv_depthwise2d`: `91.58 -> 139.24 ms`
    - take-away:
      - the remaining packed-front-end tax is not going to yield to an obvious
        input-side `.contiguous()` either
      - this is the second straight packed-conv structural tweak that looked
        better in preflight and worse in the real trainer step
      - move the next hypothesis away from packed-conv input/output surgery and
        toward the repeated fp32 shell/control casts that still show up as
        `to_copy + add + mul + unsqueeze` style kernels in the compiled H100
        trace
  - latest rejected quick-screen:
    - `GDN_USE_PACKED_QK_NORM=1` on top of the current winner
    - rejected before full local phase-1 promotion
    - preflight hybrid eager regressed from about `122.89 ms` to `140.21 ms`
    - bare-GDN self-device total regressed from `227.76 ms` to `552.22 ms`
    - most concerning delta:
      - `gdn.recurrence`: `34.71 -> 337.61 ms`
    - do not send this candidate to H100
  - latest rejected quick-screen:
    - `GDN_MANUAL_QK_NORM=1` on top of the current winner
    - idea:
      - replace the generic q/k `F.normalize` path with an explicit fp32
        `square -> sum -> rsqrt -> mul` formulation while keeping the packed
        front-end otherwise unchanged
    - result:
      - rejected at the preflight gate
      - `gdn_eager`: `1032.32 -> 1092.91 ms`
      - `hybrid_eager`: `146.49 -> 151.18 ms`
      - `hybrid_compiled`: `3223.56 -> 4966.38 ms`
    - take-away:
      - the simple manual fp32 q/k norm rewrite is not a valid next step
      - if q/k norm gets revisited again, it should be with a more targeted
        formulation than this broad `F.normalize` replacement
  - latest rejected quick-screen:
    - `RESID_MIX_FP32=0` on top of the current winner
    - idea:
      - keep only `resid_mix` on activation dtype while leaving the rest of the
        residual shell on the default fp32 restore path
      - test the smallest shell-side carve-out that still directly matches the
        remaining compiled `add + mul + unsqueeze` kernels
    - result:
      - rejected at the preflight gate
      - `gdn_eager`: `1032.32 -> 1076.62 ms`
      - `hybrid_eager`: `146.49 -> 143.97 ms`
      - `hybrid_compiled`: `3223.56 -> 9879.93 ms`
    - take-away:
      - even the narrow `resid_mix` carve-out is not a valid next step
      - stop spending time on shell-side fp32 restore removals for now
      - move the next candidate back to the packed-conv implementation itself
  - latest rejected quick-screen:
    - `GDN_USE_MANUAL_PACKED_QKV_CONV=1` on top of the current winner
    - idea:
      - replace the packed depthwise `Conv1d` with an explicit causal
        shift-and-sum implementation while reusing the same packed conv
        weights
      - test a real packed-front-end rewrite rather than another shell-side
        dtype change
    - result:
      - rejected at the preflight gate
      - `gdn_eager`: `1032.32 -> 1136.29 ms`
      - `hybrid_eager`: `146.49 -> 150.94 ms`
      - `hybrid_compiled`: `3223.56 -> 7863.94 ms`
    - take-away:
      - the simple high-level shift-and-sum rewrite is not a valid next step
      - the next packed-front-end attempt should be lower-level and closer to
        the actual kernel/memory behavior we are trying to fix
  - latest rejected quick-screen:
    - `GDN_DELAY_QKV_SPLIT=1` on top of the current winner
    - idea:
      - keep one packed q/k/v buffer through the packed depthwise conv and
        delay the q/k/v split until recurrence prep
      - remove the immediate post-conv q/k/v clone tax without changing the
        `F.normalize` operator family
    - result:
      - rejected at the preflight gate
      - `gdn_eager`: `1032.32 -> 1546.84 ms`
      - `hybrid_eager`: `146.49 -> 211.45 ms`
      - `hybrid_compiled`: `3223.56 -> 5258.08 ms`
    - take-away:
      - the current packed front-end depends on the existing post-conv
        materialization contract more than expected
      - do not revisit split timing as the next packed-front-end experiment
      - move the next structural attempt either lower-level or away from this
        exact front-end boundary
      - hybrid remains slower (`915.10 ms` vs `724.72 ms`) but keeps a clear
        quality edge on H100
      - both models are over the 16 MB limit, but hybrid is closer:
        - hybrid total bytes: `17,580,964` (`+1,580,964` over)
        - depth total bytes: `18,553,002` (`+2,553,002` over)
  - latest rejected local full-bundle screen:
    - `BLOCK_SHELL_FP32=0` on top of the current winner
    - idea:
      - keep the residual shell on activation dtype instead of restoring
        `attn_scale/mlp_scale/resid_mix/q_gain/skip_weights` to fp32
      - target the remaining compiled H100 `to_copy + add + mul + unsqueeze`
        style kernels
    - result:
      - preflight and hotpath both looked promising enough to escalate, but the
        real trainer-eager bundle regressed badly:
        - `aten::copy_`: `510.77 -> 771.96 ms`
        - `aten::mul`: `659.15 -> 1000.82 ms`
        - `gdn.qkv_conv_packed`: `234.58 -> 357.30 ms`
        - `gdn.recurrence`: `114.97 -> 173.41 ms`
        - `aten::convolution_backward`: `113.86 -> 173.00 ms`
        - `aten::_conv_depthwise2d`: `91.58 -> 139.65 ms`
        - `gdn.q_norm`: `31.74 -> 48.22 ms`
        - `gdn.k_norm`: `31.84 -> 48.20 ms`
    - take-away:
      - the broad residual-shell bf16 move is not a valid promotion candidate
      - the remaining shell-side compiled kernels, by themselves, are not
        enough reason to remove fp32 restoration wholesale
      - if the shell side gets revisited, it should be as a much narrower carve
        out, not an all-at-once flag
  - next branch pivot:
    - stop spending H100 time proving the packed-path winner again
    - keep the current winner as the HGDN systems baseline
    - keep the current winner as the systems baseline, but do **not** treat the
      branch as finished with kernel work
    - architecture retuning is prepared and bracketed, not yet the final branch
      priority
    - concrete target:
      - recover about `10%` total artifact bytes from the current hybrid
        winner while preserving as much of the `0.1172` fixed-step bpb edge as
        possible
      - treat that `~10%` cut as a starting bracket, not a claim that the best
        model must land exactly at the 16 MB ceiling
      - the size cap is a hard constraint; filling it exactly is only a
        heuristic until the compute-optimal curve is actually measured
    - allowed search space:
      - architecture can change
      - prioritize CUDA-friendly shapes when proposing resized variants
      - do not undo the packed-path kernel changes unless a new candidate beats
        them on both quality and bytes
    - new helper for this phase:
      - `conda run -s --name pg python scripts/hgdn.py arch-size-screen --config configs/hgdn/winner_20260405_11_retune.toml`
      - writes structured output to `profiles/arch_size/winner_20260405_11_retune/`
      - important caveat:
        - the screen is an initialization-time proxy only
        - use relative deltas to shortlist variants, then validate finalists
          with the trainer
    - first proxy shortlist from `winner_20260405_11_retune`:
      - `trim_layers_14`: `-12.24%` total-init proxy vs current
      - `trim_width_320`: `-11.73%` total-init proxy vs current
      - `balanced_14l_mlp3.00`: `-15.57%` total-init proxy vs current
      - keep some underfilled candidates in the final comparison set on
        purpose; earlier branch results already suggest the optimum may sit
        below full artifact utilization
      - mild trims are likely too small by themselves:
        - `trim_mlp_3.00`: `-3.82%`
        - `trim_mlp_2.75`: `-7.60%`
        - `trim_width_320_mlp3.75`: `-6.05%`
    - architecture-retune status:
      - the shortlist and launch configs are ready
      - use them after the next HGDN kernel tranche, not as proof that the
        kernel side is done
    - next kernel tranche before size lock:
      - use the current winner as the H100 profiling baseline
      - focus on remaining HGDN-native hotspots:
        - packed qkv front-end cost on H100
        - q/k normalization and gate/output glue
        - residual copy/layout churn after the packed-path win
        - after the rejected `packed_proj + separate conv` trial and the
          rejected `split -> narrow` cleanup, prioritize structural
          packed-conv transpose / clone cleanup or gate/output projection work
          over more post-split micro-optimizations
      - only promote local wins to H100, one run at a time, as before
    - first H100 retune round, once the next kernel tranche stalls:
      - run sequential fixed-step checks for:
        - `configs/hgdn/retune_trim_layers_14.toml`
        - `configs/hgdn/retune_trim_width_320.toml`
      - update after the first resize round:
        - `configs/hgdn/retune_trim_layers_14.toml` is the live winner
        - `configs/hgdn/retune_trim_width_320.toml` is rejected
        - `configs/hgdn/retune_balanced_14l_mlp3.toml` must be rerun because
          the original H100 helper ignored plain `MLP_MULT`
      - use the hybrid-only helper mode so the attention-only baseline does not
        get rerun every time:
        - `python scripts/hgdn.py h100-perf fixed2k-hybrid --config configs/hgdn/retune_trim_layers_14.toml --run-prefix h100k7a --online --set WANDB_PROJECT=pg-hconv-ablations --set WANDB_WATCH=gradients`
        - `python scripts/hgdn.py h100-perf fixed2k-hybrid --config configs/hgdn/retune_trim_width_320.toml --run-prefix h100k7b --online --set WANDB_PROJECT=pg-hconv-ablations --set WANDB_WATCH=gradients`
      - compare each completed retune back to the `h100k6` hybrid reference via:
        - `conda run -s --name pg python scripts/hgdn.py fixed2k-compare --contains h100k7 --name h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048 --reference h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048 --output-dir profiles/fixed2k_compare/h100k7_round1`
      - known W&B quirk from `h100k6`:
        - summary metrics and final eval points are reliable
        - intermediate sampled eval checkpoints at `500` and `1500` were present in console logs but not retained in W&B history, so the comparison tool currently treats missing checkpoints as missing data rather than fabricating them

### 2. Norm placement screen (`pre` vs `post` vs `keel`)

- Status: newly enabled, not quality-screened yet.
- Why: recent work argues that pre-norm may be leaving depth utilization on the table, especially when a KEEL-style residual path makes post-norm trainable again.
- Helper: `scripts/run_laptop_norm_compare.sh {hybrid|depth|both}`
- Helper default: `WANDB_WATCH=none` so the comparison screens keep online metric logging without gradient-watch stalls. Override with `WANDB_WATCH=gradients` only when histograms are actually needed.
- Scope for this branch:
  - use the hybrid trainer only
  - compare within the same residual shell
  - treat `GDN_RATIO=0` as the pure-attention control
  - treat `GDN_RATIO=1` as the current HGDN operating point
- First readout to collect:
  - fixed-step sampled-eval BPB at the same token budget
  - whether `post` is merely stable or actually better
  - whether `keel` beats both `pre` and naive `post`

Suggested local comparison contract:

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=0 MLP_MULT=4.0 NORM_STYLE=pre \
RUN_ID=norm_depth_pre_r0 scripts/sweep.sh depth
```

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=0 MLP_MULT=4.0 NORM_STYLE=post \
RUN_ID=norm_depth_post_r0 scripts/sweep.sh depth
```

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=0 MLP_MULT=4.0 NORM_STYLE=keel \
RUN_ID=norm_depth_keel_r0 scripts/sweep.sh depth
```

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=1 MLP_MULT=3.25 NORM_STYLE=pre \
RUN_ID=norm_hybrid_pre_r1 scripts/sweep.sh single
```

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=1 MLP_MULT=3.25 NORM_STYLE=post \
RUN_ID=norm_hybrid_post_r1 scripts/sweep.sh single
```

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=750 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=1 MLP_MULT=3.25 NORM_STYLE=keel \
RUN_ID=norm_hybrid_keel_r1 scripts/sweep.sh single
```

### 3. Size-matched depth-control rerun

- Status: partially resolved.
- Why: the current hybrid quality win is real, but the logged artifact sizes are still not perfectly matched.
- What was tried:
  - `MLP_MULT=4.7` at a local 600-second screen: invalid, landed at `17,092,318` total bytes and was over budget.
  - `MLP_MULT=4.0` at the proper 2k-step fixed-step contract: valid, landed at `9,668,381` total bytes and `2.6550` / `2.6715` pre/post-roundtrip BPB. It is still smaller than the hybrid and only marginally better than `MLP_MULT=3.75`.
- Current takeaway:
  - Do not use a 600-second local 4070 run as the final size-matching proxy.
  - If exact size matching still matters locally, bracket the next fixed-step depth candidate between `4.0` and a slightly larger setting.
- Last fixed-step command used:

```bash
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
WANDB_MODE=offline USE_WANDB=0 NGPU=1 ITERATIONS=2000 TRAIN_BATCH_TOKENS=65536 \
MAX_WALLCLOCK_SECONDS=0 TRAIN_SEQ_LEN=2048 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=200 \
COMPILE_STRATEGY=model MLP_MULT=4.0 RUN_ID=quality_depth_mlp40_seq2k \
scripts/sweep.sh depth
```

### 4. H100 throughput calibration

- Status: done.
- Result:
  - H100 hybrid `GDN_RATIO=1, MLP_MULT=3.25` perf: `1002.72 ms`
  - H100 depth `MLP_MULT=4.0` perf: `714.74 ms`
  - hybrid slowdown: about `1.40x`
- Current takeaway:
  - the architecture remains interesting because the quality gap on H100 is stronger than local
  - but kernel work should come before broader size sweeps because the throughput penalty is larger than the 4070 suggested

### 5. Compute-optimal size sweep

- Status: temporarily deprioritized behind kernel work.
- Why: actual trained artifacts are around `11MB`, so the branch still needs a wall-clock-vs-size sweep instead of assuming "fill 16MB" is optimal.
- Candidate HGDN wall-clock sweep family:
  - `slim`: `MODEL_DIM=320`, `MLP_MULT=3.25`
  - `current`: `MODEL_DIM=384`, `MLP_MULT=3.25`
  - `mid`: `MODEL_DIM=416`, `MLP_MULT=3.25`
  - `wide`: `MODEL_DIM=448`, `MLP_MULT=3.0`

## Already landed

- Default compile path is now `COMPILE_STRATEGY=model`.
- The branch now has a profiling harness in the trainer:
  - `PROFILE=1`
  - scheduled `torch.profiler` capture
  - trace export under `profiles/<run_id>/traces/`
  - operator summary export to `profiles/<run_id>/key_averages.json` and `key_averages.csv`
- The branch now has a dedicated H100 profiling helper:
  - `scripts/run_h100_single_gpu_hgdn_profile.sh`
  - defaults to `USE_WANDB=0`
- The trainer has a dedicated perf harness:
  - `PERF_TIMING=1` for steady-state timing
  - `PERF_IGNORE_STEPS=N` to ignore early measured steps
  - `PERF_ISOLATE_COMPILE_CACHE=1` for fresh Inductor/Triton cache dirs per run
  - `PERF_SKIP_FINAL_EVAL=1` to stop after the measured training window
- The experimental `COMPILE_STRATEGY=hybrid` path remains available:
  - each GDN module is wrapped as an explicit eager boundary because the FLA path already dispatches Triton kernels and contains an internal `torch.compiler.disable()` wrapper
  - pure attention blocks are compiled with `fullgraph=True`
  - GDN-block MLPs are compiled with `fullgraph=True`
  - the top-level hybrid model still compiles with `fullgraph=False`
- Current local result on the RTX 4070: `COMPILE_STRATEGY=hybrid` underperformed `COMPILE_STRATEGY=model` by roughly `30%` on the 16-layer HGDN throughput screen, so it should stay experimental until an H100 test says otherwise.

## Break-Glass Items

## Current HGDN CUDA fused follow-up

- The optional fused CUDA extension is now in-tree behind:
  - `GDN_USE_CUDA_FUSED_FRONTEND=1`
  - `GDN_USE_CUDA_FUSED_OUTPUT=1`
- Current validated status:
  - local build passed in `pg`
  - local direct parity passed
  - local phase-1 beat the current HGDN winner at the trainer-eager gate
  - H100 build passed
  - H100 direct parity passed
  - H100 fused preflight passed
- H100 outcome:
  - eager hybrid profile lost badly vs `h100k5`
    - `1670.55 -> 2248.98 ms`
  - compiled perf lost badly vs `h100k5`
    - `901.05 -> 1863.41 ms`
    - `581,860 -> 281,359 tok/s`
- Root cause:
  - `_PackedQKVFrontendFunctionBackward` dominates both eager and compiled H100 traces
  - `causal_dwconv_weight_backward` is the main named kernel inside that loss
- Current decision:
  - keep the extension in-tree
  - do **not** use `winner-20260405-11-cuda-fused` in the active kernel path
  - do **not** run quality/retune work on top of the fused preset
- Next isolated salvage experiment:
  - `winner-20260405-11-cuda-output-only`
  - keep the non-extension current winner front-end
  - test only `GDN_USE_CUDA_FUSED_OUTPUT=1` with `GDN_OUTPUT_NORM_FP32=1`
- Result:
  - local preflight passed
  - local phase-1 lost slightly vs the non-extension current winner
  - do **not** promote it to H100
- Current extension status:
  - full fused preset: H100 reject
  - output-only fused preset: local reject
  - keep the extension in-tree, but stop using it in the active kernel path
- If the extension is revisited later, do it in this order:
  1. packed frontend backward
  2. depthwise-conv weight gradient
  3. output fusion in isolation, only after the frontend backward is under control

### Packed `q/k` conv + separate `v` conv result

- `GDN_USE_PACKED_QK_CONV=1` is now screened and rejected at the full local
  phase-1 gate:
  - candidate bundle:
    - `profiles/rtx4070_phase1_packedqkconv_fix2/`
  - comparison:
    - `profiles/rtx4070_phase1_packedqkconv_fix2/compare_vs_rtx4070_cuda_base/comparison.md`
- main result vs the non-extension current winner:
  - trainer eager self-device total:
    - `25561.13 -> 26367.32 ms` (`+3.15%`)
  - trainer eager step average:
    - `3320.37 -> 3422.05 ms` (`+3.06%`)
- why it lost:
  - it reduced the old packed-qkv materialization bucket
  - but the required separate `v_conv` plus extra `aten::cat` made the total
    front-end subtotal worse by about `61.96 ms`
- practical consequence:
  - stay on one packed qkv front-end
  - do **not** reopen designs that split `v` back out unless they also remove
    the added packing / separate-depthwise overhead

### Depthwise attribution result and next kernel plan

- depthwise attribution screens on the current winner are now done:
  - fresh local hotpath baseline
  - `GDN_FREEZE_CONV_WEIGHTS=1` in profiler/preflight only
  - `CUDNN_BENCHMARK=1`
- what those screens said:
  - freezing conv weights did **not** create a clean enough hybrid-path
    collapse to justify a simplistic "just remove weight grad" story
  - `CUDNN_BENCHMARK=1` was the only runtime-only knob with a real hotpath win
    in `hybrid_fwd_bwd`
- full local phase-1 result for `CUDNN_BENCHMARK=1`:
  - candidate bundle:
    - `profiles/rtx4070_phase1_cudnnbench_fix1/`
  - comparison:
    - `profiles/rtx4070_phase1_cudnnbench_fix1/compare_vs_rtx4070_cuda_base/comparison.md`
  - trainer eager self-device total:
    - `25561.13 -> 26518.46 ms` (`+3.75%`)
  - trainer eager step average:
    - `3320.37 -> 3396.07 ms` (`+2.28%`)
- decision:
  - do **not** promote `CUDNN_BENCHMARK=1`
  - stop spending time on runtime toggles for this depthwise tranche

Next execution plan for `gdn.qkv_conv_depthwise` / depthwise backward:

1. Keep the active non-extension current winner unchanged.
2. Prototype a **packed depthwise causal conv custom autograd path** behind an
   explicit experiment flag.
3. Keep the current forward contract:
   - one packed q/k/v buffer
   - one packed depthwise conv
   - one contiguous packed output before split
4. Change only the backward mechanics first:
   - target the packed depthwise input-grad / weight-grad path
   - do **not** reopen split-`v`, output fusion, or extra materialization
     experiments in the same patch
5. Gate it in this order:
   - parity / grad check
   - local hotpath
   - local phase-1
   - only then H100 eager/perf/profile

The first concrete implementation target is a profiling-visible replacement for
the packed depthwise backward path, not another front-end topology change.

### 1. Graph-break audit with `TORCH_LOGS` / `tlparse`

- Trigger: hybrid remains worse than `1.3x` the depth-control baseline after the current selective-compile quick hits, or compile-time behavior becomes erratic across repeated runs.
- What to do: run short screens with `TORCH_LOGS="graph_breaks,recompiles,perf_hints"` and inspect whether breaks are only at `GDNBlock.gdn` boundaries.
- Expected upside: `2-8%` on the 4070, `5-12%` on H100 if extra Python-side breaks are still present.
- Expected cost: low. Mostly logging and a small cleanup patch.

### 2. Backward-path `compiled_autograd`

- Trigger: forward graphs look clean but hybrid throughput is still lagging, or logs suggest backward fragmentation dominates runtime.
- What to do: test PyTorch compiled autograd on the hybrid trainer in a separate branch with the same 50-step throughput harness.
- Expected upside: `0-10%`, highly dependent on whether backward graph breaks are the real bottleneck.
- Expected cost: medium. Can increase recompiles and runtime overhead if shapes or control flow drift.

### 3. Regional compilation for repeated block stacks

- Trigger: compile warmup/cold-start becomes the main pain point, or we want faster iteration on short local screens without materially changing steady-state kernels.
- What to do: trial `torch.compiler.nested_compile_region` or a block-stack regional compile pattern around the repeated encoder/decoder loops.
- Expected upside: little steady-state gain, but compile latency can drop materially on short runs.
- Expected cost: medium. Good engineering payoff only if compile startup is the bottleneck.

### 4. Dynamic shape marking

- Trigger: frequent recompiles show up when sweeping `TRAIN_SEQ_LEN`, local batch size, or eval shapes inside the same process.
- What to do: add `mark_dynamic` only at the specific tensor boundaries causing recompiles.
- Expected upside: stability and lower recompile churn, not raw peak throughput.
- Expected cost: medium. For fixed-shape runs this can be neutral or negative, so it should stay off unless logs justify it.

### 5. FLA wrapper cleanup inside `model.py`

- Trigger: module-level eager boundaries still leave noisy compile traces, or we want a cleaner OLMo-style explicit wrapper around the FLA dispatch path.
- What to do: add a small local dispatch helper around `chunk_gated_delta_rule` and mark that wrapper as compiler-disabled instead of relying only on module-level disabling from the trainer.
- Expected upside: mostly cleaner boundaries and easier debugging, maybe `1-4%` if Dynamo is still poking at surrounding glue code.
- Expected cost: low to medium. Needs care to keep dtype and API handling identical.

### 6. Compile-mode shootout on finalists only

- Trigger: the hybrid passes the quality gates and is close enough to baseline that a few percent of throughput matters.
- What to do: compare PyTorch compile modes on fixed-shape finalists only, for example the default mode against a more aggressive autotune mode.
- Expected upside: `0-5%` if the workload lines up well with the mode.
- Expected cost: medium to high. More compile/autotune time, and results can differ across the 4070 and H100.

### 7. Nsight / kernel-level profiling

- Trigger: after the first HGDN glue/kernel cleanup pass, or when we need a firmer memory-bound vs compute-bound answer on H100.
- What to do:
  - use the current torch-profiler pass first
  - then move to Nsight Compute on the top HGDN kernels, especially depthwise conv and recurrence-adjacent kernels
- Expected upside: high diagnostic clarity for the second optimization pass, not necessarily an immediate speedup by itself.
- Expected cost: medium.

### 8. cuLA backend swap on Hopper, after the in-repo kernel pass

- Trigger: only after the current HGDN kernel work has already attacked copies, depthwise convs, and elementwise glue, and only if the linear-attention backend itself still looks like a serious remaining bottleneck on H100.
- Why it is interesting:
  - cuLA targets Hopper and Blackwell directly with CUDA/CuTe/CUTLASS kernels
  - it is intended to align with the FLA interface, so the eventual integration path may be relatively shallow
- Relevant repo areas to inspect later:
  - `csrc/kda/sm90/`
    - Hopper fused forward CUTLASS path
  - `cula/kda/hopper_fused_fwd.py`
    - Hopper `kda_prefill_hopper` wrapper
  - `cula/kda/chunk.py`
    - KDA autograd wrapper and current backward path
  - `cula/ops/chunk_delta_h.py`
    - modular chunk-delta kernel decomposition
  - `docs/chunk_delta_h_pipeline.md`
    - pipeline design notes for the chunk delta-H kernel
- Why it is not the first move:
  - the current profile says the branch is losing plenty of time outside the recurrence kernel
  - cuLA is early-stage and the README still marks modular GDN forward/backward support as incomplete
  - its published requirements and tested stack differ from this branch's current environment, so it is not a low-friction drop-in today
  - for H100 training specifically, the current caveats are stronger than the
    generic “early-stage” warning:
    - the Hopper fused path is currently forward-only and positioned more like
      large-batch inference / prefill than a training backend
    - the modular KDA forward path is Blackwell-only today
    - the modular backward still delegates to FLA rather than staying inside
      cuLA
    - modular GDN forward/backward kernels are still on the roadmap rather than
      implemented
- What to do:
  - treat this as an H100-only side experiment, not a portability-preserving branch default
  - make a separate backend wrapper so the code can switch between FLA and cuLA cleanly
  - benchmark recurrence-heavy HGDN runs against the same FLA baseline on the
    same Hopper machine
  - initial use should be as a design/reference repo first, not as an assumed
    training dependency:
    - study the Hopper fused forward path for ideas
    - study the chunk decomposition/pipeline docs for how they carry state and
      structure the update
    - only attempt a direct backend swap if cuLA later lands a real Hopper
      training path or if we explicitly decide to build our own SM90 training
      wrapper around the relevant kernels
  - only keep it if it wins materially without introducing integration debt or correctness drift
- Expected upside: unknown, potentially material, but currently speculative for this exact GDN training path.
- Expected cost: high. Environment churn plus backend-integration work.
