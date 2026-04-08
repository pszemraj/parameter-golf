# HGDN Branch Status

Last updated: 2026-04-08 03:05 EDT

Branch: `exp/hgdn`

This file is the branch-local status writeup for the hybrid Gated DeltaNet (HGDN) integration work. It summarizes what has been implemented, what has been measured, what is currently believed, and what should happen next.

## Local Env Standard

For this local WSL/laptop checkout, the standard conda env is `pg`.

- Use `conda run -s --name pg ...` for local Python-dependent commands.
- The old `train` env name came from a different machine and should not be used
  on this checkout.

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
- A dedicated profiling harness exists for compiled and eager-attribution H100 traces.
- A dedicated tiny CUDA preflight now exists for direct HGDN FLA/compile validation without `torchrun`.
- The initial quality comparison at `TRAIN_SEQ_LEN=2048` favors the hybrid over the matched depth-control baseline.
- A larger attention-only baseline (`MLP_MULT=4.0`) was re-run under the same 2k-step contract and still failed to close the hybrid gap.
- The hybrid stack now has an experimental residual normalization knob: `NORM_STYLE=pre|post|keel`.
- Large feature-map `CastedLinear` weights now stay in `bf16` after model init; only low-dimensional and explicitly routed control parameters are restored to `fp32`.

## Norm Placement Aside

There is now a controlled norm-placement surface in the hybrid model family for side experiments:

- `NORM_STYLE=pre`: current default behavior
- `NORM_STYLE=post`: plain post-residual RMSNorm per sublayer
- `NORM_STYLE=keel`: KEEL-inspired variant with:
  - inner RMSNorm on the transform branch
  - post-residual RMSNorm after the sum
  - amplified residual path via `residual_alpha`
  - first block left as pre-norm for stable embedding handoff

Important scope note:

- This branch did not mutate the reference [train_gpt.py](/home/pszemraj/workspace/projects/parameter-golf/train_gpt.py) baseline script for this experiment.
- The cleanest apples-to-apples probe here is within the hybrid trainer itself:
  - attention-only baseline via `GDN_RATIO=0`
  - mixed HGDN via `GDN_RATIO=1`

Current judgment:

- There is a legitimate case for trying something more post-norm-like here.
- That case is stronger for the HGDN branch than it would be for blindly editing the repo reference trainer.
- The right first comparison is not only `pre` vs naive `post`; it is `pre` vs `post` vs `keel`.

Why this is plausible in this repo:

- Our models are shallow relative to the regimes where post-norm historically failed catastrophically.
- The stack already has branch scales (`attn_scale`, `mlp_scale`) and residual mixing, which make the residual path more controllable than a bare Transformer shell.
- The hybrid path already pays for extra normalization and state-control structure, so a KEEL-style experiment is not out of character architecturally.

What has been validated so far:

- CPU tests pass for all three styles.
- Tiny offline trainer smokes for `post` and `keel` on the hybrid path run end-to-end without immediate instability.

What has not been claimed:

- No BPB win is established yet.
- No exact `train_gpt.py` baseline result exists yet.
- No wall-clock or target-hardware claim should be made from the norm-style work until matched runs are completed.

## Important Files

- `scripts/hgdn.py`: preferred structured launcher for HGDN helpers, with subcommands, named presets, and optional TOML env configs
- `configs/hgdn/winner_20260405_19.toml`: reusable config for the active H100-confirmed HGDN kernel winner
- `configs/hgdn/winner_20260405_19_cuda_packed_conv_atenbwd.toml`: rejected H100 sidecar that keeps the exact-length CUDA packed-conv forward but hands backward back to ATen/cuDNN
- `configs/hgdn/winner_20260405_19_cuda_packed_conv_aten_weightbwd.toml`: local-only ownership split that keeps custom forward and input-grad but hands only weight-grad back to ATen
- `configs/hgdn/winner_20260405_19_cuda_fused_frontend.toml`: rejected H100 sidecar that keeps the promoted packed custom-backward conv path and fuses only the post-conv frontend shell behind the old custom-op boundary
- `configs/hgdn/winner_20260405_19_cuda_fused_frontend_lib.toml`: rejected H100 sidecar that keeps the packed frontend math but moves it to a compile-visible `torch.library` boundary
- `configs/hgdn/winner_20260405_19_cuda_split_norm_lib.toml`: next H100 sidecar candidate that keeps the promoted packed custom-backward conv winner and moves only post-conv split+q/k norm to a compile-visible `torch.library` boundary
- `configs/hgdn/winner_20260405_19_cuda_frontend_nct_custombwd.toml`: composed sidecar candidate that keeps the promoted exact-length packed custom-backward conv path and layers the compile-visible NCT frontend op above `preact_nct`
- `configs/hgdn/winner_20260405_19_cuda_frontend_nct.toml`: compile-visible NCT frontend sidecar candidate that keeps depthwise conv in ATen and moves post-conv `SiLU + split + q/k norm` one boundary earlier
- `configs/hgdn/winner_20260405_19_cuda_packed_conv.toml`: exact-length CUDA packed-conv sidecar candidate that replaces the packed qkv causal depthwise conv family itself
- `configs/hgdn/winner_20260405_19_cuda_split_norm.toml`: real-kernel H100 sidecar candidate that replaces only the post-conv packed split+q/k norm stage
- `configs/hgdn/winner_20260405_19_single_contig.toml`: rejected Python-side single-contig front-end candidate kept in-tree for reference
- `configs/hgdn/winner_20260405_19_split_copy.toml`: rejected generated-path split-copy front-end candidate kept in-tree for reference
- `configs/hgdn/winner_20260405_11.toml`: reusable config for the active timestamped HGDN kernel winner
- `configs/hgdn/winner_20260405_11_custombwd.toml`: reusable config for the packed depthwise custom-backward candidate layered on that winner
- `configs/hgdn/winner_20260405_11_cuda_fused.toml`: experimental fused-CUDA variant of that winner
- `scripts/screen_hgdn_arch_sizes.py`: CPU-only artifact-proxy screen for resized HGDN architecture candidates
- `scripts/compare_hgdn_fixed2k.py`: structured W&B comparator for completed HGDN fixed-step H100 runs
- `scripts/hgdn_kernel_scoreboard.py`: family-level H100 kernel gatekeeper that scores same-day controls and candidates against explicit `meaningful_win`, `flat_band`, copy-tax, and compile-penalty rules; accepts extracted trees and direct `.7z` bundles via `py7zr` or a system `7z` fallback
- `configs/hgdn/winner_20260405_11_retune.toml`: first-pass retune family around the active timestamped HGDN kernel winner
- `configs/hgdn/retune_*.toml`: named runnable configs for the first resized-HGDN shortlist
- `model.py`: hybrid HGDN architecture and presets
- `train_gpt_hybrid.py`: hybrid trainer
- `scripts/sweep.sh`: launch helper and perf-harness env contract
- `scripts/run_h100_single_gpu_hgdn.sh`: 1xH100 helper backend for perf and fixed-step target-hardware calibration
- `scripts/run_h100_single_gpu_hgdn_profile.sh`: 1xH100 helper backend for compiled and eager-attribution profiler captures
- `scripts/run_hgdn_cuda_preflight.sh`: single-process CUDA preflight backend for the HGDN kernel path
- `scripts/run_hgdn_local_phase1.sh`: sequential local 4070 phase-1 investigation backend
- `scripts/profile_hgdn_local_hotpath.py`: bare-GDN / hybrid-FB / optimizer-only local profiler
- `scripts/analyze_hgdn_phase1.py`: bucket-attribution and boundary-audit analyzer for phase-1 bundles
- `scripts/compare_hgdn_phase1.py`: structured before/after comparator for two local phase-1 bundles
- `scripts/run_laptop_norm_compare.sh`: 1x laptop GPU helper for fixed-step `pre/post/keel` norm screens
- `setup_hgdn_cuda.py`: in-repo build entrypoint for the optional HGDN fused CUDA extension
- `scripts/build_hgdn_cuda.sh`: thin shell wrapper around the HGDN fused CUDA build
- `scripts/hgdn_cuda_parity.py`: direct CUDA-vs-reference parity checks for the HGDN fused extension
- `docs/HGDN_CUDA_FUSED.md`: extension layout, build notes, parity commands, and staged validation guidance
- `docs/HARDWARE_TRANSFER.md`: what does and does not transfer from local 4070 profiling to 1xH100 profiling
- `docs/REDUNDANCY_AUDIT.md`: interim audit of duplication, dead code, and consolidation targets before more kernel work
- `docs/PROFILING_LOG.md`: tracked profiler checkpoints and conclusions that should survive beyond raw `profiles/` bundles
- `docs/REFERENCE.md`: architecture/reference notes
- `docs/TODO.md`: deferred follow-ups and break-glass items

## Preferred Launch Interface

The shell helpers remain the execution backend, but the preferred entrypoint is
now [scripts/hgdn.py](/home/pszemraj/workspace/projects/parameter-golf/scripts/hgdn.py).

Why:

- the old workflow required long inline env blocks
- the launcher gives named subcommands instead of remembering which shell helper
  to call
- the launcher gives named HGDN presets instead of repeating the same `GDN_*`
  overrides
- the launcher supports TOML configs when a setup should be reused

Active timestamped presets:

- `default`
- `convcontig`
- `packed-qkv`
- `winner-20260405-19`
- `winner-20260405-19-cuda-packed-conv-aten-bwd`
- `winner-20260405-19-cuda-packed-conv-aten-weight-bwd`
- `winner-20260405-19-cuda-fused-frontend`
- `winner-20260405-19-cuda-fused-frontend-lib`
- `winner-20260405-19-cuda-split-norm-lib`
- `winner-20260405-19-cuda-frontend-nct-custom-bwd`
- `winner-20260405-19-cuda-frontend-nct`
- `winner-20260405-19-cuda-packed-conv`
- `winner-20260405-19-single-contig`
- `winner-20260405-11`
- `winner-20260405-11-custom-bwd`
- `winner-20260405-11-cuda-fused`
- `winner-20260405-11-cuda-output-only`

Naming note:

- use timestamped winner names going forward so launch commands still make sense later
- the older `current-*` aliases still exist only so historical notes and old command lines do not hard-break
- the active reference stamp is `20260405-19`

Current best H100-confirmed HGDN kernel preset:

- `winner-20260405-19`
- equivalent to:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
  - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
- promotion basis:
  - repeated H100 controls:
    - `904.80 ms`
    - `904.12 ms`
  - repeated H100 candidate runs:
    - `853.23 ms`
    - `853.20 ms`
  - promoted delta:
    - `904.46 -> 853.21 ms` (`-5.67%`)

Kernel-work guardrail:

- the last few results make the practical point clear:
  - the remaining wins are not going to come from rearranging Python-side views
    or `.contiguous()` calls
  - the H100 profiles are telling us to go after the actual generated/kernel
    path
- practical rule:
  - treat Python-side layout reshuffles as suspect by default on the compiled
    HGDN path
  - prefer lower-level ATen, Triton, CUDA, or other generated-path changes for
    the next tranche
  - use the family scoreboard before declaring a kernel family exhausted:
    - meaningful win:
      - `max(5 ms, 0.5% of control_step_ms, 3 * control_noise_ms)`
    - flat band:
      - `max(2.5 ms, 0.25% of control_step_ms, 2 * control_noise_ms)`
    - derived signals:
      - `compiled_copy_tax = aten::copy_ + direct_copy_kernel_cuda`
      - `compile_specific_penalty = compiled_hotpath_delta - eager_hotpath_delta`
    - stop a family when it is `SATURATED`, after repeated `FLAT`, or after
      repeated `INTEGRATION_BOTTLENECK` results
- updated practical read:
  - the Python-side `single-contig` attempt lost
  - the generated-path `split-copy` attempt also lost locally
  - the real CUDA post-conv split+q/k norm kernel also lost on compiled H100
  - the first exact-length CUDA packed-conv replacement was locally strong
  - the first full-custom packed-conv H100 run then failed hard because the
    custom weight-backward kernel dominated the compiled step
  - the ATen-backward packed-conv follow-up also failed on H100:
    - `aten::convolution_backward` itself stayed fine
    - the packed-conv shell reopened copy and left-pad glue badly
  - the ATen-weight split did not win the clean local screen
  - the full-custom packed-conv family only became live again after a material
    rewrite of the custom weight-backward kernel
  - the rewritten full-custom packed-conv then lost only narrowly on H100:
    - `880.95 -> 903.33 ms` (`+2.54%`)
    - copy tax actually improved
    - the remaining loss moved into the post-conv frontend shell
  - the existing frontend-only fused CUDA path was locally strong on top of the
    promoted winner, but it still lost badly on H100:
    - controls:
      - `849.98 ms`
      - `850.04 ms`
    - candidate:
      - `929.75 ms`
      - `928.82 ms`
      - `931.50 ms`
    - mean delta:
      - `850.01 -> 930.02 ms` (`+9.41%`)
    - eager improved and copy tax improved, but compiled `DDP.forward`,
      `CompiledFxGraph`, and `_PackedQKVFrontendFunctionBackward` all worsened
  - the standalone compile-visible NCT frontend follow-up lost badly on H100
    because it reopened copy tax and dropped the promoted custom-backward win
  - the composed `k10 + k13` follow-up also lost on H100:
    - eager improved
    - compiled copy tax stayed flat
    - compiled `ProfilerStep*`, `CompiledFxGraph`, and `DDP.forward` got much
      worse
  - practical conclusion:
    - the old fused-frontend family is an integration bottleneck at the current
      custom-op boundary
    - keep the fused post-conv math idea only if the boundary changes
      materially
    - the compile-visible packed fused frontend also lost badly on H100:
      - controls:
        - `853.15 ms`
        - `855.07 ms`
      - candidate:
        - `931.88 ms`
        - `931.62 ms`
        - `929.84 ms`
      - mean delta:
        - `854.11 -> 931.11 ms` (`+9.02%`)
      - eager improved and copy tax improved, but the packed frontend backward,
        conv weight-backward, and surrounding `CompiledFxGraph` rows still got
        materially worse
    - the full packed frontend library family is now also an integration
      bottleneck at its current ownership split
    - the next live family is a narrower compile-visible split-norm sidecar on
      top of the promoted custom-conv winner:
      `winner-20260405-19-cuda-split-norm-lib`
    - the current compile-visible NCT frontend family is an integration
      bottleneck at this abstraction level
    - do not spend more H100 time on that family unless the boundary changes
      materially
    - do not reopen the old fused-frontend autograd island again without a
      material implementation change
    - do not send the full packed frontend library family back to H100 unless
      its backward ownership changes materially

H100 batch protocol:

- when there is only one live kernel family, prefer one larger same-day H100
  batch over multiple tiny consecutive rounds
- default shape for that batch:
  - 2 control perf runs
  - control eager and compiled profiles when attribution is still evolving
  - candidate preflight
  - candidate eager profile
  - 2 candidate perf runs, or 3 if we want tighter confidence without opening a
    second family
  - candidate compiled profile
- do not batch a second kernel family unless it is genuinely orthogonal and a
  loss would still teach us something independent
- practical reason:
  - this improves same-day noise estimates and profiler attribution without
    muddying the family decision

Training-clock reminder:

- the trainer's `MAX_WALLCLOCK_SECONDS=600` cap measures training time, not
  simple process wallclock
- warmup and compile priming happen before the training timer starts
- validation time is also excluded from the training timer
- serialization and final roundtrip eval are outside the training timer too
- the challenge still has a separate external 10-minute evaluation cap on top
  of the 10-minute training budget
- practical consequence:
  - expensive compile-heavy CUDA kernels are not automatically disqualified by
    startup cost alone
  - but compile time is not a reason to ignore steady-state runtime regressions
    or evaluation-time constraints

Latest screened front-end candidate:

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
    - `CompiledFxGraph` delta:
      - `+542.31 ms`
    - `DDP.forward` delta:
      - `+183.48 ms`
- decision:
  - keep `winner-20260405-19` active
  - park the current standalone NCT-frontend family until the compile boundary
    changes materially

Next live low-level sidecar:

- `winner-20260405-19-cuda-packed-conv`
- equivalent to:
  - `winner-20260405-19`
  - `GDN_USE_CUDA_PACKED_CONV=1`
- refreshed same-day local phase-1 result after the weight-backward rewrite:
  - baseline:
    - `profiles/rtx5090_phase1_winner20260405_19_r8/`
  - candidate:
    - `profiles/rtx5090_phase1_cuda_packedconv_wgradv3_fix2/`
  - direct comparison:
    - `profiles/rtx5090_phase1_cuda_packedconv_wgradv3_fix2/compare_vs_rtx5090_phase1_winner20260405_19_r8/comparison.md`
  - console step average:
    - `3085.62 -> 2874.43 ms` (`-6.84%`)
  - trainer `ProfilerStep*`:
    - `6154.71 -> 5707.37 ms` (`-7.27%`)
  - trainer `DistributedDataParallel.forward`:
    - `1792.51 -> 1627.82 ms`
  - trainer `aten::copy_`:
    - `646.49 -> 364.96 ms`
  - trainer `aten::mul`:
    - `1165.79 -> 969.75 ms`
  - trainer `gdn.recurrence`:
    - `164.28 -> 165.59 ms`
  - trainer `gdn.qkv_conv_output_contiguous`:
    - `76.26 -> 40.70 ms`
  - recurrence-facing boundary stayed contiguous through `conv_qkv`,
    `norm_qkv`, and `recurrence_inputs`
- decision:
  - this is now the next H100 sidecar family
  - if H100 agrees, it becomes the next promoted kernel winner
  - the family was only reopened because the custom weight-backward kernel
    changed materially, not because the earlier `h100k15` failure was ignored

Older screened packed-conv candidate:

- `winner-20260405-19-cuda-packed-conv-aten-bwd`
- equivalent to:
  - `winner-20260405-19`
  - `GDN_USE_CUDA_PACKED_CONV_ATEN_BACKWARD=1`
- purpose:
  - keep the exact-length CUDA packed-conv forward
  - hand all conv backward ownership back to ATen/cuDNN
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
- failure mode:
  - `aten::convolution_backward` itself was not the problem
  - the packed-conv shell reopened copy and left-pad glue badly
  - compiled profile top rows:
    - `aten::copy_: 491.69 ms`
    - `direct_copy_kernel_cuda: 478.43 ms`
    - `gdn.qkv_conv_cuda_aten_bwd_left_pad: 297.25 ms`
    - `gdn.qkv_conv_cuda_aten_bwd_input_grad: 201.69 ms`
- decision:
  - reject this ATen-backward ownership on H100
  - do not keep iterating shell variants at this ownership split

Older screened packed-conv candidate:

- `winner-20260405-19-cuda-packed-conv-aten-weight-bwd`
- equivalent to:
  - `winner-20260405-19`
  - `GDN_USE_CUDA_PACKED_CONV_ATEN_WEIGHT_BACKWARD=1`
- purpose:
  - keep custom forward and custom input-grad
  - hand only weight-grad back to ATen
- local result:
  - clean direct hotpath reject
  - `gdn_fwd_bwd` total self CUDA:
    - `199.41 -> 249.66 ms`
  - `hybrid_fwd_bwd` total self CUDA:
    - `351.10 -> 362.01 ms`
- decision:
  - keep this split only as a discarded ownership waypoint
  - do not spend H100 time on it

Older screened packed-conv candidate:

- `winner-20260405-19-cuda-packed-conv`
- equivalent to:
  - `winner-20260405-19`
  - `GDN_USE_CUDA_PACKED_CONV=1`
- purpose:
  - replace the packed qkv causal depthwise conv itself with a narrow
    exact-length CUDA op and a full custom backward
- first H100 result:
  - reject
  - same-day controls:
    - `848.75 ms`
    - `848.76 ms`
  - candidate:
    - `1796.70 ms`
    - `1793.67 ms`
  - mean delta:
    - `848.75 -> 1795.19 ms` (`+111.51%`)
- first failure mode:
  - the old custom weight-backward kernel dominated the compiled step
  - compiled profile top rows:
    - `_PackedQKVConvFunctionBackward: 4161.58 ms`
    - `causal_dwconv_weight_backward_kernel: 3973.02 ms`
    - `causal_dwconv_preact_forward_kernel: 138.52 ms`
    - `causal_dwconv_input_backward_kernel: 133.55 ms`
- current decision:
  - the original implementation is rejected
  - the family is only reopened because the weight-backward kernel changed
    materially and now screens positive locally

Older screened front-end candidate:

- `winner-20260405-19-cuda-frontend-nct`
- equivalent to:
  - `winner-20260405-19`
  - `GDN_USE_CUDA_FRONTEND_NCT=1`
- purpose:
  - move the post-conv `SiLU + q/k/v split + q/k L2 norm` stage one boundary
    earlier into a compile-visible `torch.library` op over `preact_nct`
  - keep the depthwise conv itself in the normal ATen path
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
- failure mode:
  - this was not a true `k10 + k13` composition
  - the path dropped `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`, so it gave
    back the promoted exact-length conv backward win
  - compiled H100 reopened copy tax and paid large frontend-op overhead
- decision:
  - reject this standalone sidecar on H100
  - keep only the compile-visible NCT frontend idea as a building block inside
    the stronger combined candidate

- `winner-20260405-19-cuda-split-norm`
- equivalent to:
  - `winner-20260405-19`
  - `GDN_USE_CUDA_SPLIT_NORM=1`
- purpose:
  - keep the promoted packed qkv front-end and custom depthwise backward
  - replace only the post-conv `split + q/k l2 norm + v materialization` stage
    with a narrow CUDA op
  - keep recurrence math unchanged and preserve the recurrence-facing contract
- status:
  - implementation and tests are in-tree
  - local phase-1 was directionally positive but still inside the rough laptop
    noise band
  - H100 sidecar rejected it on compiled perf
  - compared against same-day H100 controls:
    - controls: `879.46 ms`, `878.20 ms`
    - candidate: `959.13 ms`, `961.33 ms`
    - mean delta: `878.83 -> 960.23 ms` (`+9.26%`)
  - compiled H100 profile failure mode:
    - `aten::copy_`: `57.72 -> 210.65 ms`
    - `_PackedQKVSplitL2NormFunction: 145.83 ms`
    - `_PackedQKVSplitL2NormFunctionBackward: 194.82 ms`
    - the real depthwise-conv buckets barely moved
  - decision:
    - keep `winner-20260405-19` active
    - reject this standalone sidecar on H100
    - keep it only as a possible ingredient for a larger packed front-end
      kernel pipeline

Older screened front-end candidate:

- `winner-20260405-19-split-copy`
- equivalent to:
  - `winner-20260405-19`
  - `GDN_PACKED_QKV_SPLIT_COPY=1`
- purpose:
  - replace the packed split-plus-three-contiguous path with
    `aten.split_with_sizes_copy`
  - keep the packed recurrence-facing q/k/v contract contiguous
  - keep recurrence math unchanged
- status:
  - implementation and tests are in-tree
  - local phase-1 reject
  - compared against `profiles/rtx4070_cuda_base/`:
    - console step average: `3320.37 -> 3752.76 ms` (`+13.02%`)
    - `ProfilerStep*` self-device total: `6610.92 -> 7546.44 ms` (`+14.15%`)
  - boundary audit stayed clean, but trainer buckets still got worse:
    - `aten::mul`: `1012.30 -> 1276.17 ms`
    - `aten::copy_`: `785.65 -> 798.02 ms`
    - `gdn.recurrence`: `177.23 -> 191.34 ms`
  - do not promote this candidate to H100

- `winner-20260405-19-single-contig`
- equivalent to:
  - `winner-20260405-19`
  - `GDN_PACKED_QKV_SINGLE_CONTIG=1`
- purpose:
  - keep the promoted packed qkv front-end and custom backward
  - reduce post-conv clone/materialization by doing one packed `.contiguous()` before the q/k/v split instead of three per-output contiguous calls
  - keep q/k normalization on the same Python-side `l2_norm` path
- status:
  - implementation and tests are in-tree
  - local phase-1 reject on the RTX 4070 laptop
  - optional H100 sidecar also failed to justify promotion
  - keep `winner-20260405-19` active; do not promote this candidate

Laptop-noise note:

- this local machine is a laptop RTX 4070, not a dedicated bench box
- treat small local deltas, roughly inside `+/-5%`, as screening only
- promotion decisions for close calls should be made on H100, not on one laptop run
- practical example:
  - `winner-20260405-19-single-contig` was close enough locally to justify a belt-and-suspenders H100 sidecar
  - that sidecar still came back too flat to promote
  - `winner-20260405-19-cuda-split-norm` was close enough locally to justify
    H100 validation
  - H100 then killed it cleanly, which is why close calls should still be
    decided there rather than on the laptop

Empirical contract note:

- the trainer seeds Python/NumPy/Torch/CUDA from `SEED`, and the launch helpers now pin:
  - `SEED=1337`
  - `PYTHONHASHSEED=$SEED`
  - `CUDNN_BENCHMARK=0`
  unless explicitly overridden
- `scripts/sweep.sh` now prints `seed`, `pythonhashseed`, and `cudnn_benchmark` in the launch summary
- perf/profile/fixed-step H100 helpers now isolate compile caches per `RUN_ID` and clear those per-run cache directories before launch
- `scripts/run_hgdn_cuda_preflight.sh` now prints its seed/cache contract too
- practical read for the completed `h100k10` batch:
  - no rerun is required just for cache hygiene
  - the perf/profile runs already used isolated per-run compile caches and unique run ids
  - these changes make that contract explicit and extend the same discipline to future fixed-step runs

Experimental fused-CUDA HGDN preset:

- `winner-20260405-11-cuda-fused`
- equivalent to:
  - `winner-20260405-11`
  - `GDN_OUTPUT_NORM_FP32=1`
  - `GDN_USE_CUDA_FUSED_FRONTEND=1`
  - `GDN_USE_CUDA_FUSED_OUTPUT=1`
- status:
  - local build/parity passed in `pg`
  - local phase-1 improved the current winner on trainer eager step time
  - H100 build/parity also passed
  - H100 eager and compiled perf both regressed badly
  - keep it experimental only; do not treat it as the active path

Experimental output-only fused preset:

- `winner-20260405-11-cuda-output-only`
- equivalent to:
  - `winner-20260405-11`
  - `GDN_OUTPUT_NORM_FP32=1`
  - `GDN_USE_CUDA_FUSED_OUTPUT=1`
- status:
  - local preflight passed
  - local phase-1 lost slightly vs the non-extension current winner
  - H100 follow-up also lost:
    - compiled hybrid perf:
      - `904.46 -> 944.37 ms` (`+4.41%`)
  - keep it as a parked experiment surface, not an active H100 candidate

Experimental packed depthwise custom-backward preset:

- `winner-20260405-11-custom-bwd`
- equivalent to:
  - `winner-20260405-11`
  - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
- status:
  - local parity tests pass
  - local hotpath and local phase-1 both show real depthwise-bucket reductions
  - the short phase-1 trainer console average was noisy, so it was re-checked with a stable local compiled perf pair
  - stable local compiled perf on the 4070 with `TORCH_BLAS_PREFER_CUBLASLT=1` improved from `2191.34 ms` to `2126.57 ms` (`-2.96%`)
  - because that local win was smaller than the laptop-noise guardrail, it was forced through repeated H100 controls plus repeated candidate perf
  - that H100 gate passed, so this path is now promoted as:
    - `winner-20260405-19`

Examples:

```bash
python scripts/hgdn.py preflight --preset winner-20260405-19
```

```bash
python scripts/hgdn.py local-phase1 --preset winner-20260405-19 --run-prefix rtx4070_phase1
```

```bash
conda run -s --name pg python setup_hgdn_cuda.py build_ext --inplace
```

```bash
conda run -s --name pg python scripts/hgdn_cuda_parity.py
```

```bash
python scripts/hgdn.py preflight --preset winner-20260405-11-cuda-fused --compile-strategy hybrid
```

```bash
python scripts/hgdn.py preflight --preset winner-20260405-11-cuda-output-only --compile-strategy hybrid
```

```bash
python scripts/hgdn.py preflight --preset winner-20260405-11-custom-bwd --compile-strategy model
```

```bash
python scripts/hgdn.py h100-profile hybrid-eager --preset winner-20260405-19 --run-prefix h100k10a
```

```bash
python scripts/hgdn.py h100-perf perf --preset winner-20260405-19 --run-prefix h100k10a --offline
```

```bash
conda run -s --name pg python scripts/hgdn.py arch-size-screen \
  --config configs/hgdn/winner_20260405_11_retune.toml
```

```bash
conda run -s --name pg python scripts/hgdn.py fixed2k-compare \
  --name h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048 \
  --name h100k6_fixed2k_depth_mlp4.0_seq2048 \
  --reference h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048 \
  --output-dir profiles/fixed2k_compare/h100k6_pair
```

```bash
python scripts/hgdn.py h100-perf fixed2k-hybrid \
  --config configs/hgdn/retune_trim_layers_14.toml \
  --run-prefix h100k7a \
  --online \
  --set WANDB_PROJECT=pg-hconv-ablations \
  --set WANDB_WATCH=gradients
```

Using the reusable TOML config:

```bash
python scripts/hgdn.py h100-profile hybrid --config configs/hgdn/winner_20260405_19.toml --run-prefix h100k10a
```

For advanced cases that still need one-off passthrough envs, use:

```bash
python scripts/hgdn.py h100-profile hybrid-eager \
  --preset winner-20260405-19 \
  --run-prefix h100k10a \
  --set GDN_LOG_LAYOUTS=1 \
  --set PROFILE_ROW_LIMIT=80
```

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
- `WANDB_WATCH` is now an explicit trainer knob. The helper scripts default it to `none` so sweep screens keep normal online metric logging without gradient-histogram overhead.
- The trainer now has a profiling harness:
  - `PROFILE=1`
  - scheduled `torch.profiler` capture
  - trace export under `profiles/<run_id>/traces/`
  - operator summary export to `profiles/<run_id>/key_averages.{json,csv}`
  - one-shot HGDN FLA-boundary layout logging via `GDN_LOG_LAYOUTS=1`
- The local profiling workflow is now structured rather than ad hoc:
  - `scripts/profile_hgdn_local_hotpath.py` writes `trace + key_averages.{json,csv}`
  - `scripts/analyze_hgdn_phase1.py` turns the saved profiles into a bucket-attribution table
  - `GDN_AUDIT_BOUNDARIES=1` emits JSONL boundary records for q/k/v/gate/output dtype-layout audit
  - `scripts/run_hgdn_local_phase1.sh` runs the full local diagnostic bundle sequentially and writes analysis artifacts under one `profiles/<run_prefix>/` root
- The branch now has a tiny CUDA preflight:
  - `scripts/run_hgdn_cuda_preflight.sh`
  - runs direct single-process `gdn_eager`, `hybrid_eager`, and `hybrid_compiled` checks
  - intended to catch HGDN kernel-path regressions before handing the user longer GPU commands
- The trainer no longer blanket-promotes every `CastedLinear` module back to `fp32` after `.bfloat16()` model init:
  - large attention/MLP/GDN feature-map weights remain `bf16`
  - explicit control-path weights and low-dimensional parameters still get restored to `fp32`
  - `CastedLinear.forward()` now skips the cast path entirely when weight and activation dtypes already match
- The branch now exposes per-path GDN conv toggles:
  - `GDN_USE_Q_CONV`
  - `GDN_USE_K_CONV`
  - `GDN_USE_V_CONV`
- The branch now has an explicit conv-to-recurrence layout experiment knob:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - first local result:
    - fixes q/k/v contiguity from `conv_qkv` through `recurrence_inputs`
    - reduces local trainer self-device time from `25,990.59 ms` to `25,258.00 ms` (`-2.82%`)
    - reduces `gdn.recurrence`, `gdn.norm_qkv`, and trainer-level `aten::mul`/`aten::copy_`
    - increases `gdn.conv_qkv`, so it remains an experimental candidate pending H100 confirmation
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

### Recompile diagnosis

Using `TORCH_LOGS=recompiles` on a short compiled HGDN run shows startup-only recompiles, not a steady-state recompile loop:

- one alias-guard recompile in the GDN block because the first block initially sees `x is x0`
- one attention recompile when the rotary cache transitions from `None` to populated
- a few Muon helper recompiles because `zeropower_via_newtonschulz5` sees several distinct matrix shapes

Current conclusion:

- the branch does recompile during startup and warmup
- the branch does not currently show evidence of recompiling every 25 steps during steady-state training
- if a run appears to freeze every 25 steps, check W&B watch/logging before blaming Dynamo

Recommended diagnostics:

```bash
TORCH_LOGS=recompiles,graph_breaks \
PATH=/home/pszemraj/miniforge3/envs/pg/bin:$PATH \
USE_WANDB=0 ITERATIONS=30 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=10 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 \
COMPILE=1 COMPILE_STRATEGY=model GDN_RATIO=1 MLP_MULT=3.25 NORM_STYLE=pre \
PERF_SKIP_FINAL_EVAL=1 RUN_ID=diag_recompiles scripts/sweep.sh single
```

For a richer trace, PyTorch recommends `TORCH_TRACE=/tmp/tracedir` plus `tlparse` on the trace directory.

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
| Attention-only baseline | 1024 | `0G+16A` | `812.40` | `80,670` | `1.00x` |
| Hybrid `GDN_RATIO=3` | 1024 | `12G+4A` | `1123.78` | `58,317` | `1.38x` |
| Attention-only baseline | 2048 | `0G+16A` | `941.98` | `69,573` | `1.00x` |
| Hybrid `GDN_RATIO=3` | 2048 | `12G+4A` | `1159.84` | `56,504` | `1.23x` |

Interpretation:

- The hybrid penalty shrinks at `2048`, but the attention-only baseline is still faster on raw throughput.
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

From the logged 2k quality runs and follow-up controls:

| Config | Raw `state_dict` bytes | `int8+zlib` bytes | Code bytes | Total artifact |
|---|---:|---:|---:|---:|
| Hybrid `GDN_RATIO=1, MLP_MULT=3.25` | `100,338,699` | `10,909,417` | `53,459` | `10,962,876` |
| Attention-only baseline `MLP_MULT=3.75` | `100,047,451` | `9,401,537` | `53,459` | `9,454,996` |
| Attention-only baseline `MLP_MULT=4.0` | `104,766,043` | `9,611,898` | `56,483` | `9,668,381` |

### Exact init-state quantization audit

These numbers use the real `quantize_state_dict_int8 -> torch.save -> zlib.compress` path on the same model shapes:

| Config | Params | Int8 payload | Raw quant `torch.save` | Init `int8+zlib` |
|---|---:|---:|---:|---:|
| Hybrid `GDN_RATIO=1, MLP_MULT=3.25` | `25,279,680` | `25,650,944` | `25,750,713` | `8,140,843` |
| Attention-only baseline `MLP_MULT=3.75` | `25,193,600` | `25,374,208` | `25,453,817` | `7,271,843` |

Interpretation:

- The int8 payload itself is only about `3.95x` smaller than raw fp32 tensor bytes.
- The large extra shrink happens in zlib on the serialized quantized object.
- Training increases payload entropy, so trained checkpoints compress noticeably worse than init-state ones:
  - hybrid payload-to-zlib: about `3.15x` at init, `2.35x` after training
  - depth payload-to-zlib: about `3.49x` at init, `2.70x` after training

Current conclusion:

- The old proxy table should not be used for budget decisions.
- Future size decisions should use the trainer's real artifact audit fields, not a static heuristic.
- The hybrid is genuinely less compressible than the attention-only baseline, so size matching needs to use actual bytes, not only parameter counts.
- A 600-second local wallclock screen is good enough to reject obviously bad size candidates, but not good enough to stand in for the fixed-step artifact outcome.

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

Depth follow-up:

- `RUN_ID=quality_depth_mlp40_seq2k`
- `MLP_MULT=4.0`

### Validation BPB

| Config | 500 | 1000 | 1500 | 2000 | Final int8 roundtrip |
|---|---:|---:|---:|---:|---:|
| Hybrid `GDN_RATIO=1, MLP_MULT=3.25` | `2.9000` | `2.7492` | `2.6148` | `2.5138` | `2.5209` |
| Attention-only baseline `MLP_MULT=3.75` | `3.0342` | `2.8792` | `2.7660` | `2.6604` | `2.6778` |
| Attention-only baseline `MLP_MULT=4.0` | `3.0298` | `2.8709` | `2.7586` | `2.6550` | `2.6715` |

### Delta

| Checkpoint | Hybrid improvement over depth |
|---|---:|
| `500` | `0.1342 BPB` |
| `1000` | `0.1300 BPB` |
| `1500` | `0.1512 BPB` |
| `2000` | `0.1466 BPB` |
| Final roundtrip | `0.1569 BPB` |

### Delta vs enlarged attention-only baseline

| Checkpoint | Hybrid improvement over depth `MLP_MULT=4.0` |
|---|---:|
| `500` | `0.1298 BPB` |
| `1000` | `0.1217 BPB` |
| `1500` | `0.1438 BPB` |
| `2000` | `0.1412 BPB` |
| Final roundtrip | `0.1506 BPB` |

### Quantization sensitivity

| Config | Pre-roundtrip | Roundtrip | Degradation |
|---|---:|---:|---:|
| Hybrid | `2.5138` | `2.5209` | `+0.0071` |
| Depth `MLP_MULT=3.75` | `2.6604` | `2.6778` | `+0.0174` |
| Depth `MLP_MULT=4.0` | `2.6550` | `2.6715` | `+0.0165` |

Interpretation:

- The hybrid wins cleanly at every measured checkpoint.
- The hybrid also degrades less after the int8 roundtrip than the depth-control baseline.
- Giving the attention-only baseline more MLP width from `3.75 -> 4.0` only improved final BPB by about `0.0054`, which is far smaller than the hybrid-vs-baseline gap.
- The architecture question is currently answered in favor of the hybrid on this local quality comparison.

## Time-Matched Reindex

The existing quality comparison is step-matched. The hybrid is slower, so the fair wall-clock question is whether the attention-only baseline would erase the gap if given its extra steps.

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

## 1xH100 Calibration

Date of target-hardware calibration: `2026-04-03`

These runs used the branch helper on a single H100:

- `scripts/run_h100_single_gpu_hgdn.sh perf`
- `scripts/run_h100_single_gpu_hgdn.sh fixed2k`

Only the H100 results are treated as decision-relevant here. A100 runs were collected as a curiosity check and are not used for the branch conclusion.

### Throughput

Perf harness contract:

- `NGPU=1`
- `TRAIN_BATCH_TOKENS=524288`
- `TRAIN_SEQ_LEN=2048`
- `ITERATIONS=50`
- `PERF_IGNORE_STEPS=10`
- `COMPILE_STRATEGY=model`

| Config | Blocks | Step ms | Tokens/s | Ratio vs depth |
|---|---|---:|---:|---:|
| Attention-only baseline `MLP_MULT=4.0` | `0G+16A` | `714.74` | `733,538` | `1.00x` |
| Hybrid `GDN_RATIO=1, MLP_MULT=3.25` | `8G+8A` | `1002.72` | `522,866` | `1.40x` |

Interpretation:

- On target hardware, the HGDN throughput penalty is materially worse than the local 4070 estimate.
- The laptop did capture the direction correctly, but it understated the size of the throughput tax.
- `COMPILE_STRATEGY=model` remains the right choice; there is no evidence here that selective compile should be revisited first.

### Fixed-2k Quality

Fixed-step contract:

- `NGPU=1`
- `TRAIN_BATCH_TOKENS=524288`
- `TRAIN_SEQ_LEN=2048`
- `ITERATIONS=2000`
- `VAL_LOSS_EVERY=500`
- `COMPILE_STRATEGY=model`

| Config | Train loss @1000 | Train loss @2000 | Eval BPB @1000 | Eval BPB @1500 | Eval BPB @2000 | Final roundtrip | Artifact bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention-only baseline `MLP_MULT=4.0` | `4.4989` | `4.2910` | `2.6662` | `2.6166` | `2.5457` | `2.5950` | `9,991,658` |
| Hybrid `GDN_RATIO=1, MLP_MULT=3.25` | `4.3018` | `3.9760` | `2.5477` | `2.4594` | `2.3587` | `2.3719` | `10,936,348` |

### Granular curve comparison

The two H100 runs show the same qualitative pattern on both training and validation metrics: after the noisy startup region, the hybrid stays below the attention-only baseline.

Selected training-loss checkpoints:

| Step | Hybrid train loss | Depth train loss | Hybrid advantage |
|---|---:|---:|---:|
| `200` | `4.8443` | `5.0946` | `0.2504` |
| `400` | `4.5009` | `4.8394` | `0.3386` |
| `800` | `4.4042` | `4.5552` | `0.1510` |
| `1000` | `4.3018` | `4.4989` | `0.1970` |
| `1400` | `4.1591` | `4.4101` | `0.2510` |
| `1800` | `4.0413` | `4.3587` | `0.3174` |
| `2000` | `3.9760` | `4.2910` | `0.3150` |

Selected validation checkpoints:

| Step | Hybrid `eval/bpb` | Depth `eval/bpb` | Hybrid advantage |
|---|---:|---:|---:|
| `1000` | `2.5477` | `2.6662` | `0.1184` |
| `1500` | `2.4594` | `2.6166` | `0.1573` |
| `2000` | `2.3587` | `2.5457` | `0.1870` |
| Final roundtrip | `2.3719` | `2.5950` | `0.2231` |

Interpretation:

- The hybrid is not merely holding parity while paying a throughput tax; it is learning faster per step on H100 too.
- The H100 gap is stronger than the earlier laptop comparison on both train loss and validation BPB.
- The hybrid artifact is still larger by `944,690` bytes, so this is not a perfectly size-matched comparison yet, but the quality gap is also larger than the local branch result.
- The roundtrip gap widening from `0.1870 -> 0.2231` suggests the hybrid remains at least as quantization-robust as the attention-only baseline under this contract.

## Size-Matching Follow-Up

Even with the stronger `GDN_RATIO=1, MLP_MULT=3.25` hybrid:

- Hybrid actual total artifact bytes: `10,962,876`
- Depth actual total artifact bytes: `9,454,996`

That means the hybrid had about a `16%` artifact-size advantage in the original comparison. Some unknown fraction of the BPB gap is therefore still "more compressed params."

Two follow-up controls were run to pressure this assumption:

1. `MLP_MULT=4.7`, local 600-second calibration
   - `step 577 val_bpb: 2.7986`
   - total artifact: `17,092,318`
   - result: clearly over budget and therefore invalid as a submission-matched control
2. `MLP_MULT=4.0`, full 2k-step rerun
   - final total artifact: `9,668,381`
   - result: still about `11.8%` smaller than the hybrid, but only improves the attention-only baseline by about `0.0054` BPB at step 2000

Interpretation:

- The old `MLP_MULT=4.7` projection was wrong; init-state compression did not transfer to the trained model.
- The 600-second local screen was still useful to reject `4.7`, but it should not be treated as a representative fixed-step artifact proxy.
- The attention-only depth family appears to be relatively insensitive to extra MLP width on this local contract.
- The hybrid win survives giving the attention-only baseline more room, even though exact size matching is still unresolved.
- The best model does not have to land exactly at `16,000,000` bytes. The cap is
  a hard constraint, but the compute-optimal point may still sit below the
  ceiling.

This means:

- The quality result is already useful and meaningful.
- But the branch is not yet at a final submission-grade perfectly size-matched comparison.
- The next work should focus on whether exact local size matching is still worth the time, versus moving on to hybrid scaling and H100 calibration.

## Current Recommendation

If continuing this branch, the current best path is:

1. Keep `COMPILE_STRATEGY=model`.
2. Treat `GDN_RATIO=1, MLP_MULT=3.25, TRAIN_SEQ_LEN=2048` as the primary HGDN operating point.
3. Stop spending time on selective compile unless an H100 result contradicts the 4070 measurements.
4. Use the hybrid as the current quality winner over the depth-control baseline at this local scale, including after wall-time reindexing.
5. Treat `MLP_MULT=4.7` as ruled out locally; it overshot the artifact budget badly.
6. Treat `MLP_MULT=4.0` as a stronger but still underfilled attention-only baseline that did not materially reduce the hybrid's advantage.
7. Treat the 1xH100 calibration as a real positive signal for the architecture: the throughput tax is worse than local, but the quality gap is also stronger than local.
8. Keep the current winner as the H100 systems baseline, but do another HGDN kernel/profiling tranche before treating model-size selection as the main branch priority.
9. Use the 16 MB cap as a hard upper bound, not a hard fill target. Finalists should bracket the boundary rather than assuming that exact saturation is automatically optimal.

## Likely Next Work

- Continue HGDN-native hotspot work on top of the current winner, especially the packed qkv front-end and remaining norm/gate/layout glue.
- If exact local size matching is still required, bracket the attention-only baseline between `MLP_MULT=4.0` and a slightly larger fixed-step candidate. Do not use 600-second local runs as the final size-matching proxy.
- Add the trainer's new byte-audit fields to any future quality-run summaries when comparing branches.
- Once the next kernel tranche stalls, run a small wall-clock-capped HGDN scaling sweep to find the real compute-optimal size.
- If possible later, re-check throughput and compile strategy on the target H100 environment.
  - The branch now includes `scripts/run_h100_single_gpu_hgdn.sh` for that 1xH100 calibration step.

## Recent Branch Checkpoints

- `88fc8c6` `perf: default hgdn compile strategy to model`
- `d3bea59` `perf: add fair throughput timing harness`
- `135cd91` `perf: selectively compile hgdn attention path`
- `f5b3733` `perf: use module.compile for hgdn model`
- `5cf3b8f` `fix: allow hgdn compile graph breaks`
