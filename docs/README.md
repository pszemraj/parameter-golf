# HGDN Branch Status

Last updated: 2026-04-13

Branch: `exp/hgdn`

## Current state

- Active H100 kernel winner: [`winner_20260405_19.toml`](../configs/hgdn/winner_20260405_19.toml)
- Active winner flags:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
  - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
- The current post-conv front-end seam is closed after `h100k20`.
- The HGDN recurrence now runs through an owned compile-visible boundary that
  bypasses the upstream FLA backend-registry lock path.
- The exact `8xH100` matched-control bridge is now complete.
- Real HGDN ablations go to `pg-hgdn-ablations`.
- Local Python commands on this checkout use `conda run -s --name pg ...`.

## Active architecture bracket

- Live H100 reference:
  - `14L x 384d x mlp3.25`
    - current H100 proxy leader: `h100pack3_b_fixed2k_hybrid_r1_mlp3.25_seq2048`
    - contract: `TRAIN_BATCH_TOKENS=2097152`, `grad_accum_steps=8`, `local_batch_size=128`
- Live exact-bridge reference:
  - `h100bridge1_exact_hybrid_r1_mlp3.25_seq2048`
    - exact `8xH100` stop-step eval: `2.3949`
    - exact final roundtrip: `2.4206`
    - artifact status: `UNDER_LIMIT`
    - artifact headroom: `834,652`
- The `14L x 384d x mlp3.5` branch is no longer the live 14L winner:
  - `local64` was better than `local32`
  - the best `mlp3.5` point still lost to `mlp3.25`
  - the stronger `local64` point was over budget
- `15L x 384d` is no longer the live bracket leader:
  - both `mlp2.625` and `mlp2.875` improved at `local128`
  - neither family caught the `14L x 384d x mlp3.25` anchor on H100
- The cross-family architecture question is now resolved for this branch:
  - the live `14L x 384d x mlp3.25` HGDN finalist beat the matched attention-only baseline on the exact `8xH100` contract
  - the matched attention-only baseline also failed the artifact cap on the same contract
- The next question is more basic and more important than another finalist micro-ablations pass:
  - can the current HGDN stack get into the same order of magnitude as the
    repo's published naive baseline when trained under that baseline contract
- Scope guardrail:
  - the bounded proxy ladder is complete
  - the exact 8x matched-control bridge is complete
  - do not reopen broad H100 architecture search or another cross-family ladder without fresh contradictory exact evidence
- Exact metrics, reject rationale, and run history live in [PROFILING_LOG.md](PROFILING_LOG.md).

## Operating rules

- Use the local GPU for broad fixed-token architecture search when the candidate family fits memory.
- Use 1xH100 for finalist ranking under the same fixed-token contract.
- After fixed-token ranking, run a separate H100 batch-scale / packing pass before the final architecture call.
- The bounded H100 follow-up answered the batch-scale question cleanly:
  - the jump from `local64` to `local128` helped all three tested families
  - the `14L x 384d x mlp3.25` family stayed in front
- Because the trainer defaults to `grad_accum_steps = 8 / world_size`, the 1xH100 proxy preserves the same per-GPU local-batch mapping for a given `TRAIN_BATCH_TOKENS` unless that knob is explicitly overridden.
- The exact bridge changed the branch decision:
  - HGDN is the live record-path family
  - further architecture work should now be HGDN-only unless a later exact run says otherwise
- Before paying for more HGDN-only finalist polish, run one absolute
  competitiveness check on the official naive-baseline contract. The bridge
  answered the within-branch choice, not the absolute-score question.
- That check should be three-way:
  - the exact repo naive baseline from `train_gpt.py`
  - the hybrid-trainer attention-only control
  - the live HGDN finalist
- The exact repo naive baseline is the calibration anchor for this check.
  - Do not treat the hybrid-trainer attention-only control as a substitute for
    the published baseline.
  - Keep the attention-only control only because it isolates the architecture
    delta inside the hybrid trainer/runtime stack.
- For that check, pin the hybrid-trainer runs to `WEIGHT_DECAY=0`.
  - The baseline trainer does not apply optimizer weight decay.
  - The hybrid default `WEIGHT_DECAY=0.04` silently changes the contract and is strong enough to collapse both hybrid runs mid-training.
- Fixed-token sweeps answer learning efficiency. The final architecture call still has to survive the H100 wallclock contract.
- Low VRAM use during saturated fixed-token H100 runs is not a failure by itself. Treat it as a reason to study packing, not to discard the architecture.
- On the compiled HGDN path, default to changes that alter the generated path. Python-side view reshuffles and `.contiguous()` edits are not the main lever.
- Enable compile diagnostics only when needed:
  - `TORCH_LOGS=recompiles,graph_breaks`
  - optional `TORCH_TRACE=/tmp/tracedir`

## Competition timing

- The challenge has separate training-time and evaluation-time budgets.
- In both `train_gpt.py` and `train_gpt_hybrid.py`, the training timer excludes warmup/compile priming, validation passes, and final serialization plus roundtrip eval.
- Fixed-token sweeps are therefore a screening tool for learning efficiency, not the final leaderboard objective.

## Screening checkpoints

- Keep local and H100 screening as separate stages. They do not stabilize at the same step count.
- Local broad architecture screen:
  - default to `300` steps
  - keep `VAL_LOSS_EVERY=100`
- Local shortlist confirmation:
  - rerun the survivors for `500` steps
- H100 fixed-token finalist screen:
  - default to `1500` steps
  - keep `VAL_LOSS_EVERY=500`
- H100 follow-up passes after the first finalist ranking should keep the same `1500`-step contract unless the question explicitly changes.
- Only pay for `2000` H100 steps when the `1500`-step finalists are still crossing, the margin is tiny, or a final tie-break is needed.
- Current evidence behind those defaults:
  - local `retune2`: ranking was already mostly stable by `300` and very stable by `500`
  - H100 `retune3`: `500` was misleading, `1000` was only a coarse filter, and `1500` was the first checkpoint that looked like the `2000` result
- Do not use the `500`-step H100 screen to kill viable candidates.

## Current boundaries

- Treat norm placement as a learning-dynamics screen, not as kernel work. Compare `NORM_STYLE=pre|post|keel` inside the HGDN family first, then against the attention-only baseline via `GDN_RATIO=0`.
- Use compile/backend work only on finalists.
- Do not reopen the closed front-end seam for more ownership or packing tweaks unless the decomposition changes materially.
- Do not keep paying for matched attention-only baseline reruns now that the exact bridge has already answered the keep/kill question.

## Current compile state

- The hot HGDN recurrence no longer graph-breaks the compiled block loop on the
  upstream FLA `lock` context-manager path.
- The owned recurrence boundary lives behind
  `hgdn_fla_v1::chunk_gated_delta_rule` and is backed by
  [`../hgdn_cuda/fla_owned.py`](../hgdn_cuda/fla_owned.py).
- The custom-op backward contract needed one important fix:
  - `grad_g` must stay `float32` in the fake/meta registration
  - advertising it as `bfloat16` poisoned Inductor buffer planning in the full
    backward graph
- Muon Newton-Schulz helpers now prewarm on the live matrix-shape family before
  the training clock starts, so the optimizer-side shape-family compiles do not
  first appear mid-run.
- Small compiled trainer smoke now completes HGDN warmup plus the first real
  backward/optimizer step without HGDN graph breaks.
- Remaining compile churn is narrower and outside the closed front-end seam:
  - one-time rotary-cache recompile when `self._cos` flips from `None`
  - one eval-time `grad_mode` recompile when the same forward is reused across
    training and evaluation
- Keep auditing with `TORCH_LOGS=recompiles,graph_breaks`, but do not treat the
  recurrence seam as the live blocker anymore unless new H100 evidence says
  otherwise.

## Launch entrypoints

Structured launcher:

- [`../scripts/hgdn.py`](../scripts/hgdn.py)

Single-entry batch helpers:

- [`../scripts/run_local_hgdn_resize_round.sh`](../scripts/run_local_hgdn_resize_round.sh)
- [`../scripts/run_h100_hgdn_resize_round.sh`](../scripts/run_h100_hgdn_resize_round.sh)
- [`../scripts/run_h100_hgdn_bridge_round.sh`](../scripts/run_h100_hgdn_bridge_round.sh)
- [`../scripts/run_h100_hgdn_naive_contract_round.sh`](../scripts/run_h100_hgdn_naive_contract_round.sh)

Data / setup helpers:

- [`../scripts/bootstrap_challenge_data.sh`](../scripts/bootstrap_challenge_data.sh)

1xH100 perf / profiling helpers:

- [`../scripts/run_h100_single_gpu_hgdn.sh`](../scripts/run_h100_single_gpu_hgdn.sh)
- [`../scripts/run_h100_single_gpu_hgdn_profile.sh`](../scripts/run_h100_single_gpu_hgdn_profile.sh)
- [`../scripts/hgdn_cuda_preflight.py`](../scripts/hgdn_cuda_preflight.py)

## Related docs

- W&B schema and project rules: [WANDB_SCHEMA.md](WANDB_SCHEMA.md)
- local/H100 transfer limits: [HARDWARE_TRANSFER.md](HARDWARE_TRANSFER.md)
- CUDA extension build/runtime notes: [HGDN_CUDA_FUSED.md](HGDN_CUDA_FUSED.md)
- profiling chronology and scoreboards: [PROFILING_LOG.md](PROFILING_LOG.md)
- active next steps: [TODO.md](TODO.md)
- cleanup targets: [REDUNDANCY_AUDIT.md](REDUNDANCY_AUDIT.md)
- OLMo/Hybrid reference notes: [REFERENCE.md](REFERENCE.md)
