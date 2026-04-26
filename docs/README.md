# HGDN Branch Status

Branch: `exp/hgdn-final`

## Current Position

- Active path: packed FLA-backed sparse HGDN in `train_gpt_hybrid.py`.
- Exact comparator: `train_gpt.py`. The hybrid `GDN_RATIO=0` path is only an
  attention-only diagnostic control.
- Removed path: the owned full-block CUDA kernel and core-kernel experiments.
  The live trainer no longer has owned-kernel flags, branches, preflight, or
  tests.
- Preserved active path: packed QKV projection, packed depthwise QKV conv, and
  public-FLA recurrence support.
- GDN dynamics now match the important upstream FLA priors more closely:
  `A_log` and `dt_bias` use timescale-aware initialization, those decay params
  run in a no-weight-decay Adam group, negative eigenvalues stay enabled by
  default, and each GDN layer has a learned output RMSNorm weight.
- Candidate runs default to `GDN_FLA_RECURRENCE_MODE=direct`, and helper
  scripts also log it explicitly. `compile_visible` and `direct_fused` remain
  named ablation modes; `direct_fused` tests upstream-style in-kernel q/k L2
  normalization and decay-gate activation. All FLA recurrence modes now use the
  public FLA default scale, `1 / sqrt(head_k_dim)`.
- Local Python commands on this checkout use `conda run -s --name pg ...`.

The active branch question is narrow: can a sparse HGDN shell beat the exact
repo baseline on the exact baseline-shaped contract after accounting for speed
and artifact size?

## Next Experiment Order

1. Run the small sequential recurrence implementation matrix from
   [FLA Calibration](#fla-calibration) before reopening architecture search.
2. Rerun the local sparse exact-contract helper after the implementation matrix.
   Start with the shortlist in [Local Sparse Search](#local-sparse-search).
3. If the local rerun still supports the primary candidate, run the exact H100
   comparison in [H100 Commands](#h100-commands).
4. Run the optional quality-ceiling candidate only if the primary result leaves
   a real quality/speed tradeoff unresolved.
5. Keep native FLA and recurrence-path timing in the calibration lane described
   in [FLA Calibration](#fla-calibration). Run timing jobs sequentially with no
   other active CUDA work.

## OLMo/FLA Alignment Audit

The external GDN/OLMo design notes are in [REFERENCE.md](REFERENCE.md).

| Item | Status | Branch handling |
|---|---|---|
| Public FLA recurrence | Resolved | Custom HGDN calls public `fla.ops.gated_delta_rule.chunk_gated_delta_rule`; naive recurrence is refused for active CUDA HGDN training. |
| Decay init | Resolved | `A_log` / `dt_bias` use timescale-aware FLA-style initialization and log `gdn_decay_init` startup stats. |
| Decay weight decay | Resolved | `A_log` / `dt_bias` use a separate Adam group with `weight_decay=0.0`. |
| Recurrence scale | Resolved | Compile-visible and direct custom recurrence paths keep upstream FLA's default `1 / sqrt(head_k_dim)` scale instead of forcing `1.0`. |
| Negative eigenvalues | Resolved | Custom HGDN defaults `GDN_ALLOW_NEG_EIGVAL=1`; native FLA configs set `FLA_ALLOW_NEG_EIGVAL=true`. |
| Output norm/gate parameterization | Resolved enough for custom path | Custom HGDN now has learned per-`head_v_dim` `o_norm_weight`; it uses PyTorch norm+SiLU by default instead of `FusedRMSNormGated` directly. |
| Output norm epsilon | Documented deviation | Custom HGDN keeps the branch default `eps=1e-6`; native FLA defaults to `1e-5`. Treat `1e-5` as a strict-fidelity ablation, not the active default. |
| `expand_v=2.0` prior | Added as candidates | Practical sparse search still includes cheaper `1.0`/`1.5` variants, but OLMo-prior `v2` candidates are now first-class configs. |
| 3:1 GDN:attention prior | Added as candidate | `l8_d512_olmoish_6g2a_v2_m1p25` is the 6G/2A reality check; it is not the default promotion candidate until wallclock evidence supports it. |
| Native FLA control | Clarified | `train_gpt_fla_control.py` remains pure native GDN calibration, not OLMo Hybrid; an OLMo-ish SP1024 native config exists for dimension/value-width calibration. |
| Wrapper overhead | Measurable | `scripts/bench_fla_recurrence_paths.py` times wrapper, direct custom recurrence, direct fused FLA semantics, and native `fla.layers.GatedDeltaNet`. |
| Optional local CUDA extension | Removed | The old `hgdn_cuda_ext` sidecar, build scripts, parity scripts, and trainer flags are no longer part of this branch. |

## Local Sparse Search

Completed bundle: `local-scratch/localnaivehgdn_sparse3_bundle`.

Analyze it with:

```bash
conda run -s --name pg python scripts/analyze_local_naive_contract_bundle.py --top 20
```

The completed sparse3 bundle predates the decay-init/no-WD/output-norm fix, so
it should be treated as pre-fix evidence. It still identifies useful shapes,
but exact promotion should rerun the shortlist under the corrected GDN dynamics.

Pre-fix promotion call:

| Role | Config | Local result |
|---|---|---|
| Primary H100 sparse candidate | `configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml` | `458.84 ms/step`, sampled BPB `1.6884`, speed-rank HGDN `1`, `UNDER_LIMIT` |
| Secondary quality ceiling | `configs/hgdn/naive_contract_l9_d512_mid3_dk48_v1p5_m1p75.toml` | `586.24 ms/step`, sampled BPB `1.6839`, fixed-step rank HGDN `1`, `UNDER_LIMIT` |
| Matched attention-only diagnostic control | `configs/hgdn/naive_contract_l8_d512_r0_m2.toml` | `444.40 ms/step`, sampled BPB `1.6921`, `UNDER_LIMIT` |

The practical rerun shortlist starts with the same primary candidate, the
matched attention-only diagnostic control, and two new OLMo-prior checks:

- `configs/hgdn/naive_contract_l8_d512_mid2_dk48_v2_m1p5.toml`
- `configs/hgdn/naive_contract_l8_d512_olmoish_6g2a_v2_m1p25.toml`

The local helper includes both OLMo-prior configs plus matched attention-only
diagnostic controls for `mlp1.5` and `mlp1.25`; the H100 helper can infer those
controls when `ATTN_CONFIG` is not set.

## H100 Commands

Primary exact-baseline comparison:

```bash
USE_WANDB=0 WANDB_MODE=offline \
ATTN_USE_FLASH_ATTN3=1 \
DISTRIBUTED_MODE=parallel_muon \
MUON_DISTRIBUTED_MODE=packed_allreduce \
GDN_W_G_OPTIMIZER=matrix \
HGDN_CONFIG=configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml \
ATTN_CONFIG=configs/hgdn/naive_contract_l8_d512_r0_m2.toml \
WANDB_WATCH=none \
RUN_PREFIX_BASE=h100naive_sparse_primary \
bash scripts/run_h100_hgdn_naive_contract_round.sh
```

Optional quality-ceiling substitution:

```bash
HGDN_CONFIG=configs/hgdn/naive_contract_l9_d512_mid3_dk48_v1p5_m1p75.toml
```

The helper runs three legs: exact `train_gpt.py` baseline, config-driven sparse
HGDN, and the matched attention-only diagnostic control. It pins `DATA_PATH`,
`TOKENIZER_PATH`, and `VOCAB_SIZE` for the exact baseline leg.

Local sparse search helper:

```bash
USE_WANDB=0 WANDB_MODE=offline \
DISTRIBUTED_MODE=parallel_muon \
RUN_PREFIX_BASE=localnaivehgdn_decayfix \
bash scripts/run_local_hgdn_naive_contract_search.sh
```

Active helpers launch `torchrun train_gpt_hybrid.py` directly.

## FLA Calibration

The native FLA stack is now checked explicitly:

```bash
conda run -s --name pg python scripts/probe_fla_stack.py
```

The isolated native-FLA control lives in `train_gpt_fla_control.py` with two
separate configs:

```bash
configs/fla/native_prlike_gdn10_d544_sp8192.toml
configs/fla/native_olmoish_gdn8_d512_sp1024.toml
```

The PR-like config keeps the 10 pure native `fla.layers.GatedDeltaNet` layers,
`d_model=544`, 8 heads, `head_dim=64`, MLP multiplier `3.0`, and SP8192
data/tokenizer defaults. The OLMo-ish config uses `d_model=512`, 8 heads,
`head_dim=48`, `FLA_VALUE_EXPAND=2.0`, `FLA_ALLOW_NEG_EIGVAL=true`, and SP1024
so it can be compared against the exact-contract HGDN surface. Optional knobs
include `FLA_SHARE_QK`, `FLA_SHARE_KV`, `BIGRAM_HASH_BUCKETS`, and
`TRIGRAM_HASH_BUCKETS`.

Both native FLA configs default `FLA_STORAGE_DTYPE=fp32` for numerical sanity.
For speed calibration against bf16 custom HGDN, override
`FLA_STORAGE_DTYPE=bf16` explicitly. These configs remain pure-GDN calibration;
the custom HGDN 6G/2A config is the OLMo-style hybrid test.

Tiny smoke:

```bash
env COMPILE=0 VOCAB_SIZE=128 NUM_LAYERS=1 MODEL_DIM=64 NUM_HEADS=2 \
HEAD_DIM=32 MLP_MULT=2 FLA_VALUE_EXPAND=1 TRAIN_SEQ_LEN=32 \
SMOKE_SEQ_LEN=32 SMOKE_BATCH_SIZE=2 \
conda run -s --name pg python train_gpt_fla_control.py --smoke
```

Full calibration launch, after loading the TOML as environment:

```bash
set -a
source <(conda run -s --name pg python scripts/hgdn_helper_cli.py load-env \
  --path configs/fla/native_prlike_gdn10_d544_sp8192.toml)
set +a
torchrun --standalone --nproc_per_node=1 train_gpt_fla_control.py
```

This is calibration only. It must not change `HybridGPT` behavior or replace
the sparse HGDN finalist path unless its canonical score, artifact size, and
speed justify promotion.

Wrapper/direct/native timing ablation:

```bash
conda run -s --name pg python scripts/bench_fla_recurrence_paths.py --iters 20
```

Full-layer custom-vs-native timing calibration:

```bash
conda run -s --name pg python scripts/bench_hgdn_full_layer_paths.py --iters 20
```

Run this benchmark by itself. Timings are invalid if other tests, training jobs,
or CUDA benchmarks are active in background terminals.

For a training-level recurrence-path ablation, run the same config, seed, step
count, and compile settings sequentially with
`configs/hgdn/naive_contract_l8_d512_mid2_dk48_v2_m1p5.toml`:

```bash
GDN_FLA_RECURRENCE_MODE=direct
GDN_FLA_RECURRENCE_MODE=direct_fused
GDN_FLA_RECURRENCE_MODE=compile_visible
```

Then run the matched attention-only diagnostic control with
`configs/hgdn/naive_contract_l8_d512_r0_m1p5.toml`. Compare `ms/step`,
tokens/sec, train loss, sampled BPB, artifact proxy, and compile graph breaks.
Check for active CUDA jobs first:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

The local helper encodes that exact A/B/C/D matrix and refuses to start when
other CUDA compute jobs are visible unless `ALLOW_ACTIVE_CUDA_JOBS=1` is set:

```bash
USE_WANDB=0 WANDB_MODE=offline \
TORCH_LOGS=recompiles,graph_breaks \
RUN_PREFIX_BASE=localrecurmatrix1 \
bash scripts/run_local_hgdn_recurrence_matrix.sh
```

For an unattended local hierarchy, use the overnight pipeline. It runs:

1. the recurrence implementation matrix,
2. a bounded architecture/control screen using the selected recurrence mode,
3. a longer confirmation pass for the top HGDN configs plus their matched
   attention-only diagnostic controls,
4. a conditional OLMo-ish 6G/2A sanity check only when the confirmed primary
   HGDN beats its matched attention-only diagnostic control.

It analyzes each bundle with `scripts/analyze_hgdn_experiment_bundle.py`, writes
stage decision files under `local-scratch/<prefix>_pipeline`, and writes the
next H100 command for review instead of launching paid H100 work. The secondary
stage writes `next_h100_secondary_command.sh` when it runs.

```bash
USE_WANDB=0 WANDB_MODE=offline \
TORCH_LOGS=recompiles,graph_breaks \
RUN_PREFIX_BASE=localhgdn_overnight1 \
bash scripts/run_local_hgdn_overnight_pipeline.sh
```

The default screen is deliberately small. Override it with
`SCREEN_CANDIDATE_CONFIGS=path1.toml,path2.toml,...` when you want a different
shortlist; the underlying local search helper also accepts `CANDIDATE_CONFIGS`
for bounded one-off batches. Override the gated secondary check with
`SECONDARY_CANDIDATE_CONFIGS=path1.toml,path2.toml,...`, or set
`SECONDARY_FORCE=1` when you intentionally want the OLMo-ish sanity check even
if the primary candidate does not beat its matched control.

## Analysis And Sanity Tools

- `scripts/analyze_local_naive_contract_bundle.py`: local sparse bundle ranking.
- `scripts/analyze_hgdn_experiment_bundle.py`: generic bundle analyzer and
  promotion-decision writer for staged pipelines.
- `scripts/check_bpb_sanity.py`: nats/BPB/implied bytes-per-token checks.
- `scripts/probe_fla_stack.py`: FLA package/API/kernel import probe.
- `scripts/bench_fla_recurrence_paths.py`: wrapper versus direct public FLA
  recurrence timing, upstream-style direct fused FLA semantics timing, plus
  native `fla.layers.GatedDeltaNet` timing.
- `scripts/bench_hgdn_full_layer_paths.py`: full custom HGDN GDN layer timing
  under each recurrence mode versus native `fla.layers.GatedDeltaNet`.
- `scripts/screen_hgdn_arch_sizes.py`: artifact-size screen for active configs.

Example BPB sanity check:

```bash
conda run -s --name pg python scripts/check_bpb_sanity.py \
  local-scratch/localnaivehgdn_sparse3_bundle/logs/localnaivehgdn_sparse3_l8_d512_mid2_dk48_m2.txt
```

## Kept Entrypoints

- `train_gpt.py`: exact repo baseline.
- `train_gpt_hybrid.py`: packed sparse HGDN and attention-only diagnostic path.
- `train_gpt_fla_control.py`: isolated native-FLA calibration path.
- `scripts/run_local_hgdn_naive_contract_search.sh`: local sparse search.
- `scripts/run_local_hgdn_recurrence_matrix.sh`: local recurrence
  implementation matrix.
- `scripts/run_local_hgdn_overnight_pipeline.sh`: staged local implementation,
  architecture, and confirmation hierarchy.
- `scripts/run_h100_hgdn_naive_contract_round.sh`: H100 exact-baseline comparison.
- `scripts/bundle_hgdn_run.py`: bundle helper.
- `scripts/bootstrap_challenge_data.sh`: data bootstrap helper.

## Validation

Run these checks before handing off a branch or run bundle:

```bash
bash -n scripts/hgdn_shell_common.sh \
  scripts/run_local_hgdn_naive_contract_search.sh \
  scripts/run_local_hgdn_recurrence_matrix.sh \
  scripts/run_local_hgdn_overnight_pipeline.sh \
  scripts/run_h100_hgdn_naive_contract_round.sh \
  scripts/bootstrap_challenge_data.sh

conda run -s --name pg python -m py_compile \
  model.py train_gpt.py train_gpt_hybrid.py train_gpt_fla_control.py \
  hgdn_fla.py hgdn_runtime_utils.py scripts/hgdn_helper_cli.py \
  scripts/screen_hgdn_arch_sizes.py \
  scripts/analyze_local_naive_contract_bundle.py \
  scripts/check_bpb_sanity.py scripts/probe_fla_stack.py \
  scripts/bench_fla_recurrence_paths.py \
  scripts/bench_hgdn_full_layer_paths.py

conda run -s --name pg ruff check --fix
conda run -s --name pg ruff format
conda run -s --name pg ruff check
git diff --check
```

Related docs:

- [REFERENCE.md](REFERENCE.md)
- [WANDB_SCHEMA.md](WANDB_SCHEMA.md)
