# 5090 Full-Spec Rerun Matrix

This note tracks the cleanup after the capped frozen-spec default was invalidated.

Contract for every rerun below:
- full available `fineweb_train_*.bin` shard set on disk
- no `SPEC_MAX_TOKENS` cap unless explicitly marked as a smoke
- fresh shared spec per experiment root
- old capped-spec roots remain on disk only as invalidated historical artifacts

Status legend:
- `pending`: not started under the corrected contract
- `running`: launched under the corrected contract
- `done`: completed under the corrected contract
- `failed`: launched under the corrected contract but failed for a real reason worth preserving

Queue launcher:
- commit `adac6ee`
- command: `conda run -s --name train python tools/run_core_amp_fullspec_reruns.py`
- scope: the remaining pending controller, temporal, and structure families listed below

## Controller Families

- `done` `blocks3_followup_clean`
  - `plain3_e20`
  - `resid5_e20`
- `done` `blocks3_neighborhood_v1`
  - `plain3_e25_c8t1`
  - `plain4_e20_c8t1`
  - `resid4_e20_c8t1`
  - `resid4_e25_c8t1`
- `done` `blocks3_bptt_v2`
  - `plain4_e20_c8t1`
  - `plain4_e20_c8t2`
  - `plain4_e20_c8t4`
  - `resid4_e25_c8t1`
  - `resid4_e25_c8t2`
  - `resid4_e25_c8t4`
- `running` `blocks3_carry_v1`
  - `resid4_e20_c8t1`
  - `resid4_e20_c16t1`
  - `resid4_e20_c32t1`
  - `resid4_e25_c8t1`
  - `resid4_e25_c16t1`
  - `resid4_e25_c32t1`
- `pending` `blocks3_confirm1b_v1`
  - `resid4_e20_c16t1_1b`
  - `resid4_e25_c8t1_1b`
- `done` `blocks2_frontier`
  - `blocks2_resid5_e25_c8t1_current_512m`
  - `blocks2_resid6_e25_c8t1_current_512m`
  - `blocks2_resid5_e30_c8t1_current_512m`
- `pending` `blocks2_confirm1b`
  - `blocks2_resid6_e25_c8t1_1b`
- `done` `blocks2_radical`
  - `blocks2_resid12_e6_c8t1_r3_current_512m`
  - `blocks2_resid12_e8_c8t1_r3_current_512m`
- `pending` `blocks1_radical`
  - `blocks1_resid12_e6_c8t1_r3_current_512m`

## Temporal Family

- `pending` `blocks3_temporal`
  - `resid4_e25_c8t1_current_512m`
  - `resid4_e25_c8t1_lagged_512m`
  - `resid4_e25_c8t1_hybrid_512m`

## Structure Family

- `done` `structure_round1`
  - `blocks0`
  - `blocks3`
  - `blocks6`
  - `blocks9`
  - `branches8_pow2`
  - `readout256`
  - `readout128`
  - queue session `14136` launched from commit `e43655d`
  - final corrected ranking:
    - `blocks0`: `best_val_bpb=2.4859076582`, `tok/s=3,336,299`
    - `blocks3`: `best_val_bpb=2.4865782548`, `tok/s=1,837,963`
    - `readout256`: `best_val_bpb=2.4867157200`, `tok/s=959,199`
    - `blocks6`: `best_val_bpb=2.4867944734`, `tok/s=1,267,624`
    - `readout128`: `best_val_bpb=2.4875669087`, `tok/s=964,897`
    - `branches8_pow2`: `best_val_bpb=2.4885888269`, `tok/s=1,319,498`
    - `blocks9`: `best_val_bpb=2.4891548818`, `tok/s=966,778`
- `done` `blocks0_radical_guardrail`
  - `blocks0_resid12_e6_c8t1_r3_current_512m`
- `done` `blocks0_controller_v1`
  - `blocks0_resid12_e8_c8t1_r3_current_512m`
- `done` `blocks0_controller_v2`
  - `blocks0_resid12_e10_c8t1_r3_current_512m`
- `done` `blocks0_controller_v3`
  - `blocks0_resid10_e12_c8t1_r3_current_512m`
- `failed` `blocks0_controller_v4`
  - `blocks0_resid16_e8_c8t1_r3_current_512m`
  - `OOM` on the fixed `seq_len=512`, `batch_size=256` contract after the first step
- `done` `blocks0_controller_v5`
  - `blocks0_resid14_e8_c8t1_r3_current_512m`
- `done` `blocks0_controller_v6`
  - `blocks0_resid16_e8_c8t1_r3_current_512m_gc1`
  - checkpointed rerun of the `blocks0_controller_v4` OOM point
