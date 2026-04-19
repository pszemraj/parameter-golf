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

## Controller Families

- `pending` `blocks3_followup_clean`
  - `plain3_e20`
  - `resid5_e20`
- `pending` `blocks3_neighborhood_v1`
  - `plain3_e25_c8t1`
  - `plain4_e20_c8t1`
  - `resid4_e20_c8t1`
  - `resid4_e25_c8t1`
- `pending` `blocks3_bptt_v2`
  - `plain4_e20_c8t1`
  - `plain4_e20_c8t2`
  - `plain4_e20_c8t4`
  - `resid4_e25_c8t1`
  - `resid4_e25_c8t2`
  - `resid4_e25_c8t4`
- `pending` `blocks3_carry_v1`
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

- `pending` `structure_round1`
  - `blocks0`
  - `blocks3`
  - `blocks6`
  - `blocks9`
  - `branches8_pow2`
  - `readout256`
  - `readout128`
- `running` `blocks0_radical_guardrail`
- `done` `blocks0_radical_guardrail`
  - `blocks0_resid12_e6_c8t1_r3_current_512m`
- `done` `blocks0_controller_v1`
  - `blocks0_resid12_e8_c8t1_r3_current_512m`
