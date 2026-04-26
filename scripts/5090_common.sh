#!/usr/bin/env bash

pg_5090_lr_slug() {
  local lr="$1"
  local tag="${lr//./p}"
  tag="${tag#0p}"
  tag="${tag//-/m}"
  tag="${tag//+/}"
  printf 'lr%s' "${tag}"
}

pg_5090_fail() {
  local script_name="$1"
  local message="$2"
  cat >&2 <<EOF
${script_name}: serious-run guard failed: ${message}
Set ALLOW_DEGRADED_5090_SERIOUS=1 only for an explicit smoke/debug run.
EOF
  exit 1
}

pg_5090_expect_env() {
  local script_name="$1"
  local name="$2"
  local expected="$3"
  local actual="${!name-}"
  if [[ "${actual}" != "${expected}" ]]; then
    pg_5090_fail "${script_name}" "${name} must be ${expected}, got ${actual:-<unset>}"
  fi
}

pg_5090_expect_unset_or_empty() {
  local script_name="$1"
  local name="$2"
  local actual="${!name-}"
  if [[ -n "${actual}" ]]; then
    pg_5090_fail "${script_name}" "${name} must be unset/empty, got ${actual}"
  fi
}

pg_5090_append_bool_flag() {
  local script_name="$1"
  local array_name="$2"
  local flag_name="$3"
  local value="$4"
  local -n target_array="${array_name}"

  case "${value,,}" in
    1|true|yes|on)
      target_array+=("--${flag_name}")
      ;;
    0|false|no|off)
      target_array+=("--no-${flag_name}")
      ;;
    *)
      pg_5090_fail "${script_name}" "${flag_name} must be boolean-like, got ${value:-<unset>}"
      ;;
  esac
}

pg_5090_require_serious_launcher_defaults() {
  local script_name="$1"
  local expected_preset="${2:-controller_default}"
  if [[ "${ALLOW_DEGRADED_5090_SERIOUS:-0}" == "1" ]]; then
    echo "${script_name}: ALLOW_DEGRADED_5090_SERIOUS=1, skipping serious-run guard" >&2
    return 0
  fi

  pg_5090_expect_env "${script_name}" "PRESET" "${expected_preset}"
  pg_5090_expect_env "${script_name}" "COMPILE" "0"
  pg_5090_expect_env "${script_name}" "GRADIENT_CHECKPOINTING" "0"
  pg_5090_expect_env "${script_name}" "REBUILD_SHARED" "0"
  pg_5090_expect_env "${script_name}" "SCAN_BACKEND" "auto"
  pg_5090_expect_env "${script_name}" "TORCH_BLAS_PREFER_CUBLASLT" "1"
  pg_5090_expect_env "${script_name}" "WANDB" "1"
  pg_5090_expect_env "${script_name}" "WANDB_PROJECT" "pg-hconv-ablations"

  if [[ -n "${WANDB_MODE:-}" && "${WANDB_MODE}" != "online" ]]; then
    pg_5090_fail "${script_name}" "WANDB_MODE must be unset or online, got ${WANDB_MODE}"
  fi

  pg_5090_expect_unset_or_empty "${script_name}" "SPEC_MAX_TOKENS"
  pg_5090_expect_unset_or_empty "${script_name}" "DATA_MAX_TOKENS"
  pg_5090_expect_unset_or_empty "${script_name}" "TRAIN_FRAC"

  if [[ "${ALLOW_APPROX_BPB:-0}" != "0" ]]; then
    pg_5090_fail "${script_name}" "ALLOW_APPROX_BPB is not allowed for serious runs"
  fi
  if [[ "${ALLOW_TRAIN_FRAC_VAL_SPLIT:-0}" != "0" ]]; then
    pg_5090_fail "${script_name}" "ALLOW_TRAIN_FRAC_VAL_SPLIT is not allowed for serious runs"
  fi
}
