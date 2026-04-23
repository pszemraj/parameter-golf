#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-enqueue}"

PUEUE_BIN="${PUEUE_BIN:-${HOME}/.cargo/bin/pueue}"
PUEUED_BIN="${PUEUED_BIN:-${HOME}/.cargo/bin/pueued}"
PUEUE_ROOT="${PUEUE_ROOT:-${HOME}/.cache/parameter-golf/pueue/5090_final_week}"
PUEUE_CONFIG="${PUEUE_ROOT}/pueue.yml"
PUEUE_GROUP="${PUEUE_GROUP:-gpu5090}"
PUEUE_PARALLEL="${PUEUE_PARALLEL:-1}"
PUEUE_PORT="${PUEUE_PORT:-51239}"
QUEUE_SIDECAR="${QUEUE_SIDECAR:-1}"
FOLLOW_AFTER_ENQUEUE="${FOLLOW_AFTER_ENQUEUE:-0}"

POST_TEMPORAL_SCRIPT="${REPO_ROOT}/scripts/run_5090_post_temporal_queue.sh"
GATE_LR_SIDECAR_SCRIPT="${REPO_ROOT}/scripts/run_5090_gate_lr_sidecar.sh"

pueue_cmd() {
  "${PUEUE_BIN}" --config "${PUEUE_CONFIG}" "$@"
}

pueued_cmd() {
  "${PUEUED_BIN}" --config "${PUEUE_CONFIG}" "$@"
}

require_bins() {
  if [[ ! -x "${PUEUE_BIN}" ]]; then
    echo "Missing pueue client: ${PUEUE_BIN}" >&2
    exit 1
  fi
  if [[ ! -x "${PUEUED_BIN}" ]]; then
    echo "Missing pueue daemon: ${PUEUED_BIN}" >&2
    exit 1
  fi
}

write_config() {
  mkdir -p "${PUEUE_ROOT}/certs"
  cat > "${PUEUE_CONFIG}" <<EOF
client:
  max_status_lines: 15
  status_datetime_format: "%Y-%m-%d %H:%M:%S"
  edit_mode: files
daemon:
  callback_log_lines: 15
  shell_command:
    - bash
    - -lc
    - "{{ pueue_command_string }}"
shared:
  pueue_directory: ${PUEUE_ROOT}
  runtime_directory: ${PUEUE_ROOT}
  alias_file: ${PUEUE_ROOT}/pueue_aliases.yml
  host: "localhost"
  port: "${PUEUE_PORT}"
  daemon_cert: ${PUEUE_ROOT}/certs/daemon.cert
  daemon_key: ${PUEUE_ROOT}/certs/daemon.key
  shared_secret_path: ${PUEUE_ROOT}/secret
profiles: {}
EOF
}

ensure_daemon() {
  write_config
  if ! pueue_cmd status >/dev/null 2>&1; then
    pueued_cmd --daemonize >/dev/null
    sleep 1
  fi

  if ! pueue_cmd status >/dev/null 2>&1; then
    echo "Failed to start pueue daemon with config ${PUEUE_CONFIG}" >&2
    exit 1
  fi

  if ! pueue_cmd --color never group | rg -q "^Group \"${PUEUE_GROUP}\" \\("; then
    pueue_cmd group add "${PUEUE_GROUP}" --parallel "${PUEUE_PARALLEL}" >/dev/null
  else
    pueue_cmd parallel "${PUEUE_PARALLEL}" --group "${PUEUE_GROUP}" >/dev/null
  fi
}

print_header() {
  echo "5090 final-week pueue helper"
  echo "action=${ACTION}"
  echo "repo_root=${REPO_ROOT}"
  echo "pueue_root=${PUEUE_ROOT}"
  echo "pueue_config=${PUEUE_CONFIG}"
  echo "group=${PUEUE_GROUP} parallel=${PUEUE_PARALLEL}"
  echo "queue_sidecar=${QUEUE_SIDECAR} follow_after_enqueue=${FOLLOW_AFTER_ENQUEUE}"
  echo "finalist_script=${POST_TEMPORAL_SCRIPT}"
  echo "gate_lr_sidecar_script=${GATE_LR_SIDECAR_SCRIPT}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "dry_run=1"
  fi
}

enqueue_tasks() {
  local finalist_id sidecar_id
  local finalist_cmd="bash '${POST_TEMPORAL_SCRIPT}'"
  local sidecar_cmd="bash '${GATE_LR_SIDECAR_SCRIPT}'"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "pueue --config ${PUEUE_CONFIG} add -g ${PUEUE_GROUP} -l finalists-1b -p ${finalist_cmd}"
    if [[ "${QUEUE_SIDECAR}" == "1" ]]; then
      echo "pueue --config ${PUEUE_CONFIG} add -g ${PUEUE_GROUP} -l gate-lr-sidecar -a <finalist_id> -p ${sidecar_cmd}"
    fi
    return 0
  fi

  finalist_id="$(pueue_cmd add -g "${PUEUE_GROUP}" -l "finalists-1b" -p "${finalist_cmd}")"
  echo "queued finalists task_id=${finalist_id}"

  if [[ "${QUEUE_SIDECAR}" == "1" ]]; then
    sidecar_id="$(pueue_cmd add -g "${PUEUE_GROUP}" -l "gate-lr-sidecar" -a "${finalist_id}" -p "${sidecar_cmd}")"
    echo "queued sidecar task_id=${sidecar_id} after=${finalist_id}"
  fi

  pueue_cmd status

  if [[ "${FOLLOW_AFTER_ENQUEUE}" == "1" ]]; then
    pueue_cmd follow "${finalist_id}"
  fi
}

main() {
  require_bins
  print_header

  case "${ACTION}" in
    init)
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "mkdir -p '${PUEUE_ROOT}/certs'"
        echo "write config to ${PUEUE_CONFIG}"
        echo "pueued --config ${PUEUE_CONFIG} --daemonize"
        echo "pueue --config ${PUEUE_CONFIG} group add ${PUEUE_GROUP} --parallel ${PUEUE_PARALLEL}"
        exit 0
      fi
      ensure_daemon
      pueue_cmd status
      ;;
    status)
      ensure_daemon
      pueue_cmd status
      ;;
    enqueue)
      ensure_daemon
      enqueue_tasks
      ;;
    follow)
      ensure_daemon
      pueue_cmd follow "${2:-}"
      ;;
    wait)
      ensure_daemon
      pueue_cmd wait -g "${PUEUE_GROUP}"
      ;;
    shutdown)
      if [[ -f "${PUEUE_CONFIG}" ]] && pueue_cmd status >/dev/null 2>&1; then
        pueue_cmd shutdown
      else
        echo "pueue daemon is not running for ${PUEUE_CONFIG}"
      fi
      ;;
    *)
      echo "Unknown action: ${ACTION}" >&2
      echo "Usage: $0 [init|status|enqueue|follow|wait|shutdown]" >&2
      exit 1
      ;;
  esac
}

main "$@"
