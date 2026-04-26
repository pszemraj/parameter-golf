#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_VERSION="${RUN_VERSION:-geom1}"
SEEDS="${SEEDS:-1337}"
DRY_RUN="${DRY_RUN:-0}"
TRIGRAM_TOP_K="${TRIGRAM_TOP_K:-2}"
TRIGRAM_COUNT_WORKERS="${TRIGRAM_COUNT_WORKERS:-1}"
INCLUDE_EXTENDED_GEOMETRY="${INCLUDE_EXTENDED_GEOMETRY:-0}"

read -r -d '' DEFAULT_GEOMETRY_SPECS <<'EOF' || true
blocks0_d96_l6_i512 96 6 512
blocks0_d64_l10_i512 64 10 512
blocks0_d128_l4_i512 128 4 512
blocks0_d128_l5_i512 128 5 512
EOF

read -r -d '' EXTENDED_GEOMETRY_SPECS <<'EOF' || true
blocks0_d64_l8_i512 64 8 512
blocks0_d96_l8_i512 96 8 512
blocks0_d128_l6_i384 128 6 384
blocks0_d160_l4_i512 160 4 512
EOF

GEOMETRY_SPECS="${GEOMETRY_SPECS:-${DEFAULT_GEOMETRY_SPECS}}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --run-version VALUE
  --seeds VALUE
  --trigram-top-k VALUE
  --count-workers VALUE
  --geometry-specs VALUE
  --include-extended-geometry
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-version) RUN_VERSION="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --trigram-top-k) TRIGRAM_TOP_K="$2"; shift 2 ;;
    --trigram-count-workers|--count-workers) TRIGRAM_COUNT_WORKERS="$2"; shift 2 ;;
    --geometry-specs) GEOMETRY_SPECS="$2"; shift 2 ;;
    --include-extended-geometry) INCLUDE_EXTENDED_GEOMETRY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${INCLUDE_EXTENDED_GEOMETRY}" == "1" ]]; then
  GEOMETRY_SPECS="${GEOMETRY_SPECS}"$'\n'"${EXTENDED_GEOMETRY_SPECS}"
fi

echo "5090 trigram geometry matrix"
echo "run_version=${RUN_VERSION}"
echo "seeds=${SEEDS}"
echo "spec_columns=label core_dim core_layers inner_dim"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry_run=1"
fi

while read -r label core_dim core_layers core_inner_dim extra; do
  if [[ -z "${label:-}" || "${label:0:1}" == "#" ]]; then
    continue
  fi
  if [[ -n "${extra:-}" ]]; then
    echo "Invalid GEOMETRY_SPECS row: ${label} ${core_dim} ${core_layers} ${core_inner_dim} ${extra}" >&2
    exit 1
  fi
  echo
  echo "=== geometry ${label}: core_dim=${core_dim} layers=${core_layers} inner_dim=${core_inner_dim} ==="
  cmd=(
    bash "${SCRIPT_DIR}/run_5090_trigram_aligned_geometry_screen.sh"
    --run-version "${RUN_VERSION}"
    --seeds "${SEEDS}"
    --geometry-label "${label}"
    --geometry-core-dim "${core_dim}"
    --geometry-core-layers "${core_layers}"
    --geometry-core-inner-dim "${core_inner_dim}"
    --geometry-core-expansion ""
    --geometry-num-blocks "0"
    --trigram-top-k "${TRIGRAM_TOP_K}"
    --count-workers "${TRIGRAM_COUNT_WORKERS}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
  fi
  "${cmd[@]}"
done <<<"${GEOMETRY_SPECS}"
