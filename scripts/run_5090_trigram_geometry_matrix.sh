#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_VERSION="${RUN_VERSION:-v1}"
export SEEDS="${SEEDS:-1337}"

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
if [[ "${INCLUDE_EXTENDED_GEOMETRY:-0}" == "1" ]]; then
  GEOMETRY_SPECS="${GEOMETRY_SPECS}"$'\n'"${EXTENDED_GEOMETRY_SPECS}"
fi

echo "5090 trigram geometry matrix"
echo "run_version=${RUN_VERSION}"
echo "seeds=${SEEDS}"
echo "spec_columns=label core_dim core_layers inner_dim"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
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
  env \
    GEOMETRY_LABEL="${label}" \
    GEOMETRY_CORE_DIM="${core_dim}" \
    GEOMETRY_CORE_LAYERS="${core_layers}" \
    GEOMETRY_CORE_INNER_DIM="${core_inner_dim}" \
    GEOMETRY_CORE_EXPANSION="" \
    RUN_VERSION="${RUN_VERSION}" \
    SEEDS="${SEEDS}" \
    bash "${SCRIPT_DIR}/run_5090_trigram_aligned_geometry_screen.sh"
done <<<"${GEOMETRY_SPECS}"
