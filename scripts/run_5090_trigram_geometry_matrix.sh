#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_VERSION="${RUN_VERSION:-v1}"
export SEEDS="${SEEDS:-1337}"

read -r -d '' DEFAULT_GEOMETRY_SPECS <<'EOF' || true
blocks0_d64_l8_e8 64 8 8.0
blocks0_d64_l10_e8 64 10 8.0
blocks0_d128_l6_e4 128 6 4.0
blocks0_d128_l8_e4 128 8 4.0
EOF

GEOMETRY_SPECS="${GEOMETRY_SPECS:-${DEFAULT_GEOMETRY_SPECS}}"

echo "5090 trigram geometry matrix"
echo "run_version=${RUN_VERSION}"
echo "seeds=${SEEDS}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
fi

while read -r label core_dim core_layers core_expansion extra; do
  if [[ -z "${label:-}" || "${label:0:1}" == "#" ]]; then
    continue
  fi
  if [[ -n "${extra:-}" ]]; then
    echo "Invalid GEOMETRY_SPECS row: ${label} ${core_dim} ${core_layers} ${core_expansion} ${extra}" >&2
    exit 1
  fi
  echo
  echo "=== geometry ${label}: core_dim=${core_dim} layers=${core_layers} expansion=${core_expansion} ==="
  env \
    GEOMETRY_LABEL="${label}" \
    GEOMETRY_CORE_DIM="${core_dim}" \
    GEOMETRY_CORE_LAYERS="${core_layers}" \
    GEOMETRY_CORE_EXPANSION="${core_expansion}" \
    RUN_VERSION="${RUN_VERSION}" \
    SEEDS="${SEEDS}" \
    bash "${SCRIPT_DIR}/run_5090_trigram_aligned_geometry_screen.sh"
done <<<"${GEOMETRY_SPECS}"
