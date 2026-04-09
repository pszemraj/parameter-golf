#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

if [[ "$#" -ne 0 ]]; then
    echo "Run this script with no arguments." >&2
    echo "It always executes the full current H100 HGDN resize batch and compare step." >&2
    exit 1
fi

hgdn_require_cmd bash
hgdn_require_cmd python

python_bin="${PYTHON_BIN:-python}"
wandb_project="${WANDB_PROJECT:-pg-hconv-ablations}"
wandb_watch="${WANDB_WATCH:-gradients}"
wandb_mode="${WANDB_MODE:-online}"
run_prefix_base="${RUN_PREFIX_BASE:-h100retune2}"
compare_reference="${COMPARE_REFERENCE:-h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048}"
compare_output_dir="${COMPARE_OUTPUT_DIR:-profiles/fixed2k_compare/${run_prefix_base}_round}"

case "${wandb_mode}" in
online)
    wandb_flag="--online"
    ;;
offline)
    wandb_flag="--offline"
    ;;
*)
    echo "Unsupported WANDB_MODE: ${wandb_mode} (expected online or offline)" >&2
    exit 1
    ;;
esac

default_prefixes=(
    "${run_prefix_base}_a"
    "${run_prefix_base}_b"
    "${run_prefix_base}_c"
    "${run_prefix_base}_d"
    "${run_prefix_base}_e"
)

if [[ -n "${RUN_PREFIXES:-}" ]]; then
    IFS=',' read -r -a run_prefixes <<<"${RUN_PREFIXES}"
else
    run_prefixes=("${default_prefixes[@]}")
fi

configs=(
    "configs/hgdn/retune_balanced_14l_mlp3.toml"
    "configs/hgdn/retune_trim_layers_14.toml"
    "configs/hgdn/retune_trim_layers_14_mlp3p125.toml"
    "configs/hgdn/retune_trim_layers_14_mlp3p375.toml"
    "configs/hgdn/retune_deepen_15l_mlp2p75.toml"
)

labels=(
    "balanced rerun"
    "winner replicate"
    "winner bracket low"
    "winner bracket high"
    "deeper orthogonal"
)

if [[ "${#run_prefixes[@]}" -ne "${#configs[@]}" ]]; then
    echo "RUN_PREFIXES count (${#run_prefixes[@]}) must match config count (${#configs[@]})." >&2
    exit 1
fi

print_plan() {
    echo
    echo ">>> H100 HGDN resize round"
    echo "python_bin=${python_bin}"
    echo "wandb_project=${wandb_project}"
    echo "wandb_watch=${wandb_watch}"
    echo "wandb_mode=${wandb_mode}"
    echo "compare_reference=${compare_reference}"
    echo "compare_output_dir=${compare_output_dir}"
    echo "batch:"
    local i
    for ((i = 0; i < ${#configs[@]}; i++)); do
        echo "  - ${run_prefixes[$i]} :: ${labels[$i]} :: ${configs[$i]}"
    done
}

run_batch() {
    local i
    for ((i = 0; i < ${#configs[@]}; i++)); do
        echo
        echo ">>> 1xH100 fixed2k-hybrid: ${labels[$i]}"
        hgdn_run_with_env \
            WANDB_PROJECT="${wandb_project}" \
            WANDB_WATCH="${wandb_watch}" \
            WANDB_MODE="${wandb_mode}" \
            "${python_bin}" scripts/hgdn.py h100-perf fixed2k-hybrid \
            --config "${configs[$i]}" \
            --run-prefix "${run_prefixes[$i]}" \
            "${wandb_flag}"
    done
}

run_compare() {
    echo
    echo ">>> fixed2k compare"
    hgdn_run_with_env \
        "${python_bin}" scripts/hgdn.py fixed2k-compare \
        --contains "${run_prefix_base}_" \
        --name "${compare_reference}" \
        --reference "${compare_reference}" \
        --output-dir "${compare_output_dir}"
}

print_plan
run_batch
run_compare
