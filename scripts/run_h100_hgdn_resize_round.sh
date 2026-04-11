#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

if [[ "$#" -ne 0 ]]; then
    echo "Run this script with no arguments." >&2
    echo "It always executes the full current H100 HGDN resize batch." >&2
    exit 1
fi

hgdn_require_cmd bash
hgdn_require_cmd python

python_bin="${PYTHON_BIN:-python}"
wandb_project="${WANDB_PROJECT:-pg-hgdn-ablations}"
wandb_watch="${WANDB_WATCH:-gradients}"
wandb_mode="${WANDB_MODE:-online}"
run_prefix_base="${RUN_PREFIX_BASE:-h100retune6}"
compare_reference="${COMPARE_REFERENCE:-h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048}"
compare_reference_entity="${COMPARE_REFERENCE_ENTITY:-pszemraj}"
compare_reference_project="${COMPARE_REFERENCE_PROJECT:-}"
bundle_stage_dir="${BUNDLE_STAGE_DIR:-local-scratch/${run_prefix_base}_bundle}"
archive_output="${ARCHIVE_OUTPUT:-local-scratch/${run_prefix_base}_bundle.7z}"
command_log="${COMMAND_LOG:-local-scratch/${run_prefix_base}_commands.sh}"
torch_logs="${TORCH_LOGS:-}"
torch_trace="${TORCH_TRACE:-}"
build_hgdn_cuda="${BUILD_HGDN_CUDA:-1}"
run_hgdn_cuda_parity="${RUN_HGDN_CUDA_PARITY:-1}"
fixed2k_iterations="${FIXED2K_ITERATIONS:-1500}"
fixed2k_train_batch_tokens="${TRAIN_BATCH_TOKENS:-524288}"
fixed2k_seq_len="${FIXED2K_SEQ_LEN:-2048}"
fixed2k_val_loss_every="${FIXED2K_VAL_LOSS_EVERY:-500}"
fixed2k_train_log_every="${FIXED2K_TRAIN_LOG_EVERY:-200}"

hgdn_ensure_python_module "${python_bin}" py7zr py7zr

if [[ -z "${compare_reference_project}" \
    && "${wandb_project}" == "pg-hgdn-ablations" \
    && "${compare_reference}" == "h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048" ]]; then
    compare_reference_project="pg-hconv-ablations"
fi

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
    "${run_prefix_base}_f"
)

if [[ -n "${RUN_PREFIXES:-}" ]]; then
    IFS=',' read -r -a run_prefixes <<<"${RUN_PREFIXES}"
else
    run_prefixes=("${default_prefixes[@]}")
fi

configs=(
    "configs/hgdn/retune_deepen_15l_mlp2p375.toml"
    "configs/hgdn/retune_deepen_15l_mlp2p4375.toml"
    "configs/hgdn/retune_deepen_15l_mlp2p5.toml"
    "configs/hgdn/retune_trim_layers_14.toml"
    "configs/hgdn/retune_trim_layers_14_mlp3p375.toml"
    "configs/hgdn/retune_trim_layers_14_mlp3p5.toml"
)

labels=(
    "15L undercut probe"
    "15L low-mid probe"
    "15L incumbent anchor"
    "14L orthogonal rerun"
    "14L higher-mlp rerun"
    "14L upper-bound rerun"
)

if [[ "${#run_prefixes[@]}" -ne "${#configs[@]}" ]]; then
    echo "RUN_PREFIXES count (${#run_prefixes[@]}) must match config count (${#configs[@]})." >&2
    exit 1
fi

print_plan() {
    echo
    echo ">>> H100 HGDN resize round (1500-step finalist screen)"
    echo "python_bin=${python_bin}"
    echo "wandb_project=${wandb_project}"
    echo "wandb_watch=${wandb_watch}"
    echo "wandb_mode=${wandb_mode}"
    echo "TORCH_LOGS=${torch_logs:-<unset>}"
    echo "TORCH_TRACE=${torch_trace:-<unset>}"
    echo "build_hgdn_cuda=${build_hgdn_cuda}"
    echo "run_hgdn_cuda_parity=${run_hgdn_cuda_parity}"
    echo "fixed2k_iterations=${fixed2k_iterations}"
    echo "fixed2k_train_batch_tokens=${fixed2k_train_batch_tokens}"
    echo "fixed2k_seq_len=${fixed2k_seq_len}"
    echo "fixed2k_val_loss_every=${fixed2k_val_loss_every}"
    echo "fixed2k_train_log_every=${fixed2k_train_log_every}"
    echo "compare_reference=${compare_reference}"
    echo "compare_reference_entity=${compare_reference_entity}"
    echo "compare_reference_project=${compare_reference_project:-${wandb_project}}"
    echo "archive_output=${archive_output}"
    echo "command_log=${command_log}"
    echo "batch:"
    local i
    for ((i = 0; i < ${#configs[@]}; i++)); do
        echo "  - ${run_prefixes[$i]} :: ${labels[$i]} :: ${configs[$i]}"
    done
}

prepare_cuda() {
    mkdir -p "$(dirname "${command_log}")"
    {
        echo "#!/bin/bash"
        echo "set -euo pipefail"
    } >"${command_log}"

    if [[ "${build_hgdn_cuda}" == "1" ]]; then
        echo
        echo ">>> build HGDN CUDA extension"
        hgdn_append_plain_command \
            "${command_log}" \
            "${python_bin}" setup_hgdn_cuda.py build_ext --inplace
        "${python_bin}" setup_hgdn_cuda.py build_ext --inplace
    fi

    if [[ "${run_hgdn_cuda_parity}" == "1" ]]; then
        echo
        echo ">>> HGDN CUDA parity"
        hgdn_append_plain_command \
            "${command_log}" \
            "${python_bin}" scripts/hgdn_cuda_parity.py
        "${python_bin}" scripts/hgdn_cuda_parity.py
    fi
}

run_batch() {
    local diagnostic_env=()
    if [[ -n "${torch_logs}" ]]; then
        diagnostic_env+=("TORCH_LOGS=${torch_logs}")
    fi
    if [[ -n "${torch_trace}" ]]; then
        diagnostic_env+=("TORCH_TRACE=${torch_trace}")
    fi

    local i
    for ((i = 0; i < ${#configs[@]}; i++)); do
        echo
        echo ">>> 1xH100 fixed2k-hybrid: ${labels[$i]}"
        hgdn_append_command \
            "${command_log}" \
            "${diagnostic_env[@]}" \
            "WANDB_PROJECT=${wandb_project}" \
            "WANDB_WATCH=${wandb_watch}" \
            "WANDB_MODE=${wandb_mode}" \
            "FIXED2K_ITERATIONS=${fixed2k_iterations}" \
            "TRAIN_BATCH_TOKENS=${fixed2k_train_batch_tokens}" \
            "FIXED2K_SEQ_LEN=${fixed2k_seq_len}" \
            "FIXED2K_VAL_LOSS_EVERY=${fixed2k_val_loss_every}" \
            "FIXED2K_TRAIN_LOG_EVERY=${fixed2k_train_log_every}" \
            "${python_bin}" scripts/hgdn.py h100-perf fixed2k-hybrid \
            --config "${configs[$i]}" \
            --run-prefix "${run_prefixes[$i]}" \
            "${wandb_flag}"

        hgdn_run_with_env \
            "${diagnostic_env[@]}" \
            WANDB_PROJECT="${wandb_project}" \
            WANDB_WATCH="${wandb_watch}" \
            WANDB_MODE="${wandb_mode}" \
            FIXED2K_ITERATIONS="${fixed2k_iterations}" \
            TRAIN_BATCH_TOKENS="${fixed2k_train_batch_tokens}" \
            FIXED2K_SEQ_LEN="${fixed2k_seq_len}" \
            FIXED2K_VAL_LOSS_EVERY="${fixed2k_val_loss_every}" \
            FIXED2K_TRAIN_LOG_EVERY="${fixed2k_train_log_every}" \
            "${python_bin}" scripts/hgdn.py h100-perf fixed2k-hybrid \
            --config "${configs[$i]}" \
            --run-prefix "${run_prefixes[$i]}" \
            "${wandb_flag}"
    done
}

build_bundle() {
    echo
    echo ">>> bundle outputs"

    rm -rf "${bundle_stage_dir}"
    mkdir -p "${bundle_stage_dir}/configs" "${bundle_stage_dir}/logs"
    cp "${command_log}" "${bundle_stage_dir}/commands.sh"

    local config_path
    for config_path in "${configs[@]}"; do
        cp "${config_path}" "${bundle_stage_dir}/configs/"
    done

    local matched_logs=0
    local prefix
    for prefix in "${run_prefixes[@]}"; do
        local log_path
        for log_path in logs/"${prefix}"*.txt; do
            if [[ -f "${log_path}" ]]; then
                cp "${log_path}" "${bundle_stage_dir}/logs/"
                matched_logs=1
            fi
        done
    done

    "${python_bin}" - \
        "${bundle_stage_dir}" \
        "${run_prefix_base}" \
        "${compare_reference}" \
        "${compare_reference_entity}" \
        "${compare_reference_project:-${wandb_project}}" \
        "${archive_output}" \
        "${matched_logs}" \
        "${torch_logs}" \
        "${torch_trace}" \
        "${command_log}" \
        "${fixed2k_iterations}" \
        "${fixed2k_train_batch_tokens}" \
        "${fixed2k_seq_len}" \
        "${fixed2k_val_loss_every}" \
        "${fixed2k_train_log_every}" \
        "${configs[*]}" \
        "${run_prefixes[@]}" <<'PY'
from pathlib import Path
import json
import sys

bundle_dir = Path(sys.argv[1])
run_prefix_base = sys.argv[2]
compare_reference = sys.argv[3]
compare_reference_entity = sys.argv[4]
compare_reference_project = sys.argv[5]
archive_output = sys.argv[6]
matched_logs = bool(int(sys.argv[7]))
torch_logs = sys.argv[8]
torch_trace = sys.argv[9]
command_log = sys.argv[10]
fixed2k_iterations = int(sys.argv[11])
fixed2k_train_batch_tokens = int(sys.argv[12])
fixed2k_seq_len = int(sys.argv[13])
fixed2k_val_loss_every = int(sys.argv[14])
fixed2k_train_log_every = int(sys.argv[15])
configs = sys.argv[16].split()
run_prefixes = sys.argv[17:]

manifest = {
    "run_prefix_base": run_prefix_base,
    "run_prefixes": run_prefixes,
    "configs": configs,
    "compare_reference": compare_reference,
    "compare_reference_entity": compare_reference_entity,
    "compare_reference_project": compare_reference_project,
    "archive_output": archive_output,
    "command_log": command_log,
    "matched_logs": matched_logs,
    "torch_logs": torch_logs or None,
    "torch_trace": torch_trace or None,
    "contract": {
        "fixed2k_iterations": fixed2k_iterations,
        "train_batch_tokens": fixed2k_train_batch_tokens,
        "fixed2k_seq_len": fixed2k_seq_len,
        "fixed2k_val_loss_every": fixed2k_val_loss_every,
        "fixed2k_train_log_every": fixed2k_train_log_every,
    },
}
(bundle_dir / "bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

    hgdn_create_7z_archive "${python_bin}" "${archive_output}" "${bundle_stage_dir}"
    echo "bundle_archive=${archive_output}"
    echo "local_compare_hint=conda run -s --name pg python scripts/hgdn.py fixed2k-compare --project ${wandb_project} --reference-entity ${compare_reference_entity} --reference-project ${compare_reference_project:-${wandb_project}} --contains ${run_prefix_base}_ --name ${compare_reference} --reference ${compare_reference}"
}

print_plan
prepare_cuda
run_batch
build_bundle
