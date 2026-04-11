#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

if [[ "$#" -ne 0 ]]; then
    echo "Run this script with no arguments." >&2
    echo "It always executes the full current H100 HGDN finalist batch-scale batch." >&2
    exit 1
fi

hgdn_require_cmd bash
hgdn_require_cmd python

python_bin="${PYTHON_BIN:-python}"
wandb_project="${WANDB_PROJECT:-pg-hgdn-ablations}"
wandb_watch="${WANDB_WATCH:-gradients}"
wandb_mode="${WANDB_MODE:-online}"
run_prefix_base="${RUN_PREFIX_BASE:-h100pack1}"
compare_reference="${COMPARE_REFERENCE:-h100retune6_f_fixed2k_hybrid_r1_mlp3.5_seq2048}"
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
fixed2k_train_batch_tokens_override="${TRAIN_BATCH_TOKENS:-}"
fixed2k_seq_len="${FIXED2K_SEQ_LEN:-2048}"
fixed2k_val_loss_every="${FIXED2K_VAL_LOSS_EVERY:-500}"
fixed2k_train_log_every="${FIXED2K_TRAIN_LOG_EVERY:-200}"

hgdn_ensure_python_module "${python_bin}" py7zr py7zr

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
    "${run_prefix_base}_g"
    "${run_prefix_base}_h"
)

if [[ -n "${RUN_PREFIXES:-}" ]]; then
    IFS=',' read -r -a run_prefixes <<<"${RUN_PREFIXES}"
else
    run_prefixes=("${default_prefixes[@]}")
fi

configs=(
    "configs/hgdn/retune_trim_layers_14.toml"
    "configs/hgdn/retune_trim_layers_14.toml"
    "configs/hgdn/retune_trim_layers_14.toml"
    "configs/hgdn/retune_trim_layers_14.toml"
    "configs/hgdn/retune_trim_layers_14_mlp3p5.toml"
    "configs/hgdn/retune_trim_layers_14_mlp3p5.toml"
    "configs/hgdn/retune_trim_layers_14_mlp3p5.toml"
    "configs/hgdn/retune_trim_layers_14_mlp3p5.toml"
)

labels=(
    "14L m3.25 half-global local32"
    "14L m3.25 base-global local32"
    "14L m3.25 double-global local32"
    "14L m3.25 base-global local64"
    "14L m3.5 half-global local32"
    "14L m3.5 base-global local32"
    "14L m3.5 double-global local32"
    "14L m3.5 base-global local64"
)

batch_tokens=(
    "262144"
    "524288"
    "1048576"
    "524288"
    "262144"
    "524288"
    "1048576"
    "524288"
)

grad_accum_steps_matrix=(
    "4"
    "8"
    "16"
    "4"
    "4"
    "8"
    "16"
    "4"
)

if [[ -n "${GRAD_ACCUM_STEPS_OVERRIDE:-}" ]]; then
    grad_accum_steps_matrix=(
        "${GRAD_ACCUM_STEPS_OVERRIDE}"
        "${GRAD_ACCUM_STEPS_OVERRIDE}"
        "${GRAD_ACCUM_STEPS_OVERRIDE}"
        "${GRAD_ACCUM_STEPS_OVERRIDE}"
        "${GRAD_ACCUM_STEPS_OVERRIDE}"
        "${GRAD_ACCUM_STEPS_OVERRIDE}"
        "${GRAD_ACCUM_STEPS_OVERRIDE}"
        "${GRAD_ACCUM_STEPS_OVERRIDE}"
    )
fi

if [[ -n "${fixed2k_train_batch_tokens_override}" ]]; then
    batch_tokens=(
        "${fixed2k_train_batch_tokens_override}"
        "${fixed2k_train_batch_tokens_override}"
        "${fixed2k_train_batch_tokens_override}"
        "${fixed2k_train_batch_tokens_override}"
        "${fixed2k_train_batch_tokens_override}"
        "${fixed2k_train_batch_tokens_override}"
        "${fixed2k_train_batch_tokens_override}"
        "${fixed2k_train_batch_tokens_override}"
    )
fi

if [[ "${#run_prefixes[@]}" -ne "${#configs[@]}" \
    || "${#run_prefixes[@]}" -ne "${#labels[@]}" \
    || "${#run_prefixes[@]}" -ne "${#batch_tokens[@]}" \
    || "${#run_prefixes[@]}" -ne "${#grad_accum_steps_matrix[@]}" ]]; then
    echo "RUN_PREFIXES, configs, labels, batch_tokens, and grad_accum_steps_matrix must have the same count." >&2
    exit 1
fi

for ((i = 0; i < ${#batch_tokens[@]}; i++)); do
    batch_denom=$((grad_accum_steps_matrix[$i] * fixed2k_seq_len))
    if (( batch_tokens[$i] % batch_denom != 0 )); then
        echo "Invalid batch contract at index ${i}: TRAIN_BATCH_TOKENS=${batch_tokens[$i]} must be divisible by GRAD_ACCUM_STEPS=${grad_accum_steps_matrix[$i]} * FIXED2K_SEQ_LEN=${fixed2k_seq_len}." >&2
        exit 1
    fi
done

print_plan() {
    echo
    echo ">>> H100 HGDN finalist batch-scale round (1500-step fixed-token proxy)"
    echo "python_bin=${python_bin}"
    echo "wandb_project=${wandb_project}"
    echo "wandb_watch=${wandb_watch}"
    echo "wandb_mode=${wandb_mode}"
    echo "TORCH_LOGS=${torch_logs:-<unset>}"
    echo "TORCH_TRACE=${torch_trace:-<unset>}"
    echo "build_hgdn_cuda=${build_hgdn_cuda}"
    echo "run_hgdn_cuda_parity=${run_hgdn_cuda_parity}"
    echo "fixed2k_iterations=${fixed2k_iterations}"
    echo "fixed2k_train_batch_tokens_override=${fixed2k_train_batch_tokens_override:-<matrix defaults>}"
    echo "grad_accum_steps_override=${GRAD_ACCUM_STEPS_OVERRIDE:-<matrix defaults>}"
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
        local local_batch_size=$((batch_tokens[$i] / (grad_accum_steps_matrix[$i] * fixed2k_seq_len)))
        echo "  - ${run_prefixes[$i]} :: ${labels[$i]} :: ${configs[$i]} :: train_batch_tokens=${batch_tokens[$i]} :: grad_accum_steps=${grad_accum_steps_matrix[$i]} :: local_batch_size=${local_batch_size}"
    done
    echo "next_stage=after this proxy pass, run one exact 8x matched-control go/no-go"
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
            "TRAIN_BATCH_TOKENS=${batch_tokens[$i]}" \
            "GRAD_ACCUM_STEPS=${grad_accum_steps_matrix[$i]}" \
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
            TRAIN_BATCH_TOKENS="${batch_tokens[$i]}" \
            GRAD_ACCUM_STEPS="${grad_accum_steps_matrix[$i]}" \
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
        "${fixed2k_train_batch_tokens_override}" \
        "${fixed2k_seq_len}" \
        "${fixed2k_val_loss_every}" \
        "${fixed2k_train_log_every}" \
        "${configs[*]}" \
        "$(IFS='|'; echo "${labels[*]}")" \
        "${batch_tokens[*]}" \
        "${grad_accum_steps_matrix[*]}" \
        "${run_prefixes[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

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
fixed2k_train_batch_tokens_override = sys.argv[12]
fixed2k_seq_len = int(sys.argv[13])
fixed2k_val_loss_every = int(sys.argv[14])
fixed2k_train_log_every = int(sys.argv[15])
configs = sys.argv[16].split()
labels = sys.argv[17].split("|") if "|" in sys.argv[17] else sys.argv[17].split()
batch_tokens = [int(value) for value in sys.argv[18].split()]
grad_accum_steps_matrix = [int(value) for value in sys.argv[19].split()]
run_prefixes = sys.argv[20:]

plan = []
for prefix, config, label, train_batch_tokens, grad_accum_steps in zip(
    run_prefixes, configs, labels, batch_tokens, grad_accum_steps_matrix, strict=True
):
    plan.append(
        {
            "run_prefix": prefix,
            "config": config,
            "label": label,
            "train_batch_tokens": train_batch_tokens,
            "grad_accum_steps": grad_accum_steps,
            "local_batch_size": train_batch_tokens
            // (grad_accum_steps * fixed2k_seq_len),
        }
    )

manifest = {
    "run_prefix_base": run_prefix_base,
    "run_prefixes": run_prefixes,
    "configs": configs,
    "labels": labels,
    "plan": plan,
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
        "train_batch_tokens_override": (
            int(fixed2k_train_batch_tokens_override)
            if fixed2k_train_batch_tokens_override
            else None
        ),
        "fixed2k_seq_len": fixed2k_seq_len,
        "fixed2k_val_loss_every": fixed2k_val_loss_every,
        "fixed2k_train_log_every": fixed2k_train_log_every,
        "grad_accum_steps_override": (
            int(os.environ["GRAD_ACCUM_STEPS_OVERRIDE"])
            if "GRAD_ACCUM_STEPS_OVERRIDE" in os.environ
            else None
        ),
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
