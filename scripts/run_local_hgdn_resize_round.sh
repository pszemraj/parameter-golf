#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

if [[ "$#" -ne 0 ]]; then
    echo "Run this script with no arguments." >&2
    echo "It always executes the full current local HGDN resize batch." >&2
    exit 1
fi

hgdn_require_cmd bash
hgdn_require_cmd torchrun
hgdn_require_cmd python
hgdn_require_cmd 7z

python_bin="${PYTHON_BIN:-python}"
use_wandb="${USE_WANDB:-1}"
wandb_mode="${WANDB_MODE:-online}"
wandb_project="${WANDB_PROJECT:-pg-hgdn-ablations}"
wandb_watch="${WANDB_WATCH:-none}"
wandb_watch_log_freq="${WANDB_WATCH_LOG_FREQ:-25}"
run_prefix_base="${RUN_PREFIX_BASE:-localretune2}"
bundle_stage_dir="${BUNDLE_STAGE_DIR:-local-scratch/${run_prefix_base}_bundle}"
archive_output="${ARCHIVE_OUTPUT:-local-scratch/${run_prefix_base}_bundle.7z}"
command_log="${COMMAND_LOG:-local-scratch/${run_prefix_base}_commands.sh}"
torchinductor_max_autotune="${TORCHINDUCTOR_MAX_AUTOTUNE:-0}"
torchinductor_max_autotune_gemm="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-0}"
torch_logs="${TORCH_LOGS:-}"
torch_trace="${TORCH_TRACE:-}"

ngpu="${NGPU:-1}"
iterations="${ITERATIONS:-750}"
train_batch_tokens="${TRAIN_BATCH_TOKENS:-65536}"
train_seq_len="${TRAIN_SEQ_LEN:-1024}"
val_loss_every="${VAL_LOSS_EVERY:-100}"
train_log_every="${TRAIN_LOG_EVERY:-25}"
val_batch_size="${VAL_BATCH_SIZE:-524288}"
max_wallclock_seconds="${MAX_WALLCLOCK_SECONDS:-0}"
compile="${COMPILE:-1}"
compile_strategy="${COMPILE_STRATEGY:-model}"

case "${wandb_mode}" in
online | offline) ;;
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
    "configs/hgdn/retune_current.toml"
    "configs/hgdn/retune_balanced_14l_mlp3.toml"
    "configs/hgdn/retune_trim_layers_14_mlp3p375.toml"
    "configs/hgdn/retune_deepen_15l_mlp2p625.toml"
    "configs/hgdn/retune_deepen_15l_mlp2p75.toml"
    "configs/hgdn/retune_deepen_15l_mlp2p875.toml"
    "configs/hgdn/retune_deepen_15l_mlp3.toml"
    "configs/hgdn/retune_deepen_15l_mlp3p125.toml"
)

labels=(
    "current 16L reference"
    "fast 14L anchor"
    "best 14L anchor"
    "15L lower bracket"
    "15L local winner rerun"
    "15L upper bracket low"
    "15L upper bracket mid"
    "15L upper bracket high"
)

if [[ "${#run_prefixes[@]}" -ne "${#configs[@]}" ]]; then
    echo "RUN_PREFIXES count (${#run_prefixes[@]}) must match config count (${#configs[@]})." >&2
    exit 1
fi

resolved_run_ids=()

load_config_env() {
    local config_path="$1"
    mapfile -t config_env < <(
        "${python_bin}" - "${config_path}" <<'PY'
from pathlib import Path
import sys
import tomllib

path = Path(sys.argv[1])
data = tomllib.loads(path.read_text(encoding="utf-8"))
for key, value in data.get("env", {}).items():
    if isinstance(value, bool):
        value = "1" if value else "0"
    print(f"{key}={value}")
PY
    )
}

print_plan() {
    echo
    echo ">>> Local HGDN resize round"
    echo "python_bin=${python_bin}"
    echo "use_wandb=${use_wandb}"
    echo "wandb_mode=${wandb_mode}"
    echo "wandb_project=${wandb_project}"
    echo "wandb_watch=${wandb_watch}"
    echo "wandb_watch_log_freq=${wandb_watch_log_freq}"
    echo "TORCH_LOGS=${torch_logs:-<unset>}"
    echo "TORCH_TRACE=${torch_trace:-<unset>}"
    echo "TORCHINDUCTOR_MAX_AUTOTUNE=${torchinductor_max_autotune}"
    echo "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=${torchinductor_max_autotune_gemm}"
    echo "ngpu=${ngpu}"
    echo "iterations=${iterations}"
    echo "train_batch_tokens=${train_batch_tokens}"
    echo "train_seq_len=${train_seq_len}"
    echo "val_loss_every=${val_loss_every}"
    echo "train_log_every=${train_log_every}"
    echo "val_batch_size=${val_batch_size}"
    echo "compile=${compile}"
    echo "compile_strategy=${compile_strategy}"
    echo "max_wallclock_seconds=${max_wallclock_seconds}"
    echo "archive_output=${archive_output}"
    echo "batch:"
    local i
    for ((i = 0; i < ${#configs[@]}; i++)); do
        echo "  - ${run_prefixes[$i]} :: ${labels[$i]} :: ${configs[$i]}"
    done
}

run_batch() {
    mkdir -p "$(dirname "${command_log}")"
    {
        echo "#!/bin/bash"
        echo "set -euo pipefail"
    } >"${command_log}"

    local diagnostic_env=()
    if [[ -n "${torch_logs}" ]]; then
        diagnostic_env+=("TORCH_LOGS=${torch_logs}")
    fi
    if [[ -n "${torch_trace}" ]]; then
        diagnostic_env+=("TORCH_TRACE=${torch_trace}")
    fi

    local i
    for ((i = 0; i < ${#configs[@]}; i++)); do
        local config_path="${configs[$i]}"
        local config_stem
        config_stem="$(basename "${config_path}" .toml)"
        local run_id="${run_prefixes[$i]}_${config_stem}_seq${train_seq_len}_it${iterations}"

        load_config_env "${config_path}"
        resolved_run_ids+=("${run_id}")

        echo
        echo ">>> local fixed-data resize: ${labels[$i]}"
        echo "run_id=${run_id}"

        hgdn_append_command \
            "${command_log}" \
            "NGPU=${ngpu}" \
            "USE_WANDB=${use_wandb}" \
            "WANDB_MODE=${wandb_mode}" \
            "WANDB_PROJECT=${wandb_project}" \
            "WANDB_WATCH=${wandb_watch}" \
            "WANDB_WATCH_LOG_FREQ=${wandb_watch_log_freq}" \
            "${diagnostic_env[@]}" \
            "TORCHINDUCTOR_MAX_AUTOTUNE=${torchinductor_max_autotune}" \
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=${torchinductor_max_autotune_gemm}" \
            "COMPILE=${compile}" \
            "COMPILE_STRATEGY=${compile_strategy}" \
            "RUN_ID=${run_id}" \
            "ITERATIONS=${iterations}" \
            "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
            "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
            "TRAIN_SEQ_LEN=${train_seq_len}" \
            "VAL_LOSS_EVERY=${val_loss_every}" \
            "TRAIN_LOG_EVERY=${train_log_every}" \
            "VAL_BATCH_SIZE=${val_batch_size}" \
            "${config_env[@]}" \
            bash scripts/sweep.sh single

        hgdn_run_sweep \
            "local fixed-data resize: ${labels[$i]}" \
            single \
            "NGPU=${ngpu}" \
            "USE_WANDB=${use_wandb}" \
            "WANDB_MODE=${wandb_mode}" \
            "WANDB_PROJECT=${wandb_project}" \
            "WANDB_WATCH=${wandb_watch}" \
            "WANDB_WATCH_LOG_FREQ=${wandb_watch_log_freq}" \
            "${diagnostic_env[@]}" \
            "TORCHINDUCTOR_MAX_AUTOTUNE=${torchinductor_max_autotune}" \
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=${torchinductor_max_autotune_gemm}" \
            "COMPILE=${compile}" \
            "COMPILE_STRATEGY=${compile_strategy}" \
            "RUN_ID=${run_id}" \
            "ITERATIONS=${iterations}" \
            "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
            "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
            "TRAIN_SEQ_LEN=${train_seq_len}" \
            "VAL_LOSS_EVERY=${val_loss_every}" \
            "TRAIN_LOG_EVERY=${train_log_every}" \
            "VAL_BATCH_SIZE=${val_batch_size}" \
            "${config_env[@]}"
    done
}

build_bundle() {
    echo
    echo ">>> bundle outputs"

    rm -rf "${bundle_stage_dir}"
    mkdir -p "${bundle_stage_dir}/logs" "${bundle_stage_dir}/configs"

    local config_path
    for config_path in "${configs[@]}"; do
        cp "${config_path}" "${bundle_stage_dir}/configs/"
    done

    cp "${command_log}" "${bundle_stage_dir}/commands.sh"

    local matched_logs=0
    local run_id
    for run_id in "${resolved_run_ids[@]}"; do
        local log_path="logs/${run_id}.txt"
        if [[ -f "${log_path}" ]]; then
            cp "${log_path}" "${bundle_stage_dir}/logs/"
            matched_logs=1
        fi
    done

    "${python_bin}" - \
        "${bundle_stage_dir}" \
        "${run_prefix_base}" \
        "${wandb_project}" \
        "${wandb_mode}" \
        "${archive_output}" \
        "${matched_logs}" \
        "${torch_logs}" \
        "${torch_trace}" \
        "${torchinductor_max_autotune}" \
        "${torchinductor_max_autotune_gemm}" \
        "${iterations}" \
        "${train_batch_tokens}" \
        "${train_seq_len}" \
        "${val_loss_every}" \
        "${train_log_every}" \
        "${val_batch_size}" \
        "${compile}" \
        "${compile_strategy}" \
        "${resolved_run_ids[@]}" <<'PY'
from pathlib import Path
import json
import sys

bundle_dir = Path(sys.argv[1])
run_prefix_base = sys.argv[2]
wandb_project = sys.argv[3]
wandb_mode = sys.argv[4]
archive_output = sys.argv[5]
matched_logs = bool(int(sys.argv[6]))
torch_logs = sys.argv[7]
torch_trace = sys.argv[8]
torchinductor_max_autotune = int(sys.argv[9])
torchinductor_max_autotune_gemm = int(sys.argv[10])
iterations = int(sys.argv[11])
train_batch_tokens = int(sys.argv[12])
train_seq_len = int(sys.argv[13])
val_loss_every = int(sys.argv[14])
train_log_every = int(sys.argv[15])
val_batch_size = int(sys.argv[16])
compile_enabled = bool(int(sys.argv[17]))
compile_strategy = sys.argv[18]
run_ids = sys.argv[19:]

manifest = {
    "run_prefix_base": run_prefix_base,
    "wandb_project": wandb_project,
    "wandb_mode": wandb_mode,
    "archive_output": archive_output,
    "matched_logs": matched_logs,
    "contract": {
        "torch_logs": torch_logs or None,
        "torch_trace": torch_trace or None,
        "torchinductor_max_autotune": torchinductor_max_autotune,
        "torchinductor_max_autotune_gemm": torchinductor_max_autotune_gemm,
        "iterations": iterations,
        "train_batch_tokens": train_batch_tokens,
        "train_seq_len": train_seq_len,
        "val_loss_every": val_loss_every,
        "train_log_every": train_log_every,
        "val_batch_size": val_batch_size,
        "compile": compile_enabled,
        "compile_strategy": compile_strategy,
    },
    "run_ids": run_ids,
}
(bundle_dir / "bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

    rm -f "${archive_output}"
    mkdir -p "$(dirname "${archive_output}")"
    7z a -t7z "${archive_output}" "${bundle_stage_dir}" >/dev/null
    echo "bundle_archive=${archive_output}"
}

print_plan
run_batch
build_bundle
