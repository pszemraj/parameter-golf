#!/bin/bash

hgdn_setup_repo_root() {
    local caller_path="${1:?caller path required}"
    local caller_dir
    caller_dir="$(cd "$(dirname "$caller_path")" && pwd)"
    export HGDN_SCRIPT_DIR="$caller_dir"
    export HGDN_REPO_ROOT="$(cd "$caller_dir/.." && pwd)"
    cd "$HGDN_REPO_ROOT"
}

hgdn_require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

hgdn_validate_fla_recurrence_mode() {
    local mode="${1:?recurrence mode required}"
    case "${mode}" in
    compile_visible | direct | direct_fused) ;;
    *)
        echo "Unsupported GDN_FLA_RECURRENCE_MODE: ${mode}" >&2
        echo "Expected one of: compile_visible, direct, direct_fused" >&2
        exit 1
        ;;
    esac
}

hgdn_resolve_val_batch_size() {
    local ngpu="${1:?ngpu required}"
    local grad_accum_steps="${2:?grad accum steps required}"
    local train_seq_len="${3:?train sequence length required}"
    local min_val_batch_size=$((ngpu * grad_accum_steps * train_seq_len))
    local min_val_batch_seqs=$((ngpu * grad_accum_steps))
    local requested_tokens="${VAL_BATCH_SIZE:-}"
    local requested_seqs="${VAL_BATCH_SEQS:-}"

    if [[ -n "${requested_tokens}" && -n "${requested_seqs}" ]]; then
        echo "Set only one of VAL_BATCH_SIZE or VAL_BATCH_SEQS." >&2
        return 1
    fi

    if [[ -n "${requested_seqs}" ]]; then
        if [[ ! "${requested_seqs}" =~ ^[0-9]+$ ]]; then
            echo "VAL_BATCH_SEQS must be a positive integer, got ${requested_seqs}" >&2
            return 1
        fi
        if ((requested_seqs < 1)); then
            echo "VAL_BATCH_SEQS must be a positive integer, got ${requested_seqs}" >&2
            return 1
        fi
        if ((requested_seqs < min_val_batch_seqs)); then
            echo "VAL_BATCH_SEQS must be at least ${min_val_batch_seqs} for NGPU=${ngpu}, GRAD_ACCUM_STEPS=${grad_accum_steps}" >&2
            return 1
        fi
        echo $((requested_seqs * train_seq_len))
        return 0
    fi

    if [[ -z "${requested_tokens}" ]]; then
        echo "${min_val_batch_size}"
        return 0
    fi

    if [[ ! "${requested_tokens}" =~ ^[0-9]+$ ]]; then
        echo "VAL_BATCH_SIZE must be a positive integer token count, got ${requested_tokens}" >&2
        return 1
    fi
    if ((requested_tokens < 1)); then
        echo "VAL_BATCH_SIZE must be a positive integer token count, got ${requested_tokens}" >&2
        return 1
    fi

    if ((requested_tokens < min_val_batch_size)); then
        if ((requested_tokens >= min_val_batch_seqs)); then
            echo "Interpreting VAL_BATCH_SIZE=${requested_tokens} as validation sequences for compatibility; use VAL_BATCH_SEQS=${requested_tokens} or VAL_BATCH_SIZE=$((requested_tokens * train_seq_len))." >&2
            echo $((requested_tokens * train_seq_len))
            return 0
        fi
        echo "VAL_BATCH_SIZE is a global token count and must be at least ${min_val_batch_size} for NGPU=${ngpu}, GRAD_ACCUM_STEPS=${grad_accum_steps}, TRAIN_SEQ_LEN=${train_seq_len}; use VAL_BATCH_SEQS for sequence-count input." >&2
        return 1
    fi

    echo "${requested_tokens}"
}

hgdn_filter_recurrence_env() {
    local kv
    for kv in "$@"; do
        case "${kv}" in
        GDN_FLA_RECURRENCE_MODE=* | GDN_USE_DIRECT_FLA_LAYER_SEMANTICS=*) ;;
        *)
            printf '%s\n' "${kv}"
            ;;
        esac
    done
}

hgdn_run_hybrid_train() {
    local label="$1"
    shift

    echo
    echo ">>> $label"
    (
        for kv in "$@"; do
            export "$kv"
        done
        torchrun --standalone --nproc_per_node="${NGPU:-1}" "$HGDN_REPO_ROOT/train_gpt_hybrid.py"
    )
}

hgdn_run_with_env() {
    local extra_env=()
    while (($#)) && [[ "$1" == *=* ]]; do
        extra_env+=("$1")
        shift
    done
    env "${extra_env[@]}" "$@"
}

hgdn_python_has_module() {
    local python_bin="${1:?python binary required}"
    local module_name="${2:?module name required}"
    "${python_bin}" scripts/hgdn_helper_cli.py module-exists --module "${module_name}" >/dev/null 2>&1
}

hgdn_ensure_python_module() {
    local python_bin="${1:?python binary required}"
    local module_name="${2:?module name required}"
    local package_name="${3:-$module_name}"
    if hgdn_python_has_module "${python_bin}" "${module_name}"; then
        return 0
    fi
    echo
    echo ">>> install python package: ${package_name}"
    "${python_bin}" -m pip install "${package_name}"
}

hgdn_create_7z_archive() {
    local python_bin="${1:?python binary required}"
    local archive_output="${2:?archive output required}"
    local source_path="${3:?source path required}"
    hgdn_ensure_python_module "${python_bin}" py7zr py7zr
    rm -f "${archive_output}"
    mkdir -p "$(dirname "${archive_output}")"
    "${python_bin}" scripts/hgdn_helper_cli.py create-7z \
        --archive-output "${archive_output}" \
        --source-path "${source_path}"
}

hgdn_append_command() {
    local path="$1"
    shift
    local extra_env=()
    while (($#)) && [[ "$1" == *=* ]]; do
        extra_env+=("$1")
        shift
    done
    printf '%q ' "${extra_env[@]}" "$@" >> "$path"
    printf '\n' >> "$path"
}

hgdn_append_plain_command() {
    local path="$1"
    shift
    printf '%q ' "$@" >> "$path"
    printf '\n' >> "$path"
}
