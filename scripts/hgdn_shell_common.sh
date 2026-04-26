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
