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

hgdn_run_sweep() {
    local label="$1"
    local preset="$2"
    shift 2

    echo
    echo ">>> $label"
    (
        for kv in "$@"; do
            export "$kv"
        done
        bash "$HGDN_REPO_ROOT/scripts/sweep.sh" "$preset"
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
