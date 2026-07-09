#!/bin/bash

# Test the JSON grammar using the alpacc C backend.
# Runs lexer, parser, and combined modes on a single long input.

show_usage() {
    echo "Usage: $0"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage; exit 0
fi

export PATH="$HOME/bin:$PATH"

length=100000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
GRAMMAR="$REPO_ROOT/grammars/json.alp"

echo "Testing JSON grammar with C backend (length=$length)..."

temp_dir=$(mktemp -d)
trap "rm -rf \"$temp_dir\"" EXIT

run_test() {
    local mode_flag="$1"
    local mode_name
    if [ -z "$mode_flag" ]; then mode_name="combined"; else mode_name="${mode_flag#--}"; fi

    echo "========================================="
    echo "Testing JSON $mode_name mode..."
    echo "========================================="

    local work_dir="$temp_dir/$mode_name"
    mkdir -p "$work_dir"
    cd "$work_dir"

    # shellcheck disable=SC2086
    if ! alpacc test generate "$GRAMMAR" --single-long --length $length $mode_flag; then
        echo "ERROR: alpacc test generate failed"; return 1
    fi

    # shellcheck disable=SC2086
    if ! alpacc c "$GRAMMAR" $mode_flag -o json.c; then
        echo "ERROR: alpacc c failed"; return 1
    fi

    if ! cc -std=c99 -O2 -o json json.c &>/dev/null; then
        echo "ERROR: cc compilation failed"; return 1
    fi

    ./json < json.inputs > json.results

    # shellcheck disable=SC2086
    if ! alpacc test compare "$GRAMMAR" json.inputs json.outputs json.results $mode_flag; then
        echo "ERROR: Test FAILED for JSON $mode_name mode"; return 1
    fi

    echo "JSON $mode_name test PASSED"
}

run_test "--lexer"  || exit 1
run_test "--parser" || exit 1
run_test ""         || exit 1

echo "All JSON C backend tests passed."
