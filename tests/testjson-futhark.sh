#!/bin/bash

# Test the JSON grammar using the alpacc Futhark backend.
# Runs lexer, parser, and combined modes on a single long input.

show_usage() {
    echo "Usage: $0 [backend]"
    echo "  backend: Futhark backend (default: multicore)"
    echo "           Options: c, multicore, opencl, cuda, ispc"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage; exit 0
fi

export PATH="$HOME/bin:$PATH"

backend="${1:-multicore}"
length=100000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
GRAMMAR="$REPO_ROOT/grammars/json.alp"

echo "Testing JSON grammar with Futhark backend '$backend' (length=$length)..."

temp_dir=$(mktemp -d)
trap "rm -rf \"$temp_dir\"" EXIT

echo "Setting up Futhark packages..."
cd "$temp_dir"
futhark pkg add github.com/diku-dk/containers
futhark pkg add github.com/diku-dk/sorts
futhark pkg sync
if [ $? -ne 0 ]; then
    echo "Failed to set up Futhark packages"; exit 1
fi
echo "Futhark packages ready"

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
    cp -r "$temp_dir/lib" .
    cp "$temp_dir/futhark.pkg" .

    # shellcheck disable=SC2086
    if ! alpacc test generate "$GRAMMAR" --single-long --length $length $mode_flag; then
        echo "ERROR: alpacc test generate failed"; return 1
    fi

    # shellcheck disable=SC2086
    if ! alpacc futhark "$GRAMMAR" $mode_flag; then
        echo "ERROR: alpacc futhark failed"; return 1
    fi

    # `futhark script -b` produces binary output with a 16-byte header;
    # strip it before passing to `alpacc test compare`.
    futhark script --backend="$backend" -b json.fut 'test ($loadbytes "json.inputs")' \
        | tail -c +16 > json.results

    # shellcheck disable=SC2086
    if ! alpacc test compare "$GRAMMAR" json.inputs json.outputs json.results $mode_flag; then
        echo "ERROR: Test FAILED for JSON $mode_name mode"; return 1
    fi

    echo "JSON $mode_name test PASSED"
}

run_test "--lexer"  || exit 1
run_test "--parser" || exit 1
run_test ""         || exit 1

echo "All JSON Futhark backend tests passed."
