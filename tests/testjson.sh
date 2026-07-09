#!/bin/bash

# Test script for the JSON grammar with a configurable backend.
# Runs lexer, parser, and combined (lexer+parser) tests using a single
# parseable input of length 100000.
#
# Supported backends:
#   c          – alpacc c backend, compiled with cc
#   cuda       – alpacc cuda backend, compiled with nvcc (requires GPU)
#   multicore, opencl, ispc, <any>  – alpacc futhark backend, run via futhark script

show_usage() {
    echo "Usage: $0 [backend]"
    echo "  backend: backend to use (default: c)"
    echo "           c, cuda, or any futhark backend (multicore, opencl, ispc, ...)"
    echo "Example: $0 multicore"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

export PATH="$HOME/bin:$PATH"

backend="${1:-c}"
length=100000

echo "Testing JSON grammar with backend '$backend' and length $length..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
GRAMMAR="$REPO_ROOT/grammars/json.alp"

temp_dir=$(mktemp -d)
trap "rm -rf \"$temp_dir\"" EXIT

# Set up Futhark packages once (only needed for Futhark backends)
if [ "$backend" != "c" ] && [ "$backend" != "cuda" ]; then
    echo "Setting up Futhark packages..."
    cd "$temp_dir"
    futhark pkg add github.com/diku-dk/containers
    futhark pkg add github.com/diku-dk/sorts
    futhark pkg sync
    if [ $? -ne 0 ]; then
        echo "Failed to set up Futhark packages"
        exit 1
    fi
    echo "Futhark packages ready"
fi

# Run tests for a specific mode.
# Arguments:
#   $1 – mode flag passed to alpacc (e.g. "--lexer", "--parser", or "" for both)
run_json_test() {
    local mode_flag="$1"

    local mode_name
    if [ -z "$mode_flag" ]; then
        mode_name="combined"
    else
        mode_name="${mode_flag#--}"
    fi

    echo "========================================="
    echo "Testing JSON $mode_name mode (backend=$backend)..."
    echo "========================================="

    local work_dir="$temp_dir/$mode_name"
    mkdir -p "$work_dir"
    cd "$work_dir"

    # Generate the test input/output files
    # shellcheck disable=SC2086
    if ! alpacc test generate "$GRAMMAR" --single-long --length $length $mode_flag; then
        echo "ERROR: alpacc test generate failed for $mode_name mode"
        return 1
    fi

    if [ "$backend" = "c" ]; then
        # C backend: compile the generated .c file with cc and run it.
        # shellcheck disable=SC2086
        if ! alpacc c "$GRAMMAR" $mode_flag -o json.c; then
            echo "ERROR: alpacc c failed for $mode_name mode"
            return 1
        fi

        if ! cc -std=c99 -O2 -o json json.c &>/dev/null; then
            echo "ERROR: cc compilation failed for $mode_name mode"
            return 1
        fi

        ./json < json.inputs > json.results

    elif [ "$backend" = "cuda" ]; then
        # CUDA backend: compile the generated .cu file with nvcc and run it.
        # shellcheck disable=SC2086
        if ! alpacc cuda "$GRAMMAR" $mode_flag -o json.cu; then
            echo "ERROR: alpacc cuda failed for $mode_name mode"
            return 1
        fi

        local arch="${CUDA_ARCH:-native}"
        if ! nvcc -O3 -std=c++17 -arch="$arch" -o json json.cu &>/dev/null; then
            echo "ERROR: nvcc compilation failed for $mode_name mode"
            return 1
        fi

        ./json < json.inputs > json.results

    else
        # Futhark backend: generate .fut and run via futhark script.
        cp -r "$temp_dir/lib" .
        cp "$temp_dir/futhark.pkg" .

        # shellcheck disable=SC2086
        if ! alpacc futhark "$GRAMMAR" $mode_flag; then
            echo "ERROR: alpacc futhark failed for $mode_name mode"
            return 1
        fi

        # `futhark script -b` produces binary output with a 16-byte header;
        # strip it before passing to `alpacc test compare`.
        futhark script --backend="$backend" -b json.fut 'test ($loadbytes "json.inputs")' \
            | tail -c +16 > json.results
    fi

    # shellcheck disable=SC2086
    if ! alpacc test compare "$GRAMMAR" json.inputs json.outputs json.results $mode_flag; then
        echo "ERROR: Test FAILED for JSON $mode_name mode"
        return 1
    fi

    echo "JSON $mode_name test PASSED"
    return 0
}

run_json_test "--lexer"  || exit 1
run_json_test "--parser" || exit 1
run_json_test ""         || exit 1

echo "All JSON grammar tests passed."
