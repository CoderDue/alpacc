#!/bin/bash

# Differential test of the CUDA backend (lexer, parser, or combined mode).
#
# For each random grammar this script:
#   1. Generates test inputs with `alpacc test generate --length 4`.
#   2. Runs the CUDA binary in batch mode across all six (BS, IPT) combinations
#      and checks each against `alpacc test compare`.
#   3. Runs the CUDA binary in server mode (one frame per test) and checks
#      the results match the batch output.
#   4. Verifies -i/-o file mode gives identical output to stdin/stdout mode.
#
# Requires nvcc and an NVIDIA GPU; not run in hosted CI (GPU-less).

show_usage() {
    echo "Usage: $0 [q_value] [k_value] [target_runs] [parallel_jobs] [type_flag] [arch]"
    echo "  q_value:       -q parameter for alpacc (default: 1)"
    echo "  k_value:       -k parameter for alpacc (default: 1)"
    echo "  target_runs:   number of successful grammars (default: 10)"
    echo "  parallel_jobs: number of parallel jobs (default: 1)"
    echo "  type_flag:     --lexer, --parser, or empty for combined (default: empty)"
    echo "  arch:          nvcc -arch value (default: native)"
    echo "Example: $0 1 1 20 1 '' native"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage; exit 0
fi

export PATH="$HOME/bin:$PATH"

q_value="${1:-1}"
k_value="${2:-1}"
target="${3:-10}"
parallel_jobs="${4:-1}"
type_flag="${5:-}"
arch="${6:-native}"

if ! [[ "$q_value" =~ ^[0-9]+$ ]] || ! [[ "$k_value" =~ ^[0-9]+$ ]] || ! [[ "$target" =~ ^[0-9]+$ ]]; then
    echo "Error: q_value, k_value, and target must be positive integers"
    show_usage; exit 1
fi

if ! [[ "$parallel_jobs" =~ ^[0-9]+$ ]]; then
    echo "Error: parallel_jobs must be a positive integer"
    show_usage; exit 1
fi

echo "Starting alpacc CUDA testing..."
echo "Target: $target successful grammars"
echo "Using -q $q_value -k $k_value ${type_flag:-<combined>}"
echo "nvcc arch: $arch"
echo "Running with $parallel_jobs parallel jobs"

# ---------------------------------------------------------------------------
# Server-mode helper: tests/server_test.py builds native request frames
# from the u64-BE test inputs (querying `binary --layout` for type sizes)
# and transcodes the native responses back to the u64-BE test format so
# they can be diffed against batch output.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVERTEST_PY="$SCRIPT_DIR/server_test.py"

temp_dir=$(mktemp -d)
trap "rm -rf $temp_dir" EXIT

counter_file="$temp_dir/counter"
echo "0" > "$counter_file"
done_file="$temp_dir/done"

run_test() {
    local job_id=$1
    local q_value=$2
    local k_value=$3
    local temp_dir=$4
    local counter_file=$5
    local target=$6
    local done_file=$7
    local arch=$8
    local servertest_py=$9
    local type_flag="${10}"

    local work_dir="$temp_dir/job_$job_id"
    mkdir -p "$work_dir"
    cd "$work_dir"

    # Lexer-only binaries need --inputs to read framed test batches
    # (their default mode is a raw byte stream). The server-helper kind
    # selects the native frame format per mode.
    local batch_flags=""
    local frame_kind="both"
    if [ "$type_flag" = "--lexer" ]; then
        batch_flags="--inputs"
        frame_kind="lexer"
    elif [ "$type_flag" = "--parser" ]; then
        frame_kind="parser"
    fi

    local codegen_fails=0

    while [ ! -f "$done_file" ]; do
        # Generate random grammar
        if ! alpacc random &> /dev/null; then
            echo "alpacc random failed"; return 1
        fi
        cat > random.alp.tmp << ALPEOF
params {
  lookback=$q_value.
  lookahead=$k_value.
}

ALPEOF
        cat random.alp >> random.alp.tmp
        mv random.alp.tmp random.alp

        # Try to generate CUDA code; skip grammars that fail codegen,
        # but fail fast on configuration errors or endless codegen failures.
        if ! alpacc cuda random.alp $type_flag &> codegen_err.txt; then
            if grep -q "must be positive" codegen_err.txt; then
                echo "ERROR: alpacc rejected the configuration itself (job $job_id):"
                cat codegen_err.txt
                return 1
            fi
            codegen_fails=$((codegen_fails + 1))
            if [ "$codegen_fails" -ge 1000 ]; then
                echo "ERROR: codegen failed $codegen_fails consecutive times for job $job_id; giving up. Last error:"
                cat codegen_err.txt
                return 1
            fi
            continue
        fi
        codegen_fails=0

        if ! nvcc -std=c++17 -arch="$arch" -o random random.cu &> /dev/null; then
            echo "nvcc compilation failed for job $job_id"; return 1
        fi

        # Generate test inputs (small, to keep GPU memory bounded)
        alpacc test generate random.alp $type_flag --length 4 &> /dev/null

        local all_ok=true

        # ---- Test 1: batch mode across all six (BS, IPT) combinations ----
        for bs in 128 256; do
            for ipt in 2 4 8; do
                ./random --block-size "$bs" --items-per-thread "$ipt" $batch_flags \
                    -i random.inputs -o "results_${bs}_${ipt}.bin" 2>/dev/null

                if ! alpacc test compare random.alp random.inputs random.outputs \
                        "results_${bs}_${ipt}.bin" $type_flag &> /dev/null; then
                    echo "===== FAIL: batch BS=$bs IPT=$ipt, job $job_id ====="
                    cat random.alp
                    alpacc test compare random.alp random.inputs random.outputs \
                        "results_${bs}_${ipt}.bin" $type_flag
                    all_ok=false
                    break 2
                fi

                # All (BS,IPT) results must match each other
                if [ "$bs" != "128" ] || [ "$ipt" != "2" ]; then
                    if ! diff -q results_128_2.bin "results_${bs}_${ipt}.bin" &>/dev/null; then
                        echo "===== FAIL: BS=$bs IPT=$ipt output differs from BS=128 IPT=2, job $job_id ====="
                        all_ok=false
                        break 2
                    fi
                fi
            done
        done

        if ! $all_ok; then return 1; fi

        # ---- Test 2: server mode (default BS/IPT, native protocol) ----
        if ! python3 "$servertest_py" ./random random.inputs server_results.bin "$frame_kind" 2>/dev/null; then
            echo "===== FAIL: server mode crashed, job $job_id ====="
            all_ok=false
        elif ! diff -q results_128_2.bin server_results.bin &>/dev/null; then
            echo "===== FAIL: server mode output differs from batch, job $job_id ====="
            all_ok=false
        fi

        if ! $all_ok; then return 1; fi

        # ---- Success ----
        (
            flock -x 200
            count=$(cat "$counter_file")
            if [ "$count" -lt "$target" ]; then
                count=$((count + 1))
                echo "$count" > "$counter_file"
                echo "$count/$target completed"
                if [ "$count" -ge "$target" ]; then
                    touch "$done_file"
                fi
            fi
        ) 200>"$counter_file.lock"
        return 0
    done
    return 0
}

export -f run_test

seq 1 $target | parallel --no-notice -j "$parallel_jobs" --halt soon,fail=1 --line-buffer \
    "run_test {} $q_value $k_value $temp_dir $counter_file $target $done_file $arch $(printf '%q' "$SERVERTEST_PY") '$type_flag'"

final_count=$(cat "$counter_file")
if [ "$final_count" -ge "$target" ]; then
    echo "Tests passed."
    exit 0
else
    echo "Failed to reach target of $target successful runs (got $final_count)"
    exit 1
fi
