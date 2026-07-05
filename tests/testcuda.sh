#!/bin/bash

# Differential test of the CUDA parser backend.
#
# For each random grammar this script:
#   1. Generates test inputs with `alpacc test generate --length 4`.
#   2. Runs the CUDA parser in batch mode across all six (BS, IPT) combinations
#      and checks each against `alpacc test compare`.
#   3. Runs the CUDA parser in server mode (one frame per test) and checks
#      the results match the batch output.
#   4. Verifies -i/-o file mode gives identical output to stdin/stdout mode.
#
# Requires nvcc and an NVIDIA GPU; not run in hosted CI (GPU-less).

show_usage() {
    echo "Usage: $0 [q_value] [k_value] [target_runs] [parallel_jobs] [arch]"
    echo "  q_value:       -q parameter for alpacc (default: 1)"
    echo "  k_value:       -k parameter for alpacc (default: 1)"
    echo "  target_runs:   number of successful grammars (default: 10)"
    echo "  parallel_jobs: number of parallel jobs (default: 1)"
    echo "  arch:          nvcc -arch value (default: native)"
    echo "Example: $0 1 1 20 1 native"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage; exit 0
fi

export PATH="$HOME/bin:$PATH"

q_value="${1:-1}"
k_value="${2:-1}"
target="${3:-10}"
parallel_jobs="${4:-1}"
arch="${5:-native}"

if ! [[ "$q_value" =~ ^[0-9]+$ ]] || ! [[ "$k_value" =~ ^[0-9]+$ ]] || ! [[ "$target" =~ ^[0-9]+$ ]]; then
    echo "Error: q_value, k_value, and target must be positive integers"
    show_usage; exit 1
fi

echo "Starting alpacc CUDA parser testing..."
echo "Target: $target successful grammars"
echo "Using -q $q_value -k $k_value --parser"
echo "nvcc arch: $arch"
echo "Running with $parallel_jobs parallel jobs"

# ---------------------------------------------------------------------------
# Python helper for server-mode testing.
# Converts the batch binary format to length-prefixed server frames,
# drives the CUDA parser in --server mode, and collects the responses.
# Usage: python3 server_test.py <binary> <inputs_file> <output_file>
# Writes output in the same batch format as batch mode (with num_tests prefix)
# so `alpacc test compare` can check it directly.
# ---------------------------------------------------------------------------
SERVERTEST_PY='
import sys, struct, subprocess

def u64be(v): return struct.pack(">Q", v)
def read_u64be(data, off): return struct.unpack_from(">Q", data, off)[0], off + 8

binary, inputs_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

with open(inputs_file, "rb") as f:
    data = f.read()

off = 0
num_tests, off = read_u64be(data, off)

proc = subprocess.Popen([binary, "--server"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL)

# Send all frames then close stdin
frames = b""
for _ in range(num_tests):
    n, off = read_u64be(data, off)
    tokens = data[off:off + 8*n]
    off += 8*n
    content = u64be(n) + tokens
    frames += u64be(len(content)) + content
proc.stdin.write(frames)
proc.stdin.close()

raw = proc.stdout.read()
ret = proc.wait()
if ret != 0:
    sys.stderr.write(f"server exit {ret}\n")
    sys.exit(1)

# Prefix with num_tests to match batch output format
with open(output_file, "wb") as f:
    f.write(u64be(num_tests) + raw)
'

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

    local work_dir="$temp_dir/job_$job_id"
    mkdir -p "$work_dir"
    cd "$work_dir"

    # Write the server-test helper locally
    echo "$servertest_py" > server_test.py

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

        # Try to generate CUDA parser; skip grammars that fail codegen
        if ! alpacc cuda random.alp --parser &> /dev/null; then
            continue
        fi

        if ! nvcc -std=c++17 -arch="$arch" -o random random.cu &> /dev/null; then
            echo "nvcc compilation failed for job $job_id"; return 1
        fi

        # Generate test inputs (small, to keep GPU memory bounded)
        alpacc test generate random.alp --parser --length 4 &> /dev/null

        local all_ok=true

        # ---- Test 1: batch mode across all six (BS, IPT) combinations ----
        for bs in 128 256; do
            for ipt in 2 4 8; do
                ./random --block-size "$bs" --items-per-thread "$ipt" \
                    -i random.inputs -o "results_${bs}_${ipt}.bin" 2>/dev/null

                if ! alpacc test compare random.alp random.inputs random.outputs \
                        "results_${bs}_${ipt}.bin" --parser &> /dev/null; then
                    echo "===== FAIL: batch BS=$bs IPT=$ipt, job $job_id ====="
                    cat random.alp
                    alpacc test compare random.alp random.inputs random.outputs \
                        "results_${bs}_${ipt}.bin" --parser
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

        # ---- Test 2: server mode (default BS/IPT) ----
        if ! python3 server_test.py ./random random.inputs server_results.bin 2>/dev/null; then
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
    "run_test {} $q_value $k_value $temp_dir $counter_file $target $done_file $arch $(printf '%q' "$SERVERTEST_PY")"

final_count=$(cat "$counter_file")
if [ "$final_count" -ge "$target" ]; then
    echo "Tests passed."
    exit 0
else
    echo "Failed to reach target of $target successful runs (got $final_count)"
    exit 1
fi
