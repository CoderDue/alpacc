#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=benchmark.log

set -e

module unload cuda
module load cuda/11.8
module load futhark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/bin:$PATH"

INPUT_SIZE=104857600

make -C "$SCRIPT_DIR/benchmarks" build         FUTHARK_BACKEND=cuda INPUT_SIZE=$INPUT_SIZE
make -C "$SCRIPT_DIR/benchmarks" bench-cuda    INPUT_SIZE=$INPUT_SIZE
make -C "$SCRIPT_DIR/benchmarks" bench-futhark FUTHARK_BACKEND=cuda INPUT_SIZE=$INPUT_SIZE

echo "Benchmarking finished."
