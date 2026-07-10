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
module load gmp
module unload gcc
module load gcc/11.2.0

INPUT_SIZE=104857600

make build         FUTHARK_BACKEND=cuda INPUT_SIZE=$INPUT_SIZE
make bench-cuda    INPUT_SIZE=$INPUT_SIZE
make bench-futhark FUTHARK_BACKEND=cuda INPUT_SIZE=$INPUT_SIZE

echo "Benchmarking finished."
