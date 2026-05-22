#!/bin/bash
#SBATCH --job-name=ecv1_debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.out
#SBATCH --error=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.err
set -euo pipefail
cd /home/u6jv/harsh.u6jv/ecstasy
source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-esmfold/bin/activate
export PATH="/home/u6jv/harsh.u6jv/ecstasy/tools/hhsuite/bin:$PATH"
python scripts/build_ecstasy_v1/debug_1b70.py 2>&1
