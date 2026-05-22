#!/bin/bash
#SBATCH --job-name=ecv1_master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.out
#SBATCH --error=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.err

set -euo pipefail
cd /home/u6jv/harsh.u6jv/ecstasy
source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-boltz/bin/activate

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "----"

python scripts/build_ecstasy_v1/07_build_master.py \
    2>&1 | tee /projects/u6jv/ecstasy/benchmarks/ecstasy_v1/07_build_master.log

echo "----"
echo "Done: $(date)"
