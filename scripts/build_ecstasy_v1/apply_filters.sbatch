#!/bin/bash
#SBATCH --job-name=ecv1_filter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.out
#SBATCH --error=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.err
set -euo pipefail
cd /home/u6jv/harsh.u6jv/ecstasy
source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-boltz/bin/activate
python scripts/build_ecstasy_v1/11_apply_notebook_filters.py 2>&1
