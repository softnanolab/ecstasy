#!/bin/bash
#SBATCH --job-name=ecv1_msapf_f
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.out
#SBATCH --error=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.err
set -euo pipefail
cd /home/u6jv/harsh.u6jv/ecstasy
source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-esmfold/bin/activate
export PATH="/home/u6jv/harsh.u6jv/ecstasy/tools/hhsuite/bin:$PATH"
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
echo "----"
python scripts/build_ecstasy_v1/09b_run_msa_pairformer_filtered.py --run-id filtered \
  2>&1 | tee /projects/u6jv/ecstasy/benchmarks/ecstasy_v1/09b_msa_pairformer_filtered.log
python scripts/build_ecstasy_v1/10_score_msa_pairformer.py --run-id filtered \
  2>&1 | tee /projects/u6jv/ecstasy/benchmarks/ecstasy_v1/10_score_filtered.log
echo "Done: $(date)"
