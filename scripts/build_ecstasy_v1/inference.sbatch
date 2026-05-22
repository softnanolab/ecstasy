#!/bin/bash
#SBATCH --job-name=ecv1_msapf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.out
#SBATCH --error=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.err

set -euo pipefail
cd /home/u6jv/harsh.u6jv/ecstasy
# .venv-esmfold has msa_pairformer + openfold + torch with CUDA, per the existing config.
source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-esmfold/bin/activate

# Put hhsuite on PATH so MSA(...) finds hhfilter
export PATH="/home/u6jv/harsh.u6jv/ecstasy/tools/hhsuite/bin:$PATH"

echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
hhfilter -h 2>&1 | head -3 || echo "hhfilter NOT FOUND"
echo "----"

RUN_ID="${RUN_ID:-run0}"

python scripts/build_ecstasy_v1/09_run_msa_pairformer.py --run-id "$RUN_ID" \
  2>&1 | tee /projects/u6jv/ecstasy/benchmarks/ecstasy_v1/09_msa_pairformer_${RUN_ID}.log

python scripts/build_ecstasy_v1/10_score_msa_pairformer.py --run-id "$RUN_ID" \
  2>&1 | tee /projects/u6jv/ecstasy/benchmarks/ecstasy_v1/10_score_${RUN_ID}.log

echo "----"
echo "Done: $(date)"
