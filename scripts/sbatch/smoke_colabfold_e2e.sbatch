#!/bin/bash
#SBATCH --job-name=ecstasy_smoke_colabfold
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_colabfold/logs/%x_%j.out
#SBATCH --error=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_colabfold/logs/%x_%j.err

set -euo pipefail
mkdir -p /projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_colabfold/logs

ECSTASY_ROOT=/home/u6jv/harsh.u6jv/ecstasy
cd "$ECSTASY_ROOT"

module load gcc-native/13.2
module load cudatoolkit/24.11_12.6
# Use the ESMFold venv (has ecstasy CLI installed) to drive the outer adapter.
# The colabfold runner itself is invoked via env_path's python (3.11 colabfold venv).
source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-esmfold/bin/activate

echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "ecstasy: $(which ecstasy)"
echo "----"

START=$(date +%s)
# Diagnostic: force CPU JAX to test whether the segfault is GPU/cuDNN-specific.
export JAX_PLATFORMS=cpu
ecstasy bench predict --config configs/mint_seqid30__colabfold_smoke.yaml
echo "---- predict done ----"
ecstasy bench score   --config configs/mint_seqid30__colabfold_smoke.yaml
echo "---- DONE ----"
echo "Wallclock seconds: $(( $(date +%s) - START ))"

echo "---- contact files ----"
find /projects/u6jv/boltz_benchmarking/DATA/ecstasy/benchmarks/mint_seqid30/predictions/colabfold -name 'contact.npz' -exec ls -la {} \;
echo "---- result files ----"
ls -la /projects/u6jv/boltz_benchmarking/DATA/ecstasy/benchmarks/mint_seqid30/results/ 2>/dev/null || true
