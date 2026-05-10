#!/bin/bash

#SBATCH --job-name=ecstasy_smoke_e2e
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_e2e/logs/%x_%j.out
#SBATCH --error=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_e2e/logs/%x_%j.err

set -euo pipefail

mkdir -p /projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_e2e/logs

ECSTASY_ROOT=/home/u6jv/harsh.u6jv/ecstasy
cd "$ECSTASY_ROOT"

module load cudatoolkit/24.11_12.6
source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-boltz/bin/activate

echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "ecstasy: $(which ecstasy)"
echo "----"

START=$(date +%s)
ecstasy bench predict --config configs/mint_seqid30__boltz2_smoke.yaml
echo "---- predict done ----"
ecstasy bench score --config configs/mint_seqid30__boltz2_smoke.yaml
echo "---- DONE ----"
echo "Wallclock seconds: $(( $(date +%s) - START ))"

echo "---- contact files ----"
find /projects/u6jv/boltz_benchmarking/DATA/ecstasy/benchmarks/mint_seqid30/predictions -name 'contact.npz' -exec ls -la {} \;
echo "---- result files ----"
ls -la /projects/u6jv/boltz_benchmarking/DATA/ecstasy/benchmarks/mint_seqid30/results/ 2>/dev/null || true
