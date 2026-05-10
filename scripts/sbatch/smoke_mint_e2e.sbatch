#!/bin/bash

#SBATCH --job-name=ecstasy_smoke_mint
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_mint/logs/%x_%j.out
#SBATCH --error=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_mint/logs/%x_%j.err

set -euo pipefail
mkdir -p /projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_mint/logs

ECSTASY_ROOT=/home/u6jv/harsh.u6jv/ecstasy
cd "$ECSTASY_ROOT"

module load cudatoolkit/24.11_12.6
source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-boltz/bin/activate

CFG=/projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/config.yaml
WTS=/projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/checkpoints/last.ckpt

echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "ecstasy: $(which ecstasy)"
echo "cfg:     $CFG"
echo "wts:     $WTS"
echo "----"

START=$(date +%s)
ecstasy bench predict \
    --config configs/mint_seqid30__mint_smoke.yaml \
    --model_config "$CFG" \
    --model_weights "$WTS"
echo "---- predict done ----"
ecstasy bench score \
    --config configs/mint_seqid30__mint_smoke.yaml \
    --model_config "$CFG" \
    --model_weights "$WTS"
echo "---- DONE ----"
echo "Wallclock seconds: $(( $(date +%s) - START ))"

echo "---- contact files ----"
find /projects/u6jv/boltz_benchmarking/DATA/ecstasy/benchmarks/mint_seqid30/predictions/mint -name 'contact.npz' -exec ls -la {} \;
echo "---- result files ----"
ls -la /projects/u6jv/boltz_benchmarking/DATA/ecstasy/benchmarks/mint_seqid30/results/ 2>/dev/null || true
