#!/bin/bash
# Re-run MINT validation on the mint_seqid30 val split using ecstasy's
# benchmark enumeration + metric. One process, loads MINT once, no per-entry
# subprocess.
#
# Usage:
#   sbatch scripts/sbatch/rerun_mint_val.sbatch

#SBATCH --job-name=ecstasy_mint_rerun
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/test_runs/logs/%x_%j.out
#SBATCH --error=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/test_runs/logs/%x_%j.err

set -euo pipefail
mkdir -p /projects/u6jv/boltz_benchmarking/DATA/ecstasy/test_runs/logs

ECSTASY_ROOT=/home/u6jv/harsh.u6jv/ecstasy
cd "$ECSTASY_ROOT"

module load gcc-native/13.2
module load cudatoolkit/24.11_12.6

# MINT lives in .venv-boltz (mint is installed as a namespace package).
PY="${ECSTASY_ROOT}/envs/.venv-boltz/bin/python"

echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "python:  $PY"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "----"

START=$(date +%s)
"${PY}" analysis/rerun_mint_val.py \
    --config /projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/config.yaml \
    --weights /projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/checkpoints/last.ckpt \
    --out_json "${ECSTASY_ROOT}/analysis/mint_3khmvobe_rerun.json" \
    --out_csv "${ECSTASY_ROOT}/analysis/mint_3khmvobe_rerun.csv"
RC=$?
echo "---- DONE (rc=$RC) ----"
echo "Wallclock seconds: $(( $(date +%s) - START ))"
exit $RC
