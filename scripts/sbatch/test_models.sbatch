#!/bin/bash
# End-to-end GPU test suite: installation checks (per venv) + model-prediction
# smokes (per adapter). Runs from .venv-ecstasy; cross-spawns into each model
# venv for installation imports and via the ecstasy CLI for predictions.
#
# Usage:
#   sbatch scripts/sbatch/test_models.sbatch                 # full run
#   sbatch --export=ALL,SKIP_COLABFOLD_TEST=1 scripts/sbatch/test_models.sbatch
#   sbatch --export=ALL,PYTEST_K=boltz2 scripts/sbatch/test_models.sbatch

#SBATCH --job-name=ecstasy_test_models
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/test_runs/logs/%x_%j.out
#SBATCH --error=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/test_runs/logs/%x_%j.err

set -euo pipefail
mkdir -p /projects/u6jv/boltz_benchmarking/DATA/ecstasy/test_runs/logs

ECSTASY_ROOT=/home/u6jv/harsh.u6jv/ecstasy
cd "$ECSTASY_ROOT"

# Load CUDA so the model subprocesses (torch / jax) can find libcudnn etc.
module load gcc-native/13.2
module load cudatoolkit/24.11_12.6

# Activate the orchestrator venv (CLI + pytest live here)
source "${ECSTASY_ROOT}/envs/.venv-ecstasy/bin/activate"

echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "python:  $(which python)"
echo "pytest:  $(python -m pytest --version)"
echo "ecstasy: $(which ecstasy)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "----"

# Pytest output goes to a tmp basetemp under /projects so we can inspect
# artifacts post-mortem (compute-node /tmp is wiped at job end).
BASETEMP="/projects/u6jv/boltz_benchmarking/DATA/ecstasy/test_runs/pytest_tmp_${SLURM_JOB_ID:-local}"
mkdir -p "${BASETEMP}"

# Optional pytest -k filter via env var (e.g. PYTEST_K=boltz2)
KEXPR="${PYTEST_K:-}"
KARGS=()
if [ -n "${KEXPR}" ]; then
  KARGS=(-k "${KEXPR}")
fi

START=$(date +%s)
python -m pytest \
    tests/installation tests/integration \
    -v -ra --tb=short --color=yes \
    --basetemp="${BASETEMP}" \
    "${KARGS[@]}"
RC=$?

echo "---- DONE (rc=$RC) ----"
echo "Wallclock seconds: $(( $(date +%s) - START ))"
echo "pytest tmp:        ${BASETEMP}"
exit $RC
