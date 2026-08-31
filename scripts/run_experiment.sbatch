#!/usr/bin/env bash
#SBATCH --job-name=ecstasy-exp
#SBATCH --output=logs/ecstasy_exp_%j.out
#SBATCH --error=logs/ecstasy_exp_%j.err
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=workq
# Generic experiment runner. Usage:
#   sbatch scripts/run_experiment.sbatch experiments/boltz2_headline/manifest.yaml [extra ecstasy args]
# Runs in .venv-boltz (has ecstasy + torch + mentos, so predict subprocesses to
# each model's env and scoring can load the MENTOS GT here).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO_ROOT}/logs"
VENV="${ECSTASY_VENV:-${ENVS_ROOT:?set ECSTASY_VENV or ENVS_ROOT}/.venv-boltz}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${REPO_ROOT}"
exec ecstasy experiment "$@"
