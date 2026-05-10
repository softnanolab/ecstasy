#!/bin/bash
#SBATCH --job-name=mint_diag
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_mint/logs/%x_%j.out
#SBATCH --error=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_mint/logs/%x_%j.err

set -euo pipefail
cd /home/u6jv/harsh.u6jv
module load cudatoolkit/24.11_12.6
source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-boltz/bin/activate
srun python /home/u6jv/harsh.u6jv/ecstasy/scripts/diag_mint_inference.py
