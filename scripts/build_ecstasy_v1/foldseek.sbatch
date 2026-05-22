#!/bin/bash
#SBATCH --job-name=ecv1_foldseek
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.out
#SBATCH --error=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.err

set -euo pipefail
cd /home/u6jv/harsh.u6jv/ecstasy

source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-boltz/bin/activate
export OMP_NUM_THREADS=32

echo "Host:    $(hostname)"
echo "Cores:   $(nproc)  (using --cpus-per-task=$SLURM_CPUS_PER_TASK)"
echo "Mem:     $(free -g | awk '/^Mem/{print $2 " GB total"}')"
echo "Date:    $(date)"
echo "----"

python scripts/build_ecstasy_v1/04_run_foldseek.py \
    2>&1 | tee /projects/u6jv/ecstasy/benchmarks/ecstasy_v1/04_foldseek.log

python scripts/build_ecstasy_v1/05_compute_interface_edges.py \
    2>&1 | tee /projects/u6jv/ecstasy/benchmarks/ecstasy_v1/05_compute_edges.log

python scripts/build_ecstasy_v1/06_deleak_report.py \
    2>&1 | tee /projects/u6jv/ecstasy/benchmarks/ecstasy_v1/06_deleak_report.log

echo "----"
echo "Done: $(date)"
