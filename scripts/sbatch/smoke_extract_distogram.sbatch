#!/bin/bash

#SBATCH --job-name=ecstasy_smoke_distogram
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_extract_distogram/logs/%x_%j.out
#SBATCH --error=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_extract_distogram/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/u6jv/harsh.u6jv/ecstasy
cd "$PROJECT_ROOT"

module load cudatoolkit/24.11_12.6
source "$PROJECT_ROOT/envs/.venv-boltz/bin/activate"

SMOKE=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_extract_distogram

echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "boltz:   $(python -c 'import boltz; print(boltz.__file__)')"
echo "Inputs:  $(ls $SMOKE/inputs/)"
echo "----"

START=$(date +%s)
srun boltz predict \
    "$SMOKE/inputs" \
    --out_dir "$SMOKE/out" \
    --model boltz2 \
    --devices 1 \
    --recycling_steps 3 \
    --sampling_steps 25 \
    --diffusion_samples 1 \
    --num_workers 0 \
    --output_format mmcif \
    --no_kernels \
    --override \
    --dump_distogram
END=$(date +%s)

echo "---- DONE ----"
echo "Wallclock seconds: $((END - START))"
echo "---- distograms ----"
find "$SMOKE/out" -name 'distogram_*.npz' -exec ls -la {} \;
echo "---- shape probe ----"
python - <<'PY'
import numpy as np, glob
for p in glob.glob('/projects/u6jv/boltz_benchmarking/DATA/ecstasy/smoke_extract_distogram/out/**/distogram_*.npz', recursive=True):
    d = np.load(p)
    print(p)
    print("  probs.shape:", d['probs'].shape, "dtype:", d['probs'].dtype)
    print("  length:", int(d['length']))
    print("  bin sums (sample):", d['probs'][0, 0].sum(), d['probs'][-1, -1].sum())
PY
