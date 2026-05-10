#!/bin/bash
#SBATCH --job-name=install_openfold
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/install_logs/%x_%j.out
#SBATCH --error=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/install_logs/%x_%j.err

set -euo pipefail
mkdir -p /projects/u6jv/boltz_benchmarking/DATA/ecstasy/install_logs
cd /home/u6jv/harsh.u6jv/ecstasy

module load gcc-native/13.2  # nvcc 12.6 caps host gcc at 13; gcc 13 also supports C++17.
module load cudatoolkit/24.11_12.6
source /home/u6jv/harsh.u6jv/ecstasy/envs/.venv-esmfold/bin/activate

echo "Host:    $(hostname)"
echo "GPUs:    $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1 | head -2)"
echo "nvcc:    $(which nvcc)  $(nvcc --version | tail -1)"
echo "torch:   $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "----"

# pip install -e modules/openfold compiles attn_core_inplace_cuda;
# nvcc + CUDA_HOME + sm_90 detection all need to be present on this node.
# cudatoolkit/24.11 puts NVIDIA HPC `nvc` ahead on PATH; setuptools' gcc flags
# break under it, so force the modern gcc/g++ from gcc-native/14.2.
export CC=$(command -v gcc)
export CXX=$(command -v g++)
echo "CC=$CC  CXX=$CXX"
$CC --version | head -1
BUILD_LOG=/projects/u6jv/boltz_benchmarking/DATA/ecstasy/install_logs/openfold_build_${SLURM_JOB_ID}.log
TORCH_CUDA_ARCH_LIST="9.0" \
    pip install -v --no-cache-dir --no-build-isolation --no-deps -e ./modules/openfold > "$BUILD_LOG" 2>&1 || true
echo "---- last 80 lines of build log ($BUILD_LOG) ----"
tail -80 "$BUILD_LOG"

echo "---- post-install probe ----"
python -c "
import esm
from esm.pretrained import esmfold_v1
print('fair-esm:', esm.__version__)
print('esmfold_v1 importable:', callable(esmfold_v1))
import openfold
print('openfold:', openfold.__file__)
import attn_core_inplace_cuda
print('attn_core_inplace_cuda OK')
"
