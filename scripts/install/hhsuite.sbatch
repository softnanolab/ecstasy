#!/bin/bash
#SBATCH --job-name=ecv1_hhsuite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.out
#SBATCH --error=/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/slurm_%x_%j.err

set -euo pipefail
TOOLS=/home/u6jv/harsh.u6jv/ecstasy/tools
BUILD=/projects/u6jv/ecstasy/build/hh-suite
mkdir -p "$TOOLS" "$BUILD"
cd "$BUILD"

module load gcc-native/13.2 || true
module load cmake || true

# Fetch + extract source
if [ ! -d hh-suite-3.3.0 ]; then
  curl -fsSL https://github.com/soedinglab/hh-suite/archive/refs/tags/v3.3.0.tar.gz | tar xz
  mv hh-suite-3.3.0 src
fi
cd src
mkdir -p build_aarch64 && cd build_aarch64

# Configure for aarch64 (disable SSE/AVX flags)
cmake .. -DCMAKE_INSTALL_PREFIX="$TOOLS/hhsuite" -DCMAKE_BUILD_TYPE=Release \
   -DHAVE_AVX2=0 -DHAVE_SSE2=0 -DHAVE_SSE4=0 || cmake .. -DCMAKE_INSTALL_PREFIX="$TOOLS/hhsuite" -DCMAKE_BUILD_TYPE=Release

make -j $SLURM_CPUS_PER_TASK
make install

ls -la $TOOLS/hhsuite/bin/hhfilter
$TOOLS/hhsuite/bin/hhfilter -h 2>&1 | head -5
echo "Done: $(date)"
