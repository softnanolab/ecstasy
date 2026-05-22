#!/usr/bin/env bash
# Compile hh-suite 3.3.0 from source. Required for MSA Pairformer's hhfilter
# MSA diversification step. Source build because upstream only ships SSE2/AVX2
# binaries (no aarch64). Drop into tools/hhsuite/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
DEST="$HERE/tools/hhsuite"
BUILD="$(mktemp -d)"
trap "rm -rf $BUILD" EXIT
cd "$BUILD"

curl -fsSL https://github.com/soedinglab/hh-suite/archive/refs/tags/v3.3.0.tar.gz | tar xz
cd hh-suite-3.3.0
mkdir -p build && cd build

NCPUS="${NCPUS:-$(nproc 2>/dev/null || echo 4)}"
cmake .. -DCMAKE_INSTALL_PREFIX="$DEST" -DCMAKE_BUILD_TYPE=Release \
   -DHAVE_AVX2=0 -DHAVE_SSE2=0 -DHAVE_SSE4=0 \
  || cmake .. -DCMAKE_INSTALL_PREFIX="$DEST" -DCMAKE_BUILD_TYPE=Release

make -j "$NCPUS"
make install
"$DEST/bin/hhfilter" -h 2>&1 | head -3
echo "installed: $DEST/bin/hhfilter"
