#!/usr/bin/env bash
# Install PLMGraph-Inter (Si & Yan, eLife 2024 — PLM-embedded geometric graphs for
# inter-protein contact prediction) as an ecstasy model.
#
# PLMGraph-Inter is DRN-1D2D_Inter's successor: same MSA/coevolution pipeline
# (CCMpred + alnstats + hh-suite + ESM-1b/ESM-MSA-1b) PLUS ESM-IF1 inverse-folding
# embeddings and GVP structure graphs — so it additionally needs a per-chain PDB
# structure for each chain (ESMFold monomers for sequence-only val splits).
#
# PORT NOTE (Isambard / aarch64): upstream targets py3.8 + torch1.9 (x86). This builds
# on the project's torch2.x/cu126 stack; the runner carries a torch.load(weights_only
# =False) shim (fair-esm checkpoints) and the >h<i> a3m-header fix (LoadHHM). The PyG
# C++ extensions (torch-scatter/sparse/cluster, required by ESM-IF1) have no aarch64
# wheels, so they are compiled from source with gcc-12 + nvcc for sm_90.
set -euo pipefail
BLUE='\033[1;34m'; NC='\033[0m'; say() { echo -e "${BLUE}[plmgraph]${NC} $*"; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
ENV_PATH="${ENVS_ROOT:-$HERE/envs}/plmgraph_inter"
TOOLS="${TOOLS_ROOT:-$HERE/tools}"
WEIGHTS="${ECSTASY_ROOT:-$HERE}/weights/plmgraph_inter/esm"
PLMG="$HERE/modules/plmgraph_inter"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
PY="$ENV_PATH/bin/python"
NCPUS="${NCPUS:-$(nproc 2>/dev/null || echo 8)}"

# nvcc + CUDA-math paths (Isambard hpc_sdk). Override CUDA_HOME/MATH for another host.
HPC=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11
export CUDA_HOME="${CUDA_HOME:-$HPC/cuda/12.6}"
MATH="${MATH:-$HPC/math_libs/12.6/targets/sbsa-linux}"

[ -f "$PLMG/predict.py" ] || { echo "init submodule: git submodule update --init modules/plmgraph_inter" >&2; exit 1; }

# ---------------------------------------------------------------------------
say "1/5  venv + torch2.x + fair-esm + PyG base"
# ---------------------------------------------------------------------------
[ -x "$PY" ] || "$UV" venv --python 3.12 "$ENV_PATH"
"$UV" pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu126
"$UV" pip install --python "$PY" "fair-esm==2.0.0" biopython "numpy<2" scipy torch-geometric biotite

# ---------------------------------------------------------------------------
say "2/5  PyG C++ extensions (scatter/sparse/cluster) from source — gcc-12 + nvcc/sm_90"
# ---------------------------------------------------------------------------
# gcc-7 is too old (no -std=c++20); the hpc_sdk splits CUDA core and math libs, so the
# build can't find cusparse.h without MATH on CPATH/LIBRARY_PATH.
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:$MATH/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$MATH/lib:$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MATH/lib:${LD_LIBRARY_PATH:-}"
export CC=gcc-12 CXX=g++-12 NVCC_PREPEND_FLAGS="-ccbin g++-12"
export TORCH_CUDA_ARCH_LIST="9.0" FORCE_CUDA=1 MAX_JOBS="$NCPUS"
for pkg in torch-scatter torch-sparse torch-cluster; do
  "$PY" -c "import ${pkg//-/_}" 2>/dev/null || "$UV" pip install --python "$PY" --no-build-isolation "$pkg"
done
"$PY" -c "import gvp" 2>/dev/null || \
  "$UV" pip install --python "$PY" --no-deps "git+https://github.com/drorlab/gvp-pytorch.git"

# ---------------------------------------------------------------------------
say "3/5  external tools (CCMpred GPU, alnstats, fasta2aln-shim) — reuse hh-suite"
# ---------------------------------------------------------------------------
# These are identical to DRN's; build only if absent. CCMpred GPU build + the glibc
# fsqrt/<math.h> patch are non-trivial — if you still have them from a prior install,
# this is a no-op. (hh-suite is shared; build via scripts/install/hhsuite.sh if missing.)
[ -x "$TOOLS/hhsuite/bin/hhmake" ] || bash "$HERE/scripts/install/hhsuite.sh"
mkdir -p "$TOOLS/metapsicov/bin"
if [ ! -x "$TOOLS/metapsicov/bin/alnstats" ]; then
  B="$(mktemp -d)"; git clone --depth 1 https://github.com/psipred/metapsicov.git "$B/m"
  "${CC:-gcc}" -O3 -o "$TOOLS/metapsicov/bin/alnstats" "$B/m/src/alnstats.c" -lm; rm -rf "$B"
fi
# portable a3m->aln (replaces the x86-only fasta2aln binary)
if [ ! -x "$TOOLS/metapsicov/bin/fasta2aln" ]; then
  cat > "$TOOLS/metapsicov/bin/fasta2aln" <<'SH'
#!/usr/bin/env bash
exec python3 -c "
import sys
tab=str.maketrans('','','abcdefghijklmnopqrstuvwxyz.')
out=[];seq=[]
for line in open(sys.argv[1]):
    if line.startswith('>'):
        if seq: out.append(''.join(seq).translate(tab)); seq.clear()
    elif line.strip(): seq.append(line.strip())
if seq: out.append(''.join(seq).translate(tab))
open(sys.argv[2],'w').write('\n'.join(out)+'\n')
" "$1" "$2"
SH
  chmod +x "$TOOLS/metapsicov/bin/fasta2aln"
fi
[ -x "$TOOLS/ccmpred/bin/ccmpred" ] || { echo "WARN: build CCMpred GPU separately (CCMPRED_CUDA=1; glibc <math.h> patch)"; }

# ---------------------------------------------------------------------------
say "4/5  weights: ESM-1b/MSA-1b (reuse) + ESM-IF1 + regression"
# ---------------------------------------------------------------------------
mkdir -p "$WEIGHTS"
DRN_ESM="${ECSTASY_ROOT:-$HERE}/weights/drn_1d2d_inter/esm"
fetch() { local f="$1"
  [ -f "$WEIGHTS/$f" ] && return
  [ -f "$DRN_ESM/$f" ] && { ln -s "$DRN_ESM/$f" "$WEIGHTS/$f"; return; }   # reuse DRN copy
  curl -fL "https://dl.fbaipublicfiles.com/fair-esm/models/$f" -o "$WEIGHTS/$f"; }
fetch esm1b_t33_650M_UR50S.pt
fetch esm_msa1b_t12_100M_UR50S.pt
fetch esm_if1_gvp4_t16_142M_UR50.pt
cp -n "$PLMG/data/regression/"*contact-regression.pt "$WEIGHTS/" 2>/dev/null || true

# ---------------------------------------------------------------------------
say "5/5  trained models (7-member ensemble, Google Drive RAR)"
# ---------------------------------------------------------------------------
if [ ! -f "$PLMG/model/1" ]; then
  B="$(mktemp -d)"
  "$UV" tool run --from gdown gdown 1Y9eSlIJr-XDG5gREIEeGK4BW_Of0F_UQ -O "$B/model.rar"
  UNRAR="$(command -v unrar || true)"
  if [ -z "$UNRAR" ]; then
    curl -fsSL https://www.rarlab.com/rar/unrarsrc-6.2.12.tar.gz -o "$B/u.tgz"
    ( cd "$B" && tar xzf u.tgz && make -C unrar -j "$NCPUS" ); UNRAR="$B/unrar/unrar"
  fi
  "$UNRAR" x -y "$B/model.rar" "$PLMG/"; rm -rf "$B"
fi
say "verify"; "$PY" -c "import torch,esm,gvp,torch_cluster; import esm.inverse_folding; print('plmgraph OK; torch',torch.__version__)"
say "done -> $ENV_PATH"
