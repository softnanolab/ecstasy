#!/usr/bin/env bash
# Install DRN-1D2D_Inter (Si & Yan, Brief. Bioinform. 2023) as an ecstasy model.
#
# DRN-1D2D_Inter is SEQUENCE-ONLY inter-protein contact prediction: it needs a
# per-chain MSA for each of the two chains (ecstasy feeds it the Boltz per-chain
# CSVs, msa=boltz_csv), NO monomer structure. Inference fuses ESM-1b + ESM-MSA-1b
# embeddings/attention, a paired-MSA CCMpred coupling map, alnstats statistics and
# an HHM PSSM through a 2D ResNet ensemble (7 weights). See predict.py for the
# reference flow that the ecstasy runner (_runners/drn_1d2d_inter_runner.py) mirrors.
#
# PORT NOTE (Isambard / aarch64): upstream DRN targets py3.8 + torch1.9+cu111 (x86).
# That stack does not exist for aarch64 + Hopper, so this script builds a uv venv on
# the project's current torch (py3.12 + torch2.x/cu126, matching .venv-boltz) and the
# runner carries a torch.load(weights_only=False) shim so fair-esm's old checkpoints
# still load under torch>=2.6. Override TORCH_INDEX_URL / TORCH_PKG for another host.
#
# This script is idempotent: each stage is skipped if its artifact already exists.
# Heavy/long stages (ESM weights ~7.5 GB, trained models ~181 MB via gdown, CCMpred
# build) are clearly delimited so they can be run/inspected individually.
#
# Layout produced (all under scratch via the envs/tools symlinks):
#   envs/drn_1d2d_inter/                     uv venv (py3.12 + torch2.x + fair-esm)
#   tools/{ccmpred,metapsicov,hhsuite}/      external binaries
#   weights/drn_1d2d_inter/esm/              ESM-1b + ESM-MSA-1b + regression .pt
#   modules/drn_1d2d_inter/model/1..7        trained DRN ResNet weights (gdown)
set -euo pipefail

BLUE='\033[1;34m'; YEL='\033[1;33m'; NC='\033[0m'
say() { echo -e "${BLUE}[drn]${NC} $*"; }
warn() { echo -e "${YEL}[drn] $*${NC}"; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"   # repo root
# Honor .env's absolute scratch roots (ENVS_ROOT/TOOLS_ROOT/ECSTASY_ROOT) so the
# env/tools/weights land on scratch regardless of repo-root symlinks. This MUST
# match the registry's ${ENVS_ROOT}/${TOOLS_ROOT}/${ECSTASY_ROOT} resolution.
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
ENV_PATH="${ENVS_ROOT:-$HERE/envs}/drn_1d2d_inter"
TOOLS="${TOOLS_ROOT:-$HERE/tools}"
WEIGHTS="${ECSTASY_ROOT:-$HERE}/weights/drn_1d2d_inter"
DRN="$HERE/modules/drn_1d2d_inter"
NCPUS="${NCPUS:-$(nproc 2>/dev/null || echo 4)}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

# torch + matching CUDA is a host decision. cu126 aarch64 wheels match this project's
# other envs (.venv-boltz); override for a different CUDA / CPU-only / x86 host.
PYVER="${DRN_PYVER:-3.12}"
TORCH_PKG="${TORCH_PKG:-torch}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

[ -f "$DRN/predict.py" ] || {
  warn "submodule modules/drn_1d2d_inter is empty — run: git submodule update --init modules/drn_1d2d_inter"; exit 1; }

# ---------------------------------------------------------------------------
say "1/6  python env  ($ENV_PATH)  [uv, py$PYVER]"
# ---------------------------------------------------------------------------
# uv venv + the project's current torch (NOT upstream's py3.8/torch1.9 — see PORT NOTE).
# fair-esm is pure-python (works on torch2.x); numpy<2 avoids the removed np.float aliases
# fair-esm 2.0.0 still references.
if [ ! -x "$ENV_PATH/bin/python" ]; then
  "$UV" venv --python "$PYVER" "$ENV_PATH"
fi
export VIRTUAL_ENV="$ENV_PATH"
"$UV" pip install --python "$ENV_PATH/bin/python" $TORCH_PKG --index-url "$TORCH_INDEX_URL"
"$UV" pip install --python "$ENV_PATH/bin/python" "fair-esm==2.0.0" biopython "numpy<2"

# ---------------------------------------------------------------------------
say "2/6  hh-suite (hhmake + hhfilter)"
# ---------------------------------------------------------------------------
if [ ! -x "$TOOLS/hhsuite/bin/hhmake" ]; then
  bash "$HERE/scripts/install/hhsuite.sh"
else
  say "    hhsuite present — skip"
fi

# ---------------------------------------------------------------------------
say "3/6  CCMpred (paired-MSA coevolution coupling map)"
# ---------------------------------------------------------------------------
# CPU build by default (-DWITH_CUDA=OFF). For the GPU build set CCMPRED_CUDA=1;
# it needs a CUDA toolkit visible to cmake.
if [ ! -x "$TOOLS/ccmpred/bin/ccmpred" ]; then
  B="$(mktemp -d)"; trap 'rm -rf "$B"' EXIT
  git clone --depth 1 --recursive https://github.com/soedinglab/CCMpred.git "$B/CCMpred"
  # glibc>=2.26 declares ISO narrowing math fns (fsqrt/fadd/fmul/...) in <math.h>;
  # CCMpred's `#define fsqrt sqrtf` then rewrites glibc's own `fsqrt` decl into a
  # conflicting `sqrtf(double)`. Pull <math.h> in BEFORE the macro so those names are
  # already bound (else: "conflicting types for 'sqrtf'" on any modern-glibc host).
  CG_H="$B/CCMpred/lib/libconjugrad/include/conjugrad.h"
  grep -q '<math.h>' "$CG_H" || perl -0pi -e 's/(#define __LIBCONJ_H__\n)/$1\n#include <math.h>\n/' "$CG_H"
  # CCMpred's CLI needs terminfo (setupterm). On a spack cluster, LIBRARY_PATH/
  # CMAKE_PREFIX_PATH inject an *incompatible* ncurses, so pin the SYSTEM ncurses
  # (self-consistent) and strip the spack lib env for the build. Resolve whichever
  # libncurses soname exists (.so.6 on glibc hosts, bare .so here).
  NCURSES_LIB="$(ls /usr/lib64/libncurses.so.6 /usr/lib64/libncurses.so 2>/dev/null | head -1)"
  cmake_flags=(-DCMAKE_BUILD_TYPE=Release -DCURSES_NEED_NCURSES=TRUE
    -DCURSES_LIBRARY="$NCURSES_LIB" -DCURSES_INCLUDE_PATH=/usr/include
    -DCMAKE_EXE_LINKER_FLAGS=-ltinfo)
  [ "${CCMPRED_CUDA:-0}" = "1" ] || cmake_flags+=(-DWITH_CUDA=OFF)
  env -u LIBRARY_PATH -u CMAKE_PREFIX_PATH -u CPATH -u LD_LIBRARY_PATH \
    cmake -S "$B/CCMpred" -B "$B/CCMpred/build" "${cmake_flags[@]}"
  env -u LIBRARY_PATH -u LD_LIBRARY_PATH make -C "$B/CCMpred/build" -j "$NCPUS"
  mkdir -p "$TOOLS/ccmpred/bin"
  cp "$B/CCMpred/build/bin/ccmpred" "$TOOLS/ccmpred/bin/"
  trap - EXIT; rm -rf "$B"
else
  say "    ccmpred present — skip"
fi

# ---------------------------------------------------------------------------
say "4/6  alnstats"
# ---------------------------------------------------------------------------
# alnstats: build from PSIPRED metapsicov src. (The a3m->aln conversion upstream
# does with the x86-only fasta2aln binary is done in-runner now — _a3m_to_aln —
# so nothing to fetch for it.)
mkdir -p "$TOOLS/metapsicov/bin"
if [ ! -x "$TOOLS/metapsicov/bin/alnstats" ]; then
  B="$(mktemp -d)"; trap 'rm -rf "$B"' EXIT
  git clone --depth 1 https://github.com/psipred/metapsicov.git "$B/metapsicov"
  "${CC:-gcc}" -O3 -o "$TOOLS/metapsicov/bin/alnstats" "$B/metapsicov/src/alnstats.c" -lm
  trap - EXIT; rm -rf "$B"
fi

# ---------------------------------------------------------------------------
say "5/6  ESM-1b + ESM-MSA-1b weights + regression params"
# ---------------------------------------------------------------------------
mkdir -p "$WEIGHTS/esm"
fetch() { [ -f "$WEIGHTS/esm/$1" ] || curl -fL "https://dl.fbaipublicfiles.com/fair-esm/models/$1" -o "$WEIGHTS/esm/$1"; }
fetch esm1b_t33_650M_UR50S.pt          # ~7.0 GB
fetch esm_msa1b_t12_100M_UR50S.pt      # ~0.4 GB
# DRN ships the contact-regression params; ESM expects them alongside the weights.
cp -n "$DRN/data/regression/esm1b_t33_650M_UR50S-contact-regression.pt"     "$WEIGHTS/esm/"
cp -n "$DRN/data/regression/esm_msa1b_t12_100M_UR50S-contact-regression.pt" "$WEIGHTS/esm/"

# ---------------------------------------------------------------------------
say "6/6  trained DRN models (7-member ResNet ensemble, Google Drive)"
# ---------------------------------------------------------------------------
# https://drive.google.com/file/d/1ICqJSNc01E2cGYhVj1IxzIkmnS-FMT2C/view -> model/{1..7}
# The Drive blob is a RAR v5 archive (despite the .zip name). No system extractor on
# aarch64, so build unrar from rarlab source (tiny, no deps) and extract with it.
if [ ! -f "$DRN/model/1" ]; then
  B="$(mktemp -d)"; trap 'rm -rf "$B"' EXIT
  "$UV" tool run --from gdown gdown 1ICqJSNc01E2cGYhVj1IxzIkmnS-FMT2C -O "$B/model.rar"
  UNRAR="$(command -v unrar || true)"
  if [ -z "$UNRAR" ]; then
    curl -fsSL "${UNRAR_SRC_URL:-https://www.rarlab.com/rar/unrarsrc-6.2.12.tar.gz}" -o "$B/unrarsrc.tar.gz"
    ( cd "$B" && tar xzf unrarsrc.tar.gz && make -C unrar -j "$NCPUS" )
    UNRAR="$B/unrar/unrar"
  fi
  "$UNRAR" x -y "$B/model.rar" "$DRN/"      # archive already nests model/{1..7}
  trap - EXIT; rm -rf "$B"
else
  say "    trained models present — skip"
fi

say "done. verify:  $ENV_PATH/bin/python -c 'import torch,esm; print(torch.__version__)'"
say "registry env path -> envs/drn_1d2d_inter ; presets carry the tool/weight paths above."
