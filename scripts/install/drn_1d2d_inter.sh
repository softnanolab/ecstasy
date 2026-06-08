#!/usr/bin/env bash
# Install DRN-1D2D_Inter (Si & Yan, Brief. Bioinform. 2023) as an ecstasy model.
#
# DRN-1D2D_Inter is SEQUENCE-ONLY inter-protein contact prediction: it needs a
# per-chain MSA (UniRef90/100) for each of the two chains, NO monomer structure.
# Inference fuses ESM-1b + ESM-MSA-1b embeddings/attention, a paired-MSA CCMpred
# coupling map, alnstats statistics and an HHM PSSM through a 2D ResNet ensemble
# (7 weights). See modules/drn_1d2d_inter/predict.py for the reference flow that
# the ecstasy runner (_runners/drn_1d2d_inter_runner.py) reproduces.
#
# This script is idempotent: each stage is skipped if its artifact already exists.
# Heavy/long stages (ESM weights ~7.5 GB, trained models ~MBs via gdown, CCMpred
# build) are clearly delimited so they can be run/inspected individually.
#
# Layout produced (all under scratch via the envs/tools symlinks):
#   envs/drn_1d2d_inter/                     conda env (py3.8 + torch1.9 + fair-esm)
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

# torch1.9 + matching CUDA is a host decision. cu111 wheels are the documented
# DRN target; override TORCH_INDEX_URL for CPU-only or a different CUDA.
TORCH_PKG="${TORCH_PKG:-torch==1.9.1+cu111 torchvision==0.10.1+cu111}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/torch_stable.html}"

[ -d "$DRN/predict.py" ] || [ -f "$DRN/predict.py" ] || {
  warn "submodule modules/drn_1d2d_inter is empty — run: git submodule update --init modules/drn_1d2d_inter"; exit 1; }

# ---------------------------------------------------------------------------
say "1/6  python env  ($ENV_PATH)"
# ---------------------------------------------------------------------------
# Use conda if present, else micromamba (this cluster ships micromamba only).
if command -v conda >/dev/null 2>&1; then
  PM=conda; source "$(conda info --base)/etc/profile.d/conda.sh"
else
  PM=micromamba; eval "$("$(command -v micromamba)" shell hook -s posix)"
fi
say "    package manager: $PM"
if [ ! -d "$ENV_PATH" ]; then
  "$PM" create -p "$ENV_PATH" -c conda-forge python=3.8 -y
fi
"$PM" run -p "$ENV_PATH" pip install --no-cache-dir -f "$TORCH_INDEX_URL" $TORCH_PKG
"$PM" run -p "$ENV_PATH" pip install --no-cache-dir "fair-esm==2.0.0" biopython numpy

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
  # CCMpred's CLI needs terminfo (setupterm). On a spack cluster, LIBRARY_PATH/
  # CMAKE_PREFIX_PATH inject an *incompatible* ncurses, so pin the SYSTEM ncurses
  # (/usr/lib64, self-consistent) and strip the spack lib env for the build.
  cmake_flags=(-DCMAKE_BUILD_TYPE=Release -DCURSES_NEED_NCURSES=TRUE
    -DCURSES_LIBRARY=/usr/lib64/libncurses.so.6 -DCURSES_INCLUDE_PATH=/usr/include
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
say "4/6  alnstats + fasta2aln"
# ---------------------------------------------------------------------------
# alnstats: build from PSIPRED metapsicov src. fasta2aln: prebuilt script from
# kad-ecoli/hhsuite2 (a perl/awk converter; chmod +x).
mkdir -p "$TOOLS/metapsicov/bin"
if [ ! -x "$TOOLS/metapsicov/bin/alnstats" ]; then
  B="$(mktemp -d)"; trap 'rm -rf "$B"' EXIT
  git clone --depth 1 https://github.com/psipred/metapsicov.git "$B/metapsicov"
  "${CC:-gcc}" -O3 -o "$TOOLS/metapsicov/bin/alnstats" "$B/metapsicov/src/alnstats.c" -lm
  trap - EXIT; rm -rf "$B"
fi
if [ ! -x "$TOOLS/metapsicov/bin/fasta2aln" ]; then
  curl -fsSL https://raw.githubusercontent.com/kad-ecoli/hhsuite2/master/bin/fasta2aln \
    -o "$TOOLS/metapsicov/bin/fasta2aln"
  chmod +x "$TOOLS/metapsicov/bin/fasta2aln"
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
# NB: despite the .zip name the Drive blob is a RAR archive, so extract with 7z.
if [ ! -f "$DRN/model/1" ]; then
  "$ENV_PATH/bin/python" -m pip install --no-cache-dir gdown
  "$ENV_PATH/bin/python" -m gdown 1ICqJSNc01E2cGYhVj1IxzIkmnS-FMT2C -O "$DRN/model.zip"
  SZ="$(command -v 7z || true)"
  if [ -z "$SZ" ]; then
    "$PM" create -p "$TOOLS/p7zip-env" -c conda-forge p7zip -y
    SZ="$TOOLS/p7zip-env/bin/7z"
  fi
  ( cd "$DRN" && "$SZ" x -y model.zip && rm -f model.zip )
else
  say "    trained models present — skip"
fi

say "done. verify:  conda run -p $ENV_PATH python -c 'import torch,esm; print(torch.__version__)'"
say "registry env path -> envs/drn_1d2d_inter ; presets carry the tool/weight paths above."
