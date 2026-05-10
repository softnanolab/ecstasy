#!/usr/bin/env bash
# Install the lightweight ecstasy orchestrator venv at ./envs/.venv-ecstasy.
#
# This venv hosts the `ecstasy` CLI, the benchmark/pipeline/test code, and
# pytest. It contains no model deps (no torch, no esm, no jax). At inference
# time the CLI spawns model runners via `<model-venv>/bin/python` using the
# `env_path` set in each adapter's config.
#
# Reproducible from a fresh aarch64 GH200 login node.

set -euo pipefail

BLUE='\033[1;34m'
NC='\033[0m'

ECSTASY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PATH="${ECSTASY_ROOT}/envs/.venv-ecstasy"
BASE_PYTHON="${BASE_PYTHON:-$HOME/miniforge3/envs/mmseqs2-env/bin/python3.12}"

echo -e "${BLUE}ecstasy install${NC}"
echo "  ECSTASY_ROOT = ${ECSTASY_ROOT}"
echo "  ENV_PATH     = ${ENV_PATH}"
echo "  BASE_PYTHON  = ${BASE_PYTHON}"

if [ ! -x "${BASE_PYTHON}" ]; then
  echo "ERROR: BASE_PYTHON not found at ${BASE_PYTHON}" >&2
  echo "  Set BASE_PYTHON=<path-to-python3.12> and re-run." >&2
  exit 1
fi

mkdir -p "${ECSTASY_ROOT}/envs"
if [ -d "${ENV_PATH}" ]; then
  echo -e "${BLUE}venv already exists at ${ENV_PATH} — reusing${NC}"
else
  echo -e "${BLUE}creating venv${NC}"
  "${BASE_PYTHON}" -m venv "${ENV_PATH}"
fi

"${ENV_PATH}/bin/python" -m pip install --upgrade pip
"${ENV_PATH}/bin/python" -m pip install -e "${ECSTASY_ROOT}"

echo -e "${BLUE}verifying${NC}"
"${ENV_PATH}/bin/python" -c "import ecstasy, fire, yaml, pytest; print('ecstasy', ecstasy.__file__)"
"${ENV_PATH}/bin/ecstasy" bench list

echo -e "${BLUE}done${NC}"
echo "  activate with:  source ${ENV_PATH}/bin/activate"
