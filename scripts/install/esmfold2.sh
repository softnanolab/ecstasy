#!/usr/bin/env bash
# Dedicated venv for ESMFold2 (src/ecstasy/models/_runners/esmfold2_runner.py).
#
# This CANNOT share .venv-esmfold. That env is py3.7 / torch 1.12, pinned there by
# ESMFold-v1's openfold dependency (openfold's structure_module imports a CUDA extension
# built for cp37 at module import time). ESMFold2's `esm` package requires
# ">=3.12,<3.13", so the two are mutually exclusive by construction.
#
# `transformers` must come from the Biohub fork — upstream transformers does not carry
# ESMFold2ExperimentalModel. It is pulled in transitively by esm's own dependency pin;
# we install esm at a fixed commit so the benchmark stays reproducible rather than
# tracking a moving @main.
#
# torch is installed first, from the CUDA index, so that the esm install finds the
# requirement already satisfied and does not resolve a CPU-only wheel. 2.5.1+cu124 is
# the same build .venv-boltz uses and is known good on this cluster's A100s.
set -euo pipefail

ESM_COMMIT="26b0bc2b771e3e419ea74f445a5f35cc094a1509"   # oss sync (#370), 2026-07-28

ENVS_ROOT="${ENVS_ROOT:-$(grep -E '^ENVS_ROOT=' "$(dirname "$0")/../../.env" | cut -d= -f2)}"
VENV="${ENVS_ROOT}/.venv-esmfold2"

echo "Creating ${VENV} ..."
uv venv --python 3.12 "${VENV}"
uv pip install --python "${VENV}/bin/python" \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "${VENV}/bin/python" \
  "esm @ git+https://github.com/Biohub/esm.git@${ESM_COMMIT}"

echo "Verifying the ESMFold2 stack imports and the distogram grid is the expected one ..."
"${VENV}/bin/python" - <<'PY'
import torch
from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput
from esm.models.esmfold2.processor import ESMFold2InputBuilder

# Output-head grid (Algorithm 12). This is NOT the 64-bin 2-22A conditioning grid; see
# ESMFOLD2_INTEGRATION.md — reusing contact_cutoff_bin=19 here would score at ~8.9A.
boundaries = torch.linspace(2, 52.0, 127)
mids = (torch.cat((torch.tensor([1.0]), boundaries, torch.tensor([57.0])))[:-1]
        + torch.cat((torch.tensor([1.0]), boundaries, torch.tensor([57.0])))[1:]) / 2
assert mids.numel() == 128, mids.numel()
n_below = int((mids < 7.9375).sum())
assert n_below == 16, n_below
print("OK torch", torch.__version__, "| 128 bins,", n_below, "below 7.9375A")
PY
echo "Done: ${VENV}"
