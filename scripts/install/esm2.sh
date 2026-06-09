#!/usr/bin/env bash
# Dedicated venv for the ESM2 contact-prediction model (src/ecstasy/models/_runners/esm2_runner.py).
#
# ESM2's supervised contact head needs only torch + fair-esm — NOT the openfold/esmfold
# structure stack — so this is a tiny plain `uv` venv, independent of .venv-esmfold.
#
# torch is pinned <2.6 on purpose: fair-esm loads its pretrained checkpoints with a bare
# `torch.load(..., map_location="cpu")` (no `weights_only=False`). torch>=2.6 flipped that
# default to True and refuses the argparse Namespace inside the esm checkpoints, breaking
# `esm.pretrained.esm2_*`. 2.5.1 keeps weights_only=False by default. cu124 wheels run fine
# on the cluster's newer driver (the mentos env already runs cu130 torch on the same nodes).
set -euo pipefail

ENVS_ROOT="${ENVS_ROOT:-$(grep -E '^ENVS_ROOT=' "$(dirname "$0")/../../.env" | cut -d= -f2)}"
VENV="${ENVS_ROOT}/.venv-esm2"

echo "Creating ${VENV} ..."
uv venv --python 3.12 "${VENV}"
uv pip install --python "${VENV}/bin/python" \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "${VENV}/bin/python" fair-esm numpy

echo "Verifying ESM2 contact head loads ..."
"${VENV}/bin/python" - <<'PY'
import torch, esm
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
bc = alphabet.get_batch_converter()
_, _, toks = bc([("c", "MKTAYIAKQR" + "G" * 25 + "GGSDFAERTQ")])
with torch.no_grad():
    c = model.eval()(toks, return_contacts=True)["contacts"]
assert c.shape == (1, 45, 45), c.shape
print("OK", torch.__version__, "contacts", tuple(c.shape))
PY
echo "Done: ${VENV}"
