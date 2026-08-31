#!/usr/bin/env bash
# Dedicated venv + patched source tree for MiniFold
# (src/ecstasy/models/_runners/minifold_runner.py).
#
# Two things here are not optional and both cost real time when skipped:
#
# 1. MiniFold's published pyproject declares `packages = ["minifold"]`, so `pip install .`
#    installs ONLY the top-level package. `import minifold` then succeeds while
#    `minifold.utils`, `minifold.model` and `minifold.data` are all absent. The source
#    tree is the only complete copy, which is why the runner puts it first on sys.path
#    and asserts that `minifold` resolved there.
# 2. Upstream `FoldingTrunk.forward` hardcodes `residx = arange(L)`, so an injected
#    residue index never reaches the trunk. Without the patch below the chain break
#    degrades to linker-only — silently, producing a plausible but wrong baseline. The
#    runner refuses to start against an unpatched tree.
#
# Weights are fetched on the login node on purpose: compute nodes here have no outbound
# network, and the model load pulls ~8 GB (2.6 GB MiniFold + ~5.3 GB ESM2-3B via
# torch.hub). Fetch once, then run.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

# `.env` is gitignored, so a git worktree does not have one and every root would come
# back empty. A worktree's `.git` is a FILE holding `gitdir: <main>/.git/worktrees/<name>`,
# so the main checkout is three levels up from that. Mirrors ecstasy.config._env_path;
# the two must agree or the installer and the runner disagree about where things live.
ENV_FILE="${REPO_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" && -f "${REPO_ROOT}/.git" ]]; then
  _gitdir="$(sed -n 's/^gitdir: //p' "${REPO_ROOT}/.git")"
  [[ -n "${_gitdir}" ]] && ENV_FILE="$(cd "${_gitdir}/../../.." && pwd)/.env"
fi
_dotenv() { grep -E "^$1=" "${ENV_FILE}" 2>/dev/null | cut -d= -f2- | tr -d '[:space:]'; }

ENVS_ROOT="${ENVS_ROOT:-$(_dotenv ENVS_ROOT)}"
ECSTASY_ROOT="${ECSTASY_ROOT:-$(_dotenv ECSTASY_ROOT)}"
: "${ENVS_ROOT:?set ENVS_ROOT in the environment or .env}"
: "${ECSTASY_ROOT:?set ECSTASY_ROOT in the environment or .env}"

VENV="${ENVS_ROOT}/.venv-minifold"
SRC="${ECSTASY_ROOT}/src/minifold"
WEIGHTS="${ECSTASY_ROOT}/weights/minifold"
CACHE="${WEIGHTS}/cache"
CKPT="${WEIGHTS}/minifold_48L.ckpt"
CKPT_URL="https://huggingface.co/jwohlwend/minifold/resolve/main/minifold_48L_final.ckpt"
PATCH="${HERE}/minifold_residx.patch"

echo "==> Cloning MiniFold into ${SRC}"
mkdir -p "$(dirname "${SRC}")" "${CACHE}"
if [ ! -d "${SRC}/.git" ]; then
  git clone --depth 1 https://github.com/jwohlwend/minifold.git "${SRC}"
fi

echo "==> Applying the residx patch"
# -N makes reapplication a no-op, so the script is safe to rerun.
patch -p1 -N -d "${SRC}" < "${PATCH}" || true
grep -q "residx=None" "${SRC}/minifold/model/model.py" || {
  echo "FATAL: residx patch did not apply to ${SRC}/minifold/model/model.py" >&2
  exit 1
}

echo "==> Creating ${VENV}"
uv venv --python 3.12 "${VENV}"
VIRTUAL_ENV="${VENV}" uv pip install --python "${VENV}/bin/python" "${SRC}"
uv pip install --python "${VENV}/bin/python" numpy

echo "==> Fetching weights (login node — compute nodes have no outbound network)"
if [ ! -f "${CKPT}" ]; then
  curl -L --fail -o "${CKPT}" "${CKPT_URL}"
fi
ls -lh "${CKPT}"

echo "==> Verifying: patched source wins, residx reaches the trunk, ESM2 backbone cached"
# Deliberately does NOT construct the model. `load_model_and_alphabet` would build the
# 3B-parameter backbone in RAM (>10 GB) — fine on a compute node, antisocial on a login
# node, and liable to be killed there. The files are streamed to the hub cache instead;
# the first real run instantiates the model where it belongs.
MINIFOLD_SRC="${SRC}" TORCH_HUB_DIR="${CACHE}" "${VENV}/bin/python" - <<'PY'
import inspect
import os
import sys
from pathlib import Path

src = os.environ["MINIFOLD_SRC"]
sys.path.insert(0, src)

import torch

hub_dir = os.environ["TORCH_HUB_DIR"]
torch.hub.set_dir(hub_dir)

import minifold

tree = Path(minifold.__file__).resolve().parent.parent
assert tree == Path(src).resolve(), f"minifold resolved to {minifold.__file__}, not {src}"

# Subpackages the broken wheel omits — these resolve only from the source tree.
import minifold.data.config  # noqa: F401
import minifold.model.model as mm
import minifold.utils.protein  # noqa: F401

assert "residx" in inspect.signature(mm.FoldingTrunk.forward).parameters, \
    "FoldingTrunk.forward takes no residx — the patch is not applied"

# Stream the ESM2-3B backbone into the hub cache. These are the exact two files and
# destination fair-esm's `load_model_and_alphabet_hub` expects, so the first real run
# finds them cached and makes no network call.
ESM = "esm2_t36_3B_UR50D"
ckpt_dir = Path(hub_dir) / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
for url in (f"https://dl.fbaipublicfiles.com/fair-esm/models/{ESM}.pt",
            f"https://dl.fbaipublicfiles.com/fair-esm/regression/{ESM}-contact-regression.pt"):
    dest = ckpt_dir / url.rsplit("/", 1)[1]
    if dest.exists():
        print(f"  cached {dest.name} ({dest.stat().st_size / 1e9:.1f} GB)")
        continue
    print(f"  fetching {dest.name} ...")
    torch.hub.download_url_to_file(url, str(dest))

print("OK torch", torch.__version__, "| tree", tree, "| residx accepted")
PY

echo "Done: venv=${VENV} src=${SRC} ckpt=${CKPT}"
