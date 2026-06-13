#!/usr/bin/env bash
# Install the MENTOS scoring env at ${ENVS_ROOT}/.venv-mentos.
#
# The package is `mentos` (NOT `mint`): the GT .pt files pickle a
# `mentos.dataclasses.Sample`, and the model runner does `import mentos`, so the
# scoring env must ship `mentos` so torch.load resolves the class natively (no
# rename shim). The real package lives OUTSIDE this repo (the modules/mentos
# submodule is empty); point MENTOS_SRC at that checkout (default $HOME/mentos).
# Method: uv venv py3.12 + editable install, then re-pin torch from the cu126 index
# (this cluster's CUDA) + lightning.
set -euo pipefail
BLUE='\033[1;34m'; NC='\033[0m'
say() { echo -e "${BLUE}[mentos]${NC} $*"; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
ENV_PATH="${ENVS_ROOT:-$HERE/envs}/.venv-mentos"
# Real mentos source (editable). Override MENTOS_SRC for another host; falls back to
# the in-repo submodule if it has been populated.
MENTOS_SRC="${MENTOS_SRC:-$HOME/mentos}"
[ -f "$MENTOS_SRC/pyproject.toml" ] || MENTOS_SRC="$HERE/modules/mentos"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

[ -f "$MENTOS_SRC/pyproject.toml" ] || {
  echo "no mentos source: set MENTOS_SRC=<checkout> (or git submodule update --init modules/mentos)" >&2; exit 1; }

say "uv venv (python 3.12) -> $ENV_PATH"
"$UV" venv --python 3.12 "$ENV_PATH"
export VIRTUAL_ENV="$ENV_PATH"

say "editable install of mentos (+ deps) from $MENTOS_SRC"
( cd "$MENTOS_SRC" && "$UV" pip install -e . )

# The cu126 torch + lightning fix-up (Isambard note). Lightning FIRST: it pulls a
# default (PyPI/cu12x) torch as a transitive dep, so it must run before the cu126
# pin. Re-pin torch LAST from the cu126 index so the intended build wins and
# lightning can't clobber it.
say "force-reinstall lightning, then re-pin torch (cu126) last"
"$UV" pip install --force-reinstall lightning
"$UV" pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126

say "verify"
"$ENV_PATH/bin/python" -c "import mentos.dataclasses, torch, pandas, pyarrow; print('mentos ok; torch', torch.__version__)"
say "done -> $ENV_PATH"
