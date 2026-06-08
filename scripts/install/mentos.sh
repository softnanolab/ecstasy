#!/usr/bin/env bash
# Install the MENTOS (package name: `mint`) env at ${ENVS_ROOT}/.venv-mentos.
# Hosts the mentos model runner AND is the env that can load MENTOS ground-truth
# .pt files for scoring (they pickle torch tensors + the mint Sample dataclass).
# Method per modules/mentos/README.md: uv venv py3.12 + editable install, then
# force-reinstall torch from the cu126 index (this cluster's CUDA) + lightning.
set -euo pipefail
BLUE='\033[1;34m'; NC='\033[0m'
say() { echo -e "${BLUE}[mentos]${NC} $*"; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
ENV_PATH="${ENVS_ROOT:-$HERE/envs}/.venv-mentos"
MENTOS="$HERE/modules/mentos"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

[ -f "$MENTOS/pyproject.toml" ] || { echo "init submodule: git submodule update --init modules/mentos" >&2; exit 1; }

say "uv venv (python 3.12) -> $ENV_PATH"
"$UV" venv --python 3.12 "$ENV_PATH"
export VIRTUAL_ENV="$ENV_PATH"

say "editable install of mint (+ deps)"
( cd "$MENTOS" && "$UV" pip install -e . )

# The cu126 torch + lightning fix-up (modules/mentos/README.md, Isambard note).
# Lightning FIRST: it pulls a default (PyPI/cu12x) torch as a transitive dep, so it
# must run before the cu126 pin. Re-pin torch LAST from the cu126 index so the
# intended build wins and lightning can't clobber it.
say "force-reinstall lightning, then re-pin torch (cu126) last"
"$UV" pip install --force-reinstall lightning
"$UV" pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126

say "verify"
"$ENV_PATH/bin/python" -c "import mint, torch, pandas, pyarrow; print('mint ok; torch', torch.__version__)"
say "done -> $ENV_PATH"
