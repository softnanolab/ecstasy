#!/usr/bin/env bash
# Install DeepInteract (Morehead et al., ICLR 2022 — Geometric Transformers for
# Protein Interface Contact Prediction) as an ecstasy model.
#
# PORT NOTE (Isambard / aarch64): upstream targets py3.8 + torch1.7 + DGL 0.6 (x86),
# pytorch-lightning 1.4, torchmetrics 0.5, biopython 1.78. None of that exists for
# aarch64 + Hopper, so this builds on the project's current stack (py3.12 + torch2.6
# /cu126 + DGL 2.1) and applies a set of forward-compat shims so the 2021 code loads.
# The runner additionally carries a torch.load(weights_only=False) shim.
#
# Inputs at predict time: two per-chain PDBs + a trained ckpt + an HHsuite DB (for
# hhblits MSA/profile features). DSSP (RSA) and PSAIA (protrusion) features are
# imputed when those tools are absent (see the runner), so they are optional.
set -euo pipefail
BLUE='\033[1;34m'; NC='\033[0m'; say() { echo -e "${BLUE}[deepinteract]${NC} $*"; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
ENV_PATH="${ENVS_ROOT:-$HERE/envs}/deepinteract"
WEIGHTS="${ECSTASY_ROOT:-$HERE}/weights/deepinteract"
DI="$HERE/modules/deepinteract"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
PY="$ENV_PATH/bin/python"

[ -f "$DI/project/lit_model_predict.py" ] || {
  echo "init submodule: git submodule update --init modules/deepinteract" >&2; exit 1; }

# ---------------------------------------------------------------------------
say "1/4  venv + torch 2.6 / torchvision 0.21 (matched pair; cu126 aarch64)"
# ---------------------------------------------------------------------------
[ -x "$PY" ] || "$UV" venv --python 3.12 "$ENV_PATH"
# torch>2.6 outran torchvision's matching wheel (broke torchvision::nms, needed by the
# model's DeepLabV3Plus head) — pin the matched pair.
"$UV" pip install --python "$PY" "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu126

# ---------------------------------------------------------------------------
say "2/4  DGL 2.1 (aarch64) + lightning + feature/runtime deps"
# ---------------------------------------------------------------------------
# DGL 2.1 is the only aarch64 build; its graphbolt needs torchdata.datapipes (dropped
# in torchdata>=0.8) and pydantic, and ships a graphbolt C++ lib only up to torch 2.2 —
# DeepInteract never uses graphbolt, so pin torchdata<0.8 + add pydantic and make the
# graphbolt loader non-fatal (below).
"$UV" pip install --python "$PY" dgl "torchdata<0.8" pydantic "numpy<2"
# torchmetrics: 0.10.3 still accepts the legacy num_classes API (0.11+ requires task=);
# F1 was renamed F1Score — aliased below.
"$UV" pip install --python "$PY" pytorch-lightning "torchmetrics==0.10.3" \
  biopandas dill click networkx tqdm einops scipy pandas scikit-learn wandb timm fairscale biopython h5py
"$UV" pip install --python "$PY" --no-deps atom3-py3
"$UV" pip install --python "$PY" easy-parallel-py3 || true

# ---------------------------------------------------------------------------
say "3/4  forward-compat shims (2021 code on a 2.x stack)"
# ---------------------------------------------------------------------------
SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
# (a) DGL graphbolt: make the C++-lib load non-fatal (lib is built for torch<=2.2).
GB="$SP/dgl/graphbolt/__init__.py"
"$PY" - "$GB" <<'PYEOF'
import sys; p=sys.argv[1]; s=open(p).read()
if "try:\n    load_graphbolt()" not in s:
    s=s.replace("\nload_graphbolt()\n","\ntry:\n    load_graphbolt()\nexcept Exception:\n    pass\n")
    open(p,'w').write(s)
PYEOF
# (b) torchmetrics: F1 -> F1Score alias.
grep -q "^F1 = F1Score" "$SP/torchmetrics/__init__.py" || printf '\nF1 = F1Score  # compat alias\n' >> "$SP/torchmetrics/__init__.py"
# (c) pytorch-lightning: re-export types removed in PL>=2.0.
TYPES="$SP/pytorch_lightning/utilities/types.py"
grep -q "^EPOCH_OUTPUT" "$TYPES" || printf '\nfrom typing import Any as _Any, Dict as _Dict, List as _List\nEPOCH_OUTPUT = _List[_Dict[str, _Any]]\nSTEP_OUTPUT = _Any\n' >> "$TYPES"
# (d) biopython: re-expose protein_letters_3to1 (moved) + stub the removed Blast.Applications.
grep -q "compat re-export" "$SP/Bio/SCOP/Raf.py" || printf '\nfrom Bio.Data.PDBData import protein_letters_3to1  # compat re-export\n' >> "$SP/Bio/SCOP/Raf.py"
cat > "$SP/Bio/Blast/Applications.py" <<'PYEOF'
"""Stub for Bio.Blast.Applications (removed in BioPython>=1.80). DeepInteract
inference never runs BLAST; any real use raises a clear error."""
class _Stub:
    def __init__(self,*a,**k): pass
    def __call__(self,*a,**k):
        raise RuntimeError("Bio.Blast.Applications stubbed; not needed for inference.")
class NcbiblastpCommandline(_Stub): pass
class NcbipsiblastCommandline(_Stub): pass
def __getattr__(name): return _Stub
PYEOF

# ---------------------------------------------------------------------------
say "4/4  trained checkpoint + verify"
# ---------------------------------------------------------------------------
mkdir -p "$WEIGHTS"
[ -f "$WEIGHTS/LitGINI-GeoTran-DilResNet.ckpt" ] || \
  curl -fL https://zenodo.org/record/6671582/files/LitGINI-GeoTran-DilResNet.ckpt -o "$WEIGHTS/LitGINI-GeoTran-DilResNet.ckpt"
say "NOTE: hhblits DB (uniclust30, ~86GB) for the real eval goes under $WEIGHTS/hhsuite_db (see runner)."
PYTHONPATH="$DI" "$PY" - <<PYEOF
import warnings; warnings.filterwarnings('ignore')
import torch; _o=torch.load; torch.load=lambda *a,**k:(k.update(weights_only=False) or _o(*a,**k))
from project.utils.deepinteract_modules import LitGINI
m=LitGINI.load_from_checkpoint("$WEIGHTS/LitGINI-GeoTran-DilResNet.ckpt", map_location='cpu'); m.eval()
print("deepinteract OK — model loads:", round(sum(p.numel() for p in m.parameters())/1e6,2), "M params")
PYEOF
say "done -> $ENV_PATH"
