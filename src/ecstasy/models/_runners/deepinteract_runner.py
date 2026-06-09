"""Self-contained DeepInteract runner — invoked via its env's python by the adapter.

DeepInteract (Morehead et al., ICLR 2022) is a DGL geometric-transformer that predicts
inter-chain contacts from two per-chain PDB **structures**. It builds residue graphs
with geometric + hhblits-MSA + DSSP/PSAIA features (DSSP/PSAIA are imputed when the
tools are absent; the hhblits MSA features need an HHsuite DB).

This runner wraps DeepInteract's own ``project.lit_model_predict`` pipeline:
  * per-chain PDBs are resolved from ``cfg["struct_dir"]/<entry_id>_<chain_id>.pdb``
    (ESMFold monomers for sequence-only splits);
  * the trained LightningModule is loaded from ``cfg["ckpt"]`` and run on the two
    chains; its softmax positive-class map (graph_1_len, graph_2_len) is the
    inter-chain contact-probability block, which we embed into the full (L, L) map.

PORT (torch>=2.6 / PL>=2.x / DGL>=2.1): torch.load is wrapped weights_only=False so the
PL checkpoint loads (the env install applies the DGL/torchmetrics/PL/biopython shims).
Reads a JSON bundle on stdin; writes <out_dir>/contact.npz. Outside the FLOPs scope.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def _embed_block(block: np.ndarray, lenA: int, lenB: int) -> np.ndarray:
    """Embed the (lenA, lenB) inter-chain block into the symmetric (L, L) map.

    DeepInteract truncates very long chains (RESIDUE_COUNT_LIMIT); if the returned
    block is smaller than (lenA, lenB), pad with zeros so the scorer's (L, L) layout
    is preserved (truncated tail pairs score as non-contacts)."""
    L = lenA + lenB
    probs = np.zeros((L, L), dtype=np.float32)
    ba, bb = block.shape
    probs[:ba, lenA:lenA + bb] = block
    probs[lenA:lenA + bb, :ba] = block.T
    return probs.astype(np.float16)


def _patch_atom3_for_aarch64() -> None:
    """Runtime workarounds for running DeepInteract's 2021 atom3 pipeline on
    Linux/aarch64 (it was developed on case-insensitive macOS with x86 PSAIA):

    * PSAIA is a QT4 GUI tool with no aarch64 build — make its call a no-op so the
      protrusion (CX) features fall back to DeepInteract's DEFAULT_MISSING_PROTRUSION
      imputation (the only feature set that can't be reproduced here).
    * atom3 writes parsed pkls under a lowercase group dir but reads them via
      ``get_pdb_code(...)[1:3].upper()`` — harmless on case-insensitive FS, a
      FileNotFoundError on Linux. Make read_pickle resolve the group-dir case.
    """
    import atom3.conservation as _cons

    def _psaia_stub(psaia_dir, config_file, file_list_file):
        # Write an empty-but-valid .tbl per PDB (a 'chain' header, no residue rows) so
        # postprocessing's rglob('<pdb_code>*.tbl')[struct_idx] resolves and the parser
        # returns an empty frame -> protrusion falls back to DEFAULT_MISSING_PROTRUSION.
        out_dir = os.path.dirname(file_list_file)
        for line in open(file_list_file):
            p = line.strip()
            if not p.endswith(".pdb"):
                continue
            name = os.path.splitext(os.path.basename(p))[0]
            with open(os.path.join(out_dir, name + ".tbl"), "w") as f:
                f.write("PSAIA unavailable on aarch64 (protrusion imputed)\nchain\n")
    _cons._psaia = _psaia_stub
    import pandas as _pd
    _orig = _pd.read_pickle

    def _ci(path, *a, **k):
        if isinstance(path, (str, os.PathLike)) and not os.path.exists(path):
            d, b = os.path.split(str(path))
            par, grp = os.path.split(d)
            for alt in (grp.lower(), grp.upper()):
                cand = os.path.join(par, alt, b)
                if os.path.exists(cand):
                    path = cand
                    break
        return _orig(path, *a, **k)
    _pd.read_pickle = _ci


def main() -> None:
    import torch

    # PORT SHIM: PL load_from_checkpoint passes weights_only=True explicitly on
    # torch>=2.6, which refuses the (trusted, local) PL checkpoint — FORCE it False.
    _orig = torch.load
    torch.load = lambda *a, **k: (k.update({"weights_only": False}), _orig(*a, **k))[1]

    bundle = json.loads(sys.stdin.read())
    entry_id = bundle["entry_id"]
    sequences = bundle["sequences"]
    chain_ids = bundle["chain_ids"]
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = bundle.get("params") or {}

    # atom3 uses tempfile.mkdtemp (TMPDIR); the orchestrator may pass a stale
    # node-local TMPDIR. Point it at a valid per-entry dir.
    if not os.path.isdir(os.environ.get("TMPDIR", "")):
        _t = out_dir / "tmp"; _t.mkdir(exist_ok=True); os.environ["TMPDIR"] = str(_t)

    if len(sequences) != 2:
        raise ValueError(f"DeepInteract handles exactly 2 chains, got {len(sequences)} for {entry_id}")

    struct_dir = Path(cfg["struct_dir"])
    pdbA = struct_dir / f"{entry_id}_{chain_ids[0]}.pdb"
    pdbB = struct_dir / f"{entry_id}_{chain_ids[1]}.pdb"
    for p in (pdbA, pdbB):
        if not p.exists():
            raise FileNotFoundError(f"DeepInteract needs a per-chain PDB at {p} "
                                    "(generate ESMFold monomers first)")

    di_root = Path(cfg.get("di_root") or os.environ.get("DI_ROOT")
                   or (Path(__file__).resolve().parents[4] / "modules" / "deepinteract"))
    sys.path.insert(0, str(di_root))
    os.chdir(di_root)  # DeepInteract resolves config/test paths relative to its repo root

    # hhblits (atom3 shells out to it for the MSA/profile features) must be on PATH.
    if cfg.get("hhsuite_bin"):
        os.environ["PATH"] = f"{Path(cfg['hhsuite_bin']).parent}:{os.environ.get('PATH','')}"
    _patch_atom3_for_aarch64()

    from project.lit_model_predict import InputDataset
    from project.utils.deepinteract_modules import LitGINI
    from project.utils.deepinteract_utils import dgl_picp_collate
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl

    work = out_dir / "di_work"
    work.mkdir(parents=True, exist_ok=True)
    ds = InputDataset(
        left_pdb_filepath=str(pdbA), right_pdb_filepath=str(pdbB),
        input_dataset_dir=str(work),
        psaia_dir=cfg.get("psaia_dir", str(di_root / "project" / "datasets" / "builder" / "psaia")),
        psaia_config=str(di_root / "project" / "datasets" / "builder" / "psaia_config_file_input.txt"),
        hhsuite_db=cfg["hhsuite_db"], knn=20, geo_nbrhd_size=2, self_loops=True, force_reload=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=dgl_picp_collate)

    model = LitGINI.load_from_checkpoint(cfg["ckpt"], map_location="cpu").eval()
    # DGL's aarch64 wheel is CPU-only ("Device API cuda is not enabled"), so the graphs
    # — and hence the model — must run on CPU. The model is tiny (~5M params) and
    # hhblits dominates runtime, so this is not the bottleneck.
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False,
                         enable_progress_bar=False, enable_checkpointing=False)
    payload = trainer.predict(model=model, dataloaders=loader)[0]

    logits = payload[0][0].squeeze()                       # (2, g1_len, g2_len)
    g1, g2 = logits.shape[1:]
    flat = torch.flatten(logits, start_dim=1).transpose(1, 0)
    block = torch.softmax(flat, dim=1)[:, 1].reshape(g1, g2).cpu().numpy()

    lenA, lenB = len(sequences[0]), len(sequences[1])
    probs = _embed_block(block, lenA, lenB)
    np.savez_compressed(out_dir / "contact.npz", probs=probs, length=np.int32(lenA + lenB))
    print(f"[deepinteract] WROTE {out_dir/'contact.npz'} block={block.shape} L={lenA+lenB}", flush=True)


if __name__ == "__main__":
    main()
