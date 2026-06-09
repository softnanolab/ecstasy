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


def main() -> None:
    import torch

    # PORT SHIM: PL load_from_checkpoint uses torch.load, which defaults
    # weights_only=True on torch>=2.6 and refuses the (trusted, local) PL checkpoint.
    _orig = torch.load
    torch.load = lambda *a, **k: (k.setdefault("weights_only", False), _orig(*a, **k))[1]

    bundle = json.loads(sys.stdin.read())
    entry_id = bundle["entry_id"]
    sequences = bundle["sequences"]
    chain_ids = bundle["chain_ids"]
    out_dir = Path(bundle["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = bundle.get("params") or {}

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

    from project.lit_model_predict import InputDataset
    from project.utils.deepinteract_modules import LitGINI
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
                        collate_fn=lambda x: x[0])

    model = LitGINI.load_from_checkpoint(cfg["ckpt"], map_location="cpu").eval()
    accel = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=accel, devices=1, logger=False,
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
