"""Evaluate ONE MENTOS checkpoint on a dataset; report inter- AND intra-chain AUC/P@K.

Faithful to mentos scripts/evals/evaluate_from_wandb.py: contacts =
distogram_to_contacts(predicted_distogram), then metrics_inter_chain /
metrics_intra_chain (long-range, min_sep=24) against the raw GT distogram bins.
Native model config (a5sgd6ul: num_recycles=1). One checkpoint per invocation —
parallelize the sweep across checkpoints with a SLURM array; aggregate the JSONs after.

  python scripts/mentos_ckpt_sweep.py --ckpt <…/epoch=X-step=Y.ckpt> --dataset val_seq_pair --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="val_seq_pair")
    ap.add_argument("--run-id", default="a5sgd6ul")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from ecstasy.datasets import load_dataset
    from ecstasy.models._runners.mentos_runner import (
        _load_cfg, _residue_token_indices, _tokenize_chains)
    import mentos
    sys.path.insert(0, str(Path(mentos.__path__[0]).parent.parent))  # mentos repo for scripts.*
    from mentos.data.esm import Alphabet
    from mentos.dataclasses import ContactPredictionBatch
    from mentos.metrics.contact_prediction import (
        distogram_to_contacts, metrics_inter_chain, metrics_intra_chain)
    from scripts.evals.evaluate_from_wandb import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_cfg(None, args.run_id)                      # native num_recycles (=1)
    model = load_model(cfg, Path(args.ckpt), str(device))
    alphabet = Alphabet.from_architecture("ESM-1b")

    ds = load_dataset(args.dataset)
    gt_root = Path(os.environ["MENTOS_ROOT"]) / "pdb" / "processed" / "data"
    inter, intra = [], []      # per-protein (AUC, P@K)
    n = 0
    for e in ds.entries():
        if len(e.sequences) != 2:
            continue
        if args.limit and n >= args.limit:
            break
        tokens, chain_ids_t, chain_lengths = _tokenize_chains(e.sequences, alphabet)
        tokens, chain_ids_t = tokens.to(device), chain_ids_t.to(device)
        T = int(tokens.shape[1])
        batch = ContactPredictionBatch(
            ids=[e.id], tokens=tokens, chain_ids=chain_ids_t,
            true_contacts=torch.full((1, T, T), -1, dtype=torch.int64, device=device),
            seq_lengths=torch.tensor([T], dtype=torch.int64, device=device),
            is_homodimer=torch.zeros(1, dtype=torch.bool, device=device),
            residue_map=torch.full((1, T), -1, dtype=torch.long, device=device),
            distance_map=None)
        with torch.no_grad():
            out = model(batch, mask_inputs=False)
        keep = _residue_token_indices(chain_lengths).to(device)
        contacts = distogram_to_contacts(out.predicted_distogram)[0][keep][:, keep]  # (L, L)
        gt = torch.load(gt_root / e.id[:2] / f"{e.id}.pt",
                        weights_only=False, map_location=device).contact_map.to(device)  # (L,L) bins
        la, lb = len(e.sequences[0]), len(e.sequences[1])
        cid = torch.tensor([0] * la + [1] * lb, device=device).unsqueeze(0)  # (1, L)
        P, Tg = contacts.unsqueeze(0), gt.unsqueeze(0)
        try:                                       # inter asserts n_true>0; skip if none
            mi = metrics_inter_chain(P, Tg, cid)
            if mi:
                inter.append((float(mi["AUC"]), float(mi["P@K"])))
        except AssertionError:
            pass
        ma = metrics_intra_chain(P, Tg, cid)       # {} if no long-range intra contacts
        if ma:
            intra.append((float(ma["long_AUC"]), float(ma["long_P@K"])))
        n += 1

    digits = "".join(c for c in Path(args.ckpt).stem.split("step=")[-1] if c.isdigit())
    res = {
        "step": int(digits), "ckpt": str(args.ckpt), "n": n,
        "inter_AUC": float(np.mean([x[0] for x in inter])) if inter else None,
        "inter_P@K": float(np.mean([x[1] for x in inter])) if inter else None,
        "intra_AUC": float(np.mean([x[0] for x in intra])) if intra else None,
        "intra_P@K": float(np.mean([x[1] for x in intra])) if intra else None,
        "n_inter": len(inter), "n_intra": len(intra),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
