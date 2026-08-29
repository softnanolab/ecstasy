"""Save MENTOS predicted contact-prob maps for given dimers across training checkpoints,
to visualize how the (distogram-derived) contact map evolves. step=0 => random-init head
(MENTOS(cfg) with pretrained ESM2 but untrained pair-stack/contact-head, no training ckpt).

  python distevo_infer.py --ids 8pdc,9uc5 --steps 0,20000,40000,60000,90000 --out-dir <dir>
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--dataset", default="val_seq_pair")
    ap.add_argument("--checkpoint", default="a5sgd6ul_s90k",
                    help="registry checkpoint name; its run_id + checkpoint directory are used "
                         "(the directory is swept by --steps). No paths are hardcoded.")
    ap.add_argument("--steps", default="0,20000,40000,60000,90000")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from ecstasy.datasets import load_dataset
    from ecstasy.models._runners.mentos_runner import (
        _load_cfg, _residue_token_indices, _tokenize_chains)
    import mentos
    sys.path.insert(0, str(Path(mentos.__path__[0]).parent.parent))
    from omegaconf import OmegaConf
    from mentos.data.esm import Alphabet
    from mentos.dataclasses import ContactPredictionBatch
    from mentos.metrics.contact_prediction import (
        distogram_to_contacts, metrics_inter_chain, contact_map_to_binary_labels)
    from scripts.pretrain.pretrain_mentos import MENTOS
    from scripts.evals.evaluate_from_wandb import load_model

    from ecstasy.registry import checkpoints
    ck = checkpoints.checkpoint(args.checkpoint)                   # registry name -> concrete dir/run_id
    run_id, ckpt_dir = ck["run_id"], str(Path(ck["abs_path"]).parent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_cfg(None, run_id)                            # native num_recycles (=1)
    alphabet = Alphabet.from_architecture("ESM-1b")
    ds = load_dataset(args.dataset)
    by_id = {e.id: e for e in ds.entries()}
    ids = [i.strip() for i in args.ids.split(",")]
    steps = [int(s) for s in args.steps.split(",")]
    gt_root = Path(os.environ["MENTOS_ROOT"]) / "pdb" / "processed" / "data"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    def ckpt_for(step):
        hits = glob.glob(f"{ckpt_dir}/epoch=*-step={step}.ckpt")
        return hits[0] if hits else None

    # GT (binary contacts + bins) once per protein
    for pid in ids:
        e = by_id[pid]
        gt = torch.load(gt_root / pid[:2] / f"{pid}.pt", weights_only=False).contact_map
        gtb = (contact_map_to_binary_labels(gt.unsqueeze(0)) > 0)[0].cpu().numpy()
        la, lb = len(e.sequences[0]), len(e.sequences[1])
        np.savez(out / f"{pid}_gt.npz", gt_binary=gtb, la=la, lb=lb,
                 homo=bool(e.sequences[0] == e.sequences[1]))
        print(f"GT {pid}: L={la+lb} ({la}+{lb}) homo={e.sequences[0]==e.sequences[1]} "
              f"n_inter_true={int(gtb[:la, la:].sum())}")

    for step in steps:
        torch.manual_seed(0)
        if step == 0:
            OmegaConf.set_struct(cfg, False)
            cfg.model.setdefault("mlm", {"mask_prob": 0.0}); cfg.model.loss.setdefault("mlm", 0)
            OmegaConf.set_struct(cfg, True)
            model = MENTOS(cfg).to(device).eval()
            print("step 0: random-init head (no training ckpt)")
        else:
            cp = ckpt_for(step); assert cp, f"no ckpt for step {step}"
            model = load_model(cfg, Path(cp), str(device))
            print(f"step {step}: {cp}")
        for pid in ids:
            e = by_id[pid]
            tokens, chain_ids_t, chain_lengths = _tokenize_chains(e.sequences, alphabet)
            tokens, chain_ids_t = tokens.to(device), chain_ids_t.to(device)
            T = int(tokens.shape[1])
            batch = ContactPredictionBatch(
                ids=[pid], tokens=tokens, chain_ids=chain_ids_t,
                true_contacts=torch.full((1, T, T), -1, dtype=torch.int64, device=device),
                seq_lengths=torch.tensor([T], dtype=torch.int64, device=device),
                is_homodimer=torch.zeros(1, dtype=torch.bool, device=device),
                residue_map=torch.full((1, T), -1, dtype=torch.long, device=device),
                distance_map=None)
            with torch.no_grad():
                o = model(batch, mask_inputs=False)
            keep = _residue_token_indices(chain_lengths).to(device)
            cprob = distogram_to_contacts(o.predicted_distogram)[0][keep][:, keep]          # (L,L) contact prob
            dbin = o.predicted_distogram[0].argmax(-1)[keep][:, keep]                        # (L,L) most-likely dist bin
            gt = torch.load(gt_root / pid[:2] / f"{pid}.pt", weights_only=False).contact_map.to(device)
            la, lb = len(e.sequences[0]), len(e.sequences[1])
            cid = torch.tensor([0] * la + [1] * lb, device=device).unsqueeze(0)
            pak = np.nan
            try:
                mi = metrics_inter_chain(cprob.unsqueeze(0), gt.unsqueeze(0), cid)
                if mi:
                    pak = float(mi["P@K"])
            except AssertionError:
                pass
            np.savez(out / f"{pid}_step{step}.npz", cprob=cprob.cpu().numpy(),
                     dbin=dbin.cpu().numpy().astype(np.int16), pak=pak, step=step)
            print(f"  {pid} step{step}: P@K={pak:.3f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("DISTEVO_DONE")


if __name__ == "__main__":
    main()
