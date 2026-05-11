"""Re-run MINT validation on the mint_seqid30 val split, using ecstasy's
benchmark enumeration + metric, but loading MINT once (no per-entry subprocess).

Uses the 3khmvobe checkpoint by default:
    /projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/
        config.yaml            (Hydra-format MINT config)
        checkpoints/last.ckpt

For each val entry:
  1. enumerate via ecstasy.benchmarks.mint_seqid30.MintSeqid30Bench
  2. tokenize chains as <cls>{seq}<eos> per chain (matches the MINT runner)
  3. forward through MINT ESM2 + DualDistogramHead -> inter_logits
  4. softmax over bins, sum bins 0..(cutoff_bin-1) -> contact probability
  5. drop <cls>/<eos> token rows/cols
  6. score via ecstasy.metrics.contact.pak_inter_chain

Writes:
  analysis/mint_3khmvobe_rerun.json   summary + per_protein
  analysis/mint_3khmvobe_rerun.csv    per-entry table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# ecstasy imports (benchmark + metric)
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
from ecstasy.benchmarks.mint_seqid30 import MintSeqid30Bench  # noqa: E402
from ecstasy.metrics.contact import pak_inter_chain  # noqa: E402


def tokenize_chains(sequences, alphabet):
    chain_token_ids, chain_lengths = [], []
    for s in sequences:
        encoded = alphabet.encode(f"<cls>{s.replace('J', 'L')}<eos>")
        t = torch.tensor(encoded, dtype=torch.int64)
        chain_token_ids.append(t)
        chain_lengths.append(int(t.numel()))
    tokens = torch.cat(chain_token_ids, dim=0).unsqueeze(0)
    chain_ids = torch.cat(
        [torch.full((L,), i, dtype=torch.int64) for i, L in enumerate(chain_lengths)]
    ).unsqueeze(0)
    return tokens, chain_ids, chain_lengths


def residue_token_indices(chain_lengths):
    idx, pos = [], 0
    for L in chain_lengths:
        idx.extend(range(pos + 1, pos + L - 1))
        pos += L
    return torch.tensor(idx, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/config.yaml")
    ap.add_argument("--weights", default="/projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/checkpoints/last.ckpt")
    ap.add_argument("--cutoff_bin", type=int, default=5, help="bins 0..cutoff-1 are 'contact' (default 5 = <8Å)")
    ap.add_argument("--max_len", type=int, default=1024, help="skip entries with total token length > max_len (matches MINT eval)")
    ap.add_argument("--out_json", default=str(HERE / "mint_3khmvobe_rerun.json"))
    ap.add_argument("--out_csv", default=str(HERE / "mint_3khmvobe_rerun.csv"))
    args = ap.parse_args()

    from omegaconf import OmegaConf
    import mint
    from mint.data.esm import Alphabet

    mint_repo = Path(mint.__path__[0]).parent.parent
    if str(mint_repo) not in sys.path:
        sys.path.insert(0, str(mint_repo))
    from scripts.finetune.contact_prediction import ContactPrediction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mint-rerun] device={device}", flush=True)
    print(f"[mint-rerun] cfg={args.config}", flush=True)
    print(f"[mint-rerun] weights={args.weights}", flush=True)

    mint_cfg = OmegaConf.load(args.config)
    model = ContactPrediction(mint_cfg)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"[mint-rerun] {len(missing)} missing keys (first 3): {missing[:3]}", flush=True)
    if unexpected:
        print(f"[mint-rerun] {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}", flush=True)
    model.to(device)
    model.eval()

    alphabet = Alphabet.from_architecture("ESM-1b")
    bench = MintSeqid30Bench(data_root=Path("/projects/u6jv/boltz_benchmarking/DATA"))

    per_protein: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    rows = []

    t0 = time.time()
    entries = list(bench.entries())
    print(f"[mint-rerun] {len(entries)} val entries", flush=True)

    for i, entry in enumerate(entries):
        if i % 50 == 0:
            print(f"  [{i}/{len(entries)}]  elapsed={time.time()-t0:.0f}s  done={len(per_protein)}", flush=True)
        try:
            tokens, chain_ids_t, chain_lengths = tokenize_chains(entry.sequences, alphabet)
            if tokens.shape[1] > args.max_len:
                skipped.append((entry.id, f"len {tokens.shape[1]} > max_len {args.max_len}"))
                continue
            tokens = tokens.to(device)
            chain_ids_t = chain_ids_t.to(device)
            batch = SimpleNamespace(tokens=tokens, chain_ids=chain_ids_t)
            with torch.no_grad():
                _intra_logits, inter_logits = model.forward(batch)
            probs = torch.softmax(inter_logits[0].float(), dim=-1)
            contact_full = probs[..., : args.cutoff_bin].sum(-1).cpu()  # (T, T)
            keep = residue_token_indices(chain_lengths)
            contact_resi = contact_full[keep][:, keep].numpy().astype(np.float64)

            gt = bench.gt_for(entry.id)
            contact_gt = gt["contact_map"]
            seqs = gt["sequences"]
            if len(seqs) != 2:
                skipped.append((entry.id, "non-dimer in GT"))
                continue
            la, lb = len(seqs[0]), len(seqs[1])
            L = la + lb
            if contact_resi.shape[0] != L:
                errors.append((entry.id, f"shape mismatch: pred={contact_resi.shape}, L={L}"))
                continue
            chain_ids = np.array([0] * la + [1] * lb)
            scores = pak_inter_chain(contact_resi, contact_gt, chain_ids)
            per_protein[entry.id] = {k: float(v) for k, v in scores.items()}
            rows.append({"entry_id": entry.id, "L": L, **scores})
        except FileNotFoundError as e:
            skipped.append((entry.id, f"FileNotFoundError: {e}"))
        except Exception as e:  # noqa: BLE001
            errors.append((entry.id, f"{type(e).__name__}: {e}"))

    elapsed = time.time() - t0
    print(f"[mint-rerun] done in {elapsed:.1f}s. ok={len(per_protein)} skip={len(skipped)} err={len(errors)}", flush=True)

    # Aggregate
    keys = ["AUC", "P@K", "P@K/2", "P@K/5"]
    summary = {"n_evaluated": len(per_protein), "n_skipped": len(skipped), "n_errors": len(errors)}
    if per_protein:
        for k in keys:
            vals = np.array([v[k] for v in per_protein.values() if not np.isnan(v.get(k, np.nan))])
            if vals.size:
                summary[f"mean_{k}"] = float(vals.mean())
                summary[f"median_{k}"] = float(np.median(vals))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "metadata": {
            "config": args.config,
            "weights": args.weights,
            "cutoff_bin": args.cutoff_bin,
            "max_len": args.max_len,
            "elapsed_seconds": elapsed,
            "ecstasy_metric": "pak_inter_chain (matches mint.metrics_inter_chain)",
        },
        "summary": summary,
        "per_protein": per_protein,
        "skipped_first_20": skipped[:20],
        "errors_first_20": errors[:20],
    }, indent=2))
    print(f"[mint-rerun] wrote {out_path}", flush=True)

    if rows:
        import csv
        out_csv = Path(args.out_csv)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[mint-rerun] wrote {out_csv}", flush=True)

    # Print headline
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}" if not isinstance(v, float) else f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
