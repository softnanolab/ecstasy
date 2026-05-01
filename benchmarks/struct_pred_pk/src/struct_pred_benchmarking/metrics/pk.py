"""Per-protein P@K via MINT's own metric code.

For each protein we:
- load the (T,T) ground-truth distogram bin labels from `ground_truth/<id>.npz`
- load the (T,T) Boltz-derived contact probability from `boltz_contacts/<id>.npz`
- pass them through `mint.metrics.contact_prediction.metrics_inter_chain` and
  `granular_metrics_intra_chain(min_sep=cfg.intra_min_sep)`

Both functions expect (B, L, L) so we add a leading batch dim of size 1.
We then aggregate all per-protein rows into `metrics/results.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from mint.metrics.contact_prediction import (
    granular_metrics_intra_chain,
    metrics_inter_chain,
)

from struct_pred_benchmarking.config import BenchmarkConfig
from struct_pred_benchmarking.data.select_val import load_manifest


_CSV_COLUMNS = [
    "pdb_id",
    "total_len",
    "n_samples",
    "P_K_inter",
    "P_K2_inter",
    "P_K5_inter",
    "AUC_inter",
    "P_K_intra_LR",
    "P_K2_intra_LR",
    "P_K5_intra_LR",
    "AUC_intra_LR",
    "P_L_intra_LR",
]


def _load_pair(cfg: BenchmarkConfig, pdb_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    gt = np.load(cfg.run_dir / "ground_truth" / f"{pdb_id}.npz", allow_pickle=True)
    pred = np.load(cfg.run_dir / "boltz_contacts" / f"{pdb_id}.npz")

    targets = torch.as_tensor(gt["contact_map"], dtype=torch.long)  # (T, T) in {-1,0..9}
    chain_ids = torch.as_tensor(gt["chain_ids"], dtype=torch.long)  # (T,)
    probs = torch.as_tensor(pred["probs"], dtype=torch.float32)  # (T, T)
    n_samples = int(pred["n_samples"]) if "n_samples" in pred.files else cfg.diffusion_samples

    if probs.shape != targets.shape or probs.shape[0] != chain_ids.shape[0]:
        raise ValueError(
            f"{pdb_id}: shape mismatch — probs {probs.shape}, "
            f"targets {targets.shape}, chain_ids {chain_ids.shape}"
        )

    return probs.unsqueeze(0), targets.unsqueeze(0), chain_ids.unsqueeze(0), n_samples


def _scalar(d: dict, key: str) -> float:
    if key not in d:
        return float("nan")
    v = d[key]
    if isinstance(v, torch.Tensor):
        return float(v.item())
    return float(v)


def score_one(cfg: BenchmarkConfig, entry: dict) -> dict:
    probs, targets, chain_ids, n_samples = _load_pair(cfg, entry["id"])

    inter = metrics_inter_chain(probs, targets, chain_ids)
    intra = granular_metrics_intra_chain(probs, targets, chain_ids, min_sep=cfg.intra_min_sep) or {}

    return {
        "pdb_id": entry["id"],
        "total_len": entry["total_sequence_length"],
        "n_samples": n_samples,
        "P_K_inter": _scalar(inter, "P@K"),
        "P_K2_inter": _scalar(inter, "P@K/2"),
        "P_K5_inter": _scalar(inter, "P@K/5"),
        "AUC_inter": _scalar(inter, "AUC"),
        "P_K_intra_LR": _scalar(intra, "P@K"),
        "P_K2_intra_LR": _scalar(intra, "P@K/2"),
        "P_K5_intra_LR": _scalar(intra, "P@K/5"),
        "AUC_intra_LR": _scalar(intra, "AUC"),
        "P_L_intra_LR": _scalar(intra, "P@L"),
    }


def score_all(cfg: BenchmarkConfig) -> Path:
    rows = [score_one(cfg, e) for e in load_manifest(cfg)]
    out_path = cfg.run_dir / "metrics" / "results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path
