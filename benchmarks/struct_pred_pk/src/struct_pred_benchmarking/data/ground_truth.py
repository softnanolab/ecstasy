"""Extract per-protein ground-truth distogram bin labels from MINT processed .pt files.

Saves one .npz per protein with:
- contact_map: (T, T) int64 distogram bin labels in {-1, 0..9} (matches mint.utils.compute_distance_bins)
- chain_ids:   (T,)   int64 with values 0 / 1 for the two chains
- sequences:   tuple[str, str] (chain A, chain B) for sanity-checking downstream

T = N_A + N_B, residue level only (no ESM specials).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mint.dataclasses import Sample

from struct_pred_benchmarking.config import BenchmarkConfig
from struct_pred_benchmarking.data.select_val import load_manifest


def _load_sample(mint_data_root: Path, relative_path: str) -> Sample:
    pt_path = mint_data_root / "pdb" / "processed" / relative_path
    if not pt_path.exists():
        # `relative_path` in the parquet might already be relative to processed/
        pt_path = mint_data_root / relative_path
    if not pt_path.exists():
        raise FileNotFoundError(f"Sample .pt not found for {relative_path}")
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, Sample):
        raise TypeError(f"Expected Sample, got {type(obj)} from {pt_path}")
    return obj


def _build_chain_ids(sequences: list[str]) -> np.ndarray:
    return np.concatenate([np.full(len(seq), i, dtype=np.int64) for i, seq in enumerate(sequences)])


def extract_one(cfg: BenchmarkConfig, entry: dict) -> Path:
    sample = _load_sample(cfg.mint_data_root, entry["relative_path"])

    cm = sample.contact_map
    if isinstance(cm, torch.Tensor):
        cm = cm.detach().cpu().numpy().astype(np.int64)
    else:
        cm = np.asarray(cm, dtype=np.int64)

    expected_T = sum(len(seq) for seq in entry["sequences"])
    if cm.shape != (expected_T, expected_T):
        raise ValueError(
            f"{entry['id']}: contact_map shape {cm.shape} != ({expected_T}, {expected_T})"
        )

    chain_ids = _build_chain_ids(entry["sequences"])

    out_path = cfg.run_dir / "ground_truth" / f"{entry['id']}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        contact_map=cm,
        chain_ids=chain_ids,
        sequences=np.array(entry["sequences"], dtype=object),
    )
    return out_path


def extract_all(cfg: BenchmarkConfig) -> list[Path]:
    return [extract_one(cfg, e) for e in load_manifest(cfg)]
