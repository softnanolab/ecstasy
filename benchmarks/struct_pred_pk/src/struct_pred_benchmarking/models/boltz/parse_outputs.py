"""Parse Boltz prediction CIFs → empirical pairwise contact probability.

For each protein, glob the diffusion-sample CIFs produced by `boltz predict`,
compute synthesized Cβ-Cβ distance per residue (using the same get_cb_coord as
MINT's ground-truth pipeline), threshold at the contact distance, and average
over the samples to get P(i,j) ∈ [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure import AtomArray
from scipy.spatial.distance import pdist, squareform

from mint.utils import get_cb_coord

from struct_pred_benchmarking.config import BenchmarkConfig
from struct_pred_benchmarking.data.select_val import load_manifest


def _load_cif_atoms(cif_path: Path) -> AtomArray:
    cif = pdbx.CIFFile.read(str(cif_path))
    return pdbx.get_structure(cif, model=1)


def _per_chain_cb_coords(atoms: AtomArray, chain: str, expected_n_res: int) -> np.ndarray:
    """Return (expected_n_res, 3) Cβ coords for one chain, in residue order."""
    chain_mask = atoms.chain_id == chain
    if not chain_mask.any():
        raise ValueError(f"Chain {chain} not found in CIF (chains: {set(atoms.chain_id)})")

    sub = atoms[chain_mask]
    # res_id may not be contiguous (Boltz numbers from 1; mmCIF allows gaps).
    # Group by (res_id, ins_code) preserving first-occurrence order.
    keys: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for rid, ins in zip(sub.res_id, sub.ins_code):
        k = (int(rid), str(ins))
        if k not in seen:
            seen.add(k)
            keys.append(k)

    cb_list: list[np.ndarray] = []
    for rid, ins in keys:
        res_mask = (sub.res_id == rid) & (sub.ins_code == ins)
        res = sub[res_mask]
        n = res.coord[res.atom_name == "N"]
        ca = res.coord[res.atom_name == "CA"]
        c = res.coord[res.atom_name == "C"]
        if n.shape[0] == 0 or ca.shape[0] == 0 or c.shape[0] == 0:
            # Skip residues with incomplete backbone (mirrors mint.utils.load_structure)
            continue
        cb = get_cb_coord(C=c[0], N=n[0], CA=ca[0])
        cb_list.append(cb)

    if len(cb_list) != expected_n_res:
        raise ValueError(
            f"Chain {chain}: parsed {len(cb_list)} residues, expected {expected_n_res}"
        )
    return np.stack(cb_list, axis=0)


def cif_to_distance_matrix(cif_path: Path, sequences: list[str]) -> np.ndarray:
    """(T, T) Cβ-Cβ distance matrix for a single CIF. T = sum of chain lengths."""
    atoms = _load_cif_atoms(cif_path)
    chain_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    cbs = [
        _per_chain_cb_coords(atoms, chain_letters[i], len(seq)) for i, seq in enumerate(sequences)
    ]
    cb_all = np.concatenate(cbs, axis=0)
    return squareform(pdist(cb_all))


def _glob_sample_cifs(predictions_dir: Path, expected_count: int) -> list[Path]:
    """Find the per-sample CIFs in <predictions_dir>/predictions/<input_stem>/<input_stem>_model_*.cif.

    Falls back to a recursive glob if the layout is different.
    """
    candidates = sorted(predictions_dir.glob("boltz_results_*/predictions/*/*_model_*.cif"))
    if not candidates:
        candidates = sorted(predictions_dir.rglob("*_model_*.cif"))
    if len(candidates) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} sample CIFs under {predictions_dir}, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates


def parse_one(cfg: BenchmarkConfig, entry: dict) -> Path:
    pred_dir = cfg.run_dir / "boltz_predictions" / entry["id"]
    cif_paths = _glob_sample_cifs(pred_dir, expected_count=cfg.diffusion_samples)

    distances = np.stack(
        [cif_to_distance_matrix(p, entry["sequences"]) for p in cif_paths], axis=0
    )  # (S, T, T)
    contacts = (distances < cfg.contact_threshold_angstrom).astype(np.float32)  # (S, T, T)
    probs = contacts.mean(axis=0)  # (T, T)

    chain_ids = np.concatenate(
        [np.full(len(seq), i, dtype=np.int64) for i, seq in enumerate(entry["sequences"])]
    )

    out_path = cfg.run_dir / "boltz_contacts" / f"{entry['id']}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        probs=probs,
        chain_ids=chain_ids,
        n_samples=np.int64(distances.shape[0]),
    )
    return out_path


def parse_all(cfg: BenchmarkConfig) -> list[Path]:
    return [parse_one(cfg, e) for e in load_manifest(cfg)]
