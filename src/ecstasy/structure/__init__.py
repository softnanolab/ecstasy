"""Structural-biology primitives shared across ecstasy benchmarks.

Pinder-style interface enumeration: download bio-assembly 1 from RCSB,
parse to a single-model `AtomArray`, split into per-chain views, find
all chain pairs with any backbone-atom (N, CA, C, O) min-distance ≤ 10 Å,
and dump per-chain PDB files (used by Foldseek + downstream model runners).

Public API:
    download_cif_assembly(pdb_id, assembly_id=1) -> bytes
    parse_cif_bytes(cif_bytes) -> AtomArray
    load_local_cif(path) -> AtomArray
    split_into_chains(atoms) -> list[ChainView]
    interface_residue_indices(chain_a, chain_b, cutoff=10.0)
        -> (a_res_idx, b_res_idx, n_contact_pairs)
    enumerate_dimer_pairs(chains) -> list[(i, j)]
    write_chain_pdb(chain_atoms, out_path)
    select_chain_atoms(atoms, chain_id) -> AtomArray
    ChainView                          -- (chain_id, res_ids, sequence,
                                          backbone_xyz, backbone_res_idx)
    BACKBONE_ATOMS, INTERFACE_CUTOFF_A
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
from biotite.structure import AtomArray, filter_amino_acids
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure

BACKBONE_ATOMS = ("N", "CA", "C", "O")
INTERFACE_CUTOFF_A = 10.0


def download_cif_assembly(pdb_id: str, assembly_id: int = 1) -> bytes:
    """Fetch a biological-assembly mmCIF from RCSB. Returns raw bytes."""
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}-assembly{assembly_id}.cif.gz"
    req = Request(url, headers={"User-Agent": "ecstasy-v1/0.1"})
    with urlopen(req, timeout=60) as r:
        return gzip.decompress(r.read())


def parse_cif_bytes(cif_bytes: bytes) -> AtomArray:
    """Parse a CIF byte buffer to a single AtomArray (model 1)."""
    cif = CIFFile.read(io.StringIO(cif_bytes.decode("utf-8")))
    structure = get_structure(cif, model=1)
    return structure[filter_amino_acids(structure)]


def load_local_cif(path: Path) -> AtomArray:
    """Parse a local CIF (possibly gzipped) to a single AtomArray (model 1)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            cif = CIFFile.read(f)
    else:
        cif = CIFFile.read(str(path))
    structure = get_structure(cif, model=1)
    return structure[filter_amino_acids(structure)]


@dataclass(frozen=True)
class ChainView:
    chain_id: str  # the chain identifier as found in the assembly (may include _N suffix)
    res_ids: np.ndarray  # int residue numbers (auth_seq_id), unique, ordered
    sequence: str  # 1-letter amino acid string
    backbone_xyz: np.ndarray  # (n_backbone_atoms, 3) coords for N/CA/C/O across all residues
    backbone_res_idx: np.ndarray  # (n_backbone_atoms,) index into res_ids for each backbone atom


_AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def split_into_chains(atoms: AtomArray) -> list[ChainView]:
    """Split an AtomArray into ChainViews, one per chain_id."""
    out: list[ChainView] = []
    for cid in np.unique(atoms.chain_id):
        chain_atoms = atoms[atoms.chain_id == cid]
        if len(chain_atoms) == 0:
            continue
        # unique residue ids in insertion order
        res_ids, first_idx = np.unique(chain_atoms.res_id, return_index=True)
        order = np.argsort(first_idx)
        res_ids = res_ids[order]
        # build per-residue 3-letter -> 1-letter sequence
        seq_chars: list[str] = []
        for rid in res_ids:
            res_atoms = chain_atoms[chain_atoms.res_id == rid]
            res_name = res_atoms.res_name[0]
            seq_chars.append(_AA3to1.get(res_name, "X"))
        sequence = "".join(seq_chars)

        bb_mask = np.isin(chain_atoms.atom_name, BACKBONE_ATOMS)
        bb_atoms = chain_atoms[bb_mask]
        if len(bb_atoms) == 0:
            continue
        # map each bb atom to its res_id index in res_ids
        res_id_to_idx = {int(r): i for i, r in enumerate(res_ids)}
        bb_res_idx = np.array(
            [res_id_to_idx[int(r)] for r in bb_atoms.res_id], dtype=np.int32
        )
        out.append(
            ChainView(
                chain_id=str(cid),
                res_ids=res_ids.astype(np.int64),
                sequence=sequence,
                backbone_xyz=bb_atoms.coord.astype(np.float32),
                backbone_res_idx=bb_res_idx,
            )
        )
    return out


def interface_residue_indices(
    chain_a: ChainView, chain_b: ChainView, cutoff: float = INTERFACE_CUTOFF_A
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (interface_res_idx_a, interface_res_idx_b, n_contacting_pairs)
    where indices are 0-based positions into chain.res_ids (sequence positions).

    Interface = residues with any backbone atom within `cutoff` A of any
    backbone atom of the partner chain. The third return value is the number
    of contacting (residue_a, residue_b) pairs found.
    """
    # Pairwise distances between all backbone atoms across the two chains.
    # Memory: O(|bb_A| * |bb_B|) -- both are small (~hundreds of atoms each).
    d2 = np.sum(
        (chain_a.backbone_xyz[:, None, :] - chain_b.backbone_xyz[None, :, :]) ** 2,
        axis=-1,
    )
    close = d2 <= cutoff * cutoff
    if not close.any():
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
    a_atom_idx, b_atom_idx = np.where(close)
    a_res = np.unique(chain_a.backbone_res_idx[a_atom_idx])
    b_res = np.unique(chain_b.backbone_res_idx[b_atom_idx])
    # contacting residue-pairs (de-duplicated)
    pair_set = {
        (int(chain_a.backbone_res_idx[ai]), int(chain_b.backbone_res_idx[bi]))
        for ai, bi in zip(a_atom_idx, b_atom_idx)
    }
    return a_res.astype(np.int32), b_res.astype(np.int32), len(pair_set)


def write_chain_pdb(chain_atoms: AtomArray, out_path: Path) -> None:
    """Dump a single chain's atoms to a PDB file.

    Forces chain_id to single character "A" before writing because bio-assembly
    symmetry-mate chains can have multi-character IDs (e.g. "AA", "AB") that
    biotite's PDB writer rejects. We only write one chain per file, so the
    chain label is recoverable from the filename anyway.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    renamed = chain_atoms.copy()
    renamed.chain_id = np.array(["A"] * len(renamed), dtype=renamed.chain_id.dtype)
    pdb = PDBFile()
    pdb.set_structure(renamed)
    pdb.write(str(out_path))


def select_chain_atoms(atoms: AtomArray, chain_id: str) -> AtomArray:
    return atoms[atoms.chain_id == chain_id]


def enumerate_dimer_pairs(chains: Iterable[ChainView]) -> list[tuple[int, int]]:
    """All unordered chain-index pairs (i, j) with i < j."""
    chains = list(chains)
    return [(i, j) for i in range(len(chains)) for j in range(i + 1, len(chains))]
