"""aarch64 reimplementations of DeepInteract's two structure-derived feature sets
that depend on x86-only tools (PSAIA protrusion; DSSP via mkdssp), so the trained
model can run on Grace/Hopper with REAL — not imputed — features.

Both are written to match exactly what DeepInteract's parsers/consumers expect
(verified against atom3.conservation + project/utils/dips_plus_utils.py):

* PSAIA protrusion .tbl  -> ``write_psaia_cx_tbl``: a per-residue table with the 6 CX
  statistics (avg/s_avg/s-ch avg/s-ch s_avg/max/min) at whitespace-split indices [3:9],
  preceded by a literal ``chain`` header line (the parser turns on after that token).
  CX (Pintar et al. 2002, the index PSAIA computes): per heavy atom, count heavy-atom
  neighbours within R=10 A; CX = (Vsphere - n*Vatom) / (n*Vatom). DeepInteract min-max
  normalises each CX column PER CHAIN, so the absolute Vatom constant is irrelevant —
  only the per-chain ordering/spread matters — but all 6 distinct stats must be emitted.

* DSSP dict -> ``dssp_dict_for_pdb``: returns the same ``{(chain,(' ',resseq,' ')): tuple}``
  shape Biopython's DSSP class yields, with index [2]=SS char and [3]=relative solvent
  accessibility in [0,1]. RSA is computed with Biopython's pure-Python Shrake-Rupley SASA
  divided by the per-residue-type Sander max-ASA (the same normalisation Biopython's DSSP
  applies), clamped to 1.0. SS is filled from an optional pure-Python 3-state assignment
  (pydssp) mapped onto the 8-state alphabet, else the missing token '-'.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

_BACKBONE = {"N", "CA", "C", "O", "OXT"}
_VATOM = 20.1                       # mean heavy-atom volume, A^3 (Pintar 2002)
_R = 10.0                           # protrusion sphere radius, A (PSAIA default)
_VSPHERE = (4.0 / 3.0) * np.pi * _R ** 3


def _parse_heavy_atoms(pdb_path: str):
    """(chain, resseq_str, resname, atomname, x, y, z) for heavy atoms of model 1."""
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip()
            atomname = line[12:16].strip()
            if element == "H" or (not element and atomname[:1] == "H"):
                continue
            atoms.append((
                line[21].strip(), line[22:26].strip(), line[17:20].strip(), atomname,
                float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return atoms


def _per_atom_cx(coords: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    # neighbours within R, minus self; >=1 to avoid div-by-zero (fully isolated atom)
    counts = np.array([len(tree.query_ball_point(c, _R)) - 1 for c in coords], dtype=float)
    counts = np.maximum(counts, 1.0)
    return np.maximum(_VSPHERE / (counts * _VATOM) - 1.0, 0.0)


def write_psaia_cx_tbl(pdb_path: str, out_tbl: str) -> int:
    """Write a PSAIA-format protrusion .tbl for one PDB; returns residue count."""
    atoms = _parse_heavy_atoms(pdb_path)
    if not atoms:
        open(out_tbl, "w").write("PSAIA aarch64 CX (empty)\nchain\n")
        return 0
    coords = np.array([[a[4], a[5], a[6]] for a in atoms])
    cx = _per_atom_cx(coords)
    res_atoms: "OrderedDict[tuple, list]" = OrderedDict()
    for i, a in enumerate(atoms):
        res_atoms.setdefault((a[0], a[1], a[2]), []).append(i)
    lines = ["PSAIA aarch64 CX reimplementation (Pintar 2002)", "chain"]
    for (chain, resseq, resname), idxs in res_atoms.items():
        allcx = cx[idxs]
        sc = [i for i in idxs if atoms[i][3] not in _BACKBONE]
        sccx = cx[sc] if sc else allcx          # GLY / backbone-only -> reuse all-atom
        cid = chain if chain else "*"
        lines.append(
            f"{cid} {resseq} {resname} {allcx.mean():.4f} {allcx.std():.4f} "
            f"{sccx.mean():.4f} {sccx.std():.4f} {allcx.max():.4f} {allcx.min():.4f}")
    open(out_tbl, "w").write("\n".join(lines) + "\n")
    return len(res_atoms)


def _ss_3state_map(pdb_path: str) -> dict:
    """Optional pure-Python 3-state SS via pydssp, keyed (chain, resseq_int) -> char in
    {'H','E','-'}. Returns {} if pydssp/backbone unavailable (SS then falls back to '-')."""
    try:
        import torch
        import pydssp
    except Exception:
        return {}
    # pydssp needs ordered N,CA,C,O backbone coords per residue, per chain.
    by_chain: "OrderedDict[str, list]" = OrderedDict()
    cur = {}
    last = None
    for a in _parse_heavy_atoms(pdb_path):
        chain, resseq, _resn, name = a[0], a[1], a[2], a[3]
        key = (chain, resseq)
        if key != last and cur:
            by_chain.setdefault(last[0], []).append((last[1], cur))
            cur = {}
        last = key
        if name in ("N", "CA", "C", "O"):
            cur[name] = (a[4], a[5], a[6])
    if cur and last:
        by_chain.setdefault(last[0], []).append((last[1], cur))
    out: dict = {}
    for chain, reslist in by_chain.items():
        coords, ids = [], []
        for resseq, bb in reslist:
            if all(k in bb for k in ("N", "CA", "C", "O")):
                coords.append([bb["N"], bb["CA"], bb["C"], bb["O"]])
                ids.append(resseq)
        if len(coords) < 4:
            continue
        ss = pydssp.assign(torch.tensor(np.array(coords), dtype=torch.float32), out_type="c3")
        m = {"H": "H", "E": "E", "-": "-", "L": "-", "C": "-"}
        for resseq, s in zip(ids, ss):
            try:
                out[(chain, int(resseq))] = m.get(str(s), "-")
            except ValueError:
                pass
    return out


def dssp_dict_for_pdb(pdb_path: str) -> dict:
    """Drop-in replacement for DeepInteract's ``get_dssp_dict_for_pdb_model`` result.

    Key: ``(chain_id, (' ', resseq_int, ' '))``; value tuple has SS at [2] and relative
    solvent accessibility (Sander-normalised, [0,1]) at [3] — the indices
    ``get_dssp_value_for_residue`` reads.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.Data.PDBData import residue_sasa_scales

    sander = residue_sasa_scales["Sander"]
    model = PDBParser(QUIET=True).get_structure("x", pdb_path)[0]
    ShrakeRupley().compute(model, level="R")
    ss3 = _ss_3state_map(pdb_path)
    out: dict = {}
    for chain in model:
        cid = chain.id.strip()
        for res in chain:
            if res.id[0] != " ":
                continue
            maxasa = sander.get(res.resname)
            rsa = min(res.sasa / maxasa, 1.0) if maxasa else "NA"
            ss = ss3.get((chain.id.strip(), res.id[1]), "-")
            # (dssp_index, aa, SS, rel_acc, phi, psi, ...) — only [2],[3] are read.
            out[(cid, (" ", res.id[1], " "))] = (0, res.resname, ss, rsa,
                                                 360.0, 360.0, 0, 0, 0, 0, 0, 0, 0, 0)
    return out
