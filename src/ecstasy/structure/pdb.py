"""atom37 <-> PDB, byte-compatible with the MENTOS DockQ evaluation.

Every DockQ number ecstasy produces must be directly comparable to the 23-checkpoint
MENTOS series, which was scored on PDBs written by ``scripts/evals/eval_structure_dockq.py``
``_write_pdb`` in the ``mentos`` repo. :func:`write_atom37_pdb` is a line-for-line port of
that function, so predictions and natives are rendered by identical code and any residual
difference in a score is the structure, not the serialisation.

Do not "improve" the record layout (add TER, occupancy, altloc, renumber serials). DockQ
reads chains and residue numbers off these columns, and a change here silently shifts every
score away from the published series.

The atom37 bundle is also the contract by which a model runner hands ecstasy a structure:
runners live in their own venvs and import no ecstasy code, so they write a plain
``structure.npz`` (see :func:`load_structure_npz`) and the scoring side renders it. Writing
predictions and natives through one writer is the point — it is what makes the comparison
byte-clean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: AF2 atom37 ordering. Index into the second axis of ``atom37_positions``.
ATOM_TYPES: tuple[str, ...] = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1",
    "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1",
    "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)

#: AF2 restype ordering (20 canonical + UNK). Index by ``aatype``.
RESNAMES: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
)

CA_INDEX = ATOM_TYPES.index("CA")
CB_INDEX = ATOM_TYPES.index("CB")

#: Keys a runner's ``structure.npz`` must carry.
STRUCTURE_NPZ_KEYS = (
    "atom37_positions", "atom37_mask", "aatype", "asym_id", "residue_index",
)


def write_atom37_pdb(
    path: Path,
    positions: np.ndarray,
    atom_mask: np.ndarray,
    aatype: np.ndarray,
    asym_id: np.ndarray,
    residue_index: np.ndarray,
    residue_mask: np.ndarray | None = None,
) -> Path:
    """Write an atom37 structure to a minimal PDB. All inputs are single-sample (N, ...).

    Port of ``mentos`` ``eval_structure_dockq._write_pdb``. Residues are emitted in array
    order; a residue is skipped when ``residue_mask`` is False or its ``asym_id`` is
    negative, and within a kept residue only atoms whose ``atom_mask`` is set are written.
    ``residue_index`` is 0-based and resets at each chain break (MENTOS's convention); the
    file carries ``residue_index + 1``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    atom_mask = np.asarray(atom_mask).astype(bool)
    aatype = np.asarray(aatype).astype(int)
    asym_id = np.asarray(asym_id).astype(int)
    residue_index = np.asarray(residue_index).astype(int)
    n = positions.shape[0]
    residue_mask = (np.ones(n, dtype=bool) if residue_mask is None
                    else np.asarray(residue_mask).astype(bool))
    if positions.shape[1:] != (len(ATOM_TYPES), 3):
        raise ValueError(f"positions has shape {positions.shape}, expected {(n, 37, 3)}")
    for name, arr, want in (
        ("atom_mask", atom_mask, (n, len(ATOM_TYPES))),
        ("aatype", aatype, (n,)),
        ("asym_id", asym_id, (n,)),
        ("residue_index", residue_index, (n,)),
        ("residue_mask", residue_mask, (n,)),
    ):
        if arr.shape != want:
            raise ValueError(f"{name} has shape {arr.shape}, expected {want}")

    lines: list[str] = []
    serial = 1
    for i in range(n):
        if not bool(residue_mask[i]) or int(asym_id[i]) < 0:
            continue
        chain = chr(ord("A") + int(asym_id[i]) % 26)
        resname = RESNAMES[int(aatype[i])]
        resnum = int(residue_index[i]) + 1
        for a in range(positions.shape[1]):
            if not bool(atom_mask[i, a]):
                continue
            name = ATOM_TYPES[a]
            x, y, z = (float(v) for v in positions[i, a])
            an = f" {name}".ljust(4)[:4]
            element = name[0]
            lines.append(
                f"ATOM  {serial:>5} {an} {resname:>3} {chain}{resnum:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
            )
            serial += 1
    lines.append("END")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_structure_npz(path: Path, **arrays: np.ndarray) -> Path:
    """Write a runner's atom37 bundle. Called from inside a model venv, so it takes
    plain arrays and imports nothing but numpy."""
    missing = [k for k in STRUCTURE_NPZ_KEYS if k not in arrays]
    if missing:
        raise KeyError(f"structure.npz missing {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        atom37_positions=np.asarray(arrays["atom37_positions"], dtype=np.float32),
        atom37_mask=np.asarray(arrays["atom37_mask"]).astype(bool),
        aatype=np.asarray(arrays["aatype"]).astype(np.int8),
        asym_id=np.asarray(arrays["asym_id"]).astype(np.int8),
        residue_index=np.asarray(arrays["residue_index"]).astype(np.int32),
    )
    return path


def load_structure_npz(path: Path) -> dict[str, np.ndarray]:
    """Read a runner's ``structure.npz`` and validate the atom37 bundle it carries."""
    with np.load(path) as d:
        missing = [k for k in STRUCTURE_NPZ_KEYS if k not in d]
        if missing:
            raise KeyError(f"{path}: structure.npz missing {missing}; has {sorted(d.files)}")
        return {k: np.asarray(d[k]) for k in STRUCTURE_NPZ_KEYS}


def render_structure_npz(npz_path: Path, pdb_path: Path) -> Path:
    """Render a runner's ``structure.npz`` to PDB with the native-identical writer."""
    b = load_structure_npz(npz_path)
    return write_atom37_pdb(
        pdb_path,
        positions=b["atom37_positions"],
        atom_mask=b["atom37_mask"],
        aatype=b["aatype"],
        asym_id=b["asym_id"],
        residue_index=b["residue_index"],
    )


# --- plain-PDB helpers, used by the random-placement null ----------------------------

def read_atom_lines(path: Path) -> list[str]:
    """ATOM records only, in file order."""
    return [ln for ln in Path(path).read_text().splitlines() if ln.startswith("ATOM")]


def chain_of(line: str) -> str:
    return line[21]


def atom_coords(lines: list[str]) -> np.ndarray:
    """(n_atoms, 3) read from PDB columns 31-54."""
    if not lines:
        return np.zeros((0, 3))
    return np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
                     for ln in lines])


def replace_coords(lines: list[str], xyz: np.ndarray) -> list[str]:
    """Substitute coordinates in place, leaving every other column byte-identical."""
    if len(lines) != len(xyz):
        raise ValueError(f"{len(lines)} lines vs {len(xyz)} coordinates")
    return [f"{ln[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{ln[54:]}"
            for ln, (x, y, z) in zip(lines, xyz)]
