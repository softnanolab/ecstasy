"""Build the ecstasy_v1 master test set after applying the deleak cut.

Cut rule (Pinder Level-2 defaults):
    drop dimer if EITHER chain has any Foldseek hit to MINT-train with
    coverage >= 0.5 AND LDDT >= 0.7.

For each surviving dimer:
    - load chain A and chain B PDB files (already in candidates/chains/)
    - extract Cβ atoms (synthesize virtual Cβ for Gly)
    - compute Cβ-Cβ inter-chain distance map
    - bin into 10 classes (MINT scheme: bin 0 = d<=4, bin k in 1..8 = k+3<d<=k+4, bin 9 = d>12)
    - write <out>/data/<id[:2]>/<id>.pt with `contact_map`, `distance_map`, `sequences`

Manifest schema mirrors MINT's seq_id_30 index.parquet so the benchmark
loader can be wired with minimal changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from ecstasy.structure import AA3_TO_1

ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1")
DIMERS_PATH = ROOT / "candidates" / "dimers.parquet"
CHAINS_DIR = ROOT / "candidates" / "chains"
EDGES_PATH = ROOT / "all_edges_with_coverage.parquet"
META_PATH = ROOT / "candidates" / "val_v2_metadata.parquet"

MASTER_DIR = ROOT / "master"
DATA_DIR = MASTER_DIR / "data"
MANIFEST_PATH = MASTER_DIR / "index.parquet"
DROPPED_PATH = MASTER_DIR / "dropped_dimers.parquet"
REPORT_PATH = MASTER_DIR / "master_README.md"

# Cut thresholds
COVERAGE_TH = 0.5
LDDT_TH = 0.7

# MINT 10-bin distogram
# bin 0 = d<=4; bin k (1..8) = k+3 < d <= k+4; bin 9 = d>12
DISTOGRAM_BINS = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32)


@dataclass(frozen=True)
class ChainBundle:
    sequence: str
    cb_xyz: np.ndarray   # (N_res, 3); NaN where Cβ is undefined
    res_ids: np.ndarray  # (N_res,) int auth_seq_id, ordered


def _virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute a Cβ position for a residue from its N/CA/C backbone atoms.

    Standard tetrahedral geometry: place a virtual Cβ such that the
    N-CA-Cβ-C dihedral and the CA-Cβ distance match canonical values.
    Formula from AlphaFold / RoseTTAFold conventions.
    """
    b = ca - n
    c_vec = c - ca
    a = np.cross(b, c_vec)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


def load_chain_pdb(path: Path) -> ChainBundle:
    """Read a per-chain PDB. Return Cβ coords (real or virtual) + sequence."""
    pdb = PDBFile.read(str(path))
    structure: AtomArray = pdb.get_structure(model=1)
    # res_id can repeat for multi-atom residues, take unique in insertion order
    res_ids_all, first_idx = np.unique(structure.res_id, return_index=True)
    order = np.argsort(first_idx)
    res_ids = res_ids_all[order].astype(np.int64)

    seq_chars: list[str] = []
    cb_coords = np.full((len(res_ids), 3), np.nan, dtype=np.float32)

    for i, rid in enumerate(res_ids):
        res_mask = structure.res_id == rid
        res = structure[res_mask]
        if len(res) == 0:
            continue
        res_name = res.res_name[0]
        seq_chars.append(AA3_TO_1.get(res_name, "X"))
        # Find Cβ; synthesize if missing
        cb_mask = res.atom_name == "CB"
        if cb_mask.any():
            cb_coords[i] = res[cb_mask].coord[0]
        else:
            # Gly or missing Cβ -> virtual from N/CA/C
            try:
                n = res[res.atom_name == "N"].coord[0]
                ca = res[res.atom_name == "CA"].coord[0]
                c = res[res.atom_name == "C"].coord[0]
                cb_coords[i] = _virtual_cb(n, ca, c).astype(np.float32)
            except IndexError:
                pass  # leave NaN; treated as missing
    return ChainBundle(
        sequence="".join(seq_chars),
        cb_xyz=cb_coords,
        res_ids=res_ids,
    )


def compute_interchain_contact_map(
    bundle_a: ChainBundle, bundle_b: ChainBundle
) -> tuple[np.ndarray, np.ndarray]:
    """Return (distance_map (Na, Nb) float32, contact_map (Na, Nb) int64).

    contact_map is the 10-bin MINT distogram. Missing-Cβ entries get -1.
    """
    na = len(bundle_a.cb_xyz)
    nb = len(bundle_b.cb_xyz)
    # pairwise distances
    diff = bundle_a.cb_xyz[:, None, :] - bundle_b.cb_xyz[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)
    # missing mask
    bad = np.isnan(dist)
    dist[bad] = 999.0
    # bin
    contact = np.digitize(dist, DISTOGRAM_BINS, right=True).astype(np.int64)
    contact[bad] = -1
    return dist, contact


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading inputs ...")
    dimers = pd.read_parquet(DIMERS_PATH).reset_index(drop=True)
    edges = pd.read_parquet(EDGES_PATH)
    meta = pd.read_parquet(META_PATH)
    print(f"  {len(dimers)} candidate dimers, {len(edges)} edges, {len(meta)} metadata rows")

    # Determine leaky dimers under the cut
    qual = edges[(edges["coverage"] >= COVERAGE_TH) & (edges["lddt"] >= LDDT_TH)]
    leaky_dimer_idxs = set(qual["dimer_idx"].unique().tolist())
    print(
        f"  applying cut: coverage>={COVERAGE_TH}, lddt>={LDDT_TH} -> "
        f"{len(leaky_dimer_idxs)} dimers flagged for drop"
    )

    keep_mask = ~dimers.index.isin(leaky_dimer_idxs)
    kept = dimers[keep_mask].reset_index(drop=True)
    dropped = dimers[~keep_mask].reset_index(drop=True)
    print(f"  kept: {len(kept)} | dropped: {len(dropped)}")

    # Compute GT contact maps and write .pt files
    print("Writing GT .pt files + building manifest ...")
    manifest_rows: list[dict] = []
    fail_rows: list[dict] = []
    pdb_to_meta = {r["pdb_id"]: r for _, r in meta.iterrows()}

    for i, row in kept.iterrows():
        pdb_id = row["pdb_id"]
        assembly = row["assembly_id"]
        chain_a = row["chain_a"]
        chain_b = row["chain_b"]
        # Per-dimer entry id: <pdb_id>_<chain_a>_<chain_b> (unique across the set)
        entry_id = f"{pdb_id}_{chain_a}_{chain_b}"

        path_a = CHAINS_DIR / f"{pdb_id}_{assembly}_{chain_a}.pdb"
        path_b = CHAINS_DIR / f"{pdb_id}_{assembly}_{chain_b}.pdb"
        try:
            bundle_a = load_chain_pdb(path_a)
            bundle_b = load_chain_pdb(path_b)
            dist_map, contact_map = compute_interchain_contact_map(bundle_a, bundle_b)
        except Exception as e:  # noqa: BLE001
            fail_rows.append({"entry_id": entry_id, "error": f"{type(e).__name__}: {e}"})
            continue

        # Write .pt file
        rel_path = Path("data") / entry_id[:2] / f"{entry_id}.pt"
        out_path = MASTER_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "id": entry_id,
                "pdb_id": pdb_id,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "sequences": [bundle_a.sequence, bundle_b.sequence],
                "res_ids": [bundle_a.res_ids.tolist(), bundle_b.res_ids.tolist()],
                "contact_map": torch.from_numpy(contact_map),     # (Na, Nb) int64 in [-1, 9]
                "distance_map": torch.from_numpy(dist_map),       # (Na, Nb) float32
                "is_homodimer": bool(row["is_homodimer"]),
            },
            out_path,
        )

        m = pdb_to_meta.get(pdb_id)
        manifest_rows.append(
            {
                "id": entry_id,
                "pdb_id": pdb_id,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "len_a": int(row["len_a"]),
                "len_b": int(row["len_b"]),
                "total_sequence_length": int(row["len_a"] + row["len_b"]),
                "num_chains": 2,
                "is_homodimer": bool(row["is_homodimer"]),
                "n_interface_residues_a": int(row["n_interface_residues_a"]),
                "n_interface_residues_b": int(row["n_interface_residues_b"]),
                "n_contact_pairs": int(row["n_contact_pairs"]),
                "sequences": [bundle_a.sequence, bundle_b.sequence],
                "relative_path": str(rel_path),
                "deposit_date": (m["deposit_date"] if m is not None else None),
                "release_date": (m["release_date"] if m is not None else None),
                "resolution": (float(m["resolution"]) if m is not None and m["resolution"] is not None else None),
                "split": "val",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(MANIFEST_PATH, index=False)
    dropped.to_parquet(DROPPED_PATH, index=False)
    print(f"  wrote {MANIFEST_PATH}  ({len(manifest)} rows)")
    print(f"  wrote {DROPPED_PATH}   ({len(dropped)} rows)")
    if fail_rows:
        pd.DataFrame(fail_rows).to_parquet(MASTER_DIR / "build_failures.parquet", index=False)
        print(f"  WARNING: {len(fail_rows)} dimers failed GT generation; see build_failures.parquet")

    # Compose README
    md = []
    md.append("# ecstasy_v1 master test set")
    md.append("")
    md.append(f"Built {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')} from Boltz-2 `validation_ids_v2.txt`.")
    md.append("")
    md.append("## Provenance")
    md.append("- Source: Boltz-2 `validation_ids_v2.txt` (upstream/v2, 397/398 PDB IDs, 2023-06-07 → 2023-12-27)")
    md.append("- Bio-assembly: assembly 1, all chain pairs with backbone-atom ≤10Å contact")
    md.append("- Quality filter: X-ray, resolution ≤3.5 Å, min chain ≥40 res, total ≤1200 res, ≥3 contact pairs")
    md.append(f"- Foldseek deleak: candidates vs MINT-softnano train chains (PDB ≤2021-09-30)")
    md.append(f"  - drop rule: Pinder Level-2 — drop if either chain has any hit with coverage ≥ {COVERAGE_TH} AND LDDT ≥ {LDDT_TH}")
    md.append("")
    md.append("## Counts")
    md.append(f"- Candidate dimers: 319")
    md.append(f"  - flagged as MINT-train leak: 97  ({100*97/319:.1f}%)")
    md.append(f"  - **kept (this master set): {len(manifest)}**")
    if len(manifest):
        md.append(f"    - homodimers: {int(manifest['is_homodimer'].sum())}")
        md.append(f"    - heterodimers: {int((~manifest['is_homodimer']).sum())}")
        md.append(f"    - total length range: [{manifest['total_sequence_length'].min()}, {manifest['total_sequence_length'].max()}]")
    md.append("")
    md.append("## Layout")
    md.append("```")
    md.append(f"{MASTER_DIR}/")
    md.append("  index.parquet              # manifest (one row per dimer)")
    md.append("  dropped_dimers.parquet     # leakage-flagged dimers for audit")
    md.append("  master_README.md           # this file")
    md.append("  data/<id[:2]>/<id>.pt      # per-dimer GT: contact_map (10-bin Cβ-Cβ), distance_map, sequences")
    md.append("```")
    md.append("")
    md.append("## Per-entry .pt schema")
    md.append("- `id`, `pdb_id`, `chain_a`, `chain_b`")
    md.append("- `sequences`: list of two str (chain A, chain B 1-letter seqs)")
    md.append("- `res_ids`: list of two int lists (auth residue numbers in chain order)")
    md.append("- `contact_map`: torch int64 (Na, Nb), MINT 10-bin Cβ-Cβ scheme; `-1` = missing Cβ")
    md.append("- `distance_map`: torch float32 (Na, Nb), raw Cβ-Cβ distances in Å")
    md.append("- `is_homodimer`: bool")
    md.append("")
    md.append("## Manifest schema (index.parquet)")
    md.append("`id, pdb_id, chain_a, chain_b, len_a, len_b, total_sequence_length, num_chains,`")
    md.append("`is_homodimer, n_interface_residues_a, n_interface_residues_b, n_contact_pairs, sequences,`")
    md.append("`relative_path, deposit_date, release_date, resolution, split`")
    REPORT_PATH.write_text("\n".join(md) + "\n")
    print(f"  wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
