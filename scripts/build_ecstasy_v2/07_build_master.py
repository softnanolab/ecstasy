"""Build the ecstasy_v2 master test set after applying the deleak cut.

Cut rule (Pinder Level-2 defaults applied to BOTH train DBs):
    drop dimer if EITHER chain has any Foldseek hit, against EITHER the
    Mentos-train DB or the Boltz-2-train DB, with
        coverage >= 0.5  AND  LDDT >= 0.7

The dropped-dimer manifest tags each drop with which source(s) triggered
it ("mentos", "boltz2", or "both") for downstream reporting.

For each surviving dimer we recompute Cβ-Cβ inter-chain distance + 10-bin
distogram exactly like v1's 07 and write <out>/data/<id[:2]>/<id>.pt.
Manifest schema mirrors v1's so the existing benchmark loader works as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from ecstasy.structure import _AA3to1  # noqa: E402

ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v2")
DIMERS_PATH = ROOT / "candidates" / "dimers.parquet"
CHAINS_DIR = ROOT / "candidates" / "chains"
EDGES_PATH = ROOT / "all_edges_with_coverage.parquet"
META_PATH = ROOT / "candidates" / "rcsb_metadata.parquet"

MASTER_DIR = ROOT / "master"
DATA_DIR = MASTER_DIR / "data"
MANIFEST_PATH = MASTER_DIR / "index.parquet"
DROPPED_PATH = MASTER_DIR / "dropped_dimers.parquet"
REPORT_PATH = MASTER_DIR / "master_README.md"

COVERAGE_TH = 0.5
LDDT_TH = 0.7

DISTOGRAM_BINS = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32)


@dataclass(frozen=True)
class ChainBundle:
    sequence: str
    cb_xyz: np.ndarray
    res_ids: np.ndarray


def _virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    b = ca - n
    c_vec = c - ca
    a = np.cross(b, c_vec)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


def load_chain_pdb(path: Path) -> ChainBundle:
    pdb = PDBFile.read(str(path))
    structure: AtomArray = pdb.get_structure(model=1)
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
        seq_chars.append(_AA3to1.get(res_name, "X"))
        cb_mask = res.atom_name == "CB"
        if cb_mask.any():
            cb_coords[i] = res[cb_mask].coord[0]
        else:
            try:
                n = res[res.atom_name == "N"].coord[0]
                ca = res[res.atom_name == "CA"].coord[0]
                c = res[res.atom_name == "C"].coord[0]
                cb_coords[i] = _virtual_cb(n, ca, c).astype(np.float32)
            except IndexError:
                pass
    return ChainBundle(
        sequence="".join(seq_chars),
        cb_xyz=cb_coords,
        res_ids=res_ids,
    )


def compute_interchain_contact_map(
    bundle_a: ChainBundle, bundle_b: ChainBundle
) -> tuple[np.ndarray, np.ndarray]:
    diff = bundle_a.cb_xyz[:, None, :] - bundle_b.cb_xyz[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)
    bad = np.isnan(dist)
    dist[bad] = 999.0
    contact = np.digitize(dist, DISTOGRAM_BINS, right=True).astype(np.int64)
    contact[bad] = -1
    return dist, contact


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading inputs ...")
    dimers = pd.read_parquet(DIMERS_PATH).reset_index(drop=True)
    edges = pd.read_parquet(EDGES_PATH)
    meta = pd.read_parquet(META_PATH)
    print(f"  {len(dimers)} candidate dimers, {len(edges)} edges, {len(meta)} metadata rows")

    # leak edges = above both thresholds
    qual = edges[(edges["coverage"] >= COVERAGE_TH) & (edges["lddt"] >= LDDT_TH)]
    # per-dimer source attribution
    sources_per_dimer: dict[int, set] = {}
    for _, e in qual.iterrows():
        sources_per_dimer.setdefault(int(e["dimer_idx"]), set()).add(e["source"])
    leaky_dimer_idxs = set(sources_per_dimer.keys())
    print(
        f"  applying cut: coverage>={COVERAGE_TH}, lddt>={LDDT_TH} -> "
        f"{len(leaky_dimer_idxs)} dimers flagged for drop"
    )
    if leaky_dimer_idxs:
        src_counts: dict[str, int] = {"mentos_only": 0, "boltz2_only": 0, "both": 0}
        for s in sources_per_dimer.values():
            if s == {"mentos"}:
                src_counts["mentos_only"] += 1
            elif s == {"boltz2"}:
                src_counts["boltz2_only"] += 1
            else:
                src_counts["both"] += 1
        print(f"  drop-source breakdown: {src_counts}")

    keep_mask = ~dimers.index.isin(leaky_dimer_idxs)
    kept = dimers[keep_mask].reset_index(drop=True)
    dropped = dimers[~keep_mask].copy()
    dropped["drop_sources"] = dropped.index.map(
        lambda i: ",".join(sorted(sources_per_dimer.get(i, set()))) or None
    )
    dropped = dropped.reset_index(drop=True)
    print(f"  kept: {len(kept)} | dropped: {len(dropped)}")

    print("Writing GT .pt files + building manifest ...")
    manifest_rows: list[dict] = []
    fail_rows: list[dict] = []
    pdb_to_meta = {r["pdb_id"]: r for _, r in meta.iterrows()}

    for _, row in kept.iterrows():
        pdb_id = row["pdb_id"]
        assembly = row["assembly_id"]
        chain_a = row["chain_a"]
        chain_b = row["chain_b"]
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
                "contact_map": torch.from_numpy(contact_map),
                "distance_map": torch.from_numpy(dist_map),
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
                "resolution": (
                    float(m["resolution"])
                    if m is not None and m["resolution"] is not None
                    else None
                ),
                "split": "test",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(MANIFEST_PATH, index=False)
    dropped.to_parquet(DROPPED_PATH, index=False)
    print(f"  wrote {MANIFEST_PATH}  ({len(manifest)} rows)")
    print(f"  wrote {DROPPED_PATH}   ({len(dropped)} rows)")
    if fail_rows:
        pd.DataFrame(fail_rows).to_parquet(
            MASTER_DIR / "build_failures.parquet", index=False
        )
        print(f"  WARNING: {len(fail_rows)} dimers failed GT generation")

    md = []
    md.append("# ecstasy_v2 master test set")
    md.append("")
    md.append(
        f"Built {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')} "
        "from RCSB entries released >= 2023-06-01 (Boltz-2 training cutoff)."
    )
    md.append("")
    md.append("## Provenance")
    md.append("- Source: RCSB Search API — `initial_release_date >= 2023-06-01`, X-ray, >=2 protein entities")
    md.append("- Bio-assembly: assembly 1, all chain pairs with backbone-atom <=10 A contact")
    md.append("- Quality filter: X-ray, resolution <=3.5 A, min chain >=40 res, total <=1200 res, >=3 contact pairs")
    md.append("- Foldseek deleak: candidates vs **two** train DBs")
    md.append("  - **Mentos-train**: PDB <= 2021-09-30 (Mentos seq_id_30 train split)")
    md.append("  - **Boltz-2-train (approx)**: PDB <= 2023-06-01 — Mentos chains union all PDB chains released 2021-10-01 .. 2023-05-31 (strict superset; conservative for leakage)")
    md.append(f"- Drop rule (either DB): coverage >= {COVERAGE_TH} AND LDDT >= {LDDT_TH}")
    md.append("")
    md.append("## Counts")
    md.append(f"- Candidate dimers:                 {len(dimers)}")
    md.append(f"- Flagged as leaky (any source):    {len(dropped)}  ({100*len(dropped)/max(1,len(dimers)):.1f}%)")
    if leaky_dimer_idxs:
        md.append(f"  - mentos-only:                    {src_counts['mentos_only']}")
        md.append(f"  - boltz2-only:                    {src_counts['boltz2_only']}")
        md.append(f"  - both:                           {src_counts['both']}")
    md.append(f"- **Kept (master set):              {len(manifest)}**")
    if len(manifest):
        md.append(f"  - homodimers:                     {int(manifest['is_homodimer'].sum())}")
        md.append(f"  - heterodimers:                   {int((~manifest['is_homodimer']).sum())}")
        md.append(
            f"  - total-length range:             [{manifest['total_sequence_length'].min()}, "
            f"{manifest['total_sequence_length'].max()}]"
        )
        if manifest["release_date"].notna().any():
            md.append(
                f"  - release-date range:             {manifest['release_date'].min()} .. "
                f"{manifest['release_date'].max()}"
            )
    md.append("")
    md.append("## Layout")
    md.append("```")
    md.append(f"{MASTER_DIR}/")
    md.append("  index.parquet              # manifest (one row per dimer)")
    md.append("  dropped_dimers.parquet     # leaky dimers with `drop_sources` attribution")
    md.append("  master_README.md           # this file")
    md.append("  data/<id[:2]>/<id>.pt      # per-dimer GT: contact_map (10-bin Cb-Cb), distance_map, sequences")
    md.append("```")
    md.append("")
    md.append("## Per-entry .pt schema (same as v1)")
    md.append("- `id`, `pdb_id`, `chain_a`, `chain_b`")
    md.append("- `sequences`: list of two str (chain A, chain B 1-letter seqs)")
    md.append("- `res_ids`: list of two int lists (auth residue numbers in chain order)")
    md.append("- `contact_map`: torch int64 (Na, Nb), MINT/Mentos 10-bin Cb-Cb scheme; `-1` = missing Cb")
    md.append("- `distance_map`: torch float32 (Na, Nb), raw Cb-Cb distances in Angstrom")
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
