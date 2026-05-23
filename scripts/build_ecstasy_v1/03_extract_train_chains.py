"""Extract per-chain PDB files from MENTOS-train PDB CIFs for the Foldseek DB.

MENTOS-softnano's train split (PDB <= 2021-09-30) has 25,682 dimer entries.
CIFs live at /projects/u6jv/public/MENTOS/DATA/pdb/raw/cif_unzipped/<bucket>/<id>.cif
where <bucket> is a numeric hash bucket (000-...). We build a one-time
filename index, then for each train entry load the CIF and dump every
distinct protein chain as a PDB file.

Dedup is done after the fact by sequence (one Foldseek DB row per unique
chain sequence) to keep the search DB smaller. We still write all chain
PDB files because dumping is cheap and parallel.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ecstasy.structure import (  # noqa: E402
    load_local_cif,
    select_chain_atoms,
    split_into_chains,
    write_chain_pdb,
)

MENTOS_SPLIT_PARQUET = Path(
    "/projects/u6jv/public/MENTOS/DATA/pdb/processed/splits/seq_id_30/index.parquet"
)
MENTOS_CIF_ROOT = Path("/projects/u6jv/public/MENTOS/DATA/pdb/raw/cif_unzipped")
OUT_ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/train_db")
CHAINS_DIR = OUT_ROOT / "mentos_train_chains"
INDEX_PATH = OUT_ROOT / "mentos_train_chains_manifest.parquet"
MISSING_PATH = OUT_ROOT / "mentos_train_missing.parquet"

MIN_CHAIN_LEN = 40
DATE_CUTOFF = "2021-09-30"  # MENTOS-softnano train cutoff


def build_cif_index() -> dict[str, Path]:
    """Scan MENTOS cif_unzipped/ once to map pdb_id -> CIF path."""
    print(f"Building CIF index under {MENTOS_CIF_ROOT} ...")
    idx: dict[str, Path] = {}
    for bucket in sorted(MENTOS_CIF_ROOT.iterdir()):
        if not bucket.is_dir():
            continue
        for cif in bucket.glob("*.cif"):
            idx[cif.stem.lower()] = cif
    print(f"  indexed {len(idx)} CIFs")
    return idx


def process_one(args: tuple[str, str]) -> dict:
    """Worker: load CIF for one MENTOS-train entry, dump per-chain PDBs."""
    pdb_id, cif_path_s = args
    cif_path = Path(cif_path_s)
    out: dict = {
        "pdb_id": pdb_id,
        "status": "ok",
        "n_chains_written": 0,
        "chains": [],  # list of (chain_id, path, length, sequence)
        "error": None,
    }
    try:
        atoms = load_local_cif(cif_path)
        chains = split_into_chains(atoms)
        for c in chains:
            if len(c.res_ids) < MIN_CHAIN_LEN:
                continue
            chain_atoms = select_chain_atoms(atoms, c.chain_id)
            chain_path = CHAINS_DIR / f"{pdb_id}_{c.chain_id}.pdb"
            write_chain_pdb(chain_atoms, chain_path)
            out["chains"].append(
                {
                    "pdb_id": pdb_id,
                    "chain_id": c.chain_id,
                    "path": str(chain_path),
                    "length": int(len(c.res_ids)),
                    "sequence": c.sequence,
                }
            )
        out["n_chains_written"] = len(out["chains"])
        if out["n_chains_written"] == 0:
            out["status"] = "skip_no_long_chains"
    except Exception as e:  # noqa: BLE001
        out["status"] = "parse_error"
        out["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
    return out


def main() -> int:
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)

    # Train ID list (de-duplicated)
    split_df = pd.read_parquet(MENTOS_SPLIT_PARQUET)
    train_ids = sorted(split_df.loc[split_df["split"] == "train", "id"].unique().tolist())
    train_ids = [pid.lower() for pid in train_ids]
    print(f"MENTOS-train PDB IDs (seq_id_30): {len(train_ids)}")

    # CIF filename index (one-time scan)
    cif_idx = build_cif_index()

    # Match train IDs to CIF paths
    matched = []
    missing = []
    for pid in train_ids:
        p = cif_idx.get(pid)
        if p is None:
            missing.append(pid)
        else:
            matched.append((pid, str(p)))
    print(f"  matched to CIFs: {len(matched)}  missing: {len(missing)}")
    if missing:
        pd.DataFrame({"pdb_id": missing}).to_parquet(MISSING_PATH, index=False)
        print(f"  wrote missing list to {MISSING_PATH}")

    print(f"Extracting chains (parallel) ...")
    all_chains: list[dict] = []
    statuses: list[dict] = []
    nworkers = min(32, mp.cpu_count())
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        futures = {pool.submit(process_one, m): m[0] for m in matched}
        for n, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            all_chains.extend(res["chains"])
            statuses.append(
                {
                    "pdb_id": res["pdb_id"],
                    "status": res["status"],
                    "n_chains": res["n_chains_written"],
                    "error": res["error"],
                }
            )
            if n % 1000 == 0 or n == len(futures):
                ok = sum(1 for s in statuses if s["status"] == "ok")
                nch = sum(s["n_chains"] for s in statuses)
                print(f"  [{n}/{len(futures)}]  ok={ok}  chains_written={nch}")

    chains_df = pd.DataFrame(all_chains)
    status_df = pd.DataFrame(statuses)
    chains_df.to_parquet(INDEX_PATH, index=False)
    status_df.to_parquet(OUT_ROOT / "mentos_train_per_entry_log.parquet", index=False)

    print()
    print(f"  wrote {INDEX_PATH}  ({len(chains_df)} chain PDBs)")
    # Sequence-dedup statistic for the planning step
    n_unique_seqs = chains_df["sequence"].nunique()
    print(f"  unique chain sequences: {n_unique_seqs}")
    print(f"  status: {status_df['status'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
