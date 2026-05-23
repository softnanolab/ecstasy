"""Make Mentos-train (formerly MINT-train, PDB <= 2021-09-30) chain PDBs
available under the v2 layout.

If `build_ecstasy_v1/03_extract_train_chains.py` has already populated
`ecstasy_v1/train_db/mint_train_chains/`, we just symlink that directory
into the v2 tree so Foldseek can reuse the same DB without redoing the
~25k-entry CIF parse. Otherwise we extract from scratch with identical
logic to v1.
"""

from __future__ import annotations

import multiprocessing as mp
import os
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
    "/projects/u6jv/public/MINT/DATA/pdb/processed/splits/seq_id_30/index.parquet"
)
MENTOS_CIF_ROOT = Path("/projects/u6jv/public/MINT/DATA/pdb/raw/cif_unzipped")

V1_TRAIN_DIR = Path(
    "/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/train_db/mint_train_chains"
)
V1_MANIFEST = Path(
    "/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/train_db/mint_train_chains_manifest.parquet"
)

OUT_ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v2/train_db")
CHAINS_DIR = OUT_ROOT / "mentos_train_chains"
INDEX_PATH = OUT_ROOT / "mentos_train_chains_manifest.parquet"
MISSING_PATH = OUT_ROOT / "mentos_train_missing.parquet"

MIN_CHAIN_LEN = 40
DATE_CUTOFF = "2021-09-30"


def try_reuse_v1() -> bool:
    if not V1_TRAIN_DIR.is_dir() or not V1_MANIFEST.exists():
        return False
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if CHAINS_DIR.exists() or CHAINS_DIR.is_symlink():
        print(f"  {CHAINS_DIR} already exists; leaving as-is")
    else:
        os.symlink(V1_TRAIN_DIR, CHAINS_DIR)
        print(f"  symlinked {CHAINS_DIR} -> {V1_TRAIN_DIR}")
    if not INDEX_PATH.exists():
        df = pd.read_parquet(V1_MANIFEST)
        df.to_parquet(INDEX_PATH, index=False)
        print(f"  copied manifest -> {INDEX_PATH}  ({len(df)} rows)")
    return True


def build_cif_index() -> dict[str, Path]:
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
    pdb_id, cif_path_s = args
    cif_path = Path(cif_path_s)
    out: dict = {
        "pdb_id": pdb_id,
        "status": "ok",
        "n_chains_written": 0,
        "chains": [],
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


def extract_from_scratch() -> int:
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    split_df = pd.read_parquet(MENTOS_SPLIT_PARQUET)
    train_ids = sorted(split_df.loc[split_df["split"] == "train", "id"].unique().tolist())
    train_ids = [pid.lower() for pid in train_ids]
    print(f"Mentos-train PDB IDs (seq_id_30): {len(train_ids)}")

    cif_idx = build_cif_index()
    matched, missing = [], []
    for pid in train_ids:
        p = cif_idx.get(pid)
        if p is None:
            missing.append(pid)
        else:
            matched.append((pid, str(p)))
    print(f"  matched: {len(matched)}  missing: {len(missing)}")
    if missing:
        pd.DataFrame({"pdb_id": missing}).to_parquet(MISSING_PATH, index=False)

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
                print(f"  [{n}/{len(futures)}]  ok={ok}  chains_written={nch}", flush=True)

    chains_df = pd.DataFrame(all_chains)
    status_df = pd.DataFrame(statuses)
    chains_df.to_parquet(INDEX_PATH, index=False)
    status_df.to_parquet(OUT_ROOT / "mentos_train_per_entry_log.parquet", index=False)
    print(f"  wrote {INDEX_PATH}  ({len(chains_df)} chain PDBs)")
    print(f"  unique chain sequences: {chains_df['sequence'].nunique()}")
    print(f"  status: {status_df['status'].value_counts().to_dict()}")
    return 0


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Trying to reuse v1's mint_train_chains/ ...")
    if try_reuse_v1():
        print("  reused v1 outputs")
        return 0
    print("  v1 outputs unavailable; extracting from scratch")
    return extract_from_scratch()


if __name__ == "__main__":
    raise SystemExit(main())
