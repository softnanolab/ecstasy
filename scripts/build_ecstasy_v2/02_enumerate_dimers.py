"""Enumerate candidate dimers from the v2 RCSB candidate list (post-2023-06-01).

Mirrors `build_ecstasy_v1/02_enumerate_dimers.py` end-to-end. The only
differences are:
  - input PDB IDs come from `01_query_rcsb_candidates.py` output
  - output paths live under `/projects/u6jv/ecstasy/benchmarks/ecstasy_v2/`
  - the RCSB metadata parquet is reused from step 01 (no second roundtrip)

Filters (identical to v1):
  - X-ray, resolution <= 3.5 Å
  - min chain length 40 res
  - dimer total length <= 1200 res
  - >= 3 inter-chain backbone-atom contacts (<= 10 Å)
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError

import pandas as pd

from ecstasy.structure import (  # noqa: E402
    download_cif_assembly,
    enumerate_dimer_pairs,
    interface_residue_indices,
    parse_cif_bytes,
    select_chain_atoms,
    split_into_chains,
    write_chain_pdb,
)

OUT_ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v2/candidates")
IDS_PATH = OUT_ROOT / "pdb_ids.txt"
META_PATH = OUT_ROOT / "rcsb_metadata.parquet"
CHAINS_DIR = OUT_ROOT / "chains"
DIMERS_PATH = OUT_ROOT / "dimers.parquet"
PER_ENTRY_LOG = OUT_ROOT / "per_entry_log.parquet"

MIN_CHAIN_LEN = 40
MAX_DIMER_LEN = 1200
MIN_CONTACTS = 3
ASSEMBLY_ID = 1


def passes_quality(meta_row: pd.Series) -> tuple[bool, str]:
    if meta_row["method"] not in ("X-RAY DIFFRACTION",):
        return False, f"method={meta_row['method']}"
    if meta_row["resolution"] is None or meta_row["resolution"] > 3.5:
        return False, f"resolution={meta_row['resolution']}"
    return True, ""


def process_one(args: tuple[str, dict]) -> dict:
    pdb_id, _meta = args
    pdb_id = pdb_id.lower()
    out: dict = {
        "pdb_id": pdb_id,
        "status": "ok",
        "n_chains": 0,
        "n_dimers": 0,
        "dimer_rows": [],
        "chain_writes": [],
        "error": None,
    }
    try:
        cif_bytes = download_cif_assembly(pdb_id, ASSEMBLY_ID)
        atoms = parse_cif_bytes(cif_bytes)
        chains = split_into_chains(atoms)
        out["n_chains"] = len(chains)
        if len(chains) < 2:
            out["status"] = "skip_too_few_chains"
            return out

        short_idxs = {i for i, c in enumerate(chains) if len(c.res_ids) < MIN_CHAIN_LEN}

        for i, j in enumerate_dimer_pairs(chains):
            if i in short_idxs or j in short_idxs:
                continue
            c_a, c_b = chains[i], chains[j]
            total_len = len(c_a.res_ids) + len(c_b.res_ids)
            if total_len > MAX_DIMER_LEN:
                continue
            ia, ib, npair = interface_residue_indices(c_a, c_b)
            if npair < MIN_CONTACTS:
                continue
            out["dimer_rows"].append(
                {
                    "pdb_id": pdb_id,
                    "assembly_id": ASSEMBLY_ID,
                    "chain_a": c_a.chain_id,
                    "chain_b": c_b.chain_id,
                    "len_a": int(len(c_a.res_ids)),
                    "len_b": int(len(c_b.res_ids)),
                    "seq_a": c_a.sequence,
                    "seq_b": c_b.sequence,
                    "interface_idx_a": ia.tolist(),
                    "interface_idx_b": ib.tolist(),
                    "n_interface_residues_a": int(len(ia)),
                    "n_interface_residues_b": int(len(ib)),
                    "n_contact_pairs": int(npair),
                    "is_homodimer": bool(c_a.sequence == c_b.sequence),
                }
            )

        kept_chain_ids = {r["chain_a"] for r in out["dimer_rows"]} | {
            r["chain_b"] for r in out["dimer_rows"]
        }
        for c in chains:
            if c.chain_id not in kept_chain_ids:
                continue
            chain_atoms = select_chain_atoms(atoms, c.chain_id)
            chain_path = CHAINS_DIR / f"{pdb_id}_{ASSEMBLY_ID}_{c.chain_id}.pdb"
            write_chain_pdb(chain_atoms, chain_path)
            out["chain_writes"].append(
                {
                    "pdb_id": pdb_id,
                    "chain_id": c.chain_id,
                    "path": str(chain_path),
                    "length": int(len(c.res_ids)),
                    "sequence": c.sequence,
                }
            )

        out["n_dimers"] = len(out["dimer_rows"])
    except HTTPError as e:
        out["status"] = "http_error"
        out["error"] = f"{e.code} {e.reason}"
    except URLError as e:
        out["status"] = "url_error"
        out["error"] = str(e.reason)
    except Exception as e:  # noqa: BLE001
        out["status"] = "parse_error"
        out["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
    return out


def main() -> int:
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)

    pdb_ids = [ln.strip().lower() for ln in IDS_PATH.read_text().splitlines() if ln.strip()]
    print(f"Loaded {len(pdb_ids)} candidate PDB IDs from {IDS_PATH}")

    meta_df = pd.read_parquet(META_PATH)
    print(f"Loaded metadata for {len(meta_df)} entries from {META_PATH}")

    keep_mask = meta_df.apply(lambda r: passes_quality(r)[0], axis=1)
    kept_ids = meta_df.loc[keep_mask, "pdb_id"].tolist()
    rejected_n = len(meta_df) - len(kept_ids)
    print(
        f"  quality filter: kept {len(kept_ids)}/{len(meta_df)} "
        f"({rejected_n} rejected by method/resolution)"
    )

    meta_by_id = {r["pdb_id"]: r.to_dict() for _, r in meta_df.iterrows()}

    print(f"Enumerating dimers for {len(kept_ids)} PDB IDs (parallel) ...")
    all_dimers: list[dict] = []
    all_chains: list[dict] = []
    per_entry: list[dict] = []
    nworkers = min(16, mp.cpu_count())
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        futures = {pool.submit(process_one, (pid, meta_by_id[pid])): pid for pid in kept_ids}
        for n, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            all_dimers.extend(res["dimer_rows"])
            all_chains.extend(res["chain_writes"])
            per_entry.append(
                {
                    "pdb_id": res["pdb_id"],
                    "status": res["status"],
                    "n_chains": res["n_chains"],
                    "n_dimers": res["n_dimers"],
                    "error": res["error"],
                }
            )
            if n % 200 == 0 or n == len(futures):
                ok = sum(1 for p in per_entry if p["status"] == "ok")
                ndim = sum(p["n_dimers"] for p in per_entry)
                print(f"  [{n}/{len(futures)}]  ok={ok}  total_dimers={ndim}", flush=True)

    dimers_df = pd.DataFrame(all_dimers)
    chains_df = pd.DataFrame(all_chains)
    per_entry_df = pd.DataFrame(per_entry)
    dimers_df.to_parquet(DIMERS_PATH, index=False)
    chains_df.to_parquet(OUT_ROOT / "chains_manifest.parquet", index=False)
    per_entry_df.to_parquet(PER_ENTRY_LOG, index=False)

    print()
    print(f"  wrote {DIMERS_PATH}            ({len(dimers_df)} dimers)")
    print(f"  wrote chains_manifest.parquet  ({len(chains_df)} chain PDBs)")
    print(f"  wrote {PER_ENTRY_LOG}          (per-PDB processing log)")
    print()
    print("=== summary ===")
    print(f"PDB IDs queried:          {len(pdb_ids)}")
    print(f"  rejected by quality:    {rejected_n}")
    print(f"  processed:              {len(kept_ids)}")
    print(f"  status counts:          {per_entry_df['status'].value_counts().to_dict()}")
    print(f"Dimers (post-filter):     {len(dimers_df)}")
    if len(dimers_df):
        print(
            f"  homodimers:             {int(dimers_df['is_homodimer'].sum())} "
            f"({100 * dimers_df['is_homodimer'].mean():.1f}%)"
        )
        print(
            f"  total residues range:   [{(dimers_df['len_a'] + dimers_df['len_b']).min()}, "
            f"{(dimers_df['len_a'] + dimers_df['len_b']).max()}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
