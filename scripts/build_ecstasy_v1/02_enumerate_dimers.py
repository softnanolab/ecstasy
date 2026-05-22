"""Enumerate candidate dimers from Boltz-2 validation_ids_v2.txt.

For each PDB ID:
  1. Pull bio-assembly 1 mmCIF from RCSB.
  2. Parse chains, drop non-amino-acid atoms.
  3. Enumerate every chain pair with backbone-atom contact <= 10 A.
  4. Apply quality filters (X-ray, resolution <= 3.5 A, min chain >= 40 res).
  5. Apply size filter (total length <= 1200 residues per dimer).
  6. Dump per-chain PDB files into <out>/candidates/chains/.
  7. Write candidates/dimers.parquet with interface metadata.

Also queries RCSB GraphQL for per-PDB metadata (deposit date, resolution,
method) and writes candidates/val_v2_metadata.parquet.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

# Import the shared utilities. We add the script dir to sys.path so this can be
# run from anywhere (e.g. via `python scripts/build_ecstasy_v1/02_...`).
from ecstasy.structure import (  # noqa: E402
    download_cif_assembly,
    enumerate_dimer_pairs,
    interface_residue_indices,
    parse_cif_bytes,
    select_chain_atoms,
    split_into_chains,
    write_chain_pdb,
)

VAL_V2_PATH = Path(
    "/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/candidates/validation_ids_v2.txt"
)
OUT_ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/candidates")
CHAINS_DIR = OUT_ROOT / "chains"
META_PATH = OUT_ROOT / "val_v2_metadata.parquet"
DIMERS_PATH = OUT_ROOT / "dimers.parquet"
PER_ENTRY_LOG = OUT_ROOT / "per_entry_log.parquet"

MIN_CHAIN_LEN = 40
MAX_DIMER_LEN = 1200
MIN_CONTACTS = 3
ASSEMBLY_ID = 1


def fetch_rcsb_metadata(pdb_ids: list[str]) -> pd.DataFrame:
    """Batch-query RCSB GraphQL for deposit_date / release_date / resolution / method."""
    BATCH = 50
    rows: list[dict] = []
    for i in range(0, len(pdb_ids), BATCH):
        batch = pdb_ids[i : i + BATCH]
        ids_str = '","'.join(batch)
        query = (
            '{ entries(entry_ids: ["' + ids_str + '"]) {'
            " rcsb_id"
            " rcsb_accession_info { deposit_date initial_release_date }"
            " exptl { method }"
            " rcsb_entry_info { resolution_combined polymer_entity_count_protein }"
            " } }"
        )
        req = Request(
            "https://data.rcsb.org/graphql",
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=60) as r:
            data = json.load(r)
        for e in data["data"]["entries"] or []:
            if e is None:
                continue
            method = (e.get("exptl") or [{}])[0].get("method") if e.get("exptl") else None
            res_list = (e.get("rcsb_entry_info") or {}).get(
                "resolution_combined"
            ) or []
            resolution = res_list[0] if res_list else None
            n_prot = (e.get("rcsb_entry_info") or {}).get(
                "polymer_entity_count_protein"
            )
            ai = e["rcsb_accession_info"] or {}
            rows.append(
                {
                    "pdb_id": e["rcsb_id"].lower(),
                    "deposit_date": (ai.get("deposit_date") or "")[:10],
                    "release_date": (ai.get("initial_release_date") or "")[:10],
                    "method": method,
                    "resolution": resolution,
                    "n_protein_entities": n_prot,
                }
            )
    return pd.DataFrame(rows)


def passes_quality(meta_row: pd.Series) -> tuple[bool, str]:
    """Return (passes, reason_if_not)."""
    if meta_row["method"] not in ("X-RAY DIFFRACTION",):
        return False, f"method={meta_row['method']}"
    if meta_row["resolution"] is None or meta_row["resolution"] > 3.5:
        return False, f"resolution={meta_row['resolution']}"
    return True, ""


def process_one(args: tuple[str, dict]) -> dict:
    """Worker: download + enumerate dimers for a single PDB ID."""
    pdb_id, meta = args
    pdb_id = pdb_id.lower()
    out: dict = {
        "pdb_id": pdb_id,
        "status": "ok",
        "n_chains": 0,
        "n_dimers": 0,
        "dimer_rows": [],
        "chain_writes": [],  # list of (chain_id, out_path, sequence)
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

        # short-chain filter applied per-chain before enumerating pairs
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

        # determine which chains we actually need to dump (those appearing in a kept dimer)
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

    pdb_ids = [
        ln.strip().lower()
        for ln in VAL_V2_PATH.read_text().splitlines()
        if ln.strip()
    ]
    print(f"Loaded {len(pdb_ids)} PDB IDs from {VAL_V2_PATH}")

    print("Querying RCSB metadata...")
    meta_df = fetch_rcsb_metadata(pdb_ids)
    meta_df.to_parquet(META_PATH, index=False)
    print(f"  wrote {META_PATH}  ({len(meta_df)} rows)")

    # apply quality filter at the PDB-entry level
    keep_mask = meta_df.apply(
        lambda r: passes_quality(r)[0], axis=1
    )
    kept_ids = meta_df.loc[keep_mask, "pdb_id"].tolist()
    rejected_n = len(meta_df) - len(kept_ids)
    print(
        f"  quality filter: kept {len(kept_ids)}/{len(meta_df)} "
        f"({rejected_n} rejected by X-ray/resolution)"
    )

    meta_by_id = {r["pdb_id"]: r.to_dict() for _, r in meta_df.iterrows()}

    print(f"Enumerating dimers for {len(kept_ids)} PDB IDs (parallel)...")
    all_dimers: list[dict] = []
    all_chains: list[dict] = []
    per_entry: list[dict] = []
    nworkers = min(16, mp.cpu_count())
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        futures = {
            pool.submit(process_one, (pid, meta_by_id[pid])): pid for pid in kept_ids
        }
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
            if n % 50 == 0 or n == len(futures):
                ok = sum(1 for p in per_entry if p["status"] == "ok")
                ndim = sum(p["n_dimers"] for p in per_entry)
                print(f"  [{n}/{len(futures)}]  ok={ok}  total_dimers={ndim}")

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
