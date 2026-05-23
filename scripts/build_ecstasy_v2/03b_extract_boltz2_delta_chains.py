"""Extract per-chain PDBs for the "Boltz-2 delta" — PDB entries released
between the Mentos cutoff (2021-09-30) and the Boltz-2 cutoff (2023-06-01).

The Boltz-2 paper does not publish an explicit training chain list, so we
approximate the Boltz-2 training chain set as

    boltz2_train_chains := mentos_train_chains  ∪  delta_chains

This is a strict superset of the actual Boltz-2 training set (it includes
every PDB released up to the cutoff, irrespective of clustering), and is
therefore conservative for the purpose of deleaking ecstasy_v2 against
Boltz-2 leakage.

This script handles only the delta (2021-10-01 .. 2023-05-31, inclusive).
Step 04 (`04_run_foldseek.py`) builds the union DB by createdb'ing both
chain directories together.

For each delta PDB ID we download the asymmetric-unit CIF from RCSB
(`files.rcsb.org/download/<id>.cif.gz`) — this matches what Mentos used
upstream (no bio-assembly expansion). We apply the same chain-length
filter (>= 40 res) as the v1 train-chain extraction.
"""

from __future__ import annotations

import gzip
import io
import json
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from ecstasy.structure import (  # noqa: E402
    parse_cif_bytes,
    select_chain_atoms,
    split_into_chains,
    write_chain_pdb,
)

OUT_ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v2/train_db")
DELTA_IDS_PATH = OUT_ROOT / "boltz2_delta_ids.txt"
DELTA_META_PATH = OUT_ROOT / "boltz2_delta_metadata.parquet"
CHAINS_DIR = OUT_ROOT / "boltz2_delta_chains"
INDEX_PATH = OUT_ROOT / "boltz2_delta_chains_manifest.parquet"
PER_ENTRY_LOG = OUT_ROOT / "boltz2_delta_per_entry_log.parquet"

# Inclusive bounds. The Mentos cutoff is 2021-09-30, so the delta starts
# the next day. The Boltz-2 cutoff is 2023-06-01 (use < not <=).
DELTA_START = "2021-10-01"
DELTA_END_EXCLUSIVE = "2023-06-01"

MIN_CHAIN_LEN = 40
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
CIF_URL_TMPL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


def search_delta_ids() -> list[str]:
    """All PDB entries released in [DELTA_START, DELTA_END_EXCLUSIVE) with
    >= 1 protein entity (we don't restrict to X-ray here — the Boltz-2 train
    set spans methods)."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater_or_equal",
                        "value": f"{DELTA_START}T00:00:00Z",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "less",
                        "value": f"{DELTA_END_EXCLUSIVE}T00:00:00Z",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": 1,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "results_content_type": ["experimental"],
            "sort": [
                {"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"}
            ],
        },
    }
    all_ids: list[str] = []
    start = 0
    page = 10000
    while True:
        query["request_options"]["paginate"] = {"start": start, "rows": page}
        req = Request(
            SEARCH_URL,
            data=json.dumps(query).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=120) as r:
            data = json.load(r)
        total = data.get("total_count", 0)
        results = data.get("result_set") or []
        all_ids.extend(item["identifier"].lower() for item in results)
        print(f"  fetched {len(all_ids)}/{total} delta IDs", flush=True)
        if len(all_ids) >= total or not results:
            break
        start += page
    seen = set()
    out = []
    for pid in all_ids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def download_cif(pdb_id: str) -> bytes:
    url = CIF_URL_TMPL.format(pdb_id=pdb_id.lower())
    with urlopen(url, timeout=60) as r:
        data = r.read()
    with gzip.open(io.BytesIO(data)) as gz:
        return gz.read()


def process_one(pdb_id: str) -> dict:
    pdb_id = pdb_id.lower()
    out: dict = {
        "pdb_id": pdb_id,
        "status": "ok",
        "n_chains_written": 0,
        "chains": [],
        "error": None,
    }
    try:
        cif_bytes = download_cif(pdb_id)
        atoms = parse_cif_bytes(cif_bytes)
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
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)

    if DELTA_IDS_PATH.exists():
        delta_ids = [
            ln.strip().lower() for ln in DELTA_IDS_PATH.read_text().splitlines() if ln.strip()
        ]
        print(f"Reusing cached delta ID list: {len(delta_ids)} IDs from {DELTA_IDS_PATH}")
    else:
        print(f"Querying RCSB for PDB entries released in [{DELTA_START}, {DELTA_END_EXCLUSIVE}) ...")
        delta_ids = search_delta_ids()
        DELTA_IDS_PATH.write_text("\n".join(delta_ids) + "\n")
        print(f"  wrote {DELTA_IDS_PATH}  ({len(delta_ids)} IDs)")

    print(f"Downloading + extracting chains for {len(delta_ids)} delta PDB IDs (parallel) ...")
    all_chains: list[dict] = []
    statuses: list[dict] = []
    # I/O-bound (network + CIF parsing); use more workers than CPU count.
    nworkers = min(64, (mp.cpu_count() or 8) * 4)
    with ProcessPoolExecutor(max_workers=nworkers) as pool:
        futures = {pool.submit(process_one, pid): pid for pid in delta_ids}
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
            if n % 500 == 0 or n == len(futures):
                ok = sum(1 for s in statuses if s["status"] == "ok")
                nch = sum(s["n_chains"] for s in statuses)
                print(f"  [{n}/{len(futures)}]  ok={ok}  chains_written={nch}", flush=True)

    chains_df = pd.DataFrame(all_chains)
    status_df = pd.DataFrame(statuses)
    chains_df.to_parquet(INDEX_PATH, index=False)
    status_df.to_parquet(PER_ENTRY_LOG, index=False)
    print()
    print(f"  wrote {INDEX_PATH}  ({len(chains_df)} chain PDBs)")
    print(f"  wrote {PER_ENTRY_LOG}")
    if len(chains_df):
        print(f"  unique chain sequences (delta-only): {chains_df['sequence'].nunique()}")
    print(f"  status counts: {status_df['status'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
