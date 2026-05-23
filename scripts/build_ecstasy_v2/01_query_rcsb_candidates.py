"""Query RCSB Search API for ecstasy_v2 candidate PDB IDs.

Selection criteria (mirrors the v2 issue spec):
  - initial_release_date >= 2023-06-01 (Boltz-2 training cutoff)
  - experimental method: X-ray diffraction
  - polymer_entity_count_protein >= 2

Also pulls per-entry metadata (deposit_date, release_date, resolution,
method, n_protein_entities) so downstream scripts don't need a second
RCSB roundtrip for the same IDs.

Outputs:
  candidates/pdb_ids.txt              # newline-separated IDs (lowercase)
  candidates/rcsb_metadata.parquet    # same schema as v1 val_v2_metadata.parquet
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

OUT_ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v2/candidates")
IDS_PATH = OUT_ROOT / "pdb_ids.txt"
META_PATH = OUT_ROOT / "rcsb_metadata.parquet"

RELEASE_CUTOFF = "2023-06-01"  # Boltz-2 training cutoff
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"


def search_pdb_ids() -> list[str]:
    """Paginate RCSB Search API for entries matching the v2 criteria."""
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
                        "value": f"{RELEASE_CUTOFF}T00:00:00Z",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": 2,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"}],
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
        ids = [item["identifier"].lower() for item in results]
        all_ids.extend(ids)
        print(f"  fetched {len(all_ids)}/{total} IDs", flush=True)
        if len(all_ids) >= total or not results:
            break
        start += page
    # dedupe preserving order
    seen = set()
    out = []
    for pid in all_ids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def fetch_rcsb_metadata(pdb_ids: list[str]) -> pd.DataFrame:
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
            GRAPHQL_URL,
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=120) as r:
            data = json.load(r)
        for e in data["data"]["entries"] or []:
            if e is None:
                continue
            method = (e.get("exptl") or [{}])[0].get("method") if e.get("exptl") else None
            res_list = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or []
            resolution = res_list[0] if res_list else None
            n_prot = (e.get("rcsb_entry_info") or {}).get("polymer_entity_count_protein")
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
        print(f"  metadata: {len(rows)}/{len(pdb_ids)}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Searching RCSB for X-ray protein entries released >= {RELEASE_CUTOFF} ...")
    pdb_ids = search_pdb_ids()
    print(f"  total candidate PDB IDs: {len(pdb_ids)}")

    IDS_PATH.write_text("\n".join(pdb_ids) + "\n")
    print(f"  wrote {IDS_PATH}")

    print(f"Fetching per-entry metadata for {len(pdb_ids)} IDs ...")
    meta_df = fetch_rcsb_metadata(pdb_ids)
    meta_df.to_parquet(META_PATH, index=False)
    print(f"  wrote {META_PATH}  ({len(meta_df)} rows)")

    print()
    print("=== summary ===")
    print(f"PDB IDs:            {len(pdb_ids)}")
    print(f"release-date range: {meta_df['release_date'].min()} .. {meta_df['release_date'].max()}")
    print(f"method counts:      {meta_df['method'].value_counts().to_dict()}")
    print(f"n_protein_entities: min={meta_df['n_protein_entities'].min()} max={meta_df['n_protein_entities'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
