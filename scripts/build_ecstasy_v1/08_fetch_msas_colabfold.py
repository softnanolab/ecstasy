"""Fetch paired MSAs from ColabFold's MMseqs2 API server for ecstasy_v1.

Thin orchestration over `ecstasy.msa.colabfold` — that module owns the
HTTP client, paired-A3M parsing, and stitching. This script just:

  - loops over `master/index.parquet`
  - submits one job per dimer (2 concurrent for politeness)
  - caches raw tarballs to `msas/raw/<id>.tar.gz` for offline re-filter
  - writes the stitched A3M to `msas/<id>.a3m` (notebook format,
    concatenated chain-A + chain-B per row, insertions stripped)

Notebook fidelity: mirrors `MSA_Pairformer_with_MMseqs2.ipynb` exactly.
The `save_msa` post-filters (cov=75 / id=15 / Δgene=1) are applied
separately by `11_apply_notebook_filters.py` over the cached tarballs;
this script does no filtering so re-running `11` with different
thresholds doesn't require re-querying the server.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from ecstasy.msa.colabfold import (
    clean_sequence,
    download_results,
    make_query_fasta,
    parse_paired_a3m_bytes,
    poll_until_done,
    stitch_paired_msa,
    submit_pair,
    write_a3m,
)

MANIFEST_IN = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/master/index.parquet")
OUT_DIR = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/msas")
RAW_DIR = OUT_DIR / "raw"
OUT_MANIFEST = OUT_DIR / "_manifest.parquet"
OUT_FAIL = OUT_DIR / "_failures.parquet"

# Concurrency: notebook does 1. We do 2 to amortize polling while staying
# well below the colabfold server's rate-limit (~10 concurrent / user).
MAX_CONCURRENT = 2


def process_one(row: dict) -> dict:
    entry_id = row["id"]
    seqs = [clean_sequence(s) for s in row["sequences"]]
    chain_lens = [len(s) for s in seqs]
    out_path = OUT_DIR / f"{entry_id}.a3m"
    raw_path = RAW_DIR / f"{entry_id}.tar.gz"
    chain_break = chain_lens[0]

    # Cache: if non-trivial output already exists, skip
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with out_path.open() as f:
                depth = sum(1 for line in f if line.startswith(">"))
            if depth >= 2:
                return {"id": entry_id, "status": "cached", "depth": depth,
                        "chain_break": chain_break}
        except Exception:
            pass

    session = requests.Session()
    try:
        job_id = submit_pair(session, make_query_fasta(seqs))
        poll_until_done(session, job_id)
        tar_bytes = download_results(session, job_id)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(tar_bytes)  # cache for offline re-filter

        per_chain = parse_paired_a3m_bytes(tar_bytes, extract_metadata=False)
        if len(per_chain) != len(seqs):
            return {"id": entry_id, "status": "error",
                    "error": f"chains_parsed={len(per_chain)} != expected={len(seqs)}"}
        combined = stitch_paired_msa(per_chain, chain_lens)
        if len(combined) < 2:
            return {"id": entry_id, "status": "error",
                    "error": f"depth too low after stitching ({len(combined)})"}
        write_a3m(combined, out_path)
        return {"id": entry_id, "status": "ok", "depth": len(combined),
                "chain_break": chain_break, "job_id": job_id}
    except Exception as e:  # noqa: BLE001
        return {"id": entry_id, "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "chain_break": chain_break}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(MANIFEST_IN)
    rows = manifest.to_dict("records")
    print(f"Loaded {len(rows)} dimers from {MANIFEST_IN}")
    print(f"Concurrency: {MAX_CONCURRENT} (be polite to api.colabfold.com)")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(process_one, r): r["id"] for r in rows}
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            extra = f"depth={res.get('depth')}" if res["status"] in ("ok", "cached") else res.get("error", "")[:140]
            if done % 5 == 0 or done == len(futures) or res["status"] == "error":
                ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
                err = sum(1 for r in results if r["status"] == "error")
                print(f"  [{done}/{len(futures)}]  ok={ok}  err={err}  last={res['id']}/{res['status']}  {extra}", flush=True)

    res_df = pd.DataFrame(results)
    ok_df = res_df[res_df["status"].isin(["ok", "cached"])]
    err_df = res_df[res_df["status"] == "error"]
    ok_df.to_parquet(OUT_MANIFEST, index=False)
    err_df.to_parquet(OUT_FAIL, index=False)

    print()
    print("=== summary ===")
    print(f"  ok:     {len(ok_df)} / {len(rows)}")
    print(f"  errors: {len(err_df)}")
    if len(ok_df):
        print(f"  depth distribution:\n{ok_df['depth'].describe().to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
