"""Apply colab notebook's `save_msa` post-filters to cached raw MSA tarballs.

Reads cached raw colabfold tarballs from `msas/raw/<id>.tar.gz` and applies
the notebook's `save_msa` defaults via `ecstasy.msa.colabfold`:

    min_coverage      = 0.75
    min_identity      = 0.15
    max_genomic_distance = 1   (operon-proximity filter)

Writes filtered + stitched A3M to `msas_filtered/<id>.a3m`. The inference
runner picks the right MSA via its `--msas-dir` flag.

This script does no network I/O; rerun cheaply with different filter
thresholds by editing `FILTERS` below — the cached tarballs in `msas/raw/`
mean no colabfold server hit is required.
"""

from __future__ import annotations

import multiprocessing as mp
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ecstasy.msa.colabfold import (
    SaveMsaFilters,
    apply_save_msa_filters,
    parse_paired_a3m_bytes,
    write_a3m,
)

MASTER_INDEX = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/master/index.parquet")
RAW_DIR = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/msas/raw")
OUT_DIR = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/msas_filtered")
OUT_MANIFEST = OUT_DIR / "_manifest.parquet"

# Notebook UI cell defaults (cov=75%, id=15%, Δgene=1, neighbor_stitching=True).
FILTERS = SaveMsaFilters(min_coverage=0.75, min_identity=0.15, max_genomic_distance=1)


def process_one(row: dict) -> dict:
    eid = row["id"]
    seqs = list(row["sequences"])
    chain_lens = [len(s) for s in seqs]
    chain_break = chain_lens[0]
    raw_path = RAW_DIR / f"{eid}.tar.gz"
    out_path = OUT_DIR / f"{eid}.a3m"
    if not raw_path.exists():
        return {"id": eid, "status": "missing_raw", "depth": 0, "chain_break": chain_break}
    try:
        per_chain = parse_paired_a3m_bytes(raw_path.read_bytes(), extract_metadata=True)
        if len(per_chain) != len(seqs):
            return {"id": eid, "status": f"wrong_n_chains_{len(per_chain)}",
                    "depth": 0, "chain_break": chain_break}
        kept, stats = apply_save_msa_filters(per_chain, chain_lens, FILTERS)
        if stats.kept < 2:
            return {"id": eid, "status": "too_shallow_after_filter",
                    "depth": stats.kept, "raw_depth": stats.raw, "chain_break": chain_break,
                    **vars(stats)}
        write_a3m(kept, out_path)
        return {"id": eid, "status": "ok", "depth": stats.kept, "raw_depth": stats.raw,
                "chain_break": chain_break, **vars(stats)}
    except (tarfile.TarError, FileNotFoundError) as e:
        return {"id": eid, "status": "tar_error", "error": f"{type(e).__name__}: {e}",
                "chain_break": chain_break}
    except Exception as e:  # noqa: BLE001
        return {"id": eid, "status": "error", "error": f"{type(e).__name__}: {e}",
                "chain_break": chain_break}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(MASTER_INDEX)
    rows = manifest.to_dict("records")
    print(f"Filtering {len(rows)} entries with "
          f"cov>={FILTERS.min_coverage}, id>={FILTERS.min_identity}, "
          f"Δgene<={FILTERS.max_genomic_distance}")

    results = []
    with ProcessPoolExecutor(max_workers=min(8, mp.cpu_count())) as pool:
        futs = {pool.submit(process_one, r): r["id"] for r in rows}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if n % 25 == 0 or n == len(futs):
                ok = sum(1 for x in results if x["status"] == "ok")
                shallow = sum(1 for x in results if x["status"] == "too_shallow_after_filter")
                print(f"  [{n}/{len(futs)}]  ok={ok}  too_shallow={shallow}  last={r['id']}/{r['status']}", flush=True)

    df = pd.DataFrame(results)
    df.to_parquet(OUT_MANIFEST, index=False)
    print(f"\nwrote {OUT_MANIFEST}")
    print()
    print("=== status counts ===")
    print(df["status"].value_counts().to_string())
    ok = df[df["status"] == "ok"]
    if len(ok):
        print()
        print("=== depth distribution after filter ===")
        print(ok["depth"].describe().to_string())
        print()
        print("=== filter breakdown (mean per entry) ===")
        for col in ["raw", "filtered_cov", "filtered_id", "filtered_dist", "filtered_len", "kept"]:
            if col in ok.columns:
                print(f"  {col:18s}  mean={ok[col].mean():.1f}  median={ok[col].median():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
