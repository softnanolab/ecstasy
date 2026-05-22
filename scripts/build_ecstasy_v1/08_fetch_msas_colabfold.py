"""Fetch paired MSAs from ColabFold's MMseqs2 API server for ecstasy_v1.

Notebook-faithful — mirrors MSA_Pairformer_with_MMseqs2.ipynb.

Server format (verified empirically): the returned pair.a3m is TWO mini-a3ms
concatenated, separated by a single NUL byte (\\x00) at byte level. Each chain's
mini-a3m starts with a `>101` (or `>102`, ...) label whose sequence is the
query, followed by `>UniRef100_XXX <header_fields>` paired hits in matched
order across chains. The same UniRef ID appears once per chain side (so we can
stitch row i of chain A with row i of chain B safely).

Save:
  msas/raw/<id>.tar.gz        # the raw colabfold tarball
  msas/<id>.a3m               # concatenated A3M (chain_A_matched || chain_B_matched)
                              # with insertions (lowercase) stripped so all rows have
                              # length L = len(A) + len(B).
  msas/_manifest.parquet      # entries with depth + chain_break
  msas/_failures.parquet      # failed entries with error
"""

from __future__ import annotations

import re
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from io import BytesIO

import pandas as pd
import requests

HOST = "https://api.colabfold.com"
MODE = "paircomplete-pairfilterprox_20"  # notebook default (genomic_distance=20)
CHECK_INTERVAL = 5  # seconds
SUBMIT_TIMEOUT = 30
POLL_TIMEOUT = 60
DOWNLOAD_TIMEOUT = 120
MAX_RETRIES = 6
RETRY_BACKOFF_BASE_S = 8  # exponential backoff for 429s
MAX_WAIT_S = 1800
# Concurrency: notebook does 1. We do 2 to amortize polling while staying
# well below the colabfold server's rate-limit (~10 concurrent / user).
MAX_CONCURRENT = 2

MANIFEST_IN = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/master/index.parquet")
OUT_DIR = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/msas")
RAW_DIR = OUT_DIR / "raw"
OUT_MANIFEST = OUT_DIR / "_manifest.parquet"
OUT_FAIL = OUT_DIR / "_failures.parquet"


def clean_sequence(seq: str) -> str:
    return re.sub(r"[^A-Z]", "", seq.upper())


def make_query_fasta(seqs: list[str]) -> str:
    return "\n".join(f">{i}\n{s}" for i, s in enumerate(seqs, start=101)) + "\n"


def submit_pair(session: requests.Session, fasta: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.post(
                f"{HOST}/ticket/pair",
                data={"q": fasta, "mode": MODE},
                timeout=SUBMIT_TIMEOUT,
            )
            if r.status_code == 429:
                wait = RETRY_BACKOFF_BASE_S * (2 ** attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
            j = r.json()
            if "id" not in j:
                raise RuntimeError(f"no id in submit response: {j}")
            return j["id"]
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"submit failed: {e}")
            time.sleep(RETRY_BACKOFF_BASE_S * (2 ** attempt))
    raise RuntimeError("submit retries exhausted")


def poll_until_done(session: requests.Session, job_id: str, max_wait_s: int = MAX_WAIT_S) -> None:
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            r = session.get(f"{HOST}/ticket/{job_id}", timeout=POLL_TIMEOUT)
            if r.status_code == 429:
                time.sleep(CHECK_INTERVAL * 2)
                continue
            r.raise_for_status()
            j = r.json()
            status = j.get("status", "")
            if status == "COMPLETE":
                return
            if status == "ERROR":
                raise RuntimeError(f"server reported ERROR for job {job_id}: {j}")
        except requests.RequestException:
            pass  # transient
        time.sleep(CHECK_INTERVAL)
    raise TimeoutError(f"job {job_id} did not complete within {max_wait_s}s")


def download_results(session: requests.Session, job_id: str) -> bytes:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(f"{HOST}/result/download/{job_id}", timeout=DOWNLOAD_TIMEOUT)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("download retries exhausted")


def _strip_inserts(seq: str) -> str:
    """A3M format: lowercase letters and '.' are insertions; remove."""
    return re.sub(r"[a-z.]", "", seq)


def parse_chain_a3m_bytes(raw: bytes) -> list[tuple[str, str]]:
    """Parse a single-chain a3m chunk (after NUL split). Returns [(header, seq), ...].

    The first record is the query (label `>101` or `>102`); we keep it as the
    row-0 entry. Subsequent records are UniRef hits. We strip lowercase inserts
    so all matched sequences have the same length.
    """
    text = raw.decode("utf-8", errors="replace").strip()
    entries: list[tuple[str, str]] = []
    cur_header: str | None = None
    cur_seq_parts: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur_header is not None:
                entries.append((cur_header, _strip_inserts("".join(cur_seq_parts))))
            cur_header = line[1:]
            cur_seq_parts = []
        elif line:
            cur_seq_parts.append(line)
    if cur_header is not None:
        entries.append((cur_header, _strip_inserts("".join(cur_seq_parts))))
    return entries


def parse_paired_a3m(tar_bytes: bytes) -> list[list[tuple[str, str]]]:
    """Extract pair.a3m, split on NUL, parse each chain. Returns list-per-chain."""
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("pair.a3m"):
                raw = tf.extractfile(m).read()
                break
        else:
            raise FileNotFoundError("pair.a3m not found in tarball")

    # Trim trailing NUL if any, then split on NUL
    raw = raw.rstrip(b"\x00")
    chunks = [c for c in raw.split(b"\x00") if c.strip()]
    return [parse_chain_a3m_bytes(c) for c in chunks]


def stitch_msa(per_chain: list[list[tuple[str, str]]], expected_chain_lens: list[int]) -> tuple[list[tuple[str, str]], int]:
    """Take min depth across chains, stitch row-by-row.

    Asserts each row in chain c has matched-only length == expected_chain_lens[c].
    Rows that don't match expected length are skipped.
    """
    depth = min(len(c) for c in per_chain)
    combined: list[tuple[str, str]] = []
    skipped = 0
    for r in range(depth):
        seqs = [per_chain[c][r][1] for c in range(len(per_chain))]
        if any(len(s) != expected_chain_lens[c] for c, s in enumerate(seqs)):
            skipped += 1
            continue
        # Use chain 0's header (or join all)
        header = "|".join(per_chain[c][r][0] for c in range(len(per_chain)))
        combined.append((header, "".join(seqs)))
    if skipped:
        print(f"  (stitch: skipped {skipped}/{depth} rows due to length mismatch)", flush=True)
    return combined, len(combined)


def write_a3m(entries: list[tuple[str, str]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        for header, seq in entries:
            f.write(f">{header}\n{seq}\n")


def process_one(row: dict) -> dict:
    entry_id = row["id"]
    seqs_raw = list(row["sequences"])
    seqs = [clean_sequence(s) for s in seqs_raw]
    chain_lens = [len(s) for s in seqs]
    out_path = OUT_DIR / f"{entry_id}.a3m"
    raw_path = RAW_DIR / f"{entry_id}.tar.gz"

    # Cache: if output a3m already exists and is non-trivial, skip
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with out_path.open() as f:
                depth = sum(1 for line in f if line.startswith(">"))
            if depth >= 2:
                return {"id": entry_id, "status": "cached", "depth": depth,
                        "chain_break": chain_lens[0]}
        except Exception:
            pass

    session = requests.Session()
    fasta = make_query_fasta(seqs)
    try:
        job_id = submit_pair(session, fasta)
        poll_until_done(session, job_id)
        tar_bytes = download_results(session, job_id)
        # Save raw tarball for offline re-parse
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(tar_bytes)
        per_chain = parse_paired_a3m(tar_bytes)
        if len(per_chain) != len(seqs):
            return {"id": entry_id, "status": "error",
                    "error": f"chains_parsed={len(per_chain)} != expected={len(seqs)}"}
        combined, depth = stitch_msa(per_chain, chain_lens)
        if depth < 2:
            return {"id": entry_id, "status": "error",
                    "error": f"depth too low after stitching ({depth})"}
        write_a3m(combined, out_path)
        return {"id": entry_id, "status": "ok", "depth": depth,
                "chain_break": chain_lens[0], "job_id": job_id}
    except Exception as e:  # noqa: BLE001
        return {"id": entry_id, "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "chain_break": chain_lens[0]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(MANIFEST_IN)
    print(f"Loaded {len(manifest)} dimers from {MANIFEST_IN}")
    rows = manifest.to_dict("records")

    results: list[dict] = []
    print(f"Concurrency: {MAX_CONCURRENT} (sequential w/ tiny overlap to be polite to api.colabfold.com)")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(process_one, r): r["id"] for r in rows}
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            extra = f"depth={res.get('depth')}" if res.get("status") in ("ok", "cached") else res.get("error", "")[:140]
            if done % 5 == 0 or done == len(futures) or res["status"] == "error":
                ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
                err = sum(1 for r in results if r["status"] == "error")
                print(f"  [{done}/{len(futures)}]  ok={ok}  err={err}  last={res['id']}/{res['status']}  {extra}", flush=True)

    res_df = pd.DataFrame(results)
    ok_df = res_df[res_df["status"].isin(["ok", "cached"])].copy()
    err_df = res_df[res_df["status"] == "error"].copy()
    ok_df.to_parquet(OUT_MANIFEST, index=False)
    err_df.to_parquet(OUT_FAIL, index=False)

    print()
    print("=== summary ===")
    print(f"  ok:        {len(ok_df)} / {len(rows)}")
    print(f"  errors:    {len(err_df)}")
    if len(ok_df):
        print(f"  depth distribution:\n{ok_df['depth'].describe().to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
