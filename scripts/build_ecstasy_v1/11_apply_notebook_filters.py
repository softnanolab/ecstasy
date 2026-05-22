"""Port of the colab notebook's save_msa filters.

Reads raw colabfold tarballs from msas/raw/<id>.tar.gz, parses pair.a3m
(NUL-delimited per chain, FASTA per chain), and applies the notebook's
default `save_msa` post-filters with:

    min_coverage      = 0.75
    min_identity      = 0.15
    max_genomic_distance = 1   (operon-proximity filter — keep only rows where
                                 the two UniRef IDs come from adjacent genes)

Writes filtered + stitched A3M to msas_filtered/<id>.a3m in the same
concatenated format the inference runner expects.

The UniProt-ID-to-number conversion, _calculate_genomic_distances, and the
parsing helpers are direct ports of the notebook's ColabFoldPairedMSA class
methods (`MSA_Pairformer_with_MMseqs2.ipynb`, cell 2).
"""

from __future__ import annotations

import io
import re
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from string import ascii_uppercase
from typing import Optional

import pandas as pd

MASTER_INDEX = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/master/index.parquet")
RAW_DIR = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/msas/raw")
OUT_DIR = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/msas_filtered")
OUT_MANIFEST = OUT_DIR / "_manifest.parquet"

MIN_COVERAGE = 0.75
MIN_IDENTITY = 0.15
MAX_GENOMIC_DISTANCE = 1


# ---- UniProt-ID-to-number tables (ported verbatim from the notebook) ----

_PA = {a: 0 for a in ascii_uppercase}
for _a in ["O", "P", "Q"]:
    _PA[_a] = 1

_MA = [[{} for _ in range(6)], [{} for _ in range(6)]]
for _n, _t in enumerate(range(10)):
    for _i in [0, 1]:
        for _j in [0, 4]:
            _MA[_i][_j][str(_t)] = _n

for _n, _t in enumerate(list(ascii_uppercase) + list(range(10))):
    for _i in [0, 1]:
        for _j in [1, 2]:
            _MA[_i][_j][str(_t)] = _n
    _MA[1][3][str(_t)] = _n

for _n, _t in enumerate(ascii_uppercase):
    _MA[0][3][str(_t)] = _n
    for _i in [0, 1]:
        _MA[_i][5][str(_t)] = _n

_UPI_ENCODING = {}
_HEX = list(range(10)) + ["A", "B", "C", "D", "E", "F"]
for _n, _c in enumerate(_HEX):
    _UPI_ENCODING[str(_c)] = _n


def _extract_uniprot_id(header: str) -> str:
    pos = header.find("UniRef")
    if pos == -1:
        return ""
    start = header.find("_", pos)
    if start == -1:
        return ""
    start += 1
    end = start
    while end < len(header) and header[end] not in " _\t":
        end += 1
    uid = header[start:end]
    if len(uid) >= 3 and uid[:3] == "UPI":
        return uid
    if len(uid) not in (6, 10):
        return ""
    if not uid[0].isalpha():
        return ""
    return uid


def _uniprot_to_number(uniprot_ids: list[str]) -> list[int]:
    numbers: list[int] = []
    for uni in uniprot_ids:
        if not uni or not uni[0].isalpha():
            numbers.append(0)
            continue
        if uni.startswith("UPI") and len(uni) == 13:
            hex_part = uni[3:]
            num = 0
            tot = 1
            for u in reversed(hex_part):
                if str(u) in _UPI_ENCODING:
                    num += _UPI_ENCODING[str(u)] * tot
                    tot *= 16
                else:
                    num = 0
                    break
            numbers.append(num + 10**15)
            continue
        p = _PA.get(uni[0], 0)
        tot, num = 1, 0
        if len(uni) == 10:
            for n, u in enumerate(reversed(uni[-4:])):
                if str(u) in _MA[p][n]:
                    num += _MA[p][n][str(u)] * tot
                    tot *= len(_MA[p][n].keys())
        for n, u in enumerate(reversed(uni[:6])):
            if n < len(_MA[p]) and str(u) in _MA[p][n]:
                num += _MA[p][n][str(u)] * tot
                tot *= len(_MA[p][n].keys())
        numbers.append(num)
    return numbers


def _calc_distances(uniprot_nums: list[int]) -> list[int]:
    out = []
    for i in range(1, len(uniprot_nums)):
        if uniprot_nums[i - 1] and uniprot_nums[i]:
            out.append(abs(uniprot_nums[i] - uniprot_nums[i - 1]))
        else:
            out.append(-1)
    return out


def _strip_inserts(seq: str) -> str:
    return re.sub(r"[a-z.]", "", seq)


def _parse_msa_chunk(text: str) -> list[dict]:
    """Parse a single chain's a3m chunk into [(header, sequence, coverage, identity, uid, uniprot_num, has_uniref), ...]."""
    entries: list[dict] = []
    is_first = True
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        header = lines[i]
        seq_parts = []
        i += 1
        while i < len(lines) and not lines[i].startswith(">"):
            if lines[i].strip():
                seq_parts.append(lines[i].rstrip())
            i += 1
        sequence = _strip_inserts("".join(seq_parts))
        # parse metadata
        uid = _extract_uniprot_id(header)
        has_uniref = "UniRef" in header
        uniprot_num = _uniprot_to_number([uid])[0] if uid else 0
        if is_first:
            coverage = 1.0
            identity = 1.0
            is_first = False
        else:
            coverage = 0.0
            identity = 0.0
            parts = header.split("\t")
            if len(parts) >= 10:
                try:
                    identity = float(parts[2])
                    q_start = int(parts[4])
                    q_end = int(parts[5])
                    q_len = int(parts[6])
                    coverage = (q_end - q_start + 1) / q_len
                except Exception:  # noqa: BLE001
                    pass
        entries.append(
            {
                "header": header.lstrip(">"),
                "sequence": sequence,
                "coverage": coverage,
                "identity": identity,
                "uid": uid,
                "uniprot_num": uniprot_num,
                "has_uniref": has_uniref,
            }
        )
    return entries


def _stitch_and_filter(per_chain: list[list[dict]], expected_lens: list[int]) -> tuple[list[tuple[str, str]], dict]:
    depth = min(len(c) for c in per_chain)
    kept: list[tuple[str, str]] = []
    stats = {"raw": depth, "filtered_cov": 0, "filtered_id": 0, "filtered_dist": 0, "filtered_len": 0, "kept": 0}
    for r in range(depth):
        entries = [per_chain[c][r] for c in range(len(per_chain))]
        is_query = (r == 0)
        sequences = [e["sequence"] for e in entries]
        # length sanity (matched-only must equal expected)
        if any(len(s) != expected_lens[c] for c, s in enumerate(sequences)):
            stats["filtered_len"] += 1
            continue
        if not is_query:
            covs = [e["coverage"] for e in entries]
            ids_ = [e["identity"] for e in entries]
            if any(c < MIN_COVERAGE for c in covs):
                stats["filtered_cov"] += 1
                continue
            if any(i < MIN_IDENTITY for i in ids_):
                stats["filtered_id"] += 1
                continue
            # genomic-distance filter (only if both UniRef IDs present)
            if all(e["has_uniref"] for e in entries) and all(e["uid"] for e in entries):
                nums = [e["uniprot_num"] for e in entries]
                dists = _calc_distances(nums)
                if dists and dists[0] != -1 and dists[0] > MAX_GENOMIC_DISTANCE:
                    stats["filtered_dist"] += 1
                    continue
            else:
                # has_uniref=False or missing UID -> filter out (notebook lets through, but
                # those rows have no operon evidence; drop to mirror genomic_distance=1 strictness)
                stats["filtered_dist"] += 1
                continue
        # passed filters
        header = "|".join(e["header"] for e in entries) if not is_query else "query"
        kept.append((header, "".join(sequences)))
        stats["kept"] += 1
    return kept, stats


def process_one(row: dict) -> dict:
    eid = row["id"]
    seqs = list(row["sequences"])
    expected_lens = [len(s) for s in seqs]
    chain_break = expected_lens[0]
    raw_path = RAW_DIR / f"{eid}.tar.gz"
    out_path = OUT_DIR / f"{eid}.a3m"
    if not raw_path.exists():
        return {"id": eid, "status": "missing_raw", "depth": 0, "chain_break": chain_break}
    try:
        with tarfile.open(raw_path) as tf:
            raw = None
            for m in tf.getmembers():
                if m.name.endswith("pair.a3m"):
                    raw = tf.extractfile(m).read()
                    break
        if raw is None:
            return {"id": eid, "status": "no_pair_a3m", "depth": 0, "chain_break": chain_break}
        raw = raw.rstrip(b"\x00")
        chunks = [c for c in raw.split(b"\x00") if c.strip()]
        if len(chunks) != len(seqs):
            return {"id": eid, "status": f"wrong_n_chains_{len(chunks)}", "depth": 0, "chain_break": chain_break}
        per_chain = [_parse_msa_chunk(c.decode("utf-8", errors="replace")) for c in chunks]
        kept, stats = _stitch_and_filter(per_chain, expected_lens)
        if stats["kept"] < 2:
            return {"id": eid, "status": "too_shallow_after_filter",
                    "depth": stats["kept"], "raw_depth": stats["raw"], "chain_break": chain_break,
                    **stats}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for h, s in kept:
                f.write(f">{h}\n{s}\n")
        return {"id": eid, "status": "ok", "depth": stats["kept"], "raw_depth": stats["raw"],
                "chain_break": chain_break, **stats}
    except Exception as e:  # noqa: BLE001
        return {"id": eid, "status": "error", "error": f"{type(e).__name__}: {e}",
                "chain_break": chain_break}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(MASTER_INDEX)
    rows = manifest.to_dict("records")
    print(f"Filtering {len(rows)} entries with cov>={MIN_COVERAGE}, id>={MIN_IDENTITY}, Δgene<={MAX_GENOMIC_DISTANCE}")

    results = []
    import multiprocessing as mp
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
