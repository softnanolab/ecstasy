"""complex_api MSA backend — MSA-Pairformer paired MSA via the public ColabFold API.

⚠️  NOT how the benchmark's MSA-Pairformer MSAs were actually generated. The eval
used the LOCAL ``complex`` backend (``complex.py`` → softnanolab/colabfold-local).
This API path is kept as a network fallback / cross-check only. See ``msa/README.md``
for the model→pipeline map and why the two must not be conflated.

Reproduces the MSA Pairformer notebook/SI MMseqs2 route over the network (separate from boltz_csv):
  heterodimer -> /ticket/pair paircomplete-pairfilterprox_20 (fetch broad) -> save_msa
                 filters (coverage>=0.70, query-identity>=0.15, genomic-distance<=1) -> stitch
  homodimer   -> /ticket/msa unpaired -> tile each row (s|s)
Writes one ``#L1,L2\\t1,1``-headed a3m per complex to the store; the depth cap (512,
hhfilter) happens at model load in the runner. Needs network egress to
api.colabfold.com — run on a node with internet (login or internet-capable compute).

Backend interface: ``prepare(datasets) -> Path``, ``submit(datasets) -> None``
(fetches inline, concurrently), ``ingest(datasets, out_dir=None) -> None`` (reports
coverage; the fetch writes straight to the store).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ecstasy.config import env_value
from ecstasy.msa import colabfold as _cf
from ecstasy.msa import store
from ecstasy.msa.backends._common import collect_complexes, work_dir


def _header_and_rows(session, seqs: list[str]) -> tuple[str, list[tuple[str, str]]]:
    cleaned = [_cf.clean_sequence(s) for s in seqs]
    lengths = ",".join(str(len(s)) for s in cleaned)
    header = f"#{lengths}\t{','.join('1' for _ in cleaned)}"
    uniq = list(dict.fromkeys(cleaned))
    if len(uniq) == 1:
        # homodimer/homo-oligomer: tile the single-chain unpaired MSA
        jid = _cf.submit_msa(session, _cf.make_query_fasta([uniq[0]]), mode="env")
        _cf.poll_until_done(session, jid)
        chain_seqs = _cf.parse_unpaired_a3m_bytes(_cf.download_results(session, jid))
        L, n = len(uniq[0]), len(cleaned)
        rows = [("query" if i == 0 else f"hom{i}", s * n)
                for i, s in enumerate(chain_seqs) if len(s) == L]
        return header, rows
    # heterodimer: paired prox_20 + save_msa filters + stitch
    jid = _cf.submit_pair(session, _cf.make_query_fasta(cleaned), mode=_cf.DEFAULT_MODE)
    _cf.poll_until_done(session, jid)
    per_chain = _cf.parse_paired_a3m_bytes(_cf.download_results(session, jid), extract_metadata=True)
    rows, _stats = _cf.apply_save_msa_filters(per_chain, [len(s) for s in cleaned], _cf.SaveMsaFilters())
    return header, rows


def _fetch_one(v: dict) -> int:
    """Fetch+filter+stitch one complex into the store; -1 if already present."""
    import requests
    dst = store.path_for_pair(v["seqs"])
    if dst.exists():
        return -1
    header, rows = _header_and_rows(requests.Session(), v["seqs"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".a3m.tmp")          # write-then-rename so partials never look "done"
    with tmp.open("w") as f:
        f.write(header + "\n")
        for hdr, seq in rows:
            f.write(f">{hdr}\n{seq}\n")
    tmp.rename(dst)
    return len(rows)


def prepare(datasets: list[str]) -> Path:
    """List store-missing complexes (writes a FASTA for reference)."""
    store.complex_dir().mkdir(parents=True, exist_ok=True)
    items = collect_complexes(datasets)
    missing = {h: v for h, v in items.items() if not store.path_for_pair(v["seqs"]).exists()}
    work = work_dir("complex"); work.mkdir(parents=True, exist_ok=True)
    fasta = work / "missing.fasta"
    with fasta.open("w") as f:
        for h, v in sorted(missing.items()):
            f.write(f">{v['header']}\n{v['query']}\n")
    print(f"[msa:complex_api] datasets={datasets}")
    print(f"[msa:complex_api] unique={len(items)} already_in_store={len(items)-len(missing)} missing={len(missing)}")
    print(f"[msa:complex_api] wrote {fasta}; run --phase submit to fetch from api.colabfold.com (needs network)")
    return fasta


def submit(datasets: list[str]) -> None:
    """Fetch missing complexes from the ColabFold API into the store, concurrently.

    Resumable (skips complexes already present), tolerant of per-complex errors.
    ``COMPLEX_FETCH_WORKERS`` (default 4) concurrent API jobs; each colabfold call
    backs off on 429, so modest concurrency stays polite.
    """
    workers = int(env_value("COMPLEX_FETCH_WORKERS", "4"))
    items = collect_complexes(datasets)
    store.complex_dir().mkdir(parents=True, exist_ok=True)
    missing = [v for v in items.values() if not store.path_for_pair(v["seqs"]).exists()]
    print(f"[msa:complex_api] fetching {len(missing)} of {len(items)} complexes "
          f"from api.colabfold.com ({workers} workers)", flush=True)
    done = errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, v): v["header"] for v in missing}
        for n, fut in enumerate(as_completed(futs), 1):
            h = futs[fut]
            try:
                if fut.result() >= 0:
                    done += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"[msa:complex_api] ERROR {h}: {e}", file=sys.stderr, flush=True)
            if n % 25 == 0 or n == len(missing):
                print(f"[msa:complex_api] {n}/{len(missing)} (done={done} errors={errors})", flush=True)
    print(f"[msa:complex_api] done: wrote={done} errors={errors}")


def ingest(datasets: list[str], out_dir: str | None = None) -> None:
    """Report store coverage (the fetch in `submit` writes straight to the store)."""
    items = collect_complexes(datasets)
    have, collapsed, _ = store.depth_report(items)
    print(f"[msa:complex_api] store coverage: {have}/{len(items)} complexes "
          f"({collapsed} collapsed to query-only — proximity dropped all paired hits)")
