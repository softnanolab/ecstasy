"""The MSA stage: populate the shared store via colabfold-local.

Two phases (colabfold runs on GPU/SLURM in between — it stays external):

  prepare(datasets, kind)  collect the unique chains (per_chain) or complexes
                           (complex) across `datasets`, drop any already in the
                           store, write a FASTA of the *missing* ones to a work
                           dir, and print the exact colabfold-local sbatch.
  ingest(work_dir, kind)   match colabfold's output a3ms back to chains/pairs by
                           content hash and copy them into the store.

`run_msa(datasets, kind, submit=...)` chains prepare -> (optional sbatch) -> wait,
but by default just does prepare and tells you the command to run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ecstasy.config import settings
from ecstasy.datasets import load_dataset
from ecstasy.msa import store

COLABFOLD_LOCAL = Path(os.environ.get("COLABFOLD_LOCAL", "/home/u6jv/harsh.u6jv/colabfold-local"))


def _work_dir(kind: str) -> Path:
    return settings().msa_store / "_work" / kind


def _collect(datasets: list[str], kind: str) -> dict[str, dict]:
    """hash -> {seqs, header, query} for every unique chain/complex across datasets."""
    items: dict[str, dict] = {}
    for name in datasets:
        for e in load_dataset(name).entries():
            if kind == "per_chain":
                for seq in e.sequences:
                    h = store.chain_hash(seq)
                    items.setdefault(h, {"seqs": [seq], "header": h, "query": seq})
            elif kind == "complex":
                h = store.pair_hash(e.sequences)
                items.setdefault(h, {"seqs": list(e.sequences), "header": h,
                                     "query": ":".join(e.sequences)})
            else:
                raise ValueError(f"kind must be per_chain|complex, got {kind!r}")
    return items


def prepare(datasets: list[str], kind: str) -> Path:
    """Write a FASTA of store-missing chains/complexes; print the colabfold command."""
    store_dir = store.per_chain_dir() if kind == "per_chain" else store.complex_dir()
    store_dir.mkdir(parents=True, exist_ok=True)
    items = _collect(datasets, kind)
    missing = {h: v for h, v in items.items() if not (store_dir / f"{h}.a3m").exists()}

    work = _work_dir(kind)
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "missing.fasta"
    with fasta.open("w") as f:
        for h, v in sorted(missing.items()):
            f.write(f">{v['header']}\n{v['query']}\n")

    print(f"[msa:{kind}] datasets={datasets}")
    print(f"[msa:{kind}] unique={len(items)}  already_in_store={len(items)-len(missing)}  "
          f"missing={len(missing)}")
    print(f"[msa:{kind}] wrote {fasta}")
    if missing:
        out = work / "out"
        print("\nNext — generate MSAs (GPU/SLURM), then ingest:")
        print(f"  sbatch {COLABFOLD_LOCAL}/scripts/01_generate_msa.sh {fasta} {out}")
        print(f"  ecstasy msa --datasets {','.join(datasets)} --kind {kind} --phase ingest")
    return fasta


def submit(datasets: list[str], kind: str) -> str | None:
    """prepare(), then sbatch colabfold-local's MSA job over the missing FASTA."""
    fasta = prepare(datasets, kind)
    out = _work_dir(kind) / "out"
    script = COLABFOLD_LOCAL / "scripts" / "01_generate_msa.sh"
    if not script.exists():
        raise FileNotFoundError(f"colabfold-local MSA script not found: {script}")
    res = subprocess.run(["sbatch", "--parsable", str(script), str(fasta), str(out)],
                         capture_output=True, text=True, check=True)
    job = res.stdout.strip()
    print(f"[msa:{kind}] submitted colabfold job {job}; run --phase ingest once it finishes")
    return job


def _index_a3ms_by_hash(a3m_dir: Path, kind: str) -> dict[str, Path]:
    """Map content hash -> a3m, by hashing each a3m's query (first) sequence.

    Robust to colabfold_search's filename scheme. For complex queries the query
    line is the concatenated chains; we recover the ':' breaks from the colabfold
    header (#L1,L2) so the pair hash matches what `prepare` wrote.
    """
    index: dict[str, Path] = {}
    for p in sorted(a3m_dir.glob("*.a3m")):
        lines = p.read_text().splitlines()
        header = next((ln for ln in lines if ln.startswith("#")), None)
        # first non-comment record's sequence line
        query = None
        seen = False
        for ln in lines:
            if ln.startswith("#"):
                continue
            if ln.startswith(">"):
                if seen:
                    break
                seen = True
                continue
            if seen:
                query = "".join(c for c in ln.strip() if c.isupper())
                break
        if not query:
            continue
        if kind == "per_chain":
            index.setdefault(store.chain_hash(query), p)
        else:
            seqs = _split_complex(query, header)
            index.setdefault(store.pair_hash(seqs), p)
    return index


def _split_complex(query: str, header: str | None) -> list[str]:
    """Split a concatenated complex query into chains using colabfold '#L1,L2' header."""
    if header:
        try:
            lens = header.lstrip("#").split("\t")[0].split(",")
            lens = [int(x) for x in lens]
            out, pos = [], 0
            for L in lens:
                out.append(query[pos:pos + L])
                pos += L
            if "".join(out) == query:
                return out
        except (ValueError, IndexError):
            pass
    return [query]  # single chain / unknown break


def ingest(datasets: list[str], kind: str, a3m_dir: str | None = None) -> None:
    """Copy colabfold output a3ms into the store, keyed by content hash."""
    a3m_dir = Path(a3m_dir) if a3m_dir else _work_dir(kind) / "out"
    if not a3m_dir.exists():
        raise FileNotFoundError(f"no a3m dir at {a3m_dir} (run colabfold first)")
    store_dir = store.per_chain_dir() if kind == "per_chain" else store.complex_dir()
    store_dir.mkdir(parents=True, exist_ok=True)

    by_hash = _index_a3ms_by_hash(a3m_dir, kind)
    wanted = set(_collect(datasets, kind))
    placed = missing = 0
    for h in wanted:
        src = by_hash.get(h)
        if src is None:
            missing += 1
            continue
        dst = store_dir / f"{h}.a3m"
        if not dst.exists():
            shutil.copyfile(src, dst)
        placed += 1
    print(f"[msa:{kind}] ingested from {a3m_dir}: wanted={len(wanted)} placed={placed} missing={missing}")
    if missing:
        print(f"[msa:{kind}] {missing} had no a3m (low-hit queries) -> single-seq fallback at predict")
