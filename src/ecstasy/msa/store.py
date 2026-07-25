"""Shared, dataset-independent MSA cache.

Two flavours, keyed by content so overlapping datasets never regenerate:
  per_chain  unpaired single-chain a3m   -> msa_store/per_chain/<sha256(seq)[:16]>.a3m
  complex    paired complex a3m          -> msa_store/complex/<sha256(seqA|seqB|...)[:16]>.a3m

Boltz-2/ColabFold consume per_chain (a path per chain); MSA Pairformer consumes
complex (one path per entry). The pipeline calls :func:`lookup` to get whatever
form a model needs, falling back to single-sequence when absent.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ecstasy.config import settings


def chain_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def pair_hash(seqs) -> str:
    return hashlib.sha256("|".join(seqs).encode()).hexdigest()[:16]


def per_chain_dir() -> Path:
    return settings().msa_store / "per_chain"


def complex_dir() -> Path:
    return settings().msa_store / "complex"


def boltz_csv_dir() -> Path:
    return settings().msa_store / "boltz_csv"


def path_for_chain(seq: str) -> Path:
    return per_chain_dir() / f"{chain_hash(seq)}.a3m"


def path_for_pair(seqs) -> Path:
    return complex_dir() / f"{pair_hash(seqs)}.a3m"


def paired_depth(a3m_path: Path) -> int:
    """Number of sequence rows in a complex a3m (``>`` lines; excludes the ``#`` header).

    Depth 1 == query only (the proximity filter dropped every paired hit — a
    "collapsed" MSA, near-single-sequence for the model).
    """
    try:
        with Path(a3m_path).open() as f:
            return sum(1 for line in f if line.startswith(">"))
    except OSError:
        return 0


def depth_report(items) -> tuple[int, int, list[int]]:
    """(present, collapsed, depths) over complexes: collapsed == depth <= 1 (query only)."""
    depths = []
    for v in items.values():
        p = path_for_pair(v["seqs"])
        if p.exists():
            depths.append(paired_depth(p))
    collapsed = sum(1 for d in depths if d <= 1)
    return len(depths), collapsed, depths


def boltz_csv_complex_dir(seqs) -> Path:
    """Per-complex dir holding one ``<chain_index>.csv`` per chain (pairing-aware)."""
    return boltz_csv_dir() / pair_hash(seqs)


def path_for_boltz_csv(seqs, chain_index: int) -> Path:
    return boltz_csv_complex_dir(seqs) / f"{chain_index}.csv"


def lookup(entry, kind: str):
    """Return the MSA form a model needs, or None to trigger single-seq fallback.

    per_chain -> {chain_id: a3m_path} for chains that have one (or None)
    complex   -> a single a3m path (or None)
    boltz_csv -> {chain_id: csv_path} paired+unpaired per chain, keyed by the
                 complex (pair-hash); reproduces boltz --use_msa_server (or None)
    none      -> None
    """
    if kind == "per_chain":
        out = {}
        for cid, seq in zip(entry.chain_ids, entry.sequences):
            p = path_for_chain(seq)
            if p.exists():
                out[cid] = p
        return out or None
    if kind == "complex":
        p = path_for_pair(entry.sequences)
        return p if p.exists() else None
    if kind == "boltz_csv":
        # colabfold dedups identical chains, so a homodimer yields a single CSV.
        # Map each chain to its *unique-sequence* index (first-occurrence order),
        # matching how ingest wrote them, so repeated chains resolve correctly.
        uniq: list = []
        for s in entry.sequences:
            if s not in uniq:
                uniq.append(s)
        out = {}
        for cid, seq in zip(entry.chain_ids, entry.sequences):
            p = path_for_boltz_csv(entry.sequences, uniq.index(seq))
            if p.exists():
                out[cid] = p
        return out or None
    return None
