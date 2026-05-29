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


def path_for_chain(seq: str) -> Path:
    return per_chain_dir() / f"{chain_hash(seq)}.a3m"


def path_for_pair(seqs) -> Path:
    return complex_dir() / f"{pair_hash(seqs)}.a3m"


def lookup(entry, kind: str):
    """Return the MSA form a model needs, or None to trigger single-seq fallback.

    per_chain -> {chain_id: a3m_path} for chains that have one (or None)
    complex   -> a single a3m path (or None)
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
    return None
