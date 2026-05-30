"""Shared helpers for MSA-generation backends."""
from __future__ import annotations

from pathlib import Path

from ecstasy.config import settings
from ecstasy.datasets import load_dataset
from ecstasy.msa import store


def work_dir(kind: str) -> Path:
    return settings().msa_store / "_work" / kind


def collect_complexes(datasets: list[str]) -> dict[str, dict]:
    """pair_hash -> {seqs, header, query} for every unique complex across datasets.

    Both complex MSA backends (boltz_csv, complex) key per unique complex.
    """
    items: dict[str, dict] = {}
    for name in datasets:
        for e in load_dataset(name).entries():
            h = store.pair_hash(e.sequences)
            items.setdefault(h, {"seqs": list(e.sequences), "header": h,
                                 "query": ":".join(e.sequences)})
    return items
