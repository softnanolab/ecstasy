"""The MSA stage: populate the shared store, dispatched to a per-kind backend.

Backends live in ``msa/backends/`` (one module per ``kind``), each exposing
``prepare``/``submit``/``ingest``. This module is just the registry + dispatch:

  prepare(datasets, kind)  collect store-missing complexes, write a work FASTA.
  submit(datasets, kind)   launch generation (SLURM for boltz_csv; inline ColabFold
                           API fetch for complex).
  ingest(datasets, kind)   assemble/verify the generated MSAs into the store.

Add a kind = add a backend module + one ``BACKENDS`` entry; no new branches here.
"""
from __future__ import annotations

from pathlib import Path

from ecstasy.msa.backends import BACKENDS


def _backend(kind: str):
    try:
        return BACKENDS[kind]
    except KeyError:
        raise ValueError(f"unknown MSA kind {kind!r}; choose from {sorted(BACKENDS)}")


def prepare(datasets: list[str], kind: str) -> Path:
    """Write a FASTA of store-missing complexes for `kind`."""
    return _backend(kind).prepare(datasets)


def submit(datasets: list[str], kind: str) -> str | None:
    """Launch MSA generation for `kind` (SLURM job id, or None for inline fetch)."""
    return _backend(kind).submit(datasets)


def ingest(datasets: list[str], kind: str, a3m_dir: str | None = None) -> None:
    """Assemble/verify generated MSAs into the store for `kind`."""
    return _backend(kind).ingest(datasets, out_dir=a3m_dir)
