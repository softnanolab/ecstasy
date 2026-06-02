"""The MSA stage: populate the shared store, dispatched to a per-kind backend.

Backends live in ``msa/backends/`` (one module per ``kind``), each exposing
``prepare``/``submit``/``ingest``. This module is just the registry + dispatch:

  prepare(datasets, kind)  collect store-missing complexes, write a work FASTA.
  submit(datasets, kind)   launch generation. LOCAL routes sbatch a GPU job:
                           boltz_csv (colabfold_search) and complex (colabfold-local).
                           complex_api is the only network route (inline ColabFold API).
  ingest(datasets, kind)   assemble/verify the generated MSAs into the store.

kinds: boltz_csv -> Boltz-2 (per-chain CSVs); complex -> MSA-Pairformer (local
colabfold-local; the default + how the eval data was made); complex_api -> API
fallback. boltz_csv and complex use the SAME search engine but produce different
outputs for different models — never conflate them. See msa/README.md.

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
