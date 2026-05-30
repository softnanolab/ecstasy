"""Assemble Boltz-2 per-chain MSA CSVs from local colabfold_search output.

Boltz only carries cross-chain coevolution through the CSV ``key`` column
(read as ``taxonomy_id``); a custom ``.a3m`` is parsed with ``taxonomy=None`` and
therefore loses *all* pairing. To reproduce ``--use_msa_server`` we must hand
Boltz per-chain **CSVs** with pairing keys.

This module replicates ``boltz.main.compute_msa`` verbatim, but fed from a local
``colabfold_search`` run instead of the ColabFold API. The only difference from
the server is the search engine (local mmseqs-gpu) — the databases
(uniref30_2302 + colabfold_envdb_202108) are identical.

Pipeline (see ``msa/generate.py`` for orchestration):
  colabfold_search --unpack 0     -> result DBs ``final.a3m`` (uniref+bfd merged,
                                     per chain) and ``pair.a3m`` (per chain, row-aligned)
  mmseqs unpackdb                 -> loose per-chain a3m keyed by global qdb index
  assemble_chain_csv(...)         -> one ``<chain>.csv`` (key,sequence) per chain
"""
from __future__ import annotations

from pathlib import Path

# Boltz const.max_msa_seqs / const.max_paired_seqs (data/const.py).
MAX_MSA_SEQS = 16384
MAX_PAIRED_SEQS = 8192


def assemble_chain_csv(paired_a3m: str, unpaired_a3m: str) -> list[str]:
    """Replicate boltz.main.compute_msa's per-chain CSV assembly.

    ``paired_a3m`` is this chain's slice of the row-aligned paired alignment
    (empty for a monomer); ``unpaired_a3m`` is its unpaired alignment
    (uniref + env merged). Returns CSV lines incl. the ``key,sequence`` header.
    Paired rows are keyed by their row index (the pairing key boltz turns into a
    taxonomy id); unpaired rows are keyed ``-1``.
    """
    paired = paired_a3m.strip().splitlines()
    paired = paired[1::2]                            # sequence lines only
    paired = paired[:MAX_PAIRED_SEQS]
    keys = [i for i, s in enumerate(paired) if s != "-" * len(s)]
    paired = [s for s in paired if s != "-" * len(s)]

    unpaired = unpaired_a3m.strip().splitlines()
    unpaired = unpaired[1::2]
    unpaired = unpaired[: (MAX_MSA_SEQS - len(paired))]
    if paired:
        unpaired = unpaired[1:]                      # query already present in paired

    seqs = paired + unpaired
    keys = keys + [-1] * len(unpaired)
    return ["key,sequence"] + [f"{k},{s}" for k, s in zip(keys, seqs)]


def write_chain_csv(paired_a3m: str, unpaired_a3m: str, dest: Path) -> tuple[int, int]:
    """Assemble + write one chain CSV; return (n_rows, n_paired)."""
    rows = assemble_chain_csv(paired_a3m, unpaired_a3m)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(rows) + "\n")
    n = len(rows) - 1
    n_paired = sum(1 for r in rows[1:] if not r.startswith("-1,"))
    return n, n_paired


def parse_qdb_lookup(lookup_path: Path) -> dict[str, list[int]]:
    """Map colabfold jobname -> ordered global chain indices, from qdb.lookup.

    qdb.lookup rows are ``<global_id>\\t<jobname>\\t<file_number>`` (one row per
    chain). We preserve per-jobname chain order by global id.
    """
    by_job: dict[str, list[int]] = {}
    for line in lookup_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        gid, jobname = int(parts[0]), parts[1]
        by_job.setdefault(jobname, []).append(gid)
    for job in by_job:
        by_job[job].sort()
    return by_job
