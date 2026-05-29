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

# Legacy per_chain/complex backend (external colabfold-local). Path is machine-specific
# -> configure via COLABFOLD_LOCAL in env/.env, never committed. The boltz_csv backend
# below is self-contained and does not need this.
COLABFOLD_LOCAL = Path(os.environ.get("COLABFOLD_LOCAL", "colabfold-local"))


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
            elif kind in ("complex", "boltz_csv"):
                # both are keyed per unique complex (pair hash), colon-joined query
                h = store.pair_hash(e.sequences)
                items.setdefault(h, {"seqs": list(e.sequences), "header": h,
                                     "query": ":".join(e.sequences)})
            else:
                raise ValueError(f"kind must be per_chain|complex|boltz_csv, got {kind!r}")
    return items


def prepare(datasets: list[str], kind: str) -> Path:
    """Write a FASTA of store-missing chains/complexes; print the colabfold command."""
    if kind == "boltz_csv":
        return prepare_boltz_csv(datasets)
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
    if kind == "boltz_csv":
        return submit_boltz_csv(datasets)
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
    """Split a concatenated complex query into chains using the colabfold header.

    The header is ``#<len1>,<len2>,...\t<copies1>,<copies2>,...`` — colabfold lists
    each *unique* chain length once with a copy count, so a homodimer is
    ``#<L>\t2`` (one length, two copies), not ``#<L>,<L>\t1,1``. We expand
    lengths by copy count before slicing so the per-chain split (and thus the
    pair hash) matches what `prepare` wrote from `entry.sequences`.
    """
    if header:
        try:
            fields = header.lstrip("#").split("\t")
            lens = [int(x) for x in fields[0].split(",")]
            copies = [int(x) for x in fields[1].split(",")] if len(fields) > 1 else [1] * len(lens)
            expanded = [L for L, c in zip(lens, copies) for _ in range(c)]
            out, pos = [], 0
            for L in expanded:
                out.append(query[pos:pos + L])
                pos += L
            if "".join(out) == query and all(out):  # exact tiling, no empty chains
                return out
        except (ValueError, IndexError):
            pass
    return [query]  # single chain / unknown break


def ingest(datasets: list[str], kind: str, a3m_dir: str | None = None) -> None:
    """Copy colabfold output a3ms into the store, keyed by content hash."""
    if kind == "boltz_csv":
        return ingest_boltz_csv(datasets, out_dir=a3m_dir)
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


# ---- boltz_csv: self-contained local colabfold_search -> per-chain CSV --------
#
# Reproduces boltz `--use_msa_server` locally: paired (uniref30 greedy) + unpaired
# (uniref30 + colabfold_envdb) hits assembled into per-chain CSVs with pairing keys
# (see msa/boltz_csv.py). Uses the in-repo .venv-colabfold + vendored mmseqs-gpu +
# the server-identical ColabFold DBs (COLABFOLD_DBS); no external colabfold-local.

from ecstasy.config import env_value
from ecstasy.msa import boltz_csv as _bcsv


def _colabfold_paths() -> dict[str, str]:
    s = settings()
    dbs = env_value("COLABFOLD_DBS")
    if not dbs:
        raise RuntimeError("COLABFOLD_DBS not set (env or .env): path to uniref30 + "
                           "colabfold_envdb databases")
    return {
        "search": str(s.ENVS_ROOT / ".venv-colabfold" / "bin" / "colabfold_search"),
        "mmseqs": str(s.TOOLS_ROOT / "mmseqs-gpu" / "bin" / "mmseqs"),
        "dbs": dbs,
        "cuda_module": env_value("CUDA_MODULE", "cuda/12.6"),
        "partition": env_value("SLURM_PARTITION", "workq"),
    }


def prepare_boltz_csv(datasets: list[str]) -> Path:
    """Write a FASTA of store-missing complexes (colon-joined) for boltz_csv."""
    store.boltz_csv_dir().mkdir(parents=True, exist_ok=True)
    items = _collect(datasets, "boltz_csv")
    # a complex is present iff its first chain CSV exists in the store
    missing = {h: v for h, v in items.items()
               if not store.path_for_boltz_csv(v["seqs"], 0).exists()}

    work = _work_dir("boltz_csv")
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "missing.fasta"
    with fasta.open("w") as f:
        for h, v in sorted(missing.items()):
            f.write(f">{v['header']}\n{v['query']}\n")

    print(f"[msa:boltz_csv] datasets={datasets}")
    print(f"[msa:boltz_csv] unique={len(items)}  already_in_store={len(items)-len(missing)}  "
          f"missing={len(missing)}")
    print(f"[msa:boltz_csv] wrote {fasta}")
    return fasta


def _write_boltz_csv_sbatch(fasta: Path, out: Path) -> Path:
    """Generate the colabfold_search + unpackdb SLURM script for boltz_csv."""
    p = _colabfold_paths()
    work = fasta.parent
    (work / "logs").mkdir(parents=True, exist_ok=True)
    script = work / "generate_boltz_csv.sbatch"
    script.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=ecstasy-msa-boltzcsv
#SBATCH --output={work}/logs/msa_%j.out
#SBATCH --error={work}/logs/msa_%j.err
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition={p['partition']}
set -euo pipefail

OUT="{out}"
export TMPDIR="${{SCRATCHDIR:-/tmp}}/ecstasy_cf_boltzcsv"
mkdir -p "$TMPDIR" "$OUT"

module load {p['cuda_module']}
# shellcheck disable=SC1091
source "{settings().ENVS_ROOT}/.venv-colabfold/bin/activate"

# unpaired (uniref30 + colabfold_envdb) + paired (uniref30 greedy); --unpack 0
# keeps raw result DBs (final.a3m, pair.a3m) and qdb.lookup for faithful assembly.
srun colabfold_search \\
    --mmseqs "{p['mmseqs']}" \\
    --threads 32 \\
    --gpu 1 \\
    --db-load-mode 2 \\
    --use-env 1 \\
    --use-env-pairing 0 \\
    --pair-mode unpaired_paired \\
    --pairing_strategy 0 \\
    --unpack 0 \\
    "{fasta}" \\
    "{p['dbs']}" \\
    "$OUT"

# unpack per-chain result DBs (keyed by global qdb index). pair.a3m only exists
# when the batch has complexes (colabfold skips pairing for monomer-only inputs).
mkdir -p "$OUT/unpaired" "$OUT/paired"
"{p['mmseqs']}" unpackdb "$OUT/final.a3m" "$OUT/unpaired" --unpack-name-mode 0 --unpack-suffix .a3m
if [ -f "$OUT/pair.a3m.dbtype" ]; then
    "{p['mmseqs']}" unpackdb "$OUT/pair.a3m" "$OUT/paired" --unpack-name-mode 0 --unpack-suffix .a3m
else
    echo "no pair.a3m (monomer-only batch); paired MSAs skipped"
fi
echo "DONE boltz_csv MSA generation"
""")
    return script


def submit_boltz_csv(datasets: list[str]) -> str | None:
    """prepare_boltz_csv(), then sbatch the in-repo colabfold_search recipe."""
    fasta = prepare_boltz_csv(datasets)
    if fasta.stat().st_size == 0:
        print("[msa:boltz_csv] nothing missing; store is complete")
        return None
    out = _work_dir("boltz_csv") / "out"
    script = _write_boltz_csv_sbatch(fasta, out)
    res = subprocess.run(["sbatch", "--parsable", str(script)],
                         capture_output=True, text=True, check=True)
    job = res.stdout.strip()
    print(f"[msa:boltz_csv] submitted job {job} ({script}); run --phase ingest once it finishes")
    return job


def ingest_boltz_csv(datasets: list[str], out_dir: str | None = None) -> None:
    """Assemble per-chain CSVs from unpacked colabfold output into the store."""
    out = Path(out_dir) if out_dir else _work_dir("boltz_csv") / "out"
    lookup_path = out / "qdb.lookup"
    if not lookup_path.exists():
        raise FileNotFoundError(f"no qdb.lookup at {out} (run --phase submit first)")
    job_gids = _bcsv.parse_qdb_lookup(lookup_path)

    items = _collect(datasets, "boltz_csv")
    placed = missing = 0
    for h, v in items.items():
        gids = job_gids.get(h)
        if not gids:
            missing += 1
            continue
        for i, gid in enumerate(gids):
            up = out / "unpaired" / f"{gid}.a3m"
            pr = out / "paired" / f"{gid}.a3m"
            if not up.exists():
                continue
            paired_a3m = pr.read_text() if pr.exists() else ""
            _bcsv.write_chain_csv(paired_a3m, up.read_text(),
                                  store.path_for_boltz_csv(v["seqs"], i))
        placed += 1
    print(f"[msa:boltz_csv] ingested from {out}: wanted={len(items)} placed={placed} missing={missing}")
    if missing:
        print(f"[msa:boltz_csv] {missing} complexes absent from colabfold output "
              f"-> single-seq fallback at predict")
