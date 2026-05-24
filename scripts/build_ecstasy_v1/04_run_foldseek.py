"""Run Foldseek easy-search: candidate chains vs MENTOS-train chains.

Uses --threads from SLURM_CPUS_PER_TASK if available (else 16).
Output format: query target lddt qstart qend qlen tstart tend tlen alnlen evalue
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

FOLDSEEK = Path("/home/u6jv/harsh.u6jv/ecstasy/tools/foldseek/bin/foldseek")
CANDIDATES_DIR = Path(
    "/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/candidates/chains"
)
TRAIN_DIR = Path(
    "/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/train_db/mentos_train_chains"
)
OUT_ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1")
DB_DIR = OUT_ROOT / "foldseek_dbs"
TMP_DIR = OUT_ROOT / "foldseek_tmp"
HITS_PATH = OUT_ROOT / "foldseek_hits.m8"

FORMAT_OUTPUT = "query,target,lddt,qstart,qend,qlen,tstart,tend,tlen,alnlen,evalue"

NTHREADS = int(
    os.environ.get("SLURM_CPUS_PER_TASK")
    or os.environ.get("OMP_NUM_THREADS")
    or 16
)


def run(cmd: list[str]) -> None:
    print(" $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    cand_count = len(list(CANDIDATES_DIR.glob("*.pdb")))
    train_count = len(list(TRAIN_DIR.glob("*.pdb")))
    print(f"Candidate chain PDBs: {cand_count}")
    print(f"MENTOS-train chain PDBs: {train_count}")
    print(f"Threads:               {NTHREADS}")
    if cand_count == 0 or train_count == 0:
        print("ERROR: empty input dir", file=sys.stderr)
        return 1

    cand_db = DB_DIR / "candidates_db"
    train_db = DB_DIR / "mentos_train_db"

    if not (DB_DIR / "candidates_db").exists():
        run([str(FOLDSEEK), "createdb", str(CANDIDATES_DIR), str(cand_db),
             "--threads", str(NTHREADS)])
    else:
        print(f"  reusing existing candidates DB at {cand_db}")
    if not (DB_DIR / "mentos_train_db").exists():
        run([str(FOLDSEEK), "createdb", str(TRAIN_DIR), str(train_db),
             "--threads", str(NTHREADS)])
    else:
        print(f"  reusing existing MENTOS-train DB at {train_db}")

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    run([
        str(FOLDSEEK),
        "easy-search",
        str(CANDIDATES_DIR),
        str(train_db),
        str(HITS_PATH),
        str(TMP_DIR),
        "-s", "11.0",
        "-e", "0.05",
        "--alignment-type", "2",
        "--max-seqs", "1000",
        "--threads", str(NTHREADS),
        "--format-output", FORMAT_OUTPUT,
    ])

    print(f"\n  wrote {HITS_PATH}  ({HITS_PATH.stat().st_size / 1e6:.1f} MB)")
    with HITS_PATH.open() as f:
        n_hits = sum(1 for _ in f)
    print(f"  {n_hits} hit rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
