"""Run Foldseek easy-search twice: candidates vs Mentos-train AND vs Boltz-2-train.

Two-job approach (rather than a single union DB) so we keep per-source
attribution: downstream reports can answer "this dimer was dropped because
of a Mentos hit / a Boltz-2 hit / both."

The Boltz-2 train DB is built by createdb'ing the union of Mentos chains
+ the Boltz-2 delta chains (see `03b_extract_boltz2_delta_chains.py`).

Outputs:
  foldseek_hits_mentos.m8
  foldseek_hits_boltz2.m8
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

FOLDSEEK = Path("/home/u6jv/harsh.u6jv/ecstasy/tools/foldseek/bin/foldseek")
ROOT = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v2")
CANDIDATES_DIR = ROOT / "candidates" / "chains"
MENTOS_DIR = ROOT / "train_db" / "mentos_train_chains"
BOLTZ2_DELTA_DIR = ROOT / "train_db" / "boltz2_delta_chains"

DB_DIR = ROOT / "foldseek_dbs"
TMP_DIR = ROOT / "foldseek_tmp"
HITS_MENTOS = ROOT / "foldseek_hits_mentos.m8"
HITS_BOLTZ2 = ROOT / "foldseek_hits_boltz2.m8"

FORMAT_OUTPUT = "query,target,lddt,qstart,qend,qlen,tstart,tend,tlen,alnlen,evalue"

NTHREADS = int(
    os.environ.get("SLURM_CPUS_PER_TASK")
    or os.environ.get("OMP_NUM_THREADS")
    or 16
)


def run(cmd: list[str]) -> None:
    print(" $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def createdb_if_missing(name: str, src_dirs: list[Path]) -> Path:
    """Create a Foldseek DB from one or more directories of PDBs.

    Foldseek's `createdb` accepts multiple input paths; passing both
    `mentos_train_chains/` and `boltz2_delta_chains/` builds the union DB
    in one shot.
    """
    db_path = DB_DIR / name
    sentinel = DB_DIR / name  # foldseek's main index file uses the bare name
    if sentinel.exists():
        print(f"  reusing existing DB at {db_path}")
        return db_path
    for d in src_dirs:
        if not d.is_dir():
            raise SystemExit(f"ERROR: source dir {d} missing")
    cmd = [str(FOLDSEEK), "createdb", *[str(d) for d in src_dirs], str(db_path),
           "--threads", str(NTHREADS)]
    run(cmd)
    return db_path


def easy_search(query_dir: Path, target_db: Path, hits_path: Path) -> None:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    run([
        str(FOLDSEEK),
        "easy-search",
        str(query_dir),
        str(target_db),
        str(hits_path),
        str(TMP_DIR),
        "-s", "11.0",
        "-e", "0.05",
        "--alignment-type", "2",
        "--max-seqs", "1000",
        "--threads", str(NTHREADS),
        "--format-output", FORMAT_OUTPUT,
    ])
    print(f"  wrote {hits_path}  ({hits_path.stat().st_size / 1e6:.1f} MB)")
    with hits_path.open() as f:
        n_hits = sum(1 for _ in f)
    print(f"  {n_hits} hit rows")


def main() -> int:
    DB_DIR.mkdir(parents=True, exist_ok=True)

    cand_count = len(list(CANDIDATES_DIR.glob("*.pdb")))
    mentos_count = len(list(MENTOS_DIR.glob("*.pdb"))) if MENTOS_DIR.is_dir() else 0
    delta_count = len(list(BOLTZ2_DELTA_DIR.glob("*.pdb"))) if BOLTZ2_DELTA_DIR.is_dir() else 0
    print(f"Candidate chain PDBs:           {cand_count}")
    print(f"Mentos-train chain PDBs:        {mentos_count}")
    print(f"Boltz-2 delta chain PDBs:       {delta_count}")
    print(f"Boltz-2 train DB total (union): {mentos_count + delta_count}")
    print(f"Threads:                        {NTHREADS}")
    if cand_count == 0 or mentos_count == 0 or delta_count == 0:
        print("ERROR: empty input dir(s)", file=sys.stderr)
        return 1

    # 1. Candidates DB (queries — same for both searches)
    cand_db = createdb_if_missing("candidates_db", [CANDIDATES_DIR])
    # 2. Mentos-train DB
    mentos_db = createdb_if_missing("mentos_train_db", [MENTOS_DIR])
    # 3. Boltz-2 train DB = Mentos ∪ delta
    boltz2_db = createdb_if_missing("boltz2_train_db", [MENTOS_DIR, BOLTZ2_DELTA_DIR])

    print("\n=== Search: candidates vs Mentos-train ===")
    easy_search(CANDIDATES_DIR, mentos_db, HITS_MENTOS)

    print("\n=== Search: candidates vs Boltz-2-train (union) ===")
    easy_search(CANDIDATES_DIR, boltz2_db, HITS_BOLTZ2)

    _ = cand_db  # avoid lint
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
