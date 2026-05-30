"""boltz_csv MSA backend — faithful local reproduction of Boltz `--use_msa_server`.

Runs `colabfold_search` (SLURM/GPU) for paired (uniref30 greedy) + unpaired
(uniref30 + colabfold_envdb) hits, then assembles per-chain CSVs with pairing keys
(see ``msa/boltz_csv.py``). Self-contained: in-repo .venv-colabfold + vendored
mmseqs-gpu + the server-identical ColabFold DBs (``COLABFOLD_DBS``).

Backend interface: ``prepare(datasets) -> Path``, ``submit(datasets) -> job|None``,
``ingest(datasets, out_dir=None) -> None``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ecstasy.config import env_value, settings
from ecstasy.msa import boltz_csv as _assembly
from ecstasy.msa import store
from ecstasy.msa.backends._common import collect_complexes, work_dir


def _colabfold_paths() -> dict[str, str]:
    s = settings()
    dbs = env_value("COLABFOLD_DBS")
    if not dbs:
        raise RuntimeError("COLABFOLD_DBS not set (env or .env): path to uniref30 + "
                           "colabfold_envdb databases")
    return {
        "mmseqs": str(s.TOOLS_ROOT / "mmseqs-gpu" / "bin" / "mmseqs"),
        "dbs": dbs,
        "cuda_module": env_value("CUDA_MODULE", "cuda/12.6"),
        "partition": env_value("SLURM_PARTITION", "workq"),
        "venv": str(s.ENVS_ROOT / ".venv-colabfold"),
    }


def prepare(datasets: list[str]) -> Path:
    """Write a FASTA of store-missing complexes (colon-joined)."""
    store.boltz_csv_dir().mkdir(parents=True, exist_ok=True)
    items = collect_complexes(datasets)
    # a complex is present iff its first chain CSV exists in the store
    missing = {h: v for h, v in items.items()
               if not store.path_for_boltz_csv(v["seqs"], 0).exists()}
    work = work_dir("boltz_csv")
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


def _write_sbatch(fasta: Path, out: Path) -> Path:
    """Generate the colabfold_search + unpackdb SLURM script."""
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
source "{p['venv']}/bin/activate"

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


def submit(datasets: list[str]) -> str | None:
    """prepare(), then sbatch the in-repo colabfold_search recipe."""
    fasta = prepare(datasets)
    if fasta.stat().st_size == 0:
        print("[msa:boltz_csv] nothing missing; store is complete")
        return None
    out = work_dir("boltz_csv") / "out"
    script = _write_sbatch(fasta, out)
    res = subprocess.run(["sbatch", "--parsable", str(script)],
                         capture_output=True, text=True, check=True)
    job = res.stdout.strip()
    print(f"[msa:boltz_csv] submitted job {job} ({script}); run --phase ingest once it finishes")
    return job


def ingest(datasets: list[str], out_dir: str | None = None) -> None:
    """Assemble per-chain CSVs from unpacked colabfold output into the store."""
    out = Path(out_dir) if out_dir else work_dir("boltz_csv") / "out"
    lookup_path = out / "qdb.lookup"
    if not lookup_path.exists():
        raise FileNotFoundError(f"no qdb.lookup at {out} (run --phase submit first)")
    job_gids = _assembly.parse_qdb_lookup(lookup_path)

    items = collect_complexes(datasets)
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
            _assembly.write_chain_csv(paired_a3m, up.read_text(),
                                      store.path_for_boltz_csv(v["seqs"], i))
        placed += 1
    print(f"[msa:boltz_csv] ingested from {out}: wanted={len(items)} placed={placed} missing={missing}")
    if missing:
        print(f"[msa:boltz_csv] {missing} complexes absent from colabfold output "
              f"-> single-seq fallback at predict")
