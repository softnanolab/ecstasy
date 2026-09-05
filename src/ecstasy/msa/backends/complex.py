"""complex MSA backend — LOCAL MSA-Pairformer paired MSA via softnanolab/colabfold-local.

This is how the benchmark's MSA-Pairformer MSAs were actually generated — NOT the
ColabFold API (that's the separate ``complex_api`` backend). Pipeline: colabfold-local's
``get_paired_msa_local()`` runs local ``colabfold_search`` (mmseqs-gpu) against
``COLABFOLD_DBS`` and stitches one ``#L1,L2\\t1,1``-headed complex a3m per pair into the
store. The paired-sequence filter + chain-aware diversity selection + depth cap (512)
happen later, at model load in ``msa_pairformer_runner.py`` — not here.

Do NOT confuse this with ``boltz_csv`` (Boltz-2): same search engine, but boltz_csv keeps
paired+unpaired per-chain CSVs to reproduce ``boltz --use_msa_server``, whereas this
emits a single stitched complex a3m for MSA-Pairformer. See ``msa/README.md``.

Needs a GPU node + a colabfold-local checkout (the ``third_party/colabfold-local``
submodule, or ``$COLABFOLD_LOCAL_DIR``) and its venv (``$COLABFOLD_LOCAL_VENV``),
pinned at the SHA recorded in ``msa/README.md``.

Backend interface: ``prepare(datasets) -> Path``, ``submit(datasets) -> job|None``
(sbatch), ``ingest(datasets, out_dir=None) -> None`` (copy a3ms into the store).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ecstasy.config import env_value, settings
from ecstasy.msa import store
from ecstasy.msa.backends._common import collect_complexes, work_dir
from ecstasy.msa.backends.boltz_csv import _colabfold_paths  # shared mmseqs/dbs/cuda/partition

_DRIVER = Path(__file__).resolve().parent / "_complex_local_driver.py"


def _colabfold_local_dir() -> Path:
    """Resolve the colabfold-local checkout: $COLABFOLD_LOCAL_DIR, else the in-repo
    submodule, else ~/colabfold-local (documented fallback)."""
    explicit = env_value("COLABFOLD_LOCAL_DIR")
    if explicit:
        return Path(explicit)
    repo_root = Path(__file__).resolve().parents[4]   # …/src/ecstasy/msa/backends/complex.py -> repo
    sub = repo_root / "third_party" / "colabfold-local"
    if sub.exists():
        return sub
    return Path.home() / "colabfold-local"


def _colabfold_local_venv(cl_dir: Path) -> str:
    explicit = env_value("COLABFOLD_LOCAL_VENV")
    if explicit:
        return explicit
    for cand in (".venv", "venv", ".venv-colabfold"):
        if (cl_dir / cand / "bin" / "activate").exists():
            return str(cl_dir / cand)
    return str(cl_dir / ".venv")


def _manifest_path() -> Path:
    return work_dir("complex") / "manifest.json"


def prepare(datasets: list[str]) -> Path:
    """List store-missing complexes; write a reference FASTA + a driver manifest."""
    store.complex_dir().mkdir(parents=True, exist_ok=True)
    items = collect_complexes(datasets)
    missing = {h: v for h, v in items.items() if not store.path_for_pair(v["seqs"]).exists()}
    work = work_dir("complex")
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "missing.fasta"
    with fasta.open("w") as f:
        for h, v in sorted(missing.items()):
            f.write(f">{v['header']}\n{v['query']}\n")   # query == 'seqA:seqB' (colabfold-local format)
    manifest = [{"seqs": v["seqs"], "header": v["header"],
                 "dst": str(store.path_for_pair(v["seqs"]))} for v in missing.values()]
    _manifest_path().write_text(json.dumps(manifest))
    print(f"[msa:complex] datasets={datasets}")
    print(f"[msa:complex] unique={len(items)} already_in_store={len(items)-len(missing)} missing={len(missing)}")
    print(f"[msa:complex] wrote {fasta} + manifest; --phase submit sbatches colabfold-local (local, GPU)")
    return fasta


def _write_sbatch() -> Path:
    p = _colabfold_paths()
    cl_dir = _colabfold_local_dir()
    cl_venv = _colabfold_local_venv(cl_dir)
    work = work_dir("complex")
    (work / "logs").mkdir(parents=True, exist_ok=True)
    script = work / "generate_complex.sbatch"
    script.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=ecstasy-msa-complex
#SBATCH --output={work}/logs/msa_%j.out
#SBATCH --error={work}/logs/msa_%j.err
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition={p['partition']}
set -euo pipefail

module load {p['cuda_module']}
# colabfold-local venv (get_paired_msa_local deps); .venv-colabfold supplies colabfold_search.
# shellcheck disable=SC1091
source "{cl_venv}/bin/activate"
export PATH="{p['venv']}/bin:$PATH"
export COLABFOLD_LOCAL_DIR="{cl_dir}"
export DATA_DIR="{p['dbs']}"      # local ColabFold DBs (override colabfold-local's default)
export MMSEQS_BIN="{p['mmseqs']}" # ecstasy's vendored mmseqs-gpu
# mmseqs writes large prefilter temporaries. TMPDIR defaults to node-local
# /local/user/$UID, where the prefilter dies with "Could not open .../pref_0.0 for
# writing"; use the big shared filesystem, as boltz_csv already does.
export TMPDIR="${{SCRATCHDIR:-/tmp}}/ecstasy_cf_complex_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
# The vendored binary reports "MMseqs2 was compiled without CUDA support", so --gpu 1
# makes the prefilter die. CPU-only until a CUDA-enabled mmseqs is vendored (the
# --gpus-per-node request above is then redundant on such clusters).
export MSA_GPU=0

python "{_DRIVER}" "{_manifest_path()}"
echo "DONE complex (colabfold-local) MSA generation"
""")
    return script


def submit(datasets: list[str]) -> str | None:
    """prepare(), then sbatch the local colabfold-local generation (GPU)."""
    fasta = prepare(datasets)
    if fasta.stat().st_size == 0:
        print("[msa:complex] nothing missing; store is complete")
        return None
    script = _write_sbatch()
    res = subprocess.run(["sbatch", "--parsable", str(script)],
                         capture_output=True, text=True, check=True)
    job = res.stdout.strip()
    print(f"[msa:complex] submitted job {job} ({script}); writes straight to the store. "
          f"--phase ingest afterwards only verifies coverage.")
    return job


def ingest(datasets: list[str], out_dir: str | None = None) -> None:
    """Copy externally-generated a3ms into the store, or report coverage.

    If ``out_dir`` is given (e.g. a manual ``colabfold-local`` run, a3ms named
    ``<pair_hash>.a3m``), copy any matching missing complexes into the store. The
    sbatch path in ``submit`` writes straight to the store, so it needs no copy.
    """
    items = collect_complexes(datasets)
    copied = 0
    if out_dir:
        src = Path(out_dir)
        for v in items.values():
            dst = store.path_for_pair(v["seqs"])
            cand = src / f"{v['header']}.a3m"
            if not dst.exists() and cand.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(cand.read_text())
                copied += 1
    have, collapsed, _ = store.depth_report(items)
    if out_dir:
        print(f"[msa:complex] copied {copied} a3ms from {out_dir}")
    print(f"[msa:complex] store coverage: {have}/{len(items)} complexes "
          f"({collapsed} collapsed to query-only — proximity dropped all paired hits)")
