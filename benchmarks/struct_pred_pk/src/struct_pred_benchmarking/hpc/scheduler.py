"""qstat-capped PBS submitter — no DB, just filesystem state.

Pattern adapted from binder_design/colabfold/scheduler.py (Sarkisyan-lab):
poll user queue, sleep when at capacity, qsub one PBS script per protein.
Idempotent: a protein whose final sample CIF already exists is skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from importlib import resources
from pathlib import Path

from struct_pred_benchmarking.config import BenchmarkConfig
from struct_pred_benchmarking.data.select_val import load_manifest


POLL_SEC = 60


def _template_path() -> Path:
    pkg = resources.files("struct_pred_benchmarking.hpc")
    p = pkg / "templates" / "boltz_predict.pbs.tmpl"
    return Path(str(p))


def _user_queued_count() -> int:
    user = os.environ.get("USER", "")
    try:
        out = subprocess.check_output(["qstat", "-u", user], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    count = 0
    for line in out.splitlines():
        parts = line.split()
        # qstat -u outputs columns; the 10th (status) is one of Q, R, H, etc.
        if len(parts) >= 10 and parts[9] == "Q":
            count += 1
    return count


def _predictions_complete(run_dir: Path, pdb_id: str, n_samples: int) -> bool:
    pred_dir = run_dir / "boltz_predictions" / pdb_id
    if not pred_dir.exists():
        return False
    cifs = list(pred_dir.glob("boltz_results_*/predictions/*/*_model_*.cif"))
    if not cifs:
        cifs = list(pred_dir.rglob("*_model_*.cif"))
    return len(cifs) >= n_samples


def _render_pbs(cfg: BenchmarkConfig, pdb_id: str) -> Path:
    template = _template_path().read_text()
    text = template.format(
        id=pdb_id,
        queue=cfg.pbs.queue,
        ncpus=cfg.pbs.ncpus,
        mem_gb=cfg.pbs.mem_gb,
        ngpus=cfg.pbs.ngpus,
        walltime=cfg.pbs.walltime,
        run_dir=str(cfg.run_dir),
        repo_root=str(cfg.repo_root),
        boltz_env=str(cfg.boltz_env),
        conda_base=str(cfg.conda_base),
        diffusion_samples=cfg.diffusion_samples,
        seed=cfg.seed,
    )
    pbs_dir = cfg.run_dir / "logs" / "pbs_scripts"
    pbs_dir.mkdir(parents=True, exist_ok=True)
    out_path = pbs_dir / f"boltz_{pdb_id}.pbs"
    out_path.write_text(text)
    (cfg.run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return out_path


def submit_all(cfg: BenchmarkConfig) -> list[str]:
    if shutil.which("qsub") is None:
        raise RuntimeError("qsub not found on PATH — are you on a PBS submit host?")

    job_ids: list[str] = []
    for entry in load_manifest(cfg):
        pdb_id = entry["id"]
        if _predictions_complete(cfg.run_dir, pdb_id, cfg.diffusion_samples):
            print(f"[skip] {pdb_id}: predictions already complete")
            continue

        while _user_queued_count() >= cfg.pbs.max_queued:
            print(
                f"[wait] {_user_queued_count()} jobs queued >= cap {cfg.pbs.max_queued}; "
                f"sleeping {POLL_SEC}s"
            )
            time.sleep(POLL_SEC)

        pbs_path = _render_pbs(cfg, pdb_id)
        result = subprocess.run(
            ["qsub", str(pbs_path)], check=True, capture_output=True, text=True
        )
        job_id = result.stdout.strip()
        print(f"[qsub] {pdb_id} -> {job_id}")
        job_ids.append(job_id)

    return job_ids


def wait_for_all(cfg: BenchmarkConfig, poll_sec: int = POLL_SEC) -> None:
    """Block until every protein in the manifest has its expected sample CIFs on disk."""
    manifest = load_manifest(cfg)
    while True:
        pending = [
            e["id"]
            for e in manifest
            if not _predictions_complete(cfg.run_dir, e["id"], cfg.diffusion_samples)
        ]
        if not pending:
            return
        print(f"[wait] {len(pending)} predictions pending: {pending[:5]}...")
        time.sleep(poll_sec)
