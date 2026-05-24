"""Shared pytest fixtures for the GPU env + model-prediction smoke tests.

The suite runs from `.venv-ecstasy` — the lightweight orchestrator that hosts
the `ecstasy` CLI, pytest, and the benchmark/pipeline code (no model deps).
Each test either:
  - subprocesses into a specific model venv via `<venv>/bin/python -c "..."`
    to verify its packages can be imported (installation/), or
  - invokes `ecstasy bench predict --config <smoke yaml>`, which itself spawns
    the model's runner inside the appropriate venv (integration/).

Tests use a per-test tmp DATA_ROOT so re-runs do not hit the
contact.npz-already-exists skip in run_predict.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VENVS: dict[str, Path] = {
    "ecstasy":   REPO_ROOT / "envs" / ".venv-ecstasy",
    "boltz":     REPO_ROOT / "envs" / ".venv-boltz",
    "esmfold":   REPO_ROOT / "envs" / ".venv-esmfold",
    "colabfold": REPO_ROOT / "envs" / ".venv-colabfold",
}
SMOKE_ENTRY_ID = "10jy"  # 286-residue homodimer; mentos_seqid30 val[0]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def venvs() -> dict[str, Path]:
    return VENVS


def _venv_python(name: str) -> Path:
    p = VENVS[name] / "bin" / "python"
    if not p.exists():
        pytest.skip(f"venv {name!r} missing at {p}")
    return p


@pytest.fixture
def run_in_venv():
    """Run a Python snippet in a named venv; return CompletedProcess.

    Usage:
        r = run_in_venv("boltz", ["import torch", "print(torch.__version__)"])
        assert r.returncode == 0, r.stderr
    """
    def _run(name: str, statements: Iterable[str], timeout: int = 180) -> subprocess.CompletedProcess:
        code = "\n".join(statements)
        return subprocess.run(
            [str(_venv_python(name)), "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
    return _run


@pytest.fixture
def smoke_config(tmp_path: Path, repo_root: Path):
    """Materialize a smoke config under tmp_path with `data_root` overridden.

    Returns a callable: smoke_config(model, extra={...}) -> Path.
    """
    import yaml

    def _make(model: str, extra: dict | None = None) -> Path:
        src = repo_root / "configs" / f"mentos_seqid30__{model}_smoke.yaml"
        if not src.exists():
            pytest.skip(f"smoke config missing: {src}")
        cfg = yaml.safe_load(src.read_text())
        data_root = tmp_path / "data_root"
        data_root.mkdir(parents=True, exist_ok=True)
        cfg["data_root"] = str(data_root)
        if extra:
            cfg.setdefault("model_config", {}).update(extra)
        dst = tmp_path / f"{model}_smoke.yaml"
        dst.write_text(yaml.safe_dump(cfg))
        return dst
    return _make


@pytest.fixture
def run_ecstasy(repo_root: Path):
    """Run `ecstasy bench <args>` in the current (pytest) venv.

    Uses the venv's `ecstasy` console script. Returns CompletedProcess.
    """
    def _run(args: list[str], env: dict | None = None, timeout: int = 1800) -> subprocess.CompletedProcess:
        ecstasy_bin = Path(sys.executable).parent / "ecstasy"
        if not ecstasy_bin.exists():
            pytest.fail(f"`ecstasy` not on this venv's PATH at {ecstasy_bin}")
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        return subprocess.run(
            [str(ecstasy_bin), *args],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(repo_root), env=merged_env,
        )
    return _run


def find_contact_npz(data_root: Path, model: str, entry_id: str = SMOKE_ENTRY_ID) -> Path | None:
    """Locate <data_root>/ecstasy/benchmarks/mentos_seqid30/predictions/<model>/<run_id>/<entry_id>/contact.npz."""
    pattern = f"ecstasy/benchmarks/mentos_seqid30/predictions/{model}/*/{entry_id}/contact.npz"
    matches = list(Path(data_root).glob(pattern))
    return matches[0] if matches else None


def assert_valid_contact_npz(path: Path, expected_L: int | None = None) -> None:
    """Sanity-check a contact.npz: probs (L, L) finite in [0, 1]."""
    import numpy as np
    assert path.exists(), f"contact.npz missing: {path}"
    d = np.load(path)
    assert "probs" in d.files, f"no 'probs' in {path}: keys={d.files}"
    probs = np.asarray(d["probs"])
    assert probs.ndim == 2 and probs.shape[0] == probs.shape[1], \
        f"probs not square (L,L); got shape {probs.shape}"
    if expected_L is not None:
        assert probs.shape[0] == expected_L, \
            f"probs shape {probs.shape} != expected ({expected_L}, {expected_L})"
    assert np.isfinite(probs).all(), "non-finite values in probs"
    pmin, pmax = float(probs.min()), float(probs.max())
    assert 0.0 <= pmin <= pmax <= 1.0 + 1e-3, f"probs not in [0,1]: min={pmin} max={pmax}"
