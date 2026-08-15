"""Shared pytest fixtures.

Unit tests (test_registry, test_metrics, ...) run in any env with the ecstasy
package importable. Integration smoke tests drive the `ecstasy` CLI, which spawns
each model's runner inside the appropriate venv. Each integration test uses a
per-test tmp DATA_ROOT (via the env var) so re-runs don't hit the
contact.npz-already-exists skip.
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
    "plmgraph_inter": REPO_ROOT / "envs" / "plmgraph_inter",
    "deepinteract": REPO_ROOT / "envs" / "deepinteract",
}

# Default smoke dataset (small, single-sequence fallback works without MSAs).
SMOKE_DATASET = "recent_pp"


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
    """Run a Python snippet in a named venv; return CompletedProcess."""
    def _run(name: str, statements: Iterable[str], timeout: int = 180) -> subprocess.CompletedProcess:
        return subprocess.run([str(_venv_python(name)), "-c", "\n".join(statements)],
                              capture_output=True, text=True, timeout=timeout)
    return _run


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    d = tmp_path / "DATA"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def run_ecstasy(repo_root: Path, data_root: Path):
    """Run `ecstasy <args>` via the current venv's console script, DATA_ROOT=tmp.

    Pass venv="boltz" to use a specific venv's ecstasy (e.g. for `score`, which
    needs torch + mentos to load the GT, absent from the orchestrator env).
    """
    def _run(args: list[str], venv: str | None = None, timeout: int = 1800) -> subprocess.CompletedProcess:
        if venv:
            ecstasy_bin = VENVS[venv] / "bin" / "ecstasy"
            if not ecstasy_bin.exists():
                pytest.skip(f"ecstasy CLI not found in venv {venv!r} at {ecstasy_bin}")
        else:
            ecstasy_bin = Path(sys.executable).parent / "ecstasy"
            if not ecstasy_bin.exists():
                pytest.fail(f"`ecstasy` not on this venv's PATH at {ecstasy_bin}")
        env = {**os.environ, "DATA_ROOT": str(data_root)}
        return subprocess.run([str(ecstasy_bin), *args], capture_output=True, text=True,
                              timeout=timeout, cwd=str(repo_root), env=env)
    return _run


def first_entry_id(dataset: str = SMOKE_DATASET) -> str:
    from ecstasy.datasets import load_dataset
    return next(iter(load_dataset(dataset).entries())).id


def find_contact_npz(data_root: Path, dataset: str, model: str,
                     variant: str, entry_id: str) -> Path | None:
    p = (Path(data_root) / "runs" / dataset / model / variant
         / "predictions" / entry_id / "contact.npz")
    return p if p.exists() else None


def assert_valid_contact_npz(path: Path, expected_L: int | None = None) -> None:
    import numpy as np
    assert path and path.exists(), f"contact.npz missing: {path}"
    d = np.load(path)
    assert "probs" in d.files, f"no 'probs' in {path}: keys={d.files}"
    probs = np.asarray(d["probs"])
    assert probs.ndim == 2 and probs.shape[0] == probs.shape[1], \
        f"probs not square (L,L); got {probs.shape}"
    if expected_L is not None:
        assert probs.shape[0] == expected_L
    assert np.isfinite(probs).all(), "non-finite values in probs"
    pmin, pmax = float(probs.min()), float(probs.max())
    assert 0.0 <= pmin <= pmax <= 1.0 + 1e-3, f"probs not in [0,1]: min={pmin} max={pmax}"
