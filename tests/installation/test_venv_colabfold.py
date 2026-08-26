"""Installation checks for .venv-colabfold.

This venv hosts:
  - JAX 0.5.3 (CPU-default on aarch64; GPU plugin segfaults during AF2
    multimer inference — see TODO `AF2 (ColabFold-batch) JAX segfault`)
  - colabfold (ColabFold pipeline)
  - alphafold (the colabfold fork of alphafold weights/code)
  - colabfold_batch console script

The colabfold venv is intentionally minimal — it does NOT host the ecstasy CLI.
The orchestrator (.venv-ecstasy) spawns colabfold's runner via env_path.
"""
import pytest


@pytest.mark.installation
def test_colabfold_venv_jax_import(run_in_venv):
    r = run_in_venv("colabfold", [
        "import jax",
        "print('jax', jax.__version__)",
        # Default backend should be importable even if CUDA plugin is broken
        "import jax.numpy as jnp",
        "x = jnp.array([1.0, 2.0, 3.0])",
        "print('jnp sum:', float(x.sum()))",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_colabfold_venv_colabfold_import(run_in_venv):
    r = run_in_venv("colabfold", [
        "import colabfold",
        "print('colabfold:', colabfold.__file__)",
        "from colabfold.batch import run",
        "print('colabfold.batch.run OK')",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_colabfold_venv_alphafold_import(run_in_venv):
    r = run_in_venv("colabfold", [
        "import alphafold",
        "print('alphafold:', alphafold.__file__)",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_colabfold_venv_colabfold_batch_bin(venvs):
    """`colabfold_batch` console script must exist (used by the runner).

    Skips when the venv itself is absent, matching every other test in this file —
    they go through `run_in_venv`, which skips via `_venv_python`. Asserting here made
    "colabfold is not installed on this machine" report as a test failure, which is a
    different claim: the console script is missing only if the venv exists without it.
    """
    venv = venvs["colabfold"]
    if not (venv / "bin" / "python").exists():
        pytest.skip(f"venv 'colabfold' missing at {venv}")
    bin_path = venv / "bin" / "colabfold_batch"
    assert bin_path.exists(), f"colabfold_batch missing at {bin_path}"
