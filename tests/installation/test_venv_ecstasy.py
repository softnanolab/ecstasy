"""Installation checks for the orchestrator venv (.venv-ecstasy).

This venv hosts the CLI + pytest + benchmark/pipeline code. No model deps.
"""
import pytest


@pytest.mark.installation
def test_ecstasy_venv_cli_imports(run_in_venv):
    r = run_in_venv("ecstasy", [
        "import ecstasy",
        "from ecstasy.cli import main",
        "from ecstasy.benchmarks import BENCHMARKS",
        "from ecstasy.models import MODELS",
        "print('benchmarks:', sorted(BENCHMARKS))",
        "print('models:', sorted(MODELS))",
        "assert 'mentos_seqid30' in BENCHMARKS",
        "assert set(MODELS) >= {'boltz2', 'mentos', 'esmfold', 'colabfold', 'msa_pairformer'}",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_ecstasy_venv_deps(run_in_venv):
    r = run_in_venv("ecstasy", [
        "import fire, yaml, pytest, numpy, pandas",
        "print('fire', fire.__version__)",
        "print('yaml', yaml.__version__)",
        "print('pytest', pytest.__version__)",
        "print('numpy', numpy.__version__)",
        "print('pandas', pandas.__version__)",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_ecstasy_cli_bench_list(venvs):
    """The `ecstasy bench list` console script must print both registries."""
    import subprocess
    ecstasy_bin = venvs["ecstasy"] / "bin" / "ecstasy"
    if not ecstasy_bin.exists():
        pytest.skip(f"ecstasy console script not found at {ecstasy_bin}")
    r = subprocess.run([str(ecstasy_bin), "bench", "list"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"
    assert "mentos_seqid30" in r.stdout
    for model in ("boltz2", "mentos", "esmfold", "colabfold", "msa_pairformer"):
        assert model in r.stdout, f"model {model!r} not in `ecstasy bench list` output: {r.stdout}"
