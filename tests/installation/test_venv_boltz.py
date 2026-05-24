"""Installation checks for .venv-boltz.

This venv hosts:
  - torch (cu124) + CUDA at runtime
  - boltz (softnanolab/boltz feat/extract-distogram-2.2.1)
  - mentos (single-sequence contact-prediction model)
  - omegaconf (used by the MENTOS runner to load Hydra configs)
  - ecstasy (so the boltz/mentos runners can `import ecstasy.*` if needed)

The torch+CUDA test will fail on a login node (no GPU); it is expected to run
on a compute node inside the sbatch driver.
"""
import pytest


@pytest.mark.installation
@pytest.mark.gpu
def test_boltz_venv_torch_cuda(run_in_venv):
    r = run_in_venv("boltz", [
        "import torch",
        "print('torch', torch.__version__, 'cuda', torch.version.cuda)",
        "assert torch.cuda.is_available(), 'CUDA not available'",
        "print('device:', torch.cuda.get_device_name(0))",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_boltz_venv_boltz_import(run_in_venv):
    r = run_in_venv("boltz", [
        "import boltz",
        "print('boltz:', boltz.__file__)",
        # smoke: the contact-extraction entry points must be importable
        "from boltz.main import predict",
        "print('boltz.main.predict OK')",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_boltz_venv_mentos_import(run_in_venv):
    r = run_in_venv("boltz", [
        "import mentos",
        "from mentos.data.esm import Alphabet",
        "a = Alphabet.from_architecture('ESM-1b')",
        "print('mentos Alphabet size:', len(a))",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_boltz_venv_omegaconf(run_in_venv):
    r = run_in_venv("boltz", [
        "from omegaconf import OmegaConf",
        "print('omegaconf OK:', OmegaConf.__module__)",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_boltz_venv_ecstasy(run_in_venv):
    r = run_in_venv("boltz", [
        "import ecstasy",
        "from ecstasy.cli import main",
        "print('ecstasy:', ecstasy.__file__)",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"
