"""Installation checks for .venv-esmfold.

This venv hosts:
  - torch (cu124) + CUDA
  - fair-esm (provides ESMFold)
  - openfold (patched for aarch64; provides the structure module ESMFold uses)
  - MSA_Pairformer (capital-S import name)
  - ecstasy (so the runners can use shared helpers)

The torch+CUDA test will fail on a login node; expected on a compute node.
"""
import pytest


@pytest.mark.installation
@pytest.mark.gpu
def test_esmfold_venv_torch_cuda(run_in_venv):
    r = run_in_venv("esmfold", [
        "import torch",
        "print('torch', torch.__version__, 'cuda', torch.version.cuda)",
        "assert torch.cuda.is_available(), 'CUDA not available'",
        "print('device:', torch.cuda.get_device_name(0))",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_esmfold_venv_esm_import(run_in_venv):
    r = run_in_venv("esmfold", [
        "import esm",
        "print('esm:', esm.__file__)",
        # ESMFold entrypoint
        "from esm.pretrained import esmfold_v1",
        "print('esmfold_v1 OK')",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_esmfold_venv_openfold_import(run_in_venv):
    r = run_in_venv("esmfold", [
        "import openfold",
        "print('openfold:', openfold.__file__)",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_esmfold_venv_msa_pairformer_import(run_in_venv):
    r = run_in_venv("esmfold", [
        "from MSA_Pairformer.model import MSAPairformer",
        "from MSA_Pairformer.dataset import MSA, prepare_msa_masks",
        "print('MSA_Pairformer OK')",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_esmfold_venv_ecstasy(run_in_venv):
    r = run_in_venv("esmfold", [
        "import ecstasy",
        "from ecstasy.cli import main",
        "print('ecstasy:', ecstasy.__file__)",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"
