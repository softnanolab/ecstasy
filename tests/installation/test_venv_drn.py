"""Installation checks for the DRN-1D2D_Inter env (envs/drn_1d2d_inter).

Unlike the other models this is a conda/micromamba env (py3.8), not a uv .venv-*.
It hosts:
  - torch 1.9.1 (+cu111 on a GPU host) + CUDA at runtime
  - fair-esm (ESM-1b + ESM-MSA-1b, the embedding/attention features)
  - biopython, numpy

The torch+CUDA test fails on a login node (no GPU); expected on a compute node.
The external tool binaries (CCMpred/alnstats/fasta2aln/hhmake/hhfilter) live under
tools/ and are covered by the integration smoke, not here.
"""
import pytest


@pytest.mark.installation
@pytest.mark.gpu
def test_drn_venv_torch_cuda(run_in_venv):
    r = run_in_venv("drn_1d2d_inter", [
        "import torch",
        "print('torch', torch.__version__, 'cuda', torch.version.cuda)",
        "assert torch.cuda.is_available(), 'CUDA not available'",
        "print('device:', torch.cuda.get_device_name(0))",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_drn_venv_esm_import(run_in_venv):
    r = run_in_venv("drn_1d2d_inter", [
        "import esm",
        "print('esm:', esm.__file__)",
        # The two pretrained backbones DRN fuses.
        "from esm.pretrained import esm1b_t33_650M_UR50S, esm_msa1b_t12_100M_UR50S",
        "print('esm1b + esm_msa1b entrypoints OK')",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"


@pytest.mark.installation
def test_drn_venv_biopython(run_in_venv):
    r = run_in_venv("drn_1d2d_inter", [
        "import Bio",
        "from Bio import SeqIO",
        "print('biopython OK:', Bio.__version__)",
    ])
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"
