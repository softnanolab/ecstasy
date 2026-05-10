"""End-to-end smoke: `ecstasy bench predict` for ColabFold (AF2).

AF2 GPU JAX segfaults on aarch64 (see TODO `AF2 (ColabFold-batch) JAX segfault`).
This test forces `JAX_PLATFORMS=cpu` and uses msa_mode=single_sequence to avoid
the api.colabfold.com dependency. CPU AF2 on a 286-residue dimer takes ~5-7 min
so the timeout is generous.

Set SKIP_COLABFOLD_TEST=1 to skip entirely (useful while AF2 is in limbo).
"""
import os

import pytest

from tests.integration._common import assert_predict_succeeded


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_colabfold
@pytest.mark.slow
def test_predict_colabfold(smoke_config, run_ecstasy):
    if os.environ.get("SKIP_COLABFOLD_TEST"):
        pytest.skip("SKIP_COLABFOLD_TEST set")
    cfg = smoke_config("colabfold")
    r = run_ecstasy(
        ["bench", "predict", "--config", str(cfg)],
        env={"JAX_PLATFORMS": "cpu"},
        timeout=1800,
    )
    assert_predict_succeeded(r, cfg, "colabfold")
