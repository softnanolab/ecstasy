"""End-to-end smoke: `ecstasy bench predict` for ESMFold (single-sequence).

ESMFold downloads weights on first use (~3 GB). Allow longer timeout.
"""
import pytest

from tests.integration._common import assert_predict_succeeded


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_esmfold
def test_predict_esmfold(smoke_config, run_ecstasy):
    cfg = smoke_config("esmfold")
    r = run_ecstasy(["bench", "predict", "--config", str(cfg)], timeout=1800)
    assert_predict_succeeded(r, cfg, "esmfold")
