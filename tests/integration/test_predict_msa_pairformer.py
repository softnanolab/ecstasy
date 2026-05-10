"""End-to-end smoke: `ecstasy bench predict` for MSA Pairformer.

The smoke config does NOT set complex_a3m_dir, so the runner falls back to a
single-sequence MSA (degenerate baseline; per TODO, gives P@K=0 on signal but
still produces a valid contact.npz). The test only verifies the artifact.
"""
import pytest

from tests.integration._common import assert_predict_succeeded


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_msa_pairformer
def test_predict_msa_pairformer(smoke_config, run_ecstasy):
    cfg = smoke_config("msa_pairformer")
    r = run_ecstasy(["bench", "predict", "--config", str(cfg)], timeout=900)
    assert_predict_succeeded(r, cfg, "msa_pairformer")
