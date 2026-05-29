"""End-to-end smoke: `ecstasy run` for msa_pairformer (single-sequence, --limit 1).

With no complex MSA in the store the runner builds a single-sequence complex a3m,
which gives little signal but exercises the full path end-to-end.
"""
import pytest

from tests.conftest import SMOKE_DATASET
from tests.integration._common import assert_predict_succeeded


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_msa_pairformer
def test_predict_msa_pairformer(run_ecstasy, data_root):
    r = run_ecstasy(["run", "--dataset", SMOKE_DATASET, "--model", "msa_pairformer",
                     "--limit", "1", "--no_score"], timeout=900)
    assert_predict_succeeded(r, data_root, SMOKE_DATASET, "msa_pairformer", variant="full")
