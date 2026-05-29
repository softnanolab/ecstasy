"""End-to-end smoke: `ecstasy run` for colabfold (single-sequence, --limit 1)."""
import pytest

from tests.conftest import SMOKE_DATASET
from tests.integration._common import assert_predict_succeeded


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_colabfold
def test_predict_colabfold(run_ecstasy, data_root):
    r = run_ecstasy(["run", "--dataset", SMOKE_DATASET, "--model", "colabfold",
                     "--limit", "1", "--no_score"], timeout=1200)
    assert_predict_succeeded(r, data_root, SMOKE_DATASET, "colabfold", variant="single_seq")
