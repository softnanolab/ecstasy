"""End-to-end smoke: `ecstasy run` for mentos (single-sequence, --limit 1).

The default `pretrain_8m_35m` preset in registry/models.yaml points at a Hydra
config + Lightning checkpoint; skip if those aren't reachable on this machine.
"""
from pathlib import Path

import pytest

from tests.conftest import SMOKE_DATASET
from tests.integration._common import assert_predict_succeeded


def _skip_if_weights_missing():
    from ecstasy.models import load_model
    p = load_model("mentos").params
    for key in ("model_config_path", "model_weights_path"):
        if not Path(p[key]).exists():
            pytest.skip(f"mentos {key} not found at {p[key]}")


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_mentos
def test_predict_mentos(run_ecstasy, data_root):
    _skip_if_weights_missing()
    r = run_ecstasy(["run", "--dataset", SMOKE_DATASET, "--model", "mentos",
                     "--limit", "1", "--no_score"], timeout=900)
    assert_predict_succeeded(r, data_root, SMOKE_DATASET, "mentos", variant="pretrain_8m_35m")
