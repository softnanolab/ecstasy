"""End-to-end smoke: `ecstasy run` for boltz2 (single-sequence fallback, --limit 1)."""
import json

import pytest

from tests.conftest import SMOKE_DATASET
from tests.integration._common import assert_predict_succeeded


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_boltz2
def test_predict_boltz2(run_ecstasy, data_root):
    r = run_ecstasy(["run", "--dataset", SMOKE_DATASET, "--model", "boltz2",
                     "--limit", "1", "--no_score"], timeout=900)
    assert_predict_succeeded(r, data_root, SMOKE_DATASET, "boltz2", variant="full")


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_boltz2
def test_score_boltz2(run_ecstasy, data_root):
    """Predict (any env) then score in .venv-boltz, which has torch + mentos to
    load the MENTOS-pickled GT (the orchestrator env has neither)."""
    r = run_ecstasy(["run", "--dataset", SMOKE_DATASET, "--model", "boltz2",
                     "--limit", "1", "--no_score"], timeout=900)
    assert_predict_succeeded(r, data_root, SMOKE_DATASET, "boltz2", variant="full")

    r = run_ecstasy(["score", "--dataset", SMOKE_DATASET, "--model", "boltz2", "--limit", "1"],
                    venv="boltz", timeout=300)
    assert r.returncode == 0, f"score failed\nstdout:{r.stdout}\nstderr:{r.stderr}"

    result = (data_root / "ecstasy" / "runs" / SMOKE_DATASET / "boltz2" / "full" / "result.json")
    assert result.exists(), f"no result.json at {result}"
    summary = json.loads(result.read_text())["summary"]
    assert summary["n_evaluated"] >= 1, f"nothing scored: {summary}"
