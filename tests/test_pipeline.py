"""CPU unit coverage for pipeline.run_score / run_compare (no GPU, no torch GT).

Uses a fake in-memory Dataset whose score() ignores the contact file, so we can
exercise the scoring aggregation, result.json layout, and the comparison table
without running a model or loading pickled GT.
"""
from __future__ import annotations

import json

import pytest

from ecstasy.datasets.base import Dataset, Entry
from ecstasy.models import load_model


class _FakeDataset(Dataset):
    def __init__(self, name="faketest"):
        super().__init__(name)

    def entries(self):
        yield Entry(id="e0", sequences=("AAAA", "CCCC"), chain_ids=("A", "B"))
        yield Entry(id="e1", sequences=("AAAA", "CCCC"), chain_ids=("A", "B"))

    def gt_for(self, entry_id):  # unused by the fake score
        raise NotImplementedError

    def score(self, entry, contact_path):
        # contact_path only needs to exist; return fixed metrics
        return {"AUC": 0.8, "P@K": 0.5, "P@K/2": 0.6, "P@K/5": 0.7, "K": 4}


@pytest.fixture
def fresh_data_root(tmp_path, monkeypatch):
    from ecstasy import config
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    config.settings.cache_clear()
    yield tmp_path
    config.settings.cache_clear()


def test_run_score_and_compare(fresh_data_root):
    from ecstasy.pipeline import Run, run_score, run_compare

    run = Run(dataset=_FakeDataset(), model=load_model("boltz2"))
    # fabricate predictions the scorer will accept (file just needs to exist)
    for eid in ("e0", "e1"):
        d = run.predictions_dir / eid
        d.mkdir(parents=True, exist_ok=True)
        (d / "contact.npz").write_bytes(b"")

    run_score(run)
    result = json.loads(run.result_path.read_text())
    assert result["summary"]["n_evaluated"] == 2
    assert result["summary"]["mean"]["P@K"] == pytest.approx(0.5)
    assert result["dataset"] == "faketest" and result["variant"] == "full"

    run_compare("faketest")
    csv = (fresh_data_root / "ecstasy" / "runs" / "faketest" / "comparison.csv")
    md = (fresh_data_root / "ecstasy" / "runs" / "faketest" / "comparison.md")
    assert csv.exists() and md.exists()
    body = csv.read_text()
    assert "boltz2" in body and "full" in body


def test_run_score_skips_missing_predictions(fresh_data_root):
    from ecstasy.pipeline import Run, run_score

    run = Run(dataset=_FakeDataset("faketest2"), model=load_model("boltz2"))
    run_score(run)  # no predictions written -> all skipped, no crash
    result = json.loads(run.result_path.read_text())
    assert result["summary"]["n_evaluated"] == 0
    assert result["summary"]["n_skipped"] == 2
