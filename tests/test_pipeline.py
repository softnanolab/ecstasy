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
    csv = (fresh_data_root / "runs" / "faketest" / "comparison.csv")
    md = (fresh_data_root / "runs" / "faketest" / "comparison.md")
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


class _NDataset(_FakeDataset):
    """N two-chain entries e0..e{N-1} for shard-partitioning tests."""

    def __init__(self, name="shardtest", n=6):
        super().__init__(name)
        self._n = n

    def entries(self):
        for i in range(self._n):
            yield Entry(id=f"e{i}", sequences=("AAAA", "CCCC"), chain_ids=("A", "B"))


def _collect_shard(run, shard, monkeypatch):
    """Run run_predict with predict_one/store mocked; return the entry ids selected."""
    from ecstasy import pipeline
    seen: list[str] = []

    def _fake_predict_one(model, entry, msa, out_dir, profile=False):
        seen.append(entry.id)  # record selection; don't write contact.npz (no skips)
        return out_dir / "contact.npz"

    monkeypatch.setattr(pipeline, "predict_one", _fake_predict_one)
    monkeypatch.setattr(pipeline.store, "lookup", lambda *a, **k: None)
    pipeline.run_predict(run, shard=shard)
    return seen


def test_run_predict_shard_partitions_exactly_once(fresh_data_root, monkeypatch):
    from ecstasy.pipeline import Run
    run = Run(dataset=_NDataset(n=6), model=load_model("boltz2"))

    assert _collect_shard(run, None, monkeypatch) == [f"e{i}" for i in range(6)]
    p0 = _collect_shard(run, "0/3", monkeypatch)
    p1 = _collect_shard(run, "1/3", monkeypatch)
    p2 = _collect_shard(run, "2/3", monkeypatch)
    assert p0 == ["e0", "e3"] and p1 == ["e1", "e4"] and p2 == ["e2", "e5"]
    # disjoint union == every entry exactly once (no collision, no drop)
    assert sorted(p0 + p1 + p2) == [f"e{i}" for i in range(6)]


def test_run_predict_shard_rejects_out_of_range(fresh_data_root, monkeypatch):
    from ecstasy.pipeline import Run, run_predict
    run = Run(dataset=_NDataset(n=4), model=load_model("boltz2"))
    for bad in ("3/3", "2/2", "5/3"):
        with pytest.raises(ValueError):
            run_predict(run, shard=bad)


def test_registry_new_baselines_presets(fresh_data_root):
    """The new inter-chain baselines expose the expected presets / msa kinds."""
    from ecstasy.models import presets_for
    assert presets_for("plmgraph_inter") == ["full"]
    assert load_model("plmgraph_inter").msa == "boltz_csv"
    assert load_model("deepinteract").msa == "none"
