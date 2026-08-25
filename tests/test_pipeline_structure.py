"""CPU coverage for the structure/DockQ half of pipeline.run_score and run_compare.

Uses a fake dataset with canned per-entry structure scores, so what is under test is
ecstasy's aggregation — quality bands, the homo/hetero split, failure isolation — and
not the DockQ binary, which is external and has its own correctness story.

The load-bearing property throughout: the structure path is *additive*. A model that
emits no structure, a dataset with no full-atom GT, or a DockQ failure on one target
must all leave the contact metrics exactly as they were.
"""
from __future__ import annotations

import json

import pytest

from ecstasy.datasets.base import Dataset, Entry
from ecstasy.models import load_model


class _StructureDataset(Dataset):
    has_structure_gt = True

    def __init__(self, name="structtest", scores=None, fail_on=()):
        super().__init__(name)
        self._scores = scores or {}
        self._fail_on = set(fail_on)

    def entries(self):
        for eid in self._scores:
            yield Entry(id=eid, sequences=("AAAA", "CCCC"), chain_ids=("A", "B"))

    def gt_for(self, entry_id):                       # unused by the fake score
        raise NotImplementedError

    def score(self, entry, contact_path):
        return {"AUC": 0.8, "P@K": 0.5, "P@K/2": 0.6, "P@K/5": 0.7, "K": 4}

    def score_structure(self, entry, structure_path, work_dir=None, **kw):
        if entry.id in self._fail_on:
            raise RuntimeError("boom")
        return dict(self._scores[entry.id])


@pytest.fixture
def fresh_data_root(tmp_path, monkeypatch):
    from ecstasy import config
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    config.settings.cache_clear()
    yield tmp_path
    config.settings.cache_clear()


_TWO = {
    "e0": {"DockQ": 0.60, "Fnat": 0.5, "iRMSD": 2.0, "LRMSD": 5.0,
           "TM_mean": 0.9, "TM_min": 0.85, "CA_RMSD_mean": 1.5, "is_homodimer": 1.0},
    "e1": {"DockQ": 0.20, "Fnat": 0.1, "iRMSD": 9.0, "LRMSD": 20.0,
           "TM_mean": 0.7, "TM_min": 0.6, "CA_RMSD_mean": 4.0, "is_homodimer": 0.0},
}


def _make_run(name, scores, **kw):
    from ecstasy.pipeline import Run
    return Run(dataset=_StructureDataset(name, scores, **kw), model=load_model("boltz2"))


def _write_predictions(run, ids, structure=True):
    for eid in ids:
        d = run.predictions_dir / eid
        d.mkdir(parents=True, exist_ok=True)
        (d / "contact.npz").write_bytes(b"")
        if structure:
            (d / "structure.npz").write_bytes(b"")


def _scored(name, scores, **kw):
    """Predict-and-score shortcut: fabricate contact+structure outputs, then score."""
    from ecstasy.pipeline import run_score
    run = _make_run(name, scores, **kw)
    _write_predictions(run, list(scores))
    run_score(run)
    return run, json.loads(run.result_path.read_text())


def test_structure_metrics_reach_result_json(fresh_data_root):
    _, result = _scored("structtest", _TWO)
    st = result["summary"]["structure"]
    assert st["n"] == 2
    assert st["mean"]["DockQ"] == pytest.approx(0.40)
    assert st["median"]["DockQ"] == pytest.approx(0.40)
    assert st["mean"]["iRMSD"] == pytest.approx(5.5)
    assert result["per_protein"]["e0"]["DockQ"] == pytest.approx(0.60)


def test_contact_metrics_are_unchanged_by_the_structure_path(fresh_data_root):
    _, result = _scored("bothtest", _TWO)
    assert result["summary"]["mean"]["P@K"] == pytest.approx(0.5)
    assert result["summary"]["n_evaluated"] == 2


def test_structure_summary_reports_quality_bands(fresh_data_root):
    _, result = _scored("bandtest", _TWO)
    st = result["summary"]["structure"]
    assert st["acceptable_fraction"] == pytest.approx(0.5)      # only DockQ 0.60
    assert st["medium_fraction"] == pytest.approx(0.5)
    assert st["high_fraction"] == pytest.approx(0.0)


def test_structure_summary_splits_homo_and_hetero(fresh_data_root):
    """Under the linker hack a homodimer is one sequence duplicated around a poly-G run,
    which no language model has seen in training. Pooling the two hides that."""
    _, result = _scored("splittest", _TWO)
    st = result["summary"]["structure"]
    assert st["homodimer"]["n"] == 1
    assert st["homodimer"]["mean"]["DockQ"] == pytest.approx(0.60)
    assert st["heterodimer"]["n"] == 1
    assert st["heterodimer"]["mean"]["DockQ"] == pytest.approx(0.20)


def test_no_structure_npz_means_no_structure_summary(fresh_data_root):
    """Contact-only models must be entirely unaffected."""
    from ecstasy.pipeline import run_score
    run = _make_run("nostruct", _TWO)
    _write_predictions(run, list(_TWO), structure=False)
    run_score(run)
    summary = json.loads(run.result_path.read_text())["summary"]
    assert "structure" not in summary
    assert summary["n_evaluated"] == 2


def test_dataset_without_structure_gt_is_never_asked(fresh_data_root):
    """`has_structure_gt` gates the call, so a structure.npz beside a contact-only
    dataset is ignored rather than raising NotImplementedError."""
    from ecstasy.pipeline import Run, run_score

    ds = _StructureDataset("nogt", _TWO)
    ds.has_structure_gt = False
    run = Run(dataset=ds, model=load_model("boltz2"))
    _write_predictions(run, list(_TWO))
    run_score(run)
    summary = json.loads(run.result_path.read_text())["summary"]
    assert "structure" not in summary
    assert summary["n_evaluated"] == 2


def test_contact_metrics_survive_a_structure_failure(fresh_data_root):
    """A DockQ failure on one target must not discard contact metrics that succeeded."""
    _, result = _scored("failtest", _TWO, fail_on=("e1",))
    assert result["summary"]["n_evaluated"] == 2               # both contact-scored
    assert result["summary"]["n_structure_skipped"] == 1
    assert result["summary"]["structure"]["n"] == 1
    assert "DockQ" not in result["per_protein"]["e1"]
    assert result["per_protein"]["e1"]["P@K"] == pytest.approx(0.5)


def test_structure_skip_is_reported_not_silent(fresh_data_root):
    scores = {"e0": dict(_TWO["e0"]),
              "e1": {"_skipped": "no full-atom ground truth for this entry"}}
    _, result = _scored("skiptest", scores)
    assert result["structure_skipped_first_20"] == [
        ["e1", "no full-atom ground truth for this entry"]]
    assert result["summary"]["structure"]["n"] == 1


def test_compare_table_carries_the_dockq_columns(fresh_data_root):
    from ecstasy.pipeline import run_compare
    _scored("cmptest", _TWO)
    run_compare("cmptest")

    root = fresh_data_root / "runs" / "cmptest"
    header = (root / "comparison.csv").read_text().splitlines()[0]
    assert "mean_DockQ" in header and "median_DockQ" in header
    md = (root / "comparison.md").read_text()
    assert "## Structure (DockQ)" in md
    # iRMSD and LRMSD must sit beside DockQ: when the backbone has not formed, fnat
    # carries the score and a DockQ column read alone is misleading.
    assert "iRMSD" in md and "LRMSD" in md


def test_compare_table_omits_the_dockq_section_for_contact_only_runs(fresh_data_root):
    from ecstasy.pipeline import run_compare, run_score
    run = _make_run("cmpplain", _TWO)
    _write_predictions(run, list(_TWO), structure=False)
    run_score(run)
    run_compare("cmpplain")
    md = (fresh_data_root / "runs" / "cmpplain" / "comparison.md").read_text()
    assert "## Structure (DockQ)" not in md
