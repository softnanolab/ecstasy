"""The structure half of run_score.

The load-bearing property is that structure scoring is **additive**: a model that emits
no structure, a dataset with no full-atom GT, or a DockQ failure on one target must each
leave the contact metrics exactly as they were. That is not a nicety — a partial DockQ
failure discarding a whole run's contact numbers would be a far worse regression than not
having DockQ at all.

Also pins the reporting decisions that exist because reporting them the other way
produced wrong conclusions on this codebase (see CLAUDE.md): iRMSD/LRMSD travel with
DockQ, and the homodimer split names its definition.
"""
from __future__ import annotations

import json

import pytest

from ecstasy.datasets.base import Dataset, Entry
from ecstasy.metrics import DEFAULT_STRUCTURE_METRICS
from ecstasy.models import load_model


class _FakeDataset(Dataset):
    """Canned scores, so the aggregation is under test rather than DockQ."""

    kind = "fake_structure_split"
    has_structure_gt = True

    def __init__(self, name="structtest", scores=None, fail_on=(), structure_gt=True):
        super().__init__(name, description="fake", expected_entries=len(scores or {}))
        self._scores = scores or {}
        self._fail_on = set(fail_on)
        self.has_structure_gt = structure_gt

    def entries(self):
        for eid in self._scores:
            yield Entry(id=eid, sequences=("AAAA", "CCCC"), chain_ids=("A", "B"))

    def gt_for(self, entry_id):
        raise NotImplementedError

    def has_gt(self, entry_id):
        return True

    def score(self, entry, contact_path, metrics=None):
        return {"AUC": 0.8, "P@K": 0.5, "P@K/2": 0.6, "P@K/5": 0.7, "K": 4.0}

    def score_structure(self, entry, structure_path, work_dir=None, metrics=None,
                        null_draws=0, natives_dir=None):
        if entry.id in self._fail_on:
            raise RuntimeError("boom")
        return dict(self._scores[entry.id])


_TWO = {
    "e0": {"DockQ": 0.60, "Fnat": 0.5, "iRMSD": 2.0, "LRMSD": 5.0, "TM_mean": 0.9,
           "TM_min": 0.85, "CA_RMSD_mean": 1.5, "is_homodimer": 1.0},
    "e1": {"DockQ": 0.20, "Fnat": 0.1, "iRMSD": 9.0, "LRMSD": 20.0, "TM_mean": 0.7,
           "TM_min": 0.6, "CA_RMSD_mean": 4.0, "is_homodimer": 0.0},
}


@pytest.fixture
def fresh_data_root(tmp_path, monkeypatch):
    from ecstasy import config
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    config.settings.cache_clear()
    yield tmp_path
    config.settings.cache_clear()


def _run(name, scores, **kw):
    from ecstasy.pipeline import Run
    return Run(dataset=_FakeDataset(name, scores, **kw), model=load_model("boltz2"))


def _write_predictions(run, ids, structure=True):
    for eid in ids:
        d = run.predictions_dir / eid
        d.mkdir(parents=True, exist_ok=True)
        (d / "contact.npz").write_bytes(b"")
        if structure:
            (d / "structure.npz").write_bytes(b"")


def _scored(name, scores, structure=True, **kw):
    from ecstasy.pipeline import run_score
    run = _run(name, scores, **kw)
    _write_predictions(run, list(scores), structure=structure)
    run_score(run)
    return json.loads(run.result_path.read_text())


class TestAdditive:
    """A structure failure must never cost a target its contact metrics."""

    def test_contact_metrics_survive_a_structure_exception(self, fresh_data_root):
        r = _scored("failtest", _TWO, fail_on=("e1",))
        assert r["summary"]["n_evaluated"] == 2            # both still contact-scored
        assert r["summary"]["n_structure_skipped"] == 1
        assert r["per_protein"]["e1"]["P@K"] == pytest.approx(0.5)
        assert "DockQ" not in r["per_protein"]["e1"]
        assert r["summary"]["structure"]["n"] == 1         # the other one still counted

    def test_no_structure_npz_means_no_structure_block(self, fresh_data_root):
        r = _scored("nostruct", _TWO, structure=False)
        assert "structure" not in r["summary"]
        assert r["summary"]["n_evaluated"] == 2

    def test_dataset_without_structure_gt_is_never_asked(self, fresh_data_root):
        """has_structure_gt gates the call, so a structure.npz beside a contact-only
        dataset is ignored rather than raising NotImplementedError."""
        r = _scored("nogt", _TWO, structure_gt=False)
        assert "structure" not in r["summary"]
        assert r["summary"]["n_evaluated"] == 2

    def test_structure_skips_are_reported_not_silent(self, fresh_data_root):
        r = _scored("failtest2", _TWO, fail_on=("e0", "e1"))
        assert r["summary"]["n_structure_skipped"] == 2
        assert len(r["structure_skipped_first_20"]) == 2
        assert "structure" not in r["summary"]     # nothing scored, so no block


class TestAggregation:
    def test_reports_mean_and_median(self, fresh_data_root):
        st = _scored("agg", _TWO)["summary"]["structure"]
        assert st["n"] == 2
        assert st["mean"]["DockQ"] == pytest.approx(0.40)
        assert st["median"]["DockQ"] == pytest.approx(0.40)

    def test_rmsd_terms_travel_with_dockq(self, fresh_data_root):
        """DockQ averages fnat with two RMSD terms, so an unformed backbone still scores.
        A summary carrying DockQ without them has produced wrong conclusions here."""
        st = _scored("rmsd", _TWO)["summary"]["structure"]
        for key in ("iRMSD", "LRMSD"):
            assert key in st["mean"] and key in st["median"]
        assert st["mean"]["iRMSD"] == pytest.approx(5.5)

    def test_quality_bands(self, fresh_data_root):
        st = _scored("bands", _TWO)["summary"]["structure"]
        assert st["acceptable_fraction"] == pytest.approx(0.5)   # only DockQ 0.60
        assert st["medium_fraction"] == pytest.approx(0.5)
        assert st["high_fraction"] == pytest.approx(0.0)

    def test_split_names_its_definition(self, fresh_data_root):
        """'Homodimer' has meant two different things on the same split (129 vs 39),
        so the key must say which notion produced these numbers."""
        st = _scored("split", _TWO)["summary"]["structure"]
        assert "is_homodimer flag" in st["split_definition"]
        assert st["homodimer_flag"]["n"] == 1
        assert st["heterodimer_flag"]["n"] == 1
        assert st["homodimer_flag"]["mean"]["DockQ"] == pytest.approx(0.60)

    def test_unlabelled_rows_produce_no_split(self, fresh_data_root):
        """Defaulting a missing flag would report a split that was never measured."""
        scores = {k: {kk: vv for kk, vv in v.items() if kk != "is_homodimer"}
                  for k, v in _TWO.items()}
        st = _scored("nosplit", scores)["summary"]["structure"]
        assert "homodimer_flag" not in st and "split_definition" not in st


class TestDefaults:
    def test_default_structure_set_includes_the_rmsd_terms(self):
        for key in ("DockQ", "iRMSD", "LRMSD", "TM_mean"):
            assert key in DEFAULT_STRUCTURE_METRICS

    def test_null_floor_is_not_in_the_default_set(self):
        """It costs null_draws extra DockQ subprocesses per target; opting in is the
        point, so scoring cannot silently become 10x slower."""
        assert not any("null" in m for m in DEFAULT_STRUCTURE_METRICS)

    def test_partial_run_still_withholds_the_headline(self, fresh_data_root):
        """The coverage guard applies to structure numbers too."""
        from ecstasy.pipeline import run_score
        run = _run("partial", _TWO)
        _write_predictions(run, ["e0"])            # only one of two
        run_score(run)
        s = json.loads(run.result_path.read_text())["summary"]
        assert s.get("partial") is True
        assert "structure" not in s and "mean" not in s
