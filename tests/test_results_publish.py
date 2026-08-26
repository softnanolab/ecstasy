"""The committed benchmark record: what it refuses, and what it must not refuse.

These tests exist because publishing is the step that makes a number quotable. Every
refusal below is mutation-tested — the guard is shown failing on a mutated record, not
merely asserted to exist.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ecstasy import report, results


def _record(**over) -> dict:
    """A complete, publishable row. Tests mutate one field at a time."""
    rec = {
        "schema_version": results.SCHEMA_VERSION,
        "published_utc": "2026-08-26T19:00:00+00:00",
        "dataset": {"name": "recent_pp", "version": 1, "expected_entries": 151},
        "model": {"name": "minifold", "variant": "full"},
        "n": {"evaluated": 151, "skipped": 0, "errors": 0},
        "coverage": {"n_evaluated": 151, "n_intended": 151, "fraction": 1.0,
                     "complete": True, "limit": None},
        "metrics": {"names": ["P@K"], "mean": {"P@K": 0.2611}, "median": {"P@K": 0.14}},
        "structure": {"n": 151, "mean": {"DockQ": 0.2426, "iRMSD": 12.31,
                                         "LRMSD": 32.13, "Fnat": 0.29,
                                         "TM_mean": 0.766},
                      "acceptable_fraction": 0.391, "medium_fraction": 0.265,
                      "high_fraction": 0.040},
        "flops": None,
        "fingerprints": {"prediction": "d8e0adcf3a90235d",
                         "scoring": "057f05795464c61a"},
        "provenance": {"ecstasy_sha": "5eb9164", "ecstasy_dirty": False,
                       "model_code": {}, "weights": {}, "captured_utc": None,
                       "host": "node"},
        "run_dir": "recent_pp/minifold/full",
    }
    rec.update(over)
    return rec


class TestRefusals:
    def test_a_complete_clean_run_is_publishable(self):
        assert results.check_publishable(_record()) == []

    def test_partial_coverage_is_refused(self):
        rec = _record(coverage={"n_evaluated": 12, "n_intended": 151,
                                "fraction": 12 / 151, "complete": False, "limit": 12})
        problems = results.check_publishable(rec)
        assert problems and "coverage" in problems[0]
        # ...and the override works, because a deliberately partial run is a real case.
        assert results.check_publishable(rec, allow_partial=True) == []

    def test_a_dirty_ecstasy_tree_is_refused(self):
        rec = _record()
        rec["provenance"]["ecstasy_dirty"] = True
        problems = results.check_publishable(rec)
        assert problems and "dirty" in problems[0]
        assert results.check_publishable(rec, allow_dirty=True) == []

    def test_errored_targets_are_refused(self):
        rec = _record(n={"evaluated": 148, "skipped": 0, "errors": 3})
        assert any("errored" in p for p in results.check_publishable(rec))

    def test_a_dirty_MODEL_tree_is_NOT_refused(self):
        """MiniFold is benchmarked with the residx patch applied to its working tree.

        That is the intended experiment and it is permanent, so refusing it would make
        --allow_dirty a reflex on every MiniFold publish — and a gate everyone always
        overrides stops being a gate. It is recorded and flagged in the report instead.
        """
        rec = _record()
        rec["provenance"]["model_code"] = {
            "minifold": {"sha": "63db8b9", "dirty": True,
                         "dirty_files": ["minifold/model/model.py"]}}
        assert results.check_publishable(rec) == []


class TestStore:
    def test_publish_appends_and_load_round_trips(self, tmp_path, monkeypatch):
        store = tmp_path / "runs.jsonl"
        monkeypatch.setattr(results, "build_record", lambda p: _record())
        rec, note = results.publish(tmp_path / "result.json", store=store)
        assert note == ""
        rows = results.load(store)
        assert len(rows) == 1
        assert rows[0]["model"]["name"] == "minifold"

    def test_republishing_identical_fingerprints_is_refused(self, tmp_path, monkeypatch):
        store = tmp_path / "runs.jsonl"
        monkeypatch.setattr(results, "build_record", lambda p: _record())
        results.publish(tmp_path / "result.json", store=store)
        with pytest.raises(results.PublishRefused, match="already published"):
            results.publish(tmp_path / "result.json", store=store)
        # A deliberate repeat is allowed, and says so.
        _, note = results.publish(tmp_path / "result.json", store=store, again=True)
        assert "deliberately" in note
        assert len(results.load(store)) == 2

    def test_a_changed_scoring_fingerprint_appends_rather_than_replaces(
            self, tmp_path, monkeypatch):
        """A metric fix re-scores the SAME predictions. That is a new row, not an edit —
        it is what makes `git log -p runs.jsonl` show a number moving and why."""
        store = tmp_path / "runs.jsonl"
        monkeypatch.setattr(results, "build_record", lambda p: _record())
        results.publish(tmp_path / "result.json", store=store)

        rescored = _record()
        rescored["fingerprints"] = dict(rescored["fingerprints"], scoring="deadbeef")
        rescored["metrics"] = {"names": ["P@K"], "mean": {"P@K": 0.30}, "median": {}}
        monkeypatch.setattr(results, "build_record", lambda p: rescored)
        results.publish(tmp_path / "result.json", store=store)

        rows = results.load(store)
        assert len(rows) == 2
        assert [r["metrics"]["mean"]["P@K"] for r in rows] == [0.2611, 0.30]

    def test_a_missing_fingerprint_is_refused(self, tmp_path, monkeypatch):
        store = tmp_path / "runs.jsonl"
        rec = _record(fingerprints={"prediction": "", "scoring": ""})
        monkeypatch.setattr(results, "build_record", lambda p: rec)
        with pytest.raises(results.PublishRefused, match="missing a fingerprint"):
            results.publish(tmp_path / "result.json", store=store)

    def test_load_of_a_missing_store_is_empty_not_an_error(self, tmp_path):
        assert results.load(tmp_path / "nothing.jsonl") == []

    def test_a_corrupt_line_names_its_line_number(self, tmp_path):
        store = tmp_path / "runs.jsonl"
        store.write_text(json.dumps(_record()) + "\n{ not json\n")
        with pytest.raises(ValueError, match=":2"):
            results.load(store)

    def test_run_dir_is_recorded_relative_not_absolute(self):
        """The store is committed and read on other machines, where an absolute
        /rds/general/user/<someone>/... path means nothing."""
        assert not Path(_record()["run_dir"]).is_absolute()


class TestReport:
    def test_dockq_never_appears_without_irmsd_and_lrmsd(self):
        """DockQ averages fnat with two RMSD terms, so an undocked prediction still
        scores off fnat alone — and scores highest where its geometry is worst. This
        has produced four wrong conclusions in this project; the column order is a
        structural fix, not a convention."""
        assert "DockQ" in report._STRUCTURE
        for term in ("iRMSD", "LRMSD"):
            assert term in report._STRUCTURE
        assert report._STRUCTURE.index("DockQ") < report._STRUCTURE.index("iRMSD")

        md = report.render([_record()])
        head = next(l for l in md.splitlines() if "DockQ" in l and l.startswith("|"))
        assert "iRMSD" in head and "LRMSD" in head

    def test_a_dirty_model_tree_is_flagged_with_a_footnote(self):
        rec = _record()
        rec["provenance"]["model_code"] = {
            "minifold": {"sha": "63db8b9", "dirty": True,
                         "dirty_files": ["minifold/model/model.py"]}}
        md = report.render([rec])
        assert "minifold†" in md
        assert "residx" in md

    def test_a_clean_row_carries_no_flag(self):
        md = report.render([_record()])
        assert "minifold |" in md and "†" not in md

    def test_an_empty_store_renders_and_says_so(self):
        md = report.render([])
        assert "Nothing published yet" in md

    def test_rows_group_by_dataset(self):
        other = _record()
        other["dataset"] = {"name": "foldbench_pp", "version": 1,
                            "expected_entries": 193}
        md = report.render([_record(), other])
        assert "## recent_pp" in md and "## foldbench_pp" in md
