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


def _run_dir(tmp_path, *, flops=None, dirty_model=True) -> Path:
    """A run directory on disk, as `ecstasy score` leaves one.

    build_record's whole job is reading these files, so it has to be tested against
    real ones. Every other test in this module monkeypatches build_record out.
    """
    run = tmp_path / "runs" / "recent_pp" / "minifold" / "full"
    (run / "predictions" / "10bl").mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({
        "dataset": "recent_pp", "model": "minifold", "variant": "full",
        "metrics": ["P@K"],
        "coverage": {"n_evaluated": 151, "n_intended": 151, "fraction": 1.0,
                     "complete": True, "limit": None},
        "summary": {"n_evaluated": 151, "n_skipped": 0, "n_errors": 0,
                    "mean": {"P@K": 0.2611}, "median": {"P@K": 0.14},
                    "structure": {"n": 151, "mean": {"DockQ": 0.2426, "iRMSD": 12.31}}},
        "per_protein": {},
        "provenance": {
            "ecstasy": {"sha": "5eb9164", "dirty": False},
            "captured_utc": "2026-08-26T17:00:00+00:00",
            "env": {"host": "cx3-20-3"},
            "venv": {"packages": {"minifold": {"git": {
                "sha": "63db8b9", "dirty": dirty_model,
                "dirty_files": ["minifold/model/model.py"] if dirty_model else []}}}},
            "params_provenance": {"checkpoint": {
                "kind": "file", "path": "/w/minifold_48L.ckpt",
                "resolved": "/real/minifold_48L.ckpt", "size": 2784107297,
                "sha256_ends": "99d9db"}},
        },
    }))
    (run / "prediction_fingerprint.json").write_text(json.dumps({"digest": "d8e0adcf"}))
    (run / "scoring_fingerprint.json").write_text(json.dumps({"digest": "057f0579"}))
    if flops is not None:
        for i, v in enumerate(flops):
            d = run / "predictions" / f"t{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "flops.json").write_text(json.dumps({"flops": v, "macs": v // 2}))
    return run


class TestBuildRecord:
    """The projection layer: the only part of publish that reads real files."""

    @pytest.fixture(autouse=True)
    def _isolate_settings(self):
        """`settings()` is lru_cached, so setenv alone does not repoint DATA_ROOT.

        Without clearing, these tests pass when the file is run alone (nothing has
        called settings() yet, so the first call sees the patched env) and fail in the
        full suite once an earlier test has cached the real DATA_ROOT. Clearing on both
        sides is the pattern test_boltz_csv already uses; the trailing clear matters as
        much as the leading one, or the NEXT test inherits a tmp_path DATA_ROOT.
        """
        from ecstasy import config
        config.settings.cache_clear()
        yield
        config.settings.cache_clear()

    def _build(self, tmp_path, monkeypatch, **kw):
        # DATA_ROOT is the canonical seam the rest of the suite uses; Settings is a
        # frozen dataclass and runs_root a derived property, so it is also the only one.
        from ecstasy import config
        run = _run_dir(tmp_path, **kw)
        monkeypatch.setenv("DATA_ROOT", str(tmp_path))
        config.settings.cache_clear()
        return run, results.build_record(run / "result.json")

    def test_run_dir_is_relative_to_data_root_not_absolute(self, tmp_path, monkeypatch):
        """The store is committed and read on other machines, where an absolute
        /rds/general/user/<someone>/... path means nothing.

        This asserts on build_record's OUTPUT. An earlier version of this test asserted
        that the hand-written fixture was relative, which would have passed unchanged
        while build_record regressed to absolute paths — the exact bug it was named for.
        """
        _, rec = self._build(tmp_path, monkeypatch)
        assert not Path(rec["run_dir"]).is_absolute()
        assert rec["run_dir"] == str(Path("recent_pp") / "minifold" / "full")

    def test_fingerprints_are_read_from_their_sidecars(self, tmp_path, monkeypatch):
        _, rec = self._build(tmp_path, monkeypatch)
        assert rec["fingerprints"] == {"prediction": "d8e0adcf", "scoring": "057f0579"}

    def test_a_dirty_model_tree_is_captured_with_its_files(self, tmp_path, monkeypatch):
        """Whether the residx patch was applied is the difference between two different
        experiments. It has to survive into the row."""
        _, rec = self._build(tmp_path, monkeypatch)
        mc = rec["provenance"]["model_code"]["minifold"]
        assert mc["dirty"] is True
        assert mc["dirty_files"] == ["minifold/model/model.py"]

    def test_a_clean_model_tree_reports_not_dirty(self, tmp_path, monkeypatch):
        _, rec = self._build(tmp_path, monkeypatch, dirty_model=False)
        assert rec["provenance"]["model_code"]["minifold"]["dirty"] is False

    def test_weights_carry_the_resolved_path_and_content_hash(self, tmp_path, monkeypatch):
        """MiniFold's checkpoint is a symlink into a MENTOS log dir (#35); the row has
        to record what it actually resolved to and the bytes it hashed."""
        _, rec = self._build(tmp_path, monkeypatch)
        w = rec["provenance"]["weights"]["checkpoint"]
        assert w["resolved"] == "/real/minifold_48L.ckpt"
        assert w["sha256_ends"] == "99d9db"

    def test_an_unprofiled_run_has_no_flops(self, tmp_path, monkeypatch):
        _, rec = self._build(tmp_path, monkeypatch)
        assert rec["flops"] is None

    def test_flops_come_from_the_canonical_aggregator(self, tmp_path, monkeypatch):
        """results must not re-implement flops aggregation.

        It did once, taking sorted(vals)[n // 2] as the median — the upper-middle
        element, not the mean of the two middle values. On an even number of targets
        that disagreed with `ecstasy compare` on the same run: 30.0 against 25.0 for
        10/20/30/40. Two numbers both called "median FLOPs" is the one thing a
        published record cannot have.
        """
        from ecstasy.pipeline import flops_summary
        run, rec = self._build(tmp_path, monkeypatch, flops=[10, 20, 30, 40])
        assert rec["flops"] == flops_summary(run)
        assert rec["flops"]["median_flops"] == 25.0
        assert rec["flops"]["mean_flops"] == 25.0
        assert rec["flops"]["n_flops"] == 4


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
