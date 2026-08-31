"""Fingerprints — what makes cached work safe to reuse.

The failure being prevented: a run directory is keyed <dataset>/<model>/<variant>, which
says nothing about code, while run_predict skips any entry that already has a contact.npz.
So bumping a dependency reuses old predictions under a NEW provenance record — a
confidently false claim, and the persisted form of the mid-sweep-edit hazard.

The separation matters as much as the gating: predictions never see ground truth, so a GT
regeneration or a metric fix must NOT invalidate GPU work.
"""
from __future__ import annotations

import pytest

from ecstasy import fingerprint as fp


class TestDigest:
    def test_is_stable_across_calls(self):
        inputs = {"model": "minifold", "params": {"a": 1}}
        assert fp.digest(inputs) == fp.digest(inputs)

    def test_is_insensitive_to_key_order(self):
        assert fp.digest({"a": 1, "b": 2}) == fp.digest({"b": 2, "a": 1})

    def test_any_change_changes_it(self):
        base = {"model": "minifold", "params": {"cutoff": 17}}
        changed = {"model": "minifold", "params": {"cutoff": 19}}
        assert fp.digest(base) != fp.digest(changed)

    def test_distinguishes_a_dirty_tree_from_a_clean_one(self):
        """THE case this exists for: same commit, patch applied vs not. Those are two
        different experiments and must never share a digest."""
        clean = {"venv": {"minifold": {"sha": "63db8b91", "dirty": False}}}
        patched = {"venv": {"minifold": {"sha": "63db8b91", "dirty": True}}}
        assert fp.digest(clean) != fp.digest(patched)


class TestCompare:
    def test_names_what_changed_not_merely_that_it_did(self):
        """'The fingerprint changed' is useless; the SHA transition is actionable."""
        old = {"inputs": {"venv": {"mentos": {"sha": "2cc5309"}}}}
        new = {"inputs": {"venv": {"mentos": {"sha": "abc1234"}}}}
        diffs = fp.compare(old, new)
        assert len(diffs) == 1
        assert "venv.mentos.sha" in diffs[0]
        assert "2cc5309" in diffs[0] and "abc1234" in diffs[0]

    def test_identical_inputs_produce_no_diffs(self):
        same = {"inputs": {"a": {"b": 1}}}
        assert fp.compare(same, same) == []

    def test_reports_added_and_removed_keys(self):
        diffs = fp.compare({"inputs": {"a": 1}}, {"inputs": {"b": 2}})
        assert len(diffs) == 2

    def test_walks_nested_structures(self):
        old = {"inputs": {"dataset": {"index": {"sha256_ends": "aaa"}}}}
        new = {"inputs": {"dataset": {"index": {"sha256_ends": "bbb"}}}}
        assert "dataset.index.sha256_ends" in fp.compare(old, new)[0]

    def test_handles_missing_inputs_key(self):
        assert fp.compare({}, {}) == []
        assert fp.compare(None, None) == []


class TestMismatchError:
    def test_message_names_the_change_and_both_remedies(self):
        e = fp.FingerprintMismatch("prediction", ["venv.mentos.sha: 'a' -> 'b'"], "/run/dir")
        msg = str(e)
        assert "venv.mentos.sha" in msg
        assert "--force" in msg          # recompute in place
        assert "--variant" in msg or "--set" in msg   # or fork a new directory
        assert "looks entirely normal" in msg         # says WHY it is refusing

    def test_long_diffs_are_truncated_with_a_count(self):
        e = fp.FingerprintMismatch("prediction", [f"k{i}: 1 -> 2" for i in range(20)], "/d")
        assert "... and 8 more" in str(e)


class TestPersistence:
    def test_round_trips(self, tmp_path):
        f = fp.make("prediction", {"model": "x"})
        fp.save(tmp_path / "fp.json", f)
        assert fp.load(tmp_path / "fp.json") == f

    def test_missing_file_is_none_not_an_error(self, tmp_path):
        assert fp.load(tmp_path / "absent.json") is None

    def test_corrupt_file_is_none_not_an_error(self, tmp_path):
        p = tmp_path / "fp.json"
        p.write_text("{not json")
        assert fp.load(p) is None

    def test_make_carries_both_digest_and_inputs(self):
        """The inputs must travel with the digest, or a mismatch cannot be explained."""
        f = fp.make("prediction", {"model": "x"})
        assert f["kind"] == "prediction"
        assert f["digest"] == fp.digest({"model": "x"})
        assert f["inputs"] == {"model": "x"}


class _FakeDataset:
    name = "d"
    version = 2
    contact_bin = 19

    def __init__(self, index_hash="idx", gt_hash="gt"):
        self._index_hash = index_hash
        self._gt_hash = gt_hash

    def fingerprint(self):
        return {"index": {"sha256_ends": self._index_hash},
                "gt_root": {"sha256_ends": self._gt_hash}}


class TestSeparationOfConcerns:
    """Predictions never see ground truth; scoring never re-runs a GPU."""

    def test_gt_change_does_not_alter_the_scoring_inputs_dataset_index(self):
        a = fp.scoring_inputs(_FakeDataset(gt_hash="old"), ["P@K"])
        b = fp.scoring_inputs(_FakeDataset(gt_hash="new"), ["P@K"])
        assert fp.digest(a) != fp.digest(b), "a GT change must force a re-score"

    def test_metric_set_is_part_of_the_scoring_fingerprint(self):
        a = fp.scoring_inputs(_FakeDataset(), ["P@K"])
        b = fp.scoring_inputs(_FakeDataset(), ["P@K", "P@K(tol=2)"])
        assert fp.digest(a) != fp.digest(b)

    def test_contact_bin_is_part_of_the_scoring_fingerprint(self):
        d = _FakeDataset()
        a = fp.scoring_inputs(d, ["P@K"])
        d.contact_bin = 5
        b = fp.scoring_inputs(d, ["P@K"])
        assert fp.digest(a) != fp.digest(b)

    def test_metric_code_identity_is_included(self):
        """A metric bugfix must re-score, or old numbers silently persist."""
        got = fp.scoring_inputs(_FakeDataset(), ["P@K"])
        assert got["metric_code"]["sha256"]
        assert "contact.py" in got["metric_code"]["files"]

    def test_scoring_inputs_do_not_mention_model_or_weights(self):
        """Scoring is CPU-only and independent of which model produced the map."""
        got = fp.scoring_inputs(_FakeDataset(), ["P@K"])
        flat = str(got)
        assert "venv" not in flat and "checkpoint" not in flat


class TestCodeIdentity:
    def test_same_files_give_the_same_hash(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        first = fp._code_identity(tmp_path.glob("*.py"))
        second = fp._code_identity(tmp_path.glob("*.py"))
        assert first == second

    def test_editing_a_file_changes_the_hash(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        before = fp._code_identity([f])["sha256"]
        f.write_text("x = 2\n")
        assert fp._code_identity([f])["sha256"] != before

    def test_hash_is_order_independent(self, tmp_path):
        a, b = tmp_path / "a.py", tmp_path / "b.py"
        a.write_text("1\n")
        b.write_text("2\n")
        assert fp._code_identity([a, b]) == fp._code_identity([b, a])

    def test_missing_files_are_skipped(self, tmp_path):
        assert fp._code_identity([tmp_path / "nope.py"])["files"] == []
