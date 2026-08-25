"""Dataset identity: version, description, and the entry-count check.

A split is a file on a filesystem, and nothing stops it changing under a published
result. Before this, the only record of a split's size was a YAML comment — and those had
already drifted: they claimed val_pinder_chain was 98 rows and val_pinder_pair 474, while
the parquets hold 106 and 454. `expected_entries` turns that from a stale comment into a
failed check.

These tests use fake loaders so they run anywhere; the real splits are checked by
`ecstasy datasets --verify`, which needs MENTOS_ROOT.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ecstasy.datasets.base import Dataset, Entry, dataset_names


class _FakeSplit(Dataset):
    kind = "fake_identity_split"

    def __init__(self, name, n=3, sources=None, **meta):
        super().__init__(name, **meta)
        self._n = n
        self._sources = sources or {}

    def entries(self):
        for i in range(self._n):
            yield Entry(id=f"e{i}", sequences=("AA", "CC"))

    def gt_for(self, entry_id):
        raise NotImplementedError

    def score(self, entry, contact_path, metrics=None):
        raise NotImplementedError

    def source_paths(self):
        return self._sources


class TestMetadata:
    def test_defaults_are_conservative(self):
        d = _FakeSplit("d")
        assert d.version == 1
        assert d.expected_entries is None
        assert d.tags == []

    def test_metadata_round_trips_into_the_manifest(self):
        d = _FakeSplit("d", version=3, description="a split", expected_entries=3,
                       tags=["deleaked"])
        m = d.manifest()
        assert m["name"] == "d" and m["version"] == 3
        assert m["description"] == "a split"
        assert m["expected_entries"] == 3
        assert m["tags"] == ["deleaked"]
        assert m["kind"] == "fake_identity_split"

    def test_a_typo_in_a_row_key_is_loud(self):
        """Silently ignoring an unknown key would let `expcted_entries: 151` do nothing."""
        with pytest.raises(TypeError):
            _FakeSplit("d", verison=2)


class TestVerify:
    def test_matching_count_passes(self):
        assert _FakeSplit("d", n=3, description="x", expected_entries=3).verify()["ok"]

    def test_drift_is_reported_with_both_numbers(self):
        r = _FakeSplit("d", n=5, description="x", expected_entries=3).verify()
        assert not r["ok"]
        assert r["n_entries"] == 5 and r["expected_entries"] == 3
        assert any("drift" in p for p in r["problems"])

    def test_missing_source_is_reported_without_enumerating(self, tmp_path):
        """A missing split must say so, not fail deep inside a run."""
        r = _FakeSplit("d", description="x", expected_entries=3,
                       sources={"index": tmp_path / "nope.parquet"}).verify()
        assert not r["ok"]
        assert r["n_entries"] is None
        assert any("missing source" in p for p in r["problems"])

    def test_missing_description_is_a_problem(self):
        """Descriptions are the surface an agent reads to pick a split."""
        r = _FakeSplit("d", n=3, expected_entries=3).verify()
        assert not r["ok"]
        assert any("description" in p for p in r["problems"])

    def test_no_expected_entries_means_no_drift_check(self):
        r = _FakeSplit("d", n=7, description="x").verify()
        assert r["ok"] and r["n_entries"] == 7


class TestFingerprint:
    def test_fingerprints_a_real_source_file(self, tmp_path):
        idx = tmp_path / "index.parquet"
        idx.write_bytes(b"parquet-ish" * 20)
        fp = _FakeSplit("d", sources={"index": idx}).fingerprint()
        assert fp["index"]["size"] == 220
        assert "sha256_ends" in fp["index"]

    def test_changing_the_source_changes_the_fingerprint(self, tmp_path):
        """This is the property that lets a result be checked against its split."""
        idx = tmp_path / "index.parquet"
        idx.write_bytes(b"a" * 100)
        before = _FakeSplit("d", sources={"index": idx}).fingerprint()
        idx.write_bytes(b"b" * 100)
        after = _FakeSplit("d", sources={"index": idx}).fingerprint()
        assert before["index"]["sha256_ends"] != after["index"]["sha256_ends"]

    def test_directories_are_recorded_without_hashing(self, tmp_path):
        fp = _FakeSplit("d", sources={"gt_root": tmp_path}).fingerprint()
        assert fp["gt_root"]["kind"] == "directory" and fp["gt_root"]["exists"]


class TestRegisteredRows:
    """The committed rows must all carry identity — that is the point of the change."""

    def test_every_row_has_a_description_and_expected_entries(self):
        from ecstasy.datasets.base import _registry
        reg = _registry()
        for name in dataset_names():
            row = reg[name]
            assert row.get("description"), f"{name} has no description"
            assert row.get("expected_entries"), f"{name} has no expected_entries"
            assert row.get("version"), f"{name} has no version"

    def test_pinder_counts_are_the_measured_ones_not_the_old_comments(self):
        """Regression pin: the YAML comments said 98 and 474; the parquets hold 106/454."""
        from ecstasy.datasets.base import _registry
        reg = _registry()
        assert reg["val_pinder_chain"]["expected_entries"] == 106
        assert reg["val_pinder_pair"]["expected_entries"] == 454
