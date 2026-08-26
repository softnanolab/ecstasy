"""ecstasy's own ground-truth format, and the geometry it derives contacts with.

Two things are pinned here.

**The geometry convention**, because every published contact number depends on it: a
virtual Cβ built from N/CA/C for every residue (not the crystal atom, not CA-for-glycine),
64 bins over AF2's edges, -1 for undefined. Verified against all 151 real MENTOS ground
truth files by tests/integration/test_gt_derivation.py; here it is pinned in isolation.

**The format's refusal to guess.** A missing array, a version it does not understand, or
a pickled payload must all be errors, because ground truth silently read wrong produces
numbers that look entirely normal.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecstasy.datasets import store
from ecstasy.structure.geometry import (
    CONTACT_BIN,
    DISTANCE_BIN_EDGES,
    NUM_DISTANCE_BINS,
    bins_from_atom37,
    cb_distance_matrix,
    contacts_from_bins,
    distance_bins,
    virtual_cb,
)
from ecstasy.structure.pdb import ATOM_TYPES


class TestBinning:
    def test_edges_are_the_af2_convention(self):
        assert NUM_DISTANCE_BINS == 64
        assert len(DISTANCE_BIN_EDGES) == 63
        assert DISTANCE_BIN_EDGES[0] == pytest.approx(2.3125)
        assert DISTANCE_BIN_EDGES[-1] == pytest.approx(21.6875)

    def test_contact_bin_is_seven_point_nine(self):
        """bins 0..18 are contacts; the boundary is MENTOS's 7.9375 A threshold."""
        assert CONTACT_BIN == 19
        assert DISTANCE_BIN_EDGES[CONTACT_BIN - 1] == pytest.approx(7.9375)

    def test_digitize_is_right_inclusive(self):
        assert distance_bins(np.array([[2.0]]))[0, 0] == 0            # <= first edge
        assert distance_bins(np.array([[100.0]]))[0, 0] == 63         # > last edge

    def test_nan_becomes_minus_one(self):
        assert distance_bins(np.array([[np.nan]]))[0, 0] == -1

    def test_contacts_exclude_undefined(self):
        bins = np.array([[0, -1], [18, 19]])
        contact, valid = contacts_from_bins(bins)
        assert contact.tolist() == [[True, False], [True, False]]
        assert valid.tolist() == [[True, False], [True, True]]

    def test_undefined_is_not_a_contact_nor_a_negative(self):
        """-1 must be excluded from BOTH, or it dilutes precision as a false negative."""
        _, valid = contacts_from_bins(np.array([[-1]]))
        assert not valid[0, 0]


class TestVirtualCb:
    def test_is_offset_from_ca_by_the_bond_length(self):
        n = np.array([[0.0, 1.0, 0.0]])
        ca = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[1.0, 0.0, 0.0]])
        cb = virtual_cb(n, ca, c)
        assert np.linalg.norm(cb[0] - ca[0]) == pytest.approx(1.522, abs=1e-6)

    def test_is_applied_to_every_residue_not_only_glycine(self):
        """A single rule for all 20 residues is why the GT needs only backbone atoms."""
        rng = np.random.default_rng(0)
        n, ca, c = (rng.normal(size=(5, 3)) for _ in range(3))
        assert np.isfinite(virtual_cb(n, ca, c)).all()

    def test_nan_propagates(self):
        n = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ca = np.zeros((2, 3))
        c = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cb = virtual_cb(n, ca, c)
        assert np.isnan(cb[0]).all() and np.isfinite(cb[1]).all()

    def test_a_missing_backbone_atom_invalidates_its_whole_row(self):
        pos = np.zeros((3, len(ATOM_TYPES), 3))
        pos[:, :3] = np.array([[0.0, 1, 0], [0, 0, 0], [1, 0, 0]])
        mask = np.zeros((3, len(ATOM_TYPES)), bool)
        mask[:, :3] = True
        mask[1, 2] = False                       # residue 1 has no C
        bins, _ = bins_from_atom37(pos, mask)
        assert (bins[1] == -1).all() and (bins[:, 1] == -1).all()
        assert bins[0, 2] >= 0                   # unaffected pairs survive

    def test_distance_matrix_is_symmetric_with_zero_diagonal(self):
        rng = np.random.default_rng(1)
        n, ca, c = (rng.normal(size=(6, 3)) * 5 for _ in range(3))
        d = cb_distance_matrix(n, ca, c)
        np.testing.assert_allclose(d, d.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)


def _bundle(n_res=5):
    pos = np.zeros((n_res, len(ATOM_TYPES), 3), dtype=np.float32)
    for i in range(n_res):
        pos[i, 0] = (i * 3.8, 1.0, 0.0)      # N
        pos[i, 1] = (i * 3.8, 0.0, 0.0)      # CA
        pos[i, 2] = (i * 3.8 + 1.0, 0.0, 0.0)  # C
    mask = np.zeros((n_res, len(ATOM_TYPES)), bool)
    mask[:, :3] = True
    return {
        "sequences": ["AAA", "AA"],
        "atom37_positions": pos,
        "atom37_mask": mask,
        "aatype": np.zeros(n_res, dtype=np.int8),
        "asym_id": np.array([0, 0, 0, 1, 1], dtype=np.int8),
        "residue_index": np.array([0, 1, 2, 0, 1], dtype=np.int32),
        "chain_ids": ["A", "B"],
        "is_homodimer": False,
    }


class TestStore:
    def test_round_trips_the_arrays(self, tmp_path):
        b = _bundle()
        store.write_entry(tmp_path / "e.npz", **b)
        got = store.read_entry(tmp_path / "e.npz")
        np.testing.assert_allclose(got["atom37_positions"], b["atom37_positions"])
        np.testing.assert_array_equal(got["asym_id"], b["asym_id"])
        assert got["sequences"] == b["sequences"]
        assert got["chain_ids"] == ["A", "B"]
        assert got["is_homodimer"] is False

    def test_contacts_are_derived_not_stored(self, tmp_path):
        """One source of truth: bins are a pure function of the coordinates, so storing
        both would allow them to disagree invisibly."""
        store.write_entry(tmp_path / "e.npz", **_bundle())
        with np.load(tmp_path / "e.npz", allow_pickle=False) as d:
            assert "contact_map" not in d.files and "bins" not in d.files
        assert "contact_map" in store.read_entry(tmp_path / "e.npz")

    def test_contact_bin_is_honoured_on_read(self, tmp_path):
        store.write_entry(tmp_path / "e.npz", **_bundle())
        loose = store.read_entry(tmp_path / "e.npz", contact_bin=40)
        strict = store.read_entry(tmp_path / "e.npz", contact_bin=2)
        assert loose["contact_map"].sum() > strict["contact_map"].sum()

    def test_is_readable_without_pickle(self, tmp_path):
        """np.load(allow_pickle=False) cannot execute code and cannot depend on a class
        remaining importable — the two failure modes of the old .pt format."""
        store.write_entry(tmp_path / "e.npz", **_bundle())
        with np.load(tmp_path / "e.npz", allow_pickle=False) as d:
            assert set(d.files) >= {"meta", "atom37_positions", "asym_id"}

    def test_missing_array_is_an_error(self, tmp_path):
        np.savez_compressed(tmp_path / "bad.npz", meta=np.array('{"format_version": 1}'))
        with pytest.raises(KeyError, match="missing"):
            store.read_entry(tmp_path / "bad.npz")

    def test_unknown_format_version_is_refused(self, tmp_path):
        b = _bundle()
        store.write_entry(tmp_path / "e.npz", **b)
        import json
        with np.load(tmp_path / "e.npz", allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files}
        arrays["meta"] = np.array(json.dumps({"format_version": 999, "sequences": []}))
        np.savez_compressed(tmp_path / "future.npz", **arrays)
        with pytest.raises(ValueError, match="format version"):
            store.read_entry(tmp_path / "future.npz")

    def test_entry_path_fans_out_two_levels(self):
        p = store.entry_path("/gt", "9zdi")
        assert p.parent.name == "9z" and p.name == "9zdi.npz"

    def test_records_where_it_came_from(self, tmp_path):
        store.write_entry(tmp_path / "e.npz", source="mentos:/x/y.pt", **_bundle())
        assert store.read_entry(tmp_path / "e.npz")["source"] == "mentos:/x/y.pt"
