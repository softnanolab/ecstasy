"""Tests for ecstasy.structure — generic Pinder-style chain + interface utils.

The bigger building blocks (`download_cif_assembly`, `load_local_cif`) are
network/IO-bound and tested as part of the build_ecstasy_v1 integration job;
this file pins the pure-numpy logic (`interface_residue_indices`,
`enumerate_dimer_pairs`, `split_into_chains` indexing) that's easy to fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecstasy.structure import (
    BACKBONE_ATOMS,
    INTERFACE_CUTOFF_A,
    ChainView,
    enumerate_dimer_pairs,
    interface_residue_indices,
)


def _make_chain(n_res: int, x_offset: float = 0.0) -> ChainView:
    """Synthetic linear chain with one backbone atom per residue at (x_offset+i, 0, 0)."""
    res_ids = np.arange(1, n_res + 1, dtype=np.int64)
    xyz = np.stack([np.arange(n_res, dtype=np.float32) + x_offset,
                    np.zeros(n_res, dtype=np.float32),
                    np.zeros(n_res, dtype=np.float32)], axis=-1)
    bb_res_idx = np.arange(n_res, dtype=np.int32)
    return ChainView(
        chain_id="X",
        res_ids=res_ids,
        sequence="A" * n_res,
        backbone_xyz=xyz,
        backbone_res_idx=bb_res_idx,
    )


class TestInterfaceResidueIndices:
    def test_no_contacts_when_far(self):
        a = _make_chain(5, x_offset=0.0)
        b = _make_chain(5, x_offset=100.0)  # 100 Å away
        ia, ib, n = interface_residue_indices(a, b)
        assert n == 0
        assert ia.size == 0 and ib.size == 0

    def test_fully_overlapping_chains_all_in_contact(self):
        a = _make_chain(5, x_offset=0.0)
        b = _make_chain(5, x_offset=0.0)  # same coords
        ia, ib, n = interface_residue_indices(a, b, cutoff=10.0)
        # all 5 vs all 5 are within 10 Å (max sep is 4 Å along the line)
        assert n == 25
        assert ia.tolist() == [0, 1, 2, 3, 4]
        assert ib.tolist() == [0, 1, 2, 3, 4]

    def test_partial_overlap_returns_subset(self):
        a = _make_chain(10, x_offset=0.0)   # residues at x = 0..9
        b = _make_chain(10, x_offset=8.0)   # residues at x = 8..17
        ia, ib, n = interface_residue_indices(a, b, cutoff=2.5)
        # only residues with x-distance <= 2.5 are in contact:
        #   a[8] (x=8) is 0 away from b[0] (x=8); a[9] (x=9) is 1 away from
        #   b[0], b[1]; etc. Empirically the close set is a[6..9] vs b[0..3].
        assert n > 0
        assert max(ia) <= 9 and min(ia) >= 6
        assert max(ib) <= 3 and min(ib) >= 0

    def test_default_cutoff_matches_constant(self):
        assert INTERFACE_CUTOFF_A == 10.0


class TestEnumerateDimerPairs:
    def test_two_chains(self):
        chains = [_make_chain(1), _make_chain(1)]
        assert enumerate_dimer_pairs(chains) == [(0, 1)]

    def test_four_chains_gives_six_unordered_pairs(self):
        chains = [_make_chain(1) for _ in range(4)]
        pairs = enumerate_dimer_pairs(chains)
        assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def test_single_chain_empty(self):
        assert enumerate_dimer_pairs([_make_chain(1)]) == []


def test_backbone_atoms_constant():
    """The four-atom DockQ backbone definition we depend on for the interface
    distance check; if anyone widens this set without thought, P@K denominators
    quietly shift."""
    assert BACKBONE_ATOMS == ("N", "CA", "C", "O")
