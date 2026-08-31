"""The MiniFold chain break: residue-index construction and linker trimming.

These two helpers are the whole multimer hack. If `_residx` stops jumping, the trunk
reads the two chains as one continuous polymer and what gets measured is the
linker-only variant — which looks perfectly healthy and is a different experiment.
If `_linker_trim_index` slips, every downstream array is off by the linker length.
Neither failure raises; both are pinned here.

The expected values are written out longhand rather than derived, so a change to the
implementation cannot quietly agree with itself.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="the runner module imports torch at module scope")

from ecstasy.models._runners.minifold_runner import (  # noqa: E402
    _linker_trim_index,
    _residx,
)

LINKER = 25
OFFSET = 512


class TestResidx:
    def test_matches_the_verified_prototype_formula(self):
        """Longhand form from the CX3 prototype the design was validated with."""
        la, lb = 6, 4
        expected = np.concatenate([
            np.arange(la + LINKER),
            np.arange(lb) + la + LINKER + OFFSET,
        ])
        got = _residx(["A" * la, "C" * lb], LINKER, OFFSET)
        np.testing.assert_array_equal(got, expected)

    def test_length_covers_the_linker_joined_sequence(self):
        seqs = ["A" * 6, "C" * 4]
        got = _residx(seqs, LINKER, OFFSET)
        assert len(got) == sum(map(len, seqs)) + LINKER

    def test_step_across_the_break_saturates_the_relative_position_clamp(self):
        """RelativePosition clamps at 32 bins; the jump must dwarf it or the trunk
        still reads one continuous chain."""
        la, lb = 6, 4
        r = _residx(["A" * la, "C" * lb], LINKER, OFFSET)
        last_linker, first_b = r[la + LINKER - 1], r[la + LINKER]
        assert first_b - last_linker == OFFSET + 1 == 513
        assert first_b - r[la - 1] == OFFSET + LINKER + 1     # from the last real A
        assert first_b - last_linker > 32

    def test_within_chain_numbering_is_contiguous(self):
        la, lb = 6, 4
        r = _residx(["A" * la, "C" * lb], LINKER, OFFSET)
        assert np.all(np.diff(r[: la + LINKER]) == 1)
        assert np.all(np.diff(r[la + LINKER:]) == 1)

    def test_linker_continues_chain_a_rather_than_jumping_early(self):
        """The linker is inert padding; the jump belongs at the break, not before it."""
        la = 6
        r = _residx(["A" * la, "C" * 4], LINKER, OFFSET)
        assert r[la] == r[la - 1] + 1

    def test_a_monomer_gets_a_plain_arange(self):
        np.testing.assert_array_equal(_residx(["A" * 5], LINKER, OFFSET), np.arange(5))

    def test_three_chains_jump_at_every_break(self):
        r = _residx(["A" * 3, "C" * 2, "D" * 4], LINKER, OFFSET)
        assert len(r) == 3 + LINKER + 2 + LINKER + 4
        first_b = r[3 + LINKER]
        first_c = r[3 + LINKER + 2 + LINKER]
        assert first_b == 3 + LINKER + OFFSET
        assert first_c == first_b + 2 + LINKER + OFFSET

    @pytest.mark.parametrize("linker", [0, 25, 32])
    def test_offset_is_applied_independently_of_linker_length(self, linker):
        la, lb = 6, 4
        r = _residx(["A" * la, "C" * lb], linker, OFFSET)
        assert r[la + linker] - r[la + linker - 1] == OFFSET + 1


class TestLinkerTrim:
    def test_selects_real_residues_and_drops_the_linker(self):
        la, lb = 6, 4
        expected = np.concatenate([np.arange(la), np.arange(lb) + la + LINKER])
        np.testing.assert_array_equal(
            _linker_trim_index(["A" * la, "C" * lb], LINKER), expected)

    def test_trimmed_length_is_the_residue_count(self):
        seqs = ["A" * 6, "C" * 4]
        assert len(_linker_trim_index(seqs, LINKER)) == sum(map(len, seqs))

    def test_trimming_residx_leaves_the_jump_intact(self):
        """The trimmed arrays are what the contact map and structure are built from."""
        la, lb = 6, 4
        seqs = ["A" * la, "C" * lb]
        r = _residx(seqs, LINKER, OFFSET)[_linker_trim_index(seqs, LINKER)]
        assert len(r) == la + lb
        assert np.all(np.diff(r[:la]) == 1)
        assert np.all(np.diff(r[la:]) == 1)
        assert r[la] - r[la - 1] > 32

    def test_a_monomer_trims_to_itself(self):
        np.testing.assert_array_equal(
            _linker_trim_index(["A" * 5], LINKER), np.arange(5))

    def test_three_chains_drop_both_linkers(self):
        seqs = ["A" * 3, "C" * 2, "D" * 4]
        idx = _linker_trim_index(seqs, LINKER)
        assert len(idx) == 9
        np.testing.assert_array_equal(idx[:3], [0, 1, 2])
        np.testing.assert_array_equal(idx[3:5], [3 + LINKER, 4 + LINKER])
        np.testing.assert_array_equal(
            idx[5:], np.arange(4) + 3 + LINKER + 2 + LINKER)

    def test_a_zero_length_linker_is_a_plain_concatenation(self):
        np.testing.assert_array_equal(
            _linker_trim_index(["A" * 3, "C" * 2], 0), np.arange(5))
