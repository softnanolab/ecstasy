"""Tests for ecstasy.metrics.structure — DockQ plumbing, monomer quality, null control.

The DockQ CLI itself is not exercised here (it is an external binary, and its numbers
are its own business). What is pinned is everything ecstasy puts around it: the parsing
contract, the seeding of the random-placement null, and the geometry helpers — the parts
that can silently drift and quietly move a published comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecstasy.metrics.structure import (
    DOCKQ_BANDS,
    _DOCKQ_RE,
    ca_rmsd,
    dockq_bands,
    kabsch_superpose,
    monomer_metrics,
    random_rotation,
    stable_seed,
    tm_score,
)
from ecstasy.structure.pdb import ATOM_TYPES, CA_INDEX

# Representative DockQ CLI output: a banner and a legend precede the real numbers, which
# is exactly why the parser uses `.search()` on loose patterns.
DOCKQ_STDOUT = """\
****************************************************************
*                       DockQ                                  *
*   Scoring function for protein-protein docking models        *
*   Reference: Basu and Wallner, PLoS ONE 11(8), e0161879      *
*                                                              *
*   For the record:                                            *
*   Definition of contact <5A (Fnat)                           *
*   Definition of interface <10A all heavy atoms (iRMSD)       *
****************************************************************
Model  : /tmp/9zdi_pred.pdb
Native : /tmp/9zdi_native.pdb
Native chains: A, B
Fnat 0.234 12 correct of 51 native contacts
Fnonnat 0.700 28 non-native of 40 model contacts
iRMSD 4.321
LRMSD 12.345
DockQ 0.318
"""


class TestDockqParsing:
    """The regexes are copied from `mentos` and must stay copied — the MENTOS series
    was parsed with them, and a 'tidier' pattern lands on the banner instead."""

    def test_all_four_scores_are_recovered(self):
        got = {k: float(pat.search(DOCKQ_STDOUT).group(1))
               for k, pat in _DOCKQ_RE.items()}
        assert got == {"DockQ": 0.318, "Fnat": 0.234, "iRMSD": 4.321, "LRMSD": 12.345}

    def test_dockq_pattern_skips_the_banner_and_the_legend(self):
        """`.search()` must land on the score line, not on the word DockQ in the header."""
        assert _DOCKQ_RE["DockQ"].search(DOCKQ_STDOUT).group(1) == "0.318"


class TestDockqBands:
    def test_fractions_at_each_band(self):
        got = dockq_bands([0.10, 0.30, 0.55, 0.90])
        assert got["acceptable_fraction"] == 0.75      # >= 0.23
        assert got["medium_fraction"] == 0.50          # >= 0.49
        assert got["high_fraction"] == 0.25            # >= 0.80

    def test_thresholds_are_inclusive(self):
        got = dockq_bands(list(DOCKQ_BANDS.values()))
        assert got["acceptable_fraction"] == 1.0
        assert got["high_fraction"] == pytest.approx(1 / 3)

    def test_empty_is_nan_not_zero(self):
        """Zero would read as 'nothing was acceptable'; nan reads as 'nothing scored'."""
        assert all(np.isnan(v) for v in dockq_bands([]).values())


class TestStableSeed:
    def test_is_stable_across_calls(self):
        assert stable_seed("9zdi") == stable_seed("9zdi")

    def test_differs_between_targets(self):
        assert stable_seed("9zdi") != stable_seed("21ie")

    def test_is_a_valid_numpy_seed(self):
        rng = np.random.default_rng(stable_seed("9zdi"))
        assert 0 <= stable_seed("9zdi") < 2 ** 32
        assert rng.normal(size=3).shape == (3,)

    def test_is_not_pythons_salted_hash(self):
        """`hash` is salted per process unless PYTHONHASHSEED is set, so a hash-seeded
        null moves between runs — the floor must not drift under the result it anchors."""
        assert stable_seed("9zdi") == 1010912291


class TestRandomRotation:
    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_is_a_proper_rotation(self, seed):
        r = random_rotation(np.random.default_rng(seed))
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(r) == pytest.approx(1.0)

    def test_is_deterministic_given_the_generator(self):
        a = random_rotation(np.random.default_rng(3))
        b = random_rotation(np.random.default_rng(3))
        np.testing.assert_allclose(a, b)


class TestSuperposition:
    def test_kabsch_recovers_a_rigid_transform_exactly(self):
        rng = np.random.default_rng(0)
        target = rng.normal(size=(20, 3))
        r = random_rotation(rng)
        moved = (r @ target.T).T + np.array([10.0, -3.0, 7.0])
        np.testing.assert_allclose(kabsch_superpose(moved, target), target, atol=1e-9)

    def test_rmsd_of_a_rigid_copy_is_zero(self):
        rng = np.random.default_rng(1)
        target = rng.normal(size=(15, 3)) * 10
        moved = (random_rotation(rng) @ target.T).T + 5.0
        assert ca_rmsd(moved, target) == pytest.approx(0.0, abs=1e-9)

    def test_kabsch_does_not_mirror(self):
        """A reflection must not be used to fake a fit; det is pinned to +1.

        A mirror-permitting superposition would drive this chiral tetrahedron to
        exactly 0. Any clearly non-zero RMSD means the reflection was refused.
        """
        target = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mirrored = target * np.array([1.0, 1.0, -1.0])
        assert ca_rmsd(mirrored, target) > 0.1

    def test_tm_score_of_an_identical_trace_is_one(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(60, 3)) * 5
        assert tm_score(x.copy(), x) == pytest.approx(1.0, abs=1e-9)

    def test_tm_score_falls_as_the_trace_degrades(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 3)) * 5
        near = tm_score(x + rng.normal(size=x.shape) * 0.3, x)
        far = tm_score(x + rng.normal(size=x.shape) * 6.0, x)
        assert 0.0 < far < near < 1.0

    def test_length_mismatch_is_nan_not_an_exception(self):
        assert np.isnan(tm_score(np.zeros((4, 3)), np.zeros((5, 3))))
        assert np.isnan(ca_rmsd(np.zeros((4, 3)), np.zeros((5, 3))))


def _bundle(chain_sizes=(4, 3), coords=None, ca_present=True):
    n = sum(chain_sizes)
    pos = np.zeros((n, len(ATOM_TYPES), 3), dtype=np.float32)
    pos[:, CA_INDEX] = (np.arange(n)[:, None] * np.array([1.0, 0.0, 0.0])
                        if coords is None else coords)
    mask = np.zeros((n, len(ATOM_TYPES)), dtype=bool)
    mask[:, CA_INDEX] = ca_present
    asym = np.concatenate([np.full(s, i) for i, s in enumerate(chain_sizes)])
    return {"atom37_positions": pos, "atom37_mask": mask,
            "asym_id": asym.astype(np.int8),
            "aatype": np.zeros(n, dtype=np.int8),
            "residue_index": np.concatenate([np.arange(s) for s in chain_sizes])}


class TestMonomerMetrics:
    def test_a_perfect_prediction_scores_one_per_chain(self):
        native = _bundle()
        got = monomer_metrics(_bundle(), native)
        assert got["TM_mean"] == pytest.approx(1.0, abs=1e-6)
        assert got["TM_min"] == pytest.approx(1.0, abs=1e-6)
        assert got["CA_RMSD_mean"] == pytest.approx(0.0, abs=1e-6)
        assert [c["chain"] for c in got["per_chain"]] == [0, 1]

    def test_each_chain_is_superposed_independently(self):
        """A correctly folded pair that is docked wrongly must still score TM 1.0 per
        chain — that separation is the whole reason these run beside DockQ."""
        native = _bundle()
        pred = _bundle()
        pred["atom37_positions"][4:, CA_INDEX] += np.array([100.0, 100.0, 100.0])
        got = monomer_metrics(pred, native)
        assert got["TM_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_chain_length_mismatch_is_flagged_not_scored(self):
        got = monomer_metrics(_bundle((4, 2)), _bundle((4, 3)))
        bad = [c for c in got["per_chain"] if c["chain"] == 1][0]
        assert bad["n"] == 0
        assert "length mismatch" in bad["_note"]
        assert np.isnan(bad["TM"])

    def test_a_chain_with_no_shared_ca_is_flagged(self):
        got = monomer_metrics(_bundle(ca_present=False), _bundle())
        assert all(c["_note"] == "no shared CA" for c in got["per_chain"])
        assert np.isnan(got["TM_mean"])

    def test_tm_min_reports_the_worst_chain(self):
        rng = np.random.default_rng(5)
        native = _bundle((30, 30))
        pred = _bundle((30, 30))
        pred["atom37_positions"][30:, CA_INDEX] += rng.normal(size=(30, 3)) * 8
        got = monomer_metrics(pred, native)
        assert got["TM_min"] < got["TM_mean"] < 1.0
