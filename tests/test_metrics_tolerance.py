"""Tolerant inter-chain P@K — the metric hoisted out of the plotting script.

Two properties matter and are pinned here:

1. It reproduces the implementation it replaces. That one produced published figures, so
   a silent numerical change would retroactively invalidate them. Verified on real data
   (68 targets, 612 comparisons across tol x divisor, max |diff| 0.0); here it is pinned
   against `scipy.ndimage.binary_dilation`, the original's dilation primitive.
2. Tolerance means what it claims — a near-miss counts, a far miss does not — rather than
   being a knob that quietly inflates every score.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecstasy.metrics import ContactEval, pak_inter_chain, pak_inter_tolerant
from ecstasy.metrics.contact import _dilate_chebyshev

scipy_ndimage = pytest.importorskip("scipy.ndimage", reason="scipy not installed")


class TestDilation:
    """The numpy shift-dilation must equal scipy's, or scoring diverges from the figures."""

    @pytest.mark.parametrize("tol", [1, 2, 3, 5])
    def test_matches_scipy_on_random_masks(self, tol):
        rng = np.random.default_rng(tol)
        for _ in range(5):
            mask = rng.random((40, 37)) < 0.08
            np.testing.assert_array_equal(
                _dilate_chebyshev(mask, tol),
                scipy_ndimage.binary_dilation(mask, np.ones((3, 3), bool), iterations=tol))

    def test_matches_scipy_at_the_border(self):
        """Zero border, not wraparound — a contact at the edge must not dilate around."""
        mask = np.zeros((6, 6), bool)
        mask[0, 0] = mask[5, 5] = True
        np.testing.assert_array_equal(
            _dilate_chebyshev(mask, 2),
            scipy_ndimage.binary_dilation(mask, np.ones((3, 3), bool), iterations=2))

    def test_tol_zero_is_identity(self):
        rng = np.random.default_rng(0)
        mask = rng.random((12, 15)) < 0.2
        np.testing.assert_array_equal(_dilate_chebyshev(mask, 0), mask)

    def test_radius_is_chebyshev_not_euclidean(self):
        """A (2t+1) square, so the diagonal reaches as far as the axis."""
        mask = np.zeros((9, 9), bool)
        mask[4, 4] = True
        out = _dilate_chebyshev(mask, 2)
        assert out[2, 2] and out[6, 6]        # corners of the square are included
        assert out.sum() == 25
        assert not out[1, 4] and not out[4, 1]


def _ev(la=4, lb=5, contacts=(), probs=None, invalid=()):
    """Dimer ContactEval with GT contacts at given (a, b) inter-block positions."""
    L = la + lb
    gt = np.zeros((L, L), bool)
    valid = np.ones((L, L), bool)
    for a, b in contacts:
        gt[a, la + b] = True
    for a, b in invalid:
        valid[a, la + b] = False
    p = np.zeros((L, L), float) if probs is None else probs
    return ContactEval(probs=p, gt=gt, valid=valid, chain_lengths=(la, lb))


class TestTolerantPak:
    def test_exact_hit_scores_one_at_every_tolerance(self):
        p = np.zeros((9, 9))
        p[1, 4 + 2] = 1.0
        ev = _ev(contacts=[(1, 2)], probs=p)
        for tol in (0, 1, 2):
            assert pak_inter_tolerant(ev, tol=tol) == 1.0

    def test_a_near_miss_is_wrong_at_tol_0_and_right_at_tol_1(self):
        """The whole point: one residue off is a different error from across the complex."""
        p = np.zeros((9, 9))
        p[1, 4 + 3] = 1.0                       # GT contact is at (1, 2); predicted (1, 3)
        ev = _ev(contacts=[(1, 2)], probs=p)
        assert pak_inter_tolerant(ev, tol=0) == 0.0
        assert pak_inter_tolerant(ev, tol=1) == 1.0

    def test_a_far_miss_stays_wrong(self):
        p = np.zeros((9, 9))
        p[3, 4 + 4] = 1.0
        ev = _ev(contacts=[(0, 0)], probs=p)
        assert pak_inter_tolerant(ev, tol=2) == 0.0

    def test_tolerance_is_monotonic(self):
        rng = np.random.default_rng(3)
        la, lb = 12, 14
        gt = np.zeros((la + lb, la + lb), bool)
        gt[:la, la:][rng.random((la, lb)) < 0.1] = True
        probs = np.zeros((la + lb, la + lb))
        probs[:la, la:] = rng.random((la, lb))
        ev = ContactEval(probs=probs, gt=gt, valid=np.ones_like(gt),
                         chain_lengths=(la, lb))
        scores = [pak_inter_tolerant(ev, tol=t) for t in (0, 1, 2, 3)]
        assert scores == sorted(scores), scores

    def test_divisor_narrows_the_slice(self):
        """P@K/5 scores the top K/5 predictions, so a good top slice outscores P@K."""
        la, lb = 10, 10
        gt = np.zeros((20, 20), bool)
        probs = np.zeros((20, 20))
        for i in range(10):                       # 10 true contacts
            gt[i, la + i] = True
        probs[0, la + 0] = 1.0                    # only the top-2 predictions are right
        probs[1, la + 1] = 0.9
        probs[5, la + 7] = 0.8                    # then a wrong one
        ev = ContactEval(probs=probs, gt=gt, valid=np.ones_like(gt), chain_lengths=(la, lb))
        assert pak_inter_tolerant(ev, divisor=5) == 1.0      # top 2 of 10, both right
        assert pak_inter_tolerant(ev, divisor=1) < 1.0

    def test_undefined_cases_are_nan_not_zero(self):
        """0.0 would read as 'scored badly'; NaN reads as 'not scored'."""
        assert np.isnan(pak_inter_tolerant(_ev(contacts=[])))          # no true contacts
        mono = ContactEval(probs=np.zeros((4, 4)), gt=np.zeros((4, 4), bool),
                           valid=np.ones((4, 4), bool), chain_lengths=(4,))
        assert np.isnan(pak_inter_tolerant(mono))                      # not a dimer

    def test_invalid_pairs_are_excluded_from_positives(self):
        """An unresolved Cβ pair must not count as a true contact.

        Probabilities are distinct on purpose: with an all-zero map every pair ties and
        the stable tie-break picks index 0, which would pass by accident.
        """
        p = np.zeros((9, 9))
        p[1, 4 + 2] = 1.0        # highest, but its pair is invalid -> excluded entirely
        p[2, 4 + 3] = 0.5        # top *valid* prediction, and not a true contact
        p[0, 4 + 0] = 0.1        # the one true defined contact, ranked below
        ev = _ev(contacts=[(1, 2), (0, 0)], invalid=[(1, 2)], probs=p)
        # K counts only (0,0) -> 1, so a single prediction is scored, and it is wrong.
        assert pak_inter_tolerant(ev, tol=0) == 0.0

    def test_invalid_contacts_do_not_inflate_k(self):
        """K is the count of *defined* true contacts; masked ones must not swell it."""
        ev_all_valid = _ev(contacts=[(0, 0), (1, 2)])
        ev_one_masked = _ev(contacts=[(0, 0), (1, 2)], invalid=[(1, 2)])
        # With K=2 the top-2 are scored; with K=1 only the top-1. Give a map where the
        # single best prediction is right and the second is wrong.
        p = np.zeros((9, 9))
        p[0, 4 + 0] = 1.0
        p[3, 4 + 4] = 0.9
        ev_all_valid = _ev(contacts=[(0, 0), (1, 2)], probs=p)
        ev_one_masked = _ev(contacts=[(0, 0), (1, 2)], invalid=[(1, 2)], probs=p)
        assert pak_inter_tolerant(ev_all_valid, tol=0) == 0.5     # 1 of top 2
        assert pak_inter_tolerant(ev_one_masked, tol=0) == 1.0    # 1 of top 1

    def test_invalid_pairs_cannot_win_the_ranking(self):
        """Masked pairs are pushed below every real probability, not merely ignored."""
        p = np.zeros((9, 9))
        p[0, 4 + 4] = 99.0                        # huge, but invalid
        p[0, 4 + 0] = 0.5                         # the real best
        ev = _ev(contacts=[(0, 0)], invalid=[(0, 4)], probs=p)
        assert pak_inter_tolerant(ev, tol=0) == 1.0


class TestAgreementWithCanonicalPak:
    """tol=0 must be the canonical P@K, or the registry holds two different metrics."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_tol_zero_equals_pak_inter_chain(self, seed):
        rng = np.random.default_rng(seed)
        la, lb = 9, 11
        L = la + lb
        gt = np.zeros((L, L), bool)
        gt[:la, la:] = rng.random((la, lb)) < 0.12
        valid = np.ones((L, L), bool)
        valid[:la, la:] = rng.random((la, lb)) < 0.9
        probs = np.zeros((L, L))
        probs[:la, la:] = rng.random((la, lb))
        ev = ContactEval(probs=probs, gt=gt, valid=valid, chain_lengths=(la, lb))
        if not (gt[:la, la:] & valid[:la, la:]).any():
            pytest.skip("no defined inter contacts for this seed")
        canonical = pak_inter_chain(probs, gt, ev.chain_ids, valid=valid)["P@K"]
        assert pak_inter_tolerant(ev, tol=0) == pytest.approx(canonical)


class TestContactEval:
    def test_rejects_shape_disagreement(self):
        with pytest.raises(ValueError, match="expected"):
            ContactEval(probs=np.zeros((5, 5)), gt=np.zeros((5, 5), bool),
                        valid=np.ones((5, 5), bool), chain_lengths=(3, 3))

    def test_chain_ids_follow_the_layout(self):
        ev = _ev(la=3, lb=2)
        np.testing.assert_array_equal(ev.chain_ids, [0, 0, 0, 1, 1])

    def test_inter_block_is_the_chain_a_by_chain_b_rectangle(self):
        ev = _ev(la=3, lb=2, contacts=[(0, 1)])
        cp, gti, vi = ev.inter_block()
        assert cp.shape == gti.shape == vi.shape == (3, 2)
        assert gti[0, 1] and gti.sum() == 1

    def test_inter_block_rejects_a_monomer(self):
        mono = ContactEval(probs=np.zeros((4, 4)), gt=np.zeros((4, 4), bool),
                           valid=np.ones((4, 4), bool), chain_lengths=(4,))
        with pytest.raises(ValueError, match="dimer"):
            mono.inter_block()
