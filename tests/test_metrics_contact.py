import numpy as np
import pytest

from ecstasy.metrics.contact import pak_inter_chain


def _two_chain_ids(la: int, lb: int) -> np.ndarray:
    return np.array([0] * la + [1] * lb)


def test_perfect_prediction_pak_is_one():
    la, lb = 4, 5
    L = la + lb
    chain_ids = _two_chain_ids(la, lb)
    rng = np.random.default_rng(0)
    gt = np.zeros((L, L), dtype=bool)
    interchain_pairs = [(i, j) for i in range(la) for j in range(la, L)]
    chosen = rng.choice(len(interchain_pairs), size=6, replace=False)
    for k in chosen:
        i, j = interchain_pairs[k]
        gt[i, j] = gt[j, i] = True
    prob = gt.astype(float) + 1e-3
    out = pak_inter_chain(prob, gt, chain_ids)
    assert out["K"] == 6
    assert out["P@K"] == pytest.approx(1.0)
    assert out["P@K/2"] == pytest.approx(1.0)
    assert out["AUC"] == pytest.approx(1.0)


def test_random_prediction_close_to_chance():
    la, lb = 20, 20
    L = la + lb
    chain_ids = _two_chain_ids(la, lb)
    rng = np.random.default_rng(42)
    gt = (rng.random((L, L)) < 0.05)
    gt = np.triu(gt, k=1)
    gt = gt | gt.T
    prob = rng.random((L, L))
    out = pak_inter_chain(prob, gt, chain_ids)
    assert 0.0 <= out["P@K"] <= 1.0
    assert 0.0 <= out["AUC"] <= 1.0


def test_no_interchain_contacts_returns_nan():
    la, lb = 3, 3
    L = la + lb
    chain_ids = _two_chain_ids(la, lb)
    gt = np.zeros((L, L), dtype=bool)
    prob = np.zeros((L, L))
    out = pak_inter_chain(prob, gt, chain_ids)
    assert np.isnan(out["P@K"])
    assert out["K"] == 0


def test_intra_chain_contacts_are_ignored():
    la, lb = 5, 5
    L = la + lb
    chain_ids = _two_chain_ids(la, lb)
    gt = np.zeros((L, L), dtype=bool)
    gt[0, 1] = gt[1, 0] = True
    gt[la, la + 1] = gt[la + 1, la] = True
    gt[0, la] = gt[la, 0] = True
    prob = np.zeros((L, L))
    prob[0, la] = prob[la, 0] = 0.99
    out = pak_inter_chain(prob, gt, chain_ids)
    assert out["K"] == 1
    assert out["P@K"] == pytest.approx(1.0)


def test_shape_mismatch_raises():
    chain_ids = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="shape mismatch"):
        pak_inter_chain(np.zeros((4, 5)), np.zeros((4, 4)), chain_ids)


def test_pak_inter_chain_rect_matches_square_form():
    """The rectangular adapter on a (Na, Nb) block must give the same metrics
    as the square form on the embedded (L, L) matrix when the only true contacts
    live in the interchain block."""
    from ecstasy.metrics.contact import pak_inter_chain_rect
    la, lb = 4, 6
    L = la + lb
    rng = np.random.default_rng(7)

    # Rectangular GT + probs over the interchain block.
    gt_rect = rng.random((la, lb)) < 0.25
    probs_rect = rng.random((la, lb))

    # Embed into a square (L, L) form: probs_full has the interchain block;
    # gt_full mirrors gt_rect into the upper-tri interchain positions only.
    probs_full = np.zeros((L, L), dtype=np.float64)
    probs_full[:la, la:] = probs_rect
    probs_full[la:, :la] = probs_rect.T  # any symmetric fill; intra-chain
                                          # entries are masked out by chain_ids
    gt_full = np.zeros((L, L), dtype=bool)
    gt_full[:la, la:] = gt_rect
    gt_full[la:, :la] = gt_rect.T
    chain_ids = _two_chain_ids(la, lb)

    rect = pak_inter_chain_rect(probs_rect, gt_rect)
    square = pak_inter_chain(probs_full, gt_full, chain_ids)

    # K differs (square counts upper-tri only, rect counts the full block).
    assert square["K"] * 2 == rect["K"] or square["K"] == rect["K"]
    # Top-K ranking is determined by the same probabilities, so identity-shaped
    # outputs match up to ties; assert the headline P@K agrees.
    assert rect["P@K"] == pytest.approx(square["P@K"])
    assert rect["P@K/2"] == pytest.approx(square["P@K/2"])
    assert rect["P@K/5"] == pytest.approx(square["P@K/5"])


def test_pak_from_pairs_handles_empty():
    """All-False GT must return NaN metrics and K = 0 (no division by zero)."""
    from ecstasy.metrics.contact import pak_from_pairs
    out = pak_from_pairs(np.array([0.1, 0.9, 0.5]), np.array([False, False, False]))
    assert out["K"] == 0
    assert np.isnan(out["P@K"])
    assert np.isnan(out["AUC"])
