from __future__ import annotations

import numpy as np


def pak_from_pairs(probs: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """MENTOS-style Precision@K / P@K/2 / P@K/5 / AUC on a flat array of pairs.

    `probs` and `truth` are 1-D arrays of the same length. Each index is one
    candidate (residue-pair). Caller is responsible for selecting which pairs
    enter the metric (interchain-only, strict-upper-triangle, etc.).

    Returns the canonical metric dict used across the ecstasy benchmarks.
    "AUC" here is MENTOS's mean-precision over top-K_eff (NOT standard ROC-AUC);
    the name is kept for direct comparability with prior baselines.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    truth = np.asarray(truth).astype(bool).ravel()
    if probs.shape != truth.shape:
        raise ValueError(f"shape mismatch: probs={probs.shape}, truth={truth.shape}")

    K = int(truth.sum())
    nan = float("nan")
    if K == 0 or probs.size == 0:
        return {"AUC": nan, "P@K": nan, "P@K/2": nan, "P@K/5": nan, "K": K}

    K_eff = max(1, min(K, probs.size))
    order = np.argsort(-probs, kind="stable")[:K_eff]
    cum = np.cumsum(truth[order].astype(np.float64))
    arange = np.arange(1, K_eff + 1)

    idx_K5 = max(int(0.2 * K_eff) - 1, 0)
    idx_K2 = max(int(0.5 * K_eff) - 1, 0)
    idx_K = K_eff - 1

    return {
        "AUC": float((cum / arange).mean()),
        "P@K": float(cum[idx_K] / (idx_K + 1)),
        "P@K/2": float(cum[idx_K2] / (idx_K2 + 1)),
        "P@K/5": float(cum[idx_K5] / (idx_K5 + 1)),
        "K": K,
    }


def pak_inter_chain(
    contact_prob: np.ndarray,
    contact_gt: np.ndarray,
    chain_ids: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, float]:
    """Interchain P@K on a square (L, L) prediction + GT with per-token chain ids.

    K is the number of true interchain contacts in the strict upper triangle.
    Thin adapter around `pak_from_pairs` that selects the interchain-upper-tri
    pairs.

    `valid` (optional, (L, L) bool) gates the candidate pool to *defined* pairs —
    MENTOS's ``is_defined`` (resolved Cβ-Cβ). Pairs that are False in `valid`
    (e.g. unresolved Cβ / bin == -1) are dropped from BOTH the positives and the
    denominator, matching MENTOS exactly; without it, undefined pairs would
    linger as negatives and dilute precision differently than MENTOS.
    """
    contact_prob = np.asarray(contact_prob, dtype=np.float64)
    contact_gt = np.asarray(contact_gt).astype(bool)
    chain_ids = np.asarray(chain_ids)
    L = chain_ids.shape[0]
    if contact_prob.shape != (L, L) or contact_gt.shape != (L, L):
        raise ValueError(
            f"shape mismatch: prob={contact_prob.shape}, gt={contact_gt.shape}, L={L}"
        )
    triu = np.triu_indices(L, k=1)
    inter = chain_ids[triu[0]] != chain_ids[triu[1]]
    if valid is not None:
        inter = inter & np.asarray(valid).astype(bool)[triu]
    return pak_from_pairs(contact_prob[triu][inter], contact_gt[triu][inter])


# --- tolerant inter-chain P@K ---------------------------------------------------------
#
# Hoisted out of `scripts/mentos-perf-benchmarking/plot_pak_vs_flops.py`, where it lived
# as `_tol_inter_pak` and could only be reached by that one plotter. The behaviour is
# preserved: GT dilated by a Chebyshev radius in (chainA-res, chainB-res) space, scoring
# the top max(1, round(K/divisor)) predicted inter pairs, with invalid pairs excluded from
# both the positives and the candidate pool.
#
# ONE deliberate change: ties are broken with a stable sort. The plotter used numpy's
# default (quicksort), whose order among equal probabilities is unspecified. That matters
# here rather than being theoretical — contact.npz stores float16, which has ~3 decimal
# digits, so exact ties are common. `pak_from_pairs` already sorted stably, so the two
# implementations previously disagreed on tied pairs; this makes them agree.

def _dilate_chebyshev(mask: np.ndarray, tol: int) -> np.ndarray:
    """Dilate a 2-D bool mask by Chebyshev radius `tol` (a (2*tol+1)² square).

    Equivalent to `scipy.ndimage.binary_dilation(mask, np.ones((3, 3)), iterations=tol)`
    with the default zero border, implemented with shifts so `scipy` is not a runtime
    dependency of scoring. `tests/test_metrics_tolerance.py` pins the equivalence.
    """
    if tol <= 0:
        return mask.astype(bool)
    mask = mask.astype(bool)
    n, m = mask.shape
    out = np.zeros_like(mask)
    for dr in range(-tol, tol + 1):
        r0, r1 = max(0, dr), min(n, n + dr)
        sr0, sr1 = max(0, -dr), min(n, n - dr)
        if r0 >= r1:
            continue
        for dc in range(-tol, tol + 1):
            c0, c1 = max(0, dc), min(m, m + dc)
            sc0, sc1 = max(0, -dc), min(m, m - dc)
            if c0 >= c1:
                continue
            out[r0:r1, c0:c1] |= mask[sr0:sr1, sc0:sc1]
    return out


def pak_inter_tolerant(ev, tol: int = 0, divisor: int = 1) -> float:
    """Tolerant inter-chain P@(K/divisor) for a dimer `ContactEval`.

    K is the number of true, defined inter-chain contacts. The metric scores the top
    ``max(1, round(K / divisor))`` predicted inter pairs; a prediction counts as correct
    when a true contact lies within Chebyshev distance `tol` of it. ``tol=0`` is exact
    and reproduces the canonical P@K.

    Tolerance is not leniency for its own sake: a contact map predicted one residue off
    is a materially different error from one predicted across the complex, and exact P@K
    scores them identically. Returns NaN where the metric is undefined (non-dimer, or no
    true inter contacts) rather than a misleading 0.0.
    """
    if ev.n_chains != 2:
        return float("nan")
    cp, gti, vi = ev.inter_block()
    K = int((gti & vi).sum())
    if K == 0:
        return float("nan")
    topk = max(1, int(round(K / divisor)))
    hit = _dilate_chebyshev(gti, tol) & vi
    # Invalid pairs are pushed below every real probability rather than removed, so the
    # top-K slice keeps its shape; they can only be selected if valid pairs run out.
    ranked = np.argsort(-np.where(vi, cp, -np.inf).ravel(), kind="stable")[:topk]
    return float(hit.ravel()[ranked].sum()) / topk


def pak_inter_chain_metric(ev, key: str) -> float:
    """Registry adapter: one named value out of the `pak_inter_chain` family.

    The family (AUC, P@K, P@K/2, P@K/5) shares a single ranking, so computing them
    together is natural — but the registry addresses metrics individually, which means
    four registered names call this and the ranking is redone per name. That is an
    argsort over the inter-chain block: microseconds next to a DockQ invocation, and not
    worth the cache-invalidation risk of memoising on a struct holding numpy arrays.
    """
    d = pak_inter_chain(ev.probs, ev.gt, ev.chain_ids, valid=ev.valid)
    if key not in d:
        raise KeyError(f"{key!r} not produced by pak_inter_chain; have {sorted(d)}")
    return float(d[key])
