from __future__ import annotations

import numpy as np


def pak_from_pairs(probs: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """MINT-style Precision@K / P@K/2 / P@K/5 / AUC on a flat array of pairs.

    `probs` and `truth` are 1-D arrays of the same length. Each index is one
    candidate (residue-pair). Caller is responsible for selecting which pairs
    enter the metric (interchain-only, strict-upper-triangle, etc.).

    Returns the canonical metric dict used across the ecstasy benchmarks.
    "AUC" here is MINT's mean-precision over top-K_eff (NOT standard ROC-AUC);
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
) -> dict[str, float]:
    """Interchain P@K on a square (L, L) prediction + GT with per-token chain ids.

    K is the number of true interchain contacts in the strict upper triangle.
    Thin adapter around `pak_from_pairs` that selects the interchain-upper-tri
    pairs.
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
    return pak_from_pairs(contact_prob[triu][inter], contact_gt[triu][inter])


def pak_inter_chain_rect(probs: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Interchain P@K on a rectangular (Na, Nb) prediction + GT.

    Thin adapter around `pak_from_pairs`. Every entry is interchain by
    construction (chain A vs chain B); no triu / chain-id selection needed.
    Used by benchmarks that store rectangular interchain-only GT (e.g.
    ecstasy_v1) instead of square (L, L) GT with intra-chain blocks.
    """
    return pak_from_pairs(probs, gt)
