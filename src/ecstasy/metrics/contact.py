from __future__ import annotations

import numpy as np


def pak_inter_chain(
    contact_prob: np.ndarray,
    contact_gt: np.ndarray,
    chain_ids: np.ndarray,
) -> dict[str, float]:
    """Interchain Precision@K, P@K/2, P@K/5, AUC. Numpy-only, single sample.

    contact_prob: (L, L) float — predicted contact probability
    contact_gt:   (L, L) bool/int — 1 where C-beta < 8 A
    chain_ids:    (L,) int — chain index per token

    K is the number of true interchain contacts in the strict upper triangle.
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
    probs = contact_prob[triu][inter]
    truth = contact_gt[triu][inter]

    K = int(truth.sum())
    if K == 0 or probs.size == 0:
        return {"AUC": float("nan"), "P@K": float("nan"), "P@K/2": float("nan"), "P@K/5": float("nan"), "K": K}

    K_eff = max(1, min(K, probs.size))
    order = np.argsort(-probs, kind="stable")[:K_eff]
    sorted_truth = truth[order].astype(np.float64)
    cum = np.cumsum(sorted_truth)
    arange = np.arange(1, K_eff + 1)

    idx_K5 = max(int(0.2 * K_eff) - 1, 0)
    idx_K2 = max(int(0.5 * K_eff) - 1, 0)
    idx_K = K_eff - 1

    return {
        # MINT's "AUC" is mean P@i over i=1..K_eff, not standard ROC-AUC.
        # Kept under the same name for direct comparability with prior baselines.
        "AUC": float((cum / arange).mean()),
        "P@K": float(cum[idx_K] / (idx_K + 1)),
        "P@K/2": float(cum[idx_K2] / (idx_K2 + 1)),
        "P@K/5": float(cum[idx_K5] / (idx_K5 + 1)),
        "K": K,
    }
