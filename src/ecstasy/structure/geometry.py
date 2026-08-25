"""Cβ-Cβ distance bins — the ground-truth convention, owned by ecstasy.

This is what makes datasets independent of MENTOS. Today ``gt_for`` unpickles a
``mentos.dataclasses.Sample``, which is the only reason a scoring environment must have
MENTOS installed at all; once ecstasy can compute the ground truth itself, that
dependency disappears from scoring entirely.

The conventions below are **reproduced exactly** from ``mentos.geometry``, because every
published contact number depends on them:

* **Cβ is virtual for every residue**, built from N, CA and C with the trRosetta
  parameters (length 1.522 Å, angle 1.927 rad, dihedral -2.143 rad). Not the crystal Cβ
  atom, and not "CA for glycine" — a single rule for all 20 residues. This is why the
  ground truth needs only backbone atoms and is insensitive to sidechain resolution.
* **64 bins over 63 AF2 edges**, ``linspace(2.3125, 21.6875, 63)``, right-inclusive via
  ``np.digitize``: bin 0 is ``≤2.3125 Å``, bin 63 is ``>21.6875 Å``.
* **-1 marks undefined** — any residue missing an N, CA or C gives a NaN Cβ, and every
  pair it participates in is undefined. Undefined pairs are excluded from both the
  positives and the candidate pool downstream; they must never count as negatives.

A contact is ``0 <= bin < contact_bin``, with ``contact_bin=19`` (≤ 7.9375 Å) matching
MENTOS's ``CONTACT_LABEL_THRESHOLD_BIN``.
"""
from __future__ import annotations

import numpy as np

#: AF2-style: 63 boundaries over [2.3125, 21.6875] Å -> 64 bins.
NUM_DISTANCE_BINS = 64
DISTANCE_BIN_EDGES = np.linspace(2.3125, 21.6875, NUM_DISTANCE_BINS - 1)

#: bins 0..18 are contacts, i.e. Cβ-Cβ <= 7.9375 Å.
CONTACT_BIN = 19

#: trRosetta virtual-Cβ construction parameters.
_CB_LENGTH = 1.522
_CB_ANGLE = 1.927
_CB_DIHEDRAL = -2.143


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)


def virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Virtual Cβ from backbone N, CA, C. NaN in propagates to NaN out.

    Applied to every residue including glycine — the point of a virtual Cβ is that the
    ground truth does not depend on whether a sidechain was resolved.
    """
    n, ca, c = (np.asarray(a, dtype=np.float64) for a in (n, ca, c))
    with np.errstate(invalid="ignore", divide="ignore"):
        bc = _normalize(n - ca)
        perp = _normalize(np.cross(n - c, bc))
        basis = [bc, np.cross(perp, bc), perp]
        d = [
            _CB_LENGTH * np.cos(_CB_ANGLE),
            _CB_LENGTH * np.sin(_CB_ANGLE) * np.cos(_CB_DIHEDRAL),
            -_CB_LENGTH * np.sin(_CB_ANGLE) * np.sin(_CB_DIHEDRAL),
        ]
        return ca + sum(m * di for m, di in zip(basis, d))


def cb_distance_matrix(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(L, L) Cβ-Cβ distances in Å. NaN wherever a residue's backbone is incomplete."""
    cb = virtual_cb(n, ca, c)
    return np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)


def distance_bins(dist: np.ndarray) -> np.ndarray:
    """Bin a distance matrix; -1 where undefined."""
    bins = np.digitize(dist, DISTANCE_BIN_EDGES).astype(np.int64)
    bins[np.isnan(dist)] = -1
    return bins


def bins_from_atom37(positions: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(bins, dist)`` from an atom37 bundle.

    A residue counts as resolved only when N, CA **and** C are all present; anything less
    cannot give a virtual Cβ, so its coordinates are set to NaN and every pair it
    participates in becomes undefined.
    """
    positions = np.asarray(positions, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    backbone = positions[:, :3, :].copy()          # atom37 order: N, CA, C
    incomplete = ~mask[:, :3].all(axis=1)
    backbone[incomplete] = np.nan
    dist = cb_distance_matrix(backbone[:, 0], backbone[:, 1], backbone[:, 2])
    return distance_bins(dist), dist


def contacts_from_bins(bins: np.ndarray, contact_bin: int = CONTACT_BIN
                       ) -> tuple[np.ndarray, np.ndarray]:
    """``(contact_map, valid)`` from binned distances.

    ``valid`` is MENTOS's ``is_defined``: a pair counts only where its bin is resolved.
    Returning the two together is deliberate — a caller that takes the contact map
    without the validity mask silently treats unresolved pairs as non-contacts.
    """
    bins = np.asarray(bins)
    valid = bins >= 0
    return valid & (bins < int(contact_bin)), valid
