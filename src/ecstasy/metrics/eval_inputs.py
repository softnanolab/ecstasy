"""The inputs metrics are computed against.

One struct per metric kind, so a metric signature is ``fn(ev, **params) -> float`` and
nothing has to be re-derived per metric. Before this, every caller assembled its own
slices — which is how the tolerant P@K in the plotting script ended up operating on the
rectangular inter-chain block while ``pak_inter_chain`` operated on the upper triangle.
Those happen to select the same pairs for a dimer, but nothing said so and nothing
checked it.

Building the struct once also makes the expensive parts (GT load, chain layout) shared
across every metric in a scoring pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactEval:
    """A predicted (L, L) contact-probability map with its ground truth.

    ``valid`` is MENTOS's ``is_defined``: a pair counts only where its Cβ-Cβ bin was
    resolved. Invalid pairs are dropped from BOTH the positives and the candidate pool,
    never left in as negatives — that is what keeps P@K comparable to the MENTOS series.

    ``chain_lengths`` gives the concatenation layout, so inter-chain selection is derived
    here once rather than re-derived (differently) by each metric.
    """

    KIND = "contact"

    probs: np.ndarray            # (L, L) float, higher = more likely a contact
    gt: np.ndarray               # (L, L) bool
    valid: np.ndarray            # (L, L) bool
    chain_lengths: tuple[int, ...]

    def __post_init__(self):
        L = int(sum(self.chain_lengths))
        for name, arr in (("probs", self.probs), ("gt", self.gt), ("valid", self.valid)):
            if arr.shape != (L, L):
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected {(L, L)} from "
                    f"chain_lengths={self.chain_lengths}")

    @property
    def n_chains(self) -> int:
        return len(self.chain_lengths)

    @property
    def chain_ids(self) -> np.ndarray:
        """(L,) chain index per token."""
        return np.concatenate([np.full(n, i) for i, n in enumerate(self.chain_lengths)])

    def inter_block(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The (La, Lb) chain-A × chain-B block of (probs, gt, valid). Dimers only.

        For a dimer this block is exactly the set of inter-chain pairs in the strict
        upper triangle, so it is interchangeable with the triangle-based selection — but
        it is 2-D, which is what spatial tolerance needs: dilating GT is only meaningful
        in (chainA-residue, chainB-residue) space.
        """
        if self.n_chains != 2:
            raise ValueError(f"inter_block needs a dimer, got {self.n_chains} chains")
        la = int(self.chain_lengths[0])
        return (np.asarray(self.probs)[:la, la:],
                np.asarray(self.gt)[:la, la:].astype(bool),
                np.asarray(self.valid)[:la, la:].astype(bool))


@dataclass(frozen=True)
class StructureEval:
    """A predicted structure and its native, as atom37 bundles.

    Kept deliberately thin: DockQ is an external CLI that takes two PDB paths, so the
    rendered paths travel with the arrays rather than being re-rendered per metric.
    """

    KIND = "structure"

    pred: dict                   # atom37 bundle (see ecstasy.structure.pdb)
    native: dict                 # atom37 bundle
    pred_pdb: object = None      # Path, rendered once and shared across metrics
    native_pdb: object = None    # Path
    entry_id: str = ""
