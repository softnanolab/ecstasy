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

from dataclasses import dataclass, field

import numpy as np


# eq=False on both eval inputs: they hold numpy arrays, so a generated __eq__ raises
# "truth value of an array is ambiguous", and with frozen=True the generated __hash__
# would raise too. Identity comparison is the only sane semantics here.
@dataclass(frozen=True, eq=False)
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


@dataclass(eq=False)
class StructureEval:
    """A predicted structure and its native, as atom37 bundles.

    DockQ is an external CLI taking two PDB paths, so the rendered paths travel with the
    arrays rather than being re-rendered per metric.

    **Derived quantities are cached here, on the input they derive from.** Seven metrics
    are registered against a structure and four of them read the same DockQ run; without
    a cache, addressing them individually would invoke the binary four times. Putting the
    cache on the struct — rather than having metric functions reach in and set attributes
    on it — keeps the ownership obvious and lets the metric adapters be one-liners.
    """

    KIND = "structure"

    pred: dict                   # atom37 bundle (see ecstasy.structure.pdb)
    native: dict                 # atom37 bundle
    pred_pdb: object = None      # Path, rendered once and shared across metrics
    native_pdb: object = None    # Path
    entry_id: str = ""

    #: Lazily filled; kept out of repr so a debug print is not a wall of cached numbers.
    _dockq: dict | None = field(default=None, repr=False)
    _per_chain: list | None = field(default=None, repr=False)

    def dockq(self) -> dict[str, float]:
        """All DockQ components, from a single invocation of the binary."""
        if self._dockq is None:
            from ecstasy.metrics.structure import run_dockq
            self._dockq = run_dockq(self.pred_pdb, self.native_pdb) or {}
        return self._dockq

    def per_chain(self) -> list[dict]:
        """Per-chain fold quality, each chain superposed independently."""
        if self._per_chain is None:
            from ecstasy.metrics.structure import per_chain_quality
            self._per_chain = per_chain_quality(self.pred, self.native)
        return self._per_chain
