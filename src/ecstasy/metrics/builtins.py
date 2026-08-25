"""Registration of the metrics ecstasy ships with.

Imported for its side effects by ``ecstasy.metrics``. Keeping registration in one file
means the answer to "what metrics exist" is a single readable list rather than a search
for decorators, and adding a metric is a one-line edit here plus its implementation.

Naming: a metric's name is a stable identifier that ends up in committed results, so it
is written the way it is reported — ``P@K/2``, ``P@K(tol=2)`` — not snake_cased. Renaming
one silently orphans every historical record that used the old name.
"""
from __future__ import annotations

from ecstasy.metrics.contact import pak_inter_chain_metric, pak_inter_tolerant
from ecstasy.metrics.registry import register
from ecstasy.metrics.structure import _chain_stat, dockq_component


def _register_contact() -> None:
    for key, desc in (
        ("P@K", "Precision over the top-K predicted inter-chain pairs, K = number of "
                "true defined inter contacts. The headline contact metric."),
        ("P@K/2", "Precision over the top K/2 pairs — a stricter, higher-confidence cut."),
        ("P@K/5", "Precision over the top K/5 pairs — strictest routine cut."),
        ("AUC", "MENTOS-style mean precision over the top-K curve. NOT ROC-AUC; the name "
                "is kept for direct comparability with the published MENTOS baselines."),
    ):
        register(key, "contact", pak_inter_chain_metric, desc, key=key)

    # Spatial tolerance: a contact predicted one residue off is a different kind of error
    # from one predicted across the complex, and exact P@K cannot tell them apart.
    for tol in (1, 2):
        register(
            f"P@K(tol={tol})", "contact", pak_inter_tolerant,
            f"P@K counting a prediction correct when a true contact lies within "
            f"Chebyshev distance {tol} in (chainA-residue, chainB-residue) space.",
            tol=tol,
        )
    for tol in (1, 2):
        for divisor, label in ((2, "K/2"), (5, "K/5")):
            register(
                f"P@{label}(tol={tol})", "contact", pak_inter_tolerant,
                f"Tolerant precision (radius {tol}) over the top {label} predicted pairs.",
                tol=tol, divisor=divisor,
            )


def _register_structure() -> None:
    # DockQ and its components. All four come from ONE subprocess (the result is cached
    # on the eval input), so four registered names do not mean four invocations.
    #
    # iRMSD and LRMSD are registered as first-class metrics rather than left as
    # diagnostics because DockQ averages fnat with two RMSD terms: a prediction whose
    # backbone has not formed still scores off fnat alone while both RMSD terms give
    # near-zero credit. A DockQ number read without them is misleading.
    for key, desc, higher in (
        ("DockQ", "Overall docking quality, 0-1 (Basu & Wallner). NEVER read without "
                  "iRMSD/LRMSD beside it — an unformed backbone still scores off fnat.", True),
        ("Fnat", "Fraction of native interface contacts recovered.", True),
        ("iRMSD", "Interface backbone RMSD in Å. Lower is better.", False),
        ("LRMSD", "Ligand (mobile chain) RMSD in Å after receptor superposition. "
                  "Lower is better.", False),
    ):
        register(key, "structure", dockq_component, desc, higher_is_better=higher, key=key)

    # Per-chain fold quality, each chain superposed on its own. This is what separates
    # "folded but docked wrong" from "never folded" — two results a DockQ near zero
    # cannot tell apart, and the distinction a single-chain folder under a linker hack
    # exists to probe.
    for name, key, how, desc, higher in (
        ("TM_mean", "TM", "mean",
         "Mean per-chain TM-score, each chain superposed independently. Plain Kabsch, "
         "not TM-align's iterative search, so a slight under-estimate.", True),
        ("TM_min", "TM", "min",
         "Worst per-chain TM-score — catches one chain failing while the other folds.", True),
        ("CA_RMSD_mean", "CA_RMSD", "mean",
         "Mean per-chain CA-RMSD in Å after independent superposition. Lower is better.",
         False),
    ):
        register(name, "structure", _chain_stat, desc, higher_is_better=higher,
                 key=key, how=how)


def register_all() -> None:
    _register_contact()
    _register_structure()


register_all()
