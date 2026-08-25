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


def register_all() -> None:
    _register_contact()
    # Structure metrics (DockQ, iRMSD, LRMSD, TM, CA-RMSD) register here once the
    # structure scoring path lands on main — see PR #28. They are deliberately not
    # stubbed: an unimplemented registered name would report as a missing number rather
    # than an absent capability.


register_all()
