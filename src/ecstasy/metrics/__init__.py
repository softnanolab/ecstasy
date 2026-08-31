"""Metrics: named, reusable, and registered in exactly one place.

Importing this package registers the built-in metrics, so `registry.names()` is populated
for anything that imports `ecstasy.metrics` — scoring, plotting, the CLI, a manifest.
"""
from ecstasy.metrics import builtins  # noqa: F401  (import registers the built-ins)
from ecstasy.metrics.contact import (
    pak_from_pairs,
    pak_inter_chain,
    pak_inter_tolerant,
)
from ecstasy.metrics.eval_inputs import ContactEval, StructureEval
from ecstasy.metrics.registry import Metric, compute, describe, get, names, register

#: The default contact metric set, used when a run does not name one. Kept identical to
#: what ecstasy reported before metrics were selectable, so existing results stay
#: comparable and adding a metric to the registry never silently changes a headline
#: number.
DEFAULT_CONTACT_METRICS = ("AUC", "P@K", "P@K/2", "P@K/5")

#: The default structure metric set. iRMSD and LRMSD are in it deliberately, not as
#: optional diagnostics: DockQ averages fnat with two RMSD terms, so a prediction whose
#: backbone never formed still scores off fnat while both RMSD terms give near-zero
#: credit. A DockQ reported without them has repeatedly produced wrong conclusions on
#: this codebase — see CLAUDE.md. All of these share ONE DockQ subprocess.
#:
#: The random-placement floor is NOT here. It costs `null_draws` further subprocesses per
#: target, so it is requested explicitly rather than silently making scoring 10x slower.
DEFAULT_STRUCTURE_METRICS = ("DockQ", "Fnat", "iRMSD", "LRMSD",
                             "TM_mean", "TM_min", "CA_RMSD_mean")

__all__ = [
    "ContactEval", "StructureEval", "Metric",
    "compute", "describe", "get", "names", "register",
    "pak_from_pairs", "pak_inter_chain", "pak_inter_tolerant",
    "DEFAULT_CONTACT_METRICS", "DEFAULT_STRUCTURE_METRICS",
]
