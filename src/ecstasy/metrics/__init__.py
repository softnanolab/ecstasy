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

__all__ = [
    "ContactEval", "StructureEval", "Metric",
    "compute", "describe", "get", "names", "register",
    "pak_from_pairs", "pak_inter_chain", "pak_inter_tolerant",
    "DEFAULT_CONTACT_METRICS",
]
