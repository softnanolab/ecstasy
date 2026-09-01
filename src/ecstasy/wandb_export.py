"""Mirror the committed benchmark record into Weights & Biases.

``results/runs.jsonl`` stays the source of truth. This module *derives* a wandb run from
each published row, exactly as ``ecstasy report`` derives ``LEADERBOARD.md`` from the same
rows — so wandb is a **view**, never a second place a number can be entered. Nothing here
writes to the store, and an export failure cannot damage it.

Two consequences of being a derived view, both load-bearing:

* **Export is idempotent.** A run's wandb id is a digest of the row's identity — dataset,
  model, variant and *both* fingerprints — so re-exporting converges instead of
  multiplying. Re-scoring after a metric fix changes ``scoring`` and therefore produces a
  **new** wandb run rather than overwriting the old one, which preserves the same "the
  number moved, and here is what changed underneath it" history that the append-only
  JSONL gives you in git.
* **The caveats travel with the numbers.** The leaderboard marks a dirty model tree with
  ``†``, a dirty ecstasy tree with ``‡`` and partial coverage with ``*``. Those become
  wandb *tags*, because a reader filtering the runs table is exactly as entitled to know
  that a commit does not describe what ran as a reader of the markdown.

The projection (:func:`payload`) is a pure function with no wandb import, so it is fully
testable without the package or a network. :func:`export` is the thin shell that talks to
wandb and is the only part that needs either.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Iterable

#: Default project. Overridable per invocation; ``WANDB_PROJECT`` wins over this and the
#: ``project=`` argument wins over both, so a scratch export never lands in the real one
#: by accident.
DEFAULT_PROJECT = "ecstasy-benchmarks"

#: Contact metrics, in the reporting order used by ``ecstasy report``.
_CONTACT = ("P@K", "P@K/2", "P@K/5", "AUC")

#: Structure metrics. DockQ first, then the geometry terms that say whether it means
#: anything — the same reading order the leaderboard enforces, and for the same reason.
_STRUCTURE = ("DockQ", "iRMSD", "LRMSD", "Fnat", "TM_mean", "TM_min", "CA_RMSD_mean",
              "null_DockQ_mean", "null_DockQ_max")


class WandbUnavailable(RuntimeError):
    """The wandb package is not importable. The message says how to fix it."""


def run_id(row: dict) -> str:
    """A stable wandb run id for a published row.

    Derived from the row's identity rather than assigned, so exporting the same store
    twice updates the same runs instead of creating duplicates. Includes both
    fingerprints for the same reason :class:`ecstasy.results.Key` does: two runs of one
    model on one dataset are the same *result* only if the inputs to prediction and the
    inputs to scoring were both unchanged.

    16 hex chars of sha256. wandb ids must be ``[A-Za-z0-9_-]+``, which hex satisfies.
    """
    ident = "|".join([
        str(row.get("dataset", {}).get("name", "")),
        str(row.get("model", {}).get("name", "")),
        str(row.get("model", {}).get("variant", "")),
        str(row.get("fingerprints", {}).get("prediction", "")),
        str(row.get("fingerprints", {}).get("scoring", "")),
    ])
    return hashlib.sha256(ident.encode()).hexdigest()[:16]


def _dirty_models(row: dict) -> list[str]:
    return sorted(n for n, c in (row.get("provenance", {}).get("model_code") or {}).items()
                  if c.get("dirty"))


def tags(row: dict) -> list[str]:
    """Filterable labels, including the ones a reader is entitled to be warned by.

    The three caveat tags mirror the leaderboard's footnote marks exactly. A run whose
    recorded commit does not describe what ran must be identifiable in the runs table
    without opening it.
    """
    out = [
        f"dataset:{row.get('dataset', {}).get('name')}",
        f"model:{row.get('model', {}).get('name')}",
        f"variant:{row.get('model', {}).get('variant')}",
    ]
    if _dirty_models(row):
        out.append("dirty-model-tree")
    if row.get("provenance", {}).get("ecstasy_dirty"):
        out.append("dirty-ecstasy-tree")
    if not (row.get("coverage") or {}).get("complete", True):
        out.append("partial-coverage")
    if row.get("structure"):
        out.append("structure-scored")
    if (row.get("flops") or {}).get("mean_flops"):
        out.append("flops-measured")
    return out


def _config(row: dict) -> dict[str, Any]:
    ds, mdl = row.get("dataset", {}) or {}, row.get("model", {}) or {}
    n, cov = row.get("n", {}) or {}, row.get("coverage", {}) or {}
    prov, fps = row.get("provenance", {}) or {}, row.get("fingerprints", {}) or {}
    return {
        "dataset": ds.get("name"),
        "dataset_version": ds.get("version"),
        "dataset_expected_entries": ds.get("expected_entries"),
        "model": mdl.get("name"),
        "variant": mdl.get("variant"),
        "n_evaluated": n.get("evaluated"),
        "n_skipped": n.get("skipped"),
        "n_errors": n.get("errors"),
        "coverage_complete": cov.get("complete"),
        "coverage_fraction": cov.get("fraction"),
        "fingerprint_prediction": fps.get("prediction"),
        "fingerprint_scoring": fps.get("scoring"),
        "ecstasy_sha": prov.get("ecstasy_sha"),
        "ecstasy_dirty": bool(prov.get("ecstasy_dirty")),
        "dirty_model_trees": _dirty_models(row),
        "host": prov.get("host"),
        "captured_utc": prov.get("captured_utc"),
        "published_utc": row.get("published_utc"),
        "run_dir": row.get("run_dir"),
        "schema_version": row.get("schema_version"),
        "metric_names": (row.get("metrics") or {}).get("names") or [],
    }


def _summary(row: dict) -> dict[str, Any]:
    """The numbers, flattened into namespaced keys.

    **An unmeasured quantity is omitted, never zero.** A missing FLOPs figure written as
    ``0`` would place a model at the origin of the compute axis and read as a measurement;
    an absent key plus an explicit ``*_measured: False`` reads as what it is. This matters
    concretely today: ESMFold2's FLOPs are refused because its ESMC-6B backbone is
    uncounted, so the strongest model in the campaign has no cost figure at all, and a
    zero there would be a fabricated one.
    """
    out: dict[str, Any] = {}

    metrics = row.get("metrics") or {}
    for stat in ("mean", "median"):
        block = metrics.get(stat) or {}
        for m in _CONTACT:
            if block.get(m) is not None:
                out[f"contact/{stat}/{m}"] = block[m]

    struct = row.get("structure") or {}
    out["structure/scored"] = bool(struct)
    if struct:
        out["structure/n"] = struct.get("n")
        for stat in ("mean", "median"):
            block = struct.get(stat) or {}
            for m in _STRUCTURE:
                if block.get(m) is not None:
                    out[f"structure/{stat}/{m}"] = block[m]
        for frac in ("acceptable_fraction", "medium_fraction", "high_fraction"):
            if struct.get(frac) is not None:
                out[f"structure/{frac}"] = struct[frac]
        # Homo/hetero split: kept to the headline number and n per side rather than the
        # full metric set. On recent_pp the two differ on DockQ *and* on TM_min, i.e.
        # heterodimer failure is partly a folding failure — a distinction that vanishes
        # if only the pooled figure is exported.
        for side in ("homodimer", "heterodimer"):
            blk = struct.get(f"{side}_flag") or {}
            if not blk:
                continue
            out[f"structure/{side}/n"] = blk.get("n")
            for stat in ("mean", "median"):
                inner = blk.get(stat) or {}
                for m in ("DockQ", "TM_min"):
                    if inner.get(m) is not None:
                        out[f"structure/{side}/{stat}/{m}"] = inner[m]

    fl = row.get("flops") or {}
    measured = fl.get("mean_flops") is not None
    out["flops/measured"] = measured
    if measured:
        out["flops/mean_flops"] = fl["mean_flops"]
        out["flops/mean_tflops"] = fl["mean_flops"] / 1e12
        if fl.get("median_flops") is not None:
            out["flops/median_flops"] = fl["median_flops"]
            out["flops/median_tflops"] = fl["median_flops"] / 1e12
    return out


def payload(row: dict) -> dict[str, Any]:
    """Project one published row into what wandb needs. Pure; no wandb import.

    Returns ``{id, name, tags, config, summary}``. Keeping this separate from
    :func:`export` is what makes the projection testable in an environment with neither
    the package nor a network — which is the only environment the test gate has.
    """
    ds = (row.get("dataset") or {}).get("name")
    mdl = (row.get("model") or {}).get("name")
    var = (row.get("model") or {}).get("variant")
    return {
        "id": run_id(row),
        "name": f"{ds}/{mdl}/{var}",
        "tags": tags(row),
        "config": _config(row),
        "summary": _summary(row),
    }


def export(rows: Iterable[dict], project: str | None = None, entity: str | None = None,
           dry_run: bool = False) -> list[dict]:
    """Create or update one wandb run per row. Returns the payloads that were sent.

    ``dry_run`` returns the payloads without importing wandb or touching the network,
    which is both the smoke test and the way to inspect what would be sent.

    Resolution order for the project is argument, then ``WANDB_PROJECT``, then
    :data:`DEFAULT_PROJECT`; the entity falls back to ``WANDB_ENTITY`` and then to
    whatever the local login implies. ``WANDB_MODE=offline`` is honoured by wandb itself,
    so a sweep on a node without egress can log locally and be synced later — though on
    Isambard compute nodes egress to ``api.wandb.ai`` is direct and this is unnecessary.
    """
    payloads = [payload(r) for r in rows]
    if dry_run:
        return payloads

    try:
        import wandb
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise WandbUnavailable(
            "wandb is not installed. It is an opt-in extra so that the CLI does not "
            "require it:\n    uv pip install -e '.[wandb]'\n"
            "Then authenticate once with `wandb login` (a ~/.netrc entry for "
            "api.wandb.ai also works)."
        ) from e

    project = project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT
    entity = entity or os.environ.get("WANDB_ENTITY") or None

    for p in payloads:
        run = wandb.init(project=project, entity=entity, id=p["id"], name=p["name"],
                         tags=p["tags"], config=p["config"], resume="allow",
                         reinit=True)
        try:
            run.summary.update(p["summary"])
        finally:
            run.finish()
    return payloads
