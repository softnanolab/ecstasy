"""Resolve a MENTOS checkpoint *name* to concrete params from the committed
``registry/checkpoints.yaml`` (${VAR} paths, same interpolation as models.yaml/datasets.yaml).

checkpoints.yaml IS the source of truth — there is no external system behind it. Add a
checkpoint by adding a row (by hand, or via the `/experiment` command, which asks for the
required fields before appending one) and commit it, the same way a new dataset or model
preset is added to datasets.yaml / models.yaml.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from ecstasy.config import resolve

_REGISTRY = Path(__file__).resolve().parent / "checkpoints.yaml"


@lru_cache(maxsize=1)
def _load() -> dict:
    if not _REGISTRY.exists():
        raise FileNotFoundError(
            f"{_REGISTRY} not found — it should be committed at "
            "src/ecstasy/registry/checkpoints.yaml. If it's missing, recreate it with a "
            "top-level `checkpoints: {}` and add rows as you register checkpoints.")
    return yaml.safe_load(_REGISTRY.read_text()) or {}


def checkpoint(name: str) -> dict:
    cks = _load().get("checkpoints", {})
    if name not in cks:
        raise KeyError(f"checkpoint {name!r} not in registry (have: {sorted(cks)}). "
                       "Add a row to src/ecstasy/registry/checkpoints.yaml, then retry.")
    return resolve(dict(cks[name]))


def checkpoint_params(name: str) -> dict:
    """A registry checkpoint row -> the params the mentos runner expects (the keys a hardcoded
    preset used to carry). Raises if the checkpoint has no weights file (e.g. the init baseline)."""
    c = checkpoint(name)
    if not c.get("abs_path"):
        raise ValueError(f"checkpoint {name!r} has no abs_path (e.g. the random-init baseline) — "
                         "not runnable via `ecstasy run`.")
    return {
        "model_weights_path": c["abs_path"],
        "run_id": c.get("run_id"),
        "num_recycles": c.get("num_recycles"),
        "model_config_path": c.get("model_config_path"),
    }
