"""Resolve checkpoint/dataset *names* to concrete paths from the gitignored local cache
(``registry.local.yaml`` at the repo root), which
``scripts/mentos-perf-benchmarking/notion_pull.py`` writes from the Notion benchmarking
Registry. Notion is the source of truth — nothing concrete (paths, run ids, steps) is committed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_CACHE = Path(__file__).resolve().parents[3] / "registry.local.yaml"


@lru_cache(maxsize=1)
def _load() -> dict:
    if not _CACHE.exists():
        raise FileNotFoundError(
            f"{_CACHE} not found — run "
            "`PYTHONPATH=src python scripts/mentos-perf-benchmarking/notion_pull.py` to pull the "
            "Notion benchmarking Registry into a local cache (see README → Configuration).")
    return yaml.safe_load(_CACHE.read_text()) or {}


def checkpoint(name: str) -> dict:
    cks = _load().get("checkpoints", {})
    if name not in cks:
        raise KeyError(f"checkpoint {name!r} not in registry (have: {sorted(cks)}). "
                       "Add it in the Notion Checkpoints table, then re-run notion_pull.")
    return cks[name]


def dataset(name: str) -> dict:
    ds = _load().get("datasets", {})
    if name not in ds:
        raise KeyError(f"dataset {name!r} not in registry (have: {sorted(ds)}).")
    return ds[name]


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
