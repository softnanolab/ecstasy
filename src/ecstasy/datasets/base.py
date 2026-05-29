"""Dataset abstraction + registry loader.

A *dataset* is a named evaluation set (a split of dimers with ground-truth
interchain contact maps). Datasets are declared as rows in
``registry/datasets.yaml`` — not as one Python subclass per split — and loaded
by :func:`load_dataset`, which resolves ``${VAR}`` paths and instantiates the
loader class named by the row's ``kind``.

A loader implements three methods:
  entries()          -> iterable of Entry (id, sequences, chain_ids)
  gt_for(entry_id)   -> {"contact_map": bool ndarray, "sequences": [...]}
  score(entry, npz)  -> {AUC, P@K, P@K/2, P@K/5, K} (or {"_skipped"/"_error": ...})
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterable

import yaml

from ecstasy.config import resolve

_REGISTRY = Path(__file__).resolve().parent.parent / "registry" / "datasets.yaml"


@dataclass(frozen=True)
class Entry:
    id: str
    sequences: tuple[str, ...]
    chain_ids: tuple[str, ...] = ("A", "B")
    metadata: dict = field(default_factory=dict)


class Dataset(ABC):
    #: maps the registry ``kind`` string to the loader class
    KINDS: ClassVar[dict[str, type["Dataset"]]] = {}
    kind: ClassVar[str]

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if getattr(cls, "kind", None):
            Dataset.KINDS[cls.kind] = cls

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def entries(self) -> Iterable[Entry]: ...

    @abstractmethod
    def gt_for(self, entry_id: str) -> dict: ...

    @abstractmethod
    def score(self, entry: Entry, contact_path: Path) -> dict[str, float]: ...


def _registry() -> dict:
    return yaml.safe_load(_REGISTRY.read_text())


def dataset_names() -> list[str]:
    return sorted(k for k in _registry() if not k.startswith("_"))


def load_dataset(name: str) -> Dataset:
    reg = _registry()
    if name not in reg or name.startswith("_"):
        raise KeyError(f"unknown dataset {name!r}; registered: {dataset_names()}")
    row = resolve(dict(reg[name]))  # expand ${VAR}; copy so we can pop
    kind = row.pop("kind")
    # import loaders so they self-register via __init_subclass__
    from ecstasy.datasets import mentos  # noqa: F401

    if kind not in Dataset.KINDS:
        raise KeyError(f"dataset {name!r} has unknown kind {kind!r}; have {sorted(Dataset.KINDS)}")
    return Dataset.KINDS[kind](name=name, **row)
