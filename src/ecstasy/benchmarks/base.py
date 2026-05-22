from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterable


@dataclass(frozen=True)
class Entry:
    id: str
    sequences: tuple[str, ...]
    chain_ids: tuple[str, ...] = ("A", "B")
    metadata: dict = field(default_factory=dict)


class Benchmark(ABC):
    name: ClassVar[str]

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    @property
    def out_root(self) -> Path:
        return self.data_root / "ecstasy" / "benchmarks" / self.name

    @abstractmethod
    def entries(self) -> Iterable[Entry]: ...

    @abstractmethod
    def gt_for(self, entry_id: str) -> dict: ...

    @abstractmethod
    def score(self, entry: Entry, contact_path: Path) -> dict[str, float]: ...


BENCHMARKS: dict[str, type[Benchmark]] = {}


def register_benchmark(cls: type[Benchmark]) -> type[Benchmark]:
    BENCHMARKS[cls.name] = cls
    return cls


def load_benchmark(name: str, data_root: Path) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; registered: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name](data_root=data_root)
