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

Every row also carries **identity**: a ``version``, a human ``description`` and an
``expected_entries`` count. These are not decoration. A split is a file on a filesystem
that nothing stops from changing underneath a published number, and the drift is already
observable — the comments in ``datasets.yaml`` claimed ``val_pinder_chain`` was 98 rows
and ``val_pinder_pair`` 474, while the parquets hold 106 and 454. ``expected_entries``
turns that class of drift from a stale comment into a failed check, and
:meth:`Dataset.fingerprint` records which bytes were actually read.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterable

import yaml

from ecstasy.config import resolve

#: Row keys that describe the dataset rather than tell a loader where to look. Split out
#: in `load_dataset` so loaders keep narrow, explicit signatures.
_META_KEYS = ("version", "description", "expected_entries", "tags")

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

    def __init__(self, name: str, version: int = 1, description: str = "",
                 expected_entries: int | None = None, tags: list | None = None):
        self.name = name
        self.version = int(version)
        self.description = description
        self.expected_entries = None if expected_entries is None else int(expected_entries)
        self.tags = list(tags or [])

    @abstractmethod
    def entries(self) -> Iterable[Entry]: ...

    @abstractmethod
    def gt_for(self, entry_id: str) -> dict: ...

    def score(self, entry: Entry, contact_path: Path,
              metrics: tuple[str, ...] | None = None) -> dict[str, float]:
        """Score one prediction against this dataset's ground truth.

        Concrete, not abstract: scoring is a pure function of what ``gt_for`` returns and
        the predicted map, so every loader would otherwise reimplement it identically.
        A loader supplies ``gt_for``; this supplies the rest.

        `metrics` names registered contact metrics. Defaulting to
        ``DEFAULT_CONTACT_METRICS`` keeps the reported set identical to what ecstasy
        produced before metrics were selectable, so adding a metric to the registry can
        never silently change a headline number.
        """
        import numpy as np

        from ecstasy.metrics import DEFAULT_CONTACT_METRICS, ContactEval
        from ecstasy.metrics import registry as metric_registry

        probs = np.asarray(np.load(contact_path)["probs"], dtype=np.float32)
        gt = self.gt_for(entry.id)
        seqs = gt["sequences"]
        if len(seqs) != 2:
            return {"_skipped": "non-dimer"}
        la, lb = len(seqs[0]), len(seqs[1])
        L = la + lb
        if probs.shape[0] != L or gt["contact_map"].shape[0] != L:
            return {"_error": f"shape mismatch: probs={probs.shape}, "
                              f"gt={gt['contact_map'].shape}, L={L}"}

        ev = ContactEval(probs=probs, gt=gt["contact_map"], valid=gt["valid"],
                         chain_lengths=(la, lb))
        out = metric_registry.compute(metrics or DEFAULT_CONTACT_METRICS, ev)
        # K is not a metric — it is the denominator every P@K is taken over, and it says
        # whether a target had enough signal to be scored at all.
        _, gti, vi = ev.inter_block()
        out["K"] = float(int((gti & vi).sum()))
        if gt.get("is_homodimer") is not None:
            out["is_homodimer"] = float(bool(gt["is_homodimer"]))
        return out

    # --- identity -------------------------------------------------------------------

    def source_paths(self) -> dict[str, Path]:
        """The files/directories this dataset is defined by. Loaders should override.

        Used to fingerprint what was actually read, so a result can be checked against
        the split it claims to describe.
        """
        return {}

    def fingerprint(self) -> dict:
        """Content identity of this dataset's sources, for the provenance record."""
        from ecstasy.provenance import file_identity

        out: dict[str, dict] = {}
        for key, path in self.source_paths().items():
            p = Path(path)
            out[key] = (file_identity(p) if p.is_file()
                        else {"path": str(p), "exists": p.exists(), "kind": "directory"})
        return out

    def composition(self) -> dict:
        """What this split is made of — computed ONCE and stored, never per campaign.

        This exists because of a concrete failure. A campaign doc asserted "40 true
        homodimers, 111 heterodimers" for the 151-dimer val split, derived ad hoc by
        testing exact chain-sequence equality. The dataset's own flag says 129/22. Both
        numbers were reported as fact, and downstream a homo/hetero result table was
        computed on the wrong one.

        Reconciled: of the 90 that the flag calls homodimeric while their sequences
        differ, **86 are >= 90% identical** — the same protein, differing only in how many
        residues were experimentally resolved (10bl: 344 vs 345). Exact string equality
        was therefore measuring crystallography, not biology.

        The lesson is not "pick the right boolean". It is that a boolean bakes one
        definition into the data and hides the rest, so the next person picks a different
        one. The **distribution** is recorded instead, and any threshold can be applied
        afterwards without re-deriving anything:

            n_entries, n_dimers, chain_lengths (min/median/max),
            n_homodimer_flag, chain_identity histogram + per-entry values

        Note the flag is not perfectly clean either: 4 of the 151 fall below 90% identity,
        one at 0.128 (699 vs 96 residues — a chain almost entirely unresolved).
        """
        from difflib import SequenceMatcher

        ids, lengths, identity, flags = [], [], {}, {}
        for entry in self.entries():
            if not self.has_gt(entry.id):
                continue
            gt = self.gt_for(entry.id)
            seqs = gt["sequences"]
            ids.append(entry.id)
            lengths.append(sum(len(s) for s in seqs))
            if len(seqs) == 2:
                identity[entry.id] = round(
                    1.0 if seqs[0] == seqs[1]
                    else SequenceMatcher(None, seqs[0], seqs[1], autojunk=False).ratio(), 4)
            if gt.get("is_homodimer") is not None:
                flags[entry.id] = bool(gt["is_homodimer"])

        vals = sorted(identity.values())
        bands = {"identical": 0, ">=0.99": 0, ">=0.95": 0, ">=0.90": 0, "<0.90": 0}
        for r in vals:
            key = ("identical" if r == 1.0 else ">=0.99" if r >= 0.99
                   else ">=0.95" if r >= 0.95 else ">=0.90" if r >= 0.90 else "<0.90")
            bands[key] += 1
        return {
            "n_entries": len(ids),
            "n_dimers": len(identity),
            "chain_identity_bands": bands,
            "chain_identity": identity,
            "n_homodimer_flag": sum(flags.values()) if flags else None,
            "total_length": {
                "min": min(lengths) if lengths else None,
                "median": (sorted(lengths)[len(lengths) // 2] if lengths else None),
                "max": max(lengths) if lengths else None,
            },
        }

    def manifest(self) -> dict:
        """Machine-readable description — what `ecstasy datasets` emits.

        This is the surface an agent reads to answer "what evaluation sets exist and what
        are they" without opening the YAML or guessing from a name.
        """
        return {
            "name": self.name,
            "kind": getattr(self, "kind", None),
            "version": self.version,
            "description": self.description,
            "expected_entries": self.expected_entries,
            "tags": self.tags,
            "sources": {k: str(v) for k, v in self.source_paths().items()},
        }

    def has_gt(self, entry_id: str) -> bool:
        """Is ground truth actually present for this entry? Loaders should override."""
        return True

    def coverage(self) -> dict:
        """How much of this split can actually be scored here.

        Entry count and ground-truth presence are different questions, and conflating
        them hides the more dangerous one: an index can list 930 entries while only 76
        have GT, in which case a run predicts 930 targets on a GPU and reports a mean
        over 8% of the split that prints identically to a mean over all of it.
        """
        ids = [e.id for e in self.entries()]
        missing = [i for i in ids if not self.has_gt(i)]
        n = len(ids)
        return {
            "n_entries": n,
            "n_gt_present": n - len(missing),
            "n_gt_missing": len(missing),
            "fraction": (n - len(missing)) / n if n else 0.0,
            "missing_first_20": missing[:20],
        }

    def verify(self) -> dict:
        """Check the split on disk still matches what the row claims.

        Returns ``{ok, n_entries, expected_entries, coverage, problems[]}``. Walks the
        index and stats the GT, so it is a deliberate command rather than something
        scoring pays for on every run.
        """
        problems: list[str] = []
        for key, path in self.source_paths().items():
            if not Path(path).exists():
                problems.append(f"missing source {key}: {path}")
        n, cov = None, None
        if not problems:
            try:
                cov = self.coverage()
                n = cov["n_entries"]
            except Exception as e:  # noqa: BLE001
                problems.append(f"could not enumerate entries: {type(e).__name__}: {e}")
        if n is not None and self.expected_entries is not None and n != self.expected_entries:
            problems.append(
                f"entry count drift: found {n}, row declares expected_entries="
                f"{self.expected_entries}. Either the split changed underneath published "
                f"results, or the row is stale — resolve before trusting new numbers.")
        if cov and cov["n_gt_missing"]:
            problems.append(
                f"ground truth missing for {cov['n_gt_missing']}/{cov['n_entries']} "
                f"entries ({cov['fraction']:.1%} present) — this split cannot produce a "
                f"complete result here.")
        if not self.description:
            problems.append("row has no description")
        return {"name": self.name, "ok": not problems, "n_entries": n,
                "expected_entries": self.expected_entries, "coverage": cov,
                "problems": problems}


def _registry() -> dict:
    return yaml.safe_load(_REGISTRY.read_text())


def dataset_names() -> list[str]:
    return sorted(k for k in _registry() if not k.startswith("_"))


def dataset_manifests() -> list[dict]:
    """Manifests for every registered dataset, without touching the filesystem."""
    return [load_dataset(n).manifest() for n in dataset_names()]


def load_dataset(name: str) -> Dataset:
    reg = _registry()
    if name not in reg or name.startswith("_"):
        raise KeyError(f"unknown dataset {name!r}; registered: {dataset_names()}")
    row = resolve(dict(reg[name]))  # expand ${VAR}; copy so we can pop
    kind = row.pop("kind")
    # import loaders so they self-register via __init_subclass__
    from ecstasy.datasets import ecstasy_native, mentos  # noqa: F401

    if kind not in Dataset.KINDS:
        raise KeyError(f"dataset {name!r} has unknown kind {kind!r}; have {sorted(Dataset.KINDS)}")
    return Dataset.KINDS[kind](name=name, **row)
