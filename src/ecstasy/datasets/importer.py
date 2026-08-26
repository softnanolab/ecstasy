"""Materialise a dataset into one self-contained folder.

Before this, "the dataset" was not one thing you could point at: the index parquet lived
in one MENTOS subtree, the ground truth in another, natives somewhere else, predictions
under a separate root. That is *why* four registered splits turned out to be missing 84-92%
of their ground truth with nothing noticing.

An imported dataset is one directory that owns everything a model or a metric could need:

    datasets/<name>/
        dataset.yaml        identity, source, coverage, provenance of the import
        index.parquet
        gt/<xx>/<id>.npz    pickle-free ground truth (see ecstasy.datasets.store)

Duplication across splits is accepted deliberately — the validation sets are small, and a
folder you can copy to another machine and run is worth more than the disk it costs.

Import is explicit and reports exactly what it could not find, so a partial split is a
visible fact rather than a silent 8% mean.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ecstasy.datasets import store


@dataclass
class ImportReport:
    name: str
    dest: Path
    n_entries: int = 0
    n_written: int = 0
    n_already_present: int = 0
    missing: list = field(default_factory=list)
    failed: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing and not self.failed

    @property
    def coverage(self) -> float:
        done = self.n_written + self.n_already_present
        return done / self.n_entries if self.n_entries else 0.0

    def summary(self) -> str:
        done = self.n_written + self.n_already_present
        line = (f"{self.name}: {done}/{self.n_entries} entries "
                f"({self.coverage:.1%}) -> {self.dest}")
        if self.missing:
            line += f"\n  ground truth absent for {len(self.missing)}: " \
                    f"{', '.join(self.missing[:8])}" \
                    + (" ..." if len(self.missing) > 8 else "")
        if self.failed:
            line += f"\n  failed for {len(self.failed)}: " + "; ".join(
                f"{i}: {e}" for i, e in self.failed[:5])
        return line


class SourceChanged(RuntimeError):
    """The import source no longer matches what the row's ``built_from`` recorded."""


def source_from_spec(spec: dict, name: str):
    """Build the (unregistered) loader an import reads from, guarding its identity.

    The spec is a row's ``built_from`` block. It is deliberately NOT a registered
    dataset: a MENTOS-backed row that anything could score against is exactly what we
    are getting rid of. This constructs the loader for the length of one import and
    throws it away.

    ``index_sha256``, when present, is asserted before a single entry is read. That is
    what makes an imported dataset safe to keep: if MENTOS rebuilds its split under the
    same path, the re-import stops rather than quietly replacing your dataset with a
    different one that still answers to the same name.
    """
    from ecstasy.datasets.base import Dataset
    from ecstasy.datasets import ecstasy_native, mentos  # noqa: F401

    spec = dict(spec)
    expect = spec.pop("index_sha256", None)
    kind = spec.pop("kind", "mentos_square")
    if kind not in Dataset.KINDS:
        raise KeyError(f"{name}: built_from names unknown kind {kind!r}; "
                       f"have {sorted(Dataset.KINDS)}")
    src = Dataset.KINDS[kind](name=f"{name}@source", **spec)

    index = Path(getattr(src, "index", ""))
    if not index.exists():
        raise FileNotFoundError(
            f"{name}: import source index not found: {index}. The source is a "
            f"prerequisite for BUILDING this dataset, not for using one already built.")
    if expect:
        from ecstasy.provenance import sha256_file

        actual = sha256_file(index)
        if actual != expect:
            raise SourceChanged(
                f"{name}: import source changed underneath the recipe.\n"
                f"  index:    {index}\n"
                f"  recorded: {expect}\n"
                f"  on disk:  {actual}\n"
                f"This is not a corrupted file — it is a DIFFERENT split at the same "
                f"path. Importing over the existing folder would silently replace a "
                f"dataset that published results refer to. Import under a new name and "
                f"bump the version, or update index_sha256 deliberately if you have "
                f"confirmed the change is intended.")
    return src


def _sample_to_arrays(sample) -> dict | None:
    """A MENTOS Sample -> the arrays ecstasy stores. None if it lacks full-atom GT."""
    needed = ("aatype", "atom37_positions", "atom37_mask", "residue_index", "asym_id")
    if any(getattr(sample, f, None) is None for f in needed):
        return None
    return {
        "sequences": list(sample.sequences),
        "atom37_positions": sample.atom37_positions.numpy(),
        "atom37_mask": sample.atom37_mask.numpy(),
        "aatype": sample.aatype.numpy(),
        "asym_id": sample.asym_id.numpy(),
        "residue_index": sample.residue_index.numpy(),
        "chain_ids": list(sample.chain_ids) if sample.chain_ids else None,
        "is_homodimer": sample.is_homodimer,
    }


def import_from_mentos(source, dest: Path, name: str | None = None,
                       overwrite: bool = False, limit: int | None = None,
                       identity: dict | None = None) -> ImportReport:
    """Convert a MENTOS-backed split into a self-contained ecstasy dataset folder.

    `source` is a loaded ``MentosSquareDataset``. Reading its pickles requires MENTOS
    installed **here, once, at import time** — which is the point: after this, scoring the
    imported folder never needs it again.

    `identity` carries version/description/expected_entries/tags/contact_bin from the
    registry row that OWNS the folder. Identity belongs to the ecstasy dataset, not to
    whatever it was converted from — the source is a throwaway reader.
    """
    import torch

    dest = Path(dest)
    name = name or source.name
    gt_dir = dest / "gt"
    dest.mkdir(parents=True, exist_ok=True)

    entries = list(source.entries())
    if limit is not None:
        entries = entries[: int(limit)]
    report = ImportReport(name=name, dest=dest, n_entries=len(entries))

    for entry in entries:
        out = store.entry_path(gt_dir, entry.id)
        if out.exists() and not overwrite:
            report.n_already_present += 1
            continue
        src_pt = source.gt_path(entry.id)
        if not src_pt.exists():
            report.missing.append(entry.id)
            continue
        try:
            sample = torch.load(src_pt, weights_only=False, map_location="cpu")
            arrays = _sample_to_arrays(sample)
            if arrays is None:
                report.failed.append((entry.id, "no full-atom ground truth in sample"))
                continue
            store.write_entry(out, source=f"mentos:{src_pt}", **arrays)
            report.n_written += 1
        except Exception as e:  # noqa: BLE001
            report.failed.append((entry.id, f"{type(e).__name__}: {e}"))

    # The index travels with the dataset, filtered to THIS dataset's rows.
    #
    # Copying the source index wholesale would carry the split column with it, and the
    # folder would only be interpretable by re-applying the source's split value — a
    # MENTOS convention leaking into a dataset that is supposed to have stopped
    # depending on MENTOS. recent_pp's source index holds 23,463 PDB-train rows beside
    # its 151 val rows; a reader that forgot `split: val` would silently enumerate all
    # of them. Filtering here means the folder's row count IS the dataset.
    index_dest = dest / "index.parquet"
    if not index_dest.exists() or overwrite:
        _write_index(source, index_dest, [e.id for e in entries])

    _write_manifest(dest, name, source, report, identity or {})

    # Composition is computed here, once, against the folder that was just written — so
    # every later reader gets the same numbers instead of re-deriving them with whatever
    # definition they happen to pick. Written separately from dataset.yaml because the
    # per-entry identity table is long and dataset.yaml should stay scannable.
    from ecstasy.datasets.ecstasy_native import EcstasyDataset

    # split=None deliberately: the written index IS this dataset's rows, so there is
    # nothing left to select from. Composition is read back through the same loader
    # every later reader uses, over the folder that was just written.
    imported = EcstasyDataset(name=name, root=dest)
    (dest / "composition.json").write_text(json.dumps(imported.composition(), indent=1))
    return report


#: Source columns describing where the SOURCE kept its data, not what the entry is.
#: Dropped on import. `relative_path` points into the MENTOS tree — a stale pointer that
#: someone will eventually follow — and `split` is now the folder itself, so keeping it
#: would only invite re-filtering a dataset that is already exactly its own rows.
_SOURCE_ONLY_COLUMNS = ("relative_path", "split")


def _write_index(source, dest: Path, ids: list[str]) -> None:
    """Write this dataset's own index: the source's rows for `ids`, in `ids` order.

    Filtering on the split BEFORE the ids matters: FoldBench's index carries the same
    entry under several split values (`foldbench` is the union of `foldbench_pp` and
    `foldbench_abag`), so an id-only filter returns each row two or three times. The
    length assertion below catches that rather than writing a silently inflated index.
    """
    import pandas as pd

    df = pd.read_parquet(source.index)
    split = getattr(source, "split", None)
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    df = df[df["id"].astype(str).isin(set(ids))].copy()
    df["id"] = df["id"].astype(str)
    order = {i: n for n, i in enumerate(ids)}
    df = df.sort_values("id", key=lambda c: c.map(order)).reset_index(drop=True)
    if len(df) != len(ids):
        raise RuntimeError(
            f"index filter produced {len(df)} rows for {len(ids)} entries — duplicate "
            f"or missing ids in {source.index} (split={split!r})")
    df = df.drop(columns=[c for c in _SOURCE_ONLY_COLUMNS if c in df.columns])
    df.to_parquet(dest, index=False)


def _write_manifest(dest: Path, name: str, source, report: ImportReport,
                    identity: dict) -> None:
    """dataset.yaml: what this folder is, where it came from, and how complete it is.

    Coverage is recorded at import time so a later reader does not have to stat thousands
    of files to discover the folder is 8% populated.

    The folder is self-describing on purpose: copied to another machine it still knows
    its version, its contact-bin convention and how complete it is, with no registry row
    and no MENTOS anywhere in sight.
    """
    from ecstasy import provenance

    index_src = Path(getattr(source, "index", ""))
    manifest = {
        "name": name,
        "version": identity.get("version", getattr(source, "version", 1)),
        "description": identity.get("description", getattr(source, "description", "")),
        "expected_entries": report.n_entries,
        "tags": list(identity.get("tags", getattr(source, "tags", []))),
        "contact_bin": identity.get("contact_bin", getattr(source, "contact_bin", 19)),
        "gt_format_version": store.FORMAT_VERSION,
        # A --limit import writes a folder whose expected_entries EQUALS its truncated
        # count, so it would otherwise verify as a perfectly healthy dataset of the
        # wrong size. Marked here so `datasets --verify` can refuse it.
        "partial_import": bool(identity.get("partial_import")),
        # A truncated import is a different dataset wearing the same name. Recorded so
        # `ecstasy datasets --verify` calls it out: expected_entries would otherwise
        # equal the truncated count and the folder would verify as perfectly healthy.
        "partial_import": bool(identity.get("partial_import")),
        "imported": {
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "from": str(getattr(source, "gt_root", "")),
            "index_source": str(index_src),
            # Recorded so the folder can say which bytes it was built from long after
            # that path has changed or gone. This is the only trace of the source that
            # survives the import, and nothing reads it to find data.
            "index_sha256": (provenance.sha256_file(index_src)
                             if index_src.is_file() else None),
            "source_split": getattr(source, "split", None),
            "ecstasy": provenance.git_state(Path(__file__).resolve().parents[3]),
            "n_written": report.n_written,
            "n_already_present": report.n_already_present,
            "n_missing": len(report.missing),
            "n_failed": len(report.failed),
            "coverage": report.coverage,
            "complete": report.complete,
        },
    }
    if report.missing:
        manifest["imported"]["missing_first_50"] = report.missing[:50]
    (dest / "dataset.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (dest / "import_report.json").write_text(json.dumps({
        "missing": report.missing, "failed": report.failed}, indent=1))
