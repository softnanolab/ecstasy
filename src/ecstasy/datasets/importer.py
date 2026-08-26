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
import shutil
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
                       overwrite: bool = False, limit: int | None = None) -> ImportReport:
    """Convert a MENTOS-backed split into a self-contained ecstasy dataset folder.

    `source` is a loaded ``MentosSquareDataset``. Reading its pickles requires MENTOS
    installed **here, once, at import time** — which is the point: after this, scoring the
    imported folder never needs it again.
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

    # The index travels with the dataset; it is what defines the entry list.
    index_dest = dest / "index.parquet"
    if not index_dest.exists() or overwrite:
        shutil.copy2(source.index, index_dest)

    _write_manifest(dest, name, source, report)

    # Composition is computed here, once, against the folder that was just written — so
    # every later reader gets the same numbers instead of re-deriving them with whatever
    # definition they happen to pick. Written separately from dataset.yaml because the
    # per-entry identity table is long and dataset.yaml should stay scannable.
    from ecstasy.datasets.ecstasy_native import EcstasyDataset

    imported = EcstasyDataset(name=name, root=dest, split=getattr(source, "split", None))
    (dest / "composition.json").write_text(json.dumps(imported.composition(), indent=1))
    return report


def _write_manifest(dest: Path, name: str, source, report: ImportReport) -> None:
    """dataset.yaml: what this folder is, where it came from, and how complete it is.

    Coverage is recorded at import time so a later reader does not have to stat thousands
    of files to discover the folder is 8% populated.
    """
    from ecstasy import provenance

    manifest = {
        "name": name,
        "version": getattr(source, "version", 1),
        "description": getattr(source, "description", ""),
        "expected_entries": report.n_entries,
        "tags": list(getattr(source, "tags", [])),
        "contact_bin": getattr(source, "contact_bin", 19),
        "gt_format_version": store.FORMAT_VERSION,
        "imported": {
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "from": str(getattr(source, "gt_root", "")),
            "index_source": str(getattr(source, "index", "")),
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
