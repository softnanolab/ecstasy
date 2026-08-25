"""Loader for a self-contained ecstasy dataset folder.

The point of this loader is what it does NOT import. Reading ground truth here is
``np.load(..., allow_pickle=False)`` over arrays ecstasy wrote — no ``torch``, no
``mentos``, no pickled classes that must remain importable. A scoring environment for an
imported dataset needs numpy and pandas.

The folder is described by its own ``dataset.yaml``, so a dataset copied to another
machine carries its identity, its contact-bin convention and its import coverage with it,
rather than depending on a registry row that may not exist there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from ecstasy.datasets import store
from ecstasy.datasets.base import Dataset, Entry


class EcstasyDataset(Dataset):
    """A dataset materialised by ``ecstasy datasets import``."""

    kind = "ecstasy"
    has_structure_gt = True

    def __init__(self, name: str, root: str, split: str | None = None,
                 contact_bin: int | None = None, **meta):
        self.root = Path(root)
        self.manifest_path = self.root / "dataset.yaml"
        folder = {}
        if self.manifest_path.exists():
            folder = yaml.safe_load(self.manifest_path.read_text()) or {}
        # The folder's own manifest is authoritative for what it contains; a registry row
        # may override identity but must not silently disagree about the GT convention.
        merged = {
            "version": folder.get("version", 1),
            "description": folder.get("description", ""),
            "expected_entries": folder.get("expected_entries"),
            "tags": folder.get("tags", []),
        }
        merged.update({k: v for k, v in meta.items() if v is not None})
        super().__init__(name, **merged)
        self.split = split
        self.contact_bin = int(contact_bin if contact_bin is not None
                               else folder.get("contact_bin", 19))
        self.folder_manifest = folder

    # --- identity -------------------------------------------------------------------

    @property
    def index(self) -> Path:
        return self.root / "index.parquet"

    @property
    def gt_root(self) -> Path:
        return self.root / "gt"

    def source_paths(self) -> dict[str, Path]:
        return {"index": self.index, "gt_root": self.gt_root, "manifest": self.manifest_path}

    def gt_path(self, entry_id: str) -> Path:
        return store.entry_path(self.gt_root, entry_id)

    def has_gt(self, entry_id: str) -> bool:
        return self.gt_path(entry_id).exists()

    # --- data -----------------------------------------------------------------------

    def entries(self) -> Iterable[Entry]:
        import pandas as pd

        df = pd.read_parquet(self.index)
        if self.split is not None and "split" in df.columns:
            df = df[df["split"] == self.split]
        for row in df.itertuples():
            seqs = tuple(row.sequences)
            yield Entry(id=str(row.id), sequences=seqs,
                        chain_ids=tuple(["A", "B"][: len(seqs)]))

    def gt_for(self, entry_id: str) -> dict:
        return store.read_entry(self.gt_path(entry_id), contact_bin=self.contact_bin)

    def native_bundle(self, entry_id: str) -> dict | None:
        gt = self.gt_for(entry_id)
        return {k: gt[k] for k in ("atom37_positions", "atom37_mask", "aatype",
                                   "asym_id", "residue_index")}
