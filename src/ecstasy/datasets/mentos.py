"""MENTOS square-GT dataset loader (seq_id_30 + the four deleaked val splits).

All MENTOS PDB-processed splits share one format: an ``index.parquet`` with a
``split`` column and a ``sequences`` array per row, and per-entry ground truth
at ``<gt_root>/<id[:2]>/<id>.pt`` holding a *square* (L, L) binned Cβ-Cβ distance
map (-1 marks unresolved Cβ). One class serves every such split; the split is
chosen by the registry row, not by subclassing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ecstasy.datasets.base import Dataset, Entry


class MentosSquareDataset(Dataset):
    kind = "mentos_square"

    def __init__(self, name: str, index: str, gt_root: str, split: str = "val",
                 contact_bin: int = 5, swap_chains: bool = False, **meta):
        # `meta` carries the row's identity fields (version/description/expected_entries/
        # tags). Passing them through rather than accepting **kwargs blindly means a
        # typo'd key in datasets.yaml raises here instead of being silently ignored.
        super().__init__(name, **meta)
        self.index = Path(index)
        self.gt_root = Path(gt_root)
        self.split = split
        self.contact_bin = int(contact_bin)
        # swap_chains: chain-order-permutation experiment. Reverse each dimer's chain
        # order (A,B)->(B,A) at input AND reindex the square GT to match, so the model
        # is scored on the same interface seen in flipped order. Monomers pass through.
        self.swap_chains = bool(swap_chains)

    def source_paths(self) -> dict[str, Path]:
        return {"index": self.index, "gt_root": self.gt_root}

    def gt_path(self, entry_id: str) -> Path:
        return self.gt_root / entry_id[:2] / f"{entry_id}.pt"

    def has_gt(self, entry_id: str) -> bool:
        return self.gt_path(entry_id).exists()

    @staticmethod
    def _swap_perm(la: int, L: int) -> np.ndarray:
        # new concat order = chainB (orig [la:L)) then chainA (orig [0:la))
        return np.r_[np.arange(la, L), np.arange(0, la)]

    def entries(self) -> Iterable[Entry]:
        import pandas as pd

        df = pd.read_parquet(self.index)
        df = df[df["split"] == self.split]
        for row in df.itertuples():
            seqs = tuple(row.sequences)
            if self.swap_chains and len(seqs) == 2:
                seqs = (seqs[1], seqs[0])
            chain_ids = tuple(["A", "B"][: len(seqs)])
            yield Entry(id=str(row.id), sequences=seqs, chain_ids=chain_ids)

    def gt_for(self, entry_id: str) -> dict:
        import torch

        # GT .pt files pickle a `mentos.dataclasses.Sample`. The scoring env ships the
        # `mentos` package (.venv-boltz / .venv-mentos both have it editable from
        # /home/.../mentos), so torch.load resolves the class natively — no rename
        # shim. (Lazy torch import: only a scoring env reaches here, never the
        # torch-less orchestrator. See the mentos_package_and_venvs memory.)
        sample = torch.load(self.gt_path(entry_id), weights_only=False, map_location="cpu")
        # bin < contact_bin == contact; -1 (unresolved) must NOT count as contact.
        raw = sample.contact_map.numpy()
        contact_map = (raw >= 0) & (raw < self.contact_bin)
        # `valid` = MENTOS is_defined: a pair is defined iff its Cβ-Cβ bin is resolved
        # (raw >= 0). Unresolved (-1) pairs are dropped from the candidate pool so they
        # never count as negatives (matches mentos.metrics_inter_chain).
        valid = raw >= 0
        seqs = list(sample.sequences)
        if self.swap_chains and len(seqs) == 2:
            perm = self._swap_perm(len(seqs[0]), contact_map.shape[0])
            contact_map = contact_map[np.ix_(perm, perm)]
            valid = valid[np.ix_(perm, perm)]
            seqs = [seqs[1], seqs[0]]
        return {"contact_map": contact_map, "valid": valid, "sequences": seqs,
                "is_homodimer": sample.is_homodimer}
