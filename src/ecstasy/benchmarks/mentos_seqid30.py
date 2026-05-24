from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ecstasy.benchmarks.base import Benchmark, Entry, register_benchmark
from ecstasy.metrics.contact import pak_inter_chain


@register_benchmark
class MentosSeqid30Bench(Benchmark):
    name = "mentos_seqid30"
    parquet = Path("/projects/u6jv/public/MENTOS/DATA/pdb/processed/splits/seq_id_30/index.parquet")
    gt_root = Path("/projects/u6jv/public/MENTOS/DATA/pdb/processed/data")
    split = "val"

    def entries(self) -> Iterable[Entry]:
        import pandas as pd
        df = pd.read_parquet(self.parquet)
        df = df[df["split"] == self.split]
        for row in df.itertuples():
            seqs = tuple(row.sequences)
            chain_ids = tuple(["A", "B"][: len(seqs)])
            yield Entry(id=str(row.id), sequences=seqs, chain_ids=chain_ids)

    # MENTOS stores `contact_map` as a binned distance index in [0, 9] over edges
    # [4, 5, 6, 7, 8, 9, 10, 11, 12]. Bins 0-4 (< 8 Å Cβ-Cβ) are "contacts".
    contact_threshold_bin: int = 5

    def gt_for(self, entry_id: str) -> dict:
        import torch
        p = self.gt_root / entry_id[:2] / f"{entry_id}.pt"
        sample = torch.load(p, weights_only=False, map_location="cpu")
        # MENTOS marks unresolved Cβ positions with bin = -1; those must NOT be
        # counted as contacts. Without the `>= 0` guard, `-1 < threshold` is
        # True and K gets inflated → P@K is systematically wrong for entries
        # with missing residues.
        raw = sample.contact_map.numpy()
        contact_map = (raw >= 0) & (raw < self.contact_threshold_bin)
        sequences = list(sample.sequences)
        return {"contact_map": contact_map, "sequences": sequences}

    def score(self, entry: Entry, contact_path: Path) -> dict[str, float]:
        d = np.load(contact_path)
        probs = np.asarray(d["probs"], dtype=np.float32)
        gt = self.gt_for(entry.id)
        contact_gt = gt["contact_map"]
        seqs = gt["sequences"]
        if len(seqs) != 2:
            return {"_skipped": "non-dimer"}
        la, lb = len(seqs[0]), len(seqs[1])
        L = la + lb
        if probs.shape[0] != L or contact_gt.shape[0] != L:
            return {"_error": f"shape mismatch: probs={probs.shape}, gt={contact_gt.shape}, L={L}"}
        chain_ids = np.array([0] * la + [1] * lb)
        return pak_inter_chain(probs, contact_gt, chain_ids)
