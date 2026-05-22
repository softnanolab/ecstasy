"""ecstasy_v1: leakage-free dimer benchmark.

Built from Boltz-2's validation_ids_v2.txt (deposited 2023-06-07 -> 2023-12-27,
post-Boltz-2 training cutoff) and Foldseek-deleaked against MINT-softnano
training chains at Pinder defaults (coverage >= 0.5, LDDT >= 0.7). 222 dimers
total (83 homo / 139 hetero).

GT files store rectangular interchain Cβ-Cβ contact + distance maps (Na, Nb)
rather than the full square (L, L) -- only the interchain block matters for
dimer evaluation. score() extracts the corresponding block from each model's
(L, L) prediction and computes Pinder-style Precision@K.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ecstasy.benchmarks.base import Benchmark, Entry, register_benchmark
from ecstasy.metrics.contact import pak_inter_chain_rect


@register_benchmark
class EcstasyV1Bench(Benchmark):
    name = "ecstasy_v1"
    root = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/master")
    parquet = root / "index.parquet"

    # Cβ-Cβ contact threshold in distogram bins.
    # MINT binning: bin 0 = d <= 4, bin k in 1..8 = k+3 < d <= k+4, bin 9 = d > 12
    # "contact" defined as Cβ-Cβ < 8 Å -> bins 0..4 (i.e. < 5).
    contact_threshold_bin: int = 5

    def entries(self) -> Iterable[Entry]:
        import pandas as pd

        df = pd.read_parquet(self.parquet)
        for row in df.itertuples():
            seqs = tuple(row.sequences)
            chain_ids = tuple([row.chain_a, row.chain_b])
            yield Entry(id=str(row.id), sequences=seqs, chain_ids=chain_ids)

    def gt_for(self, entry_id: str) -> dict:
        import torch

        rel = Path("data") / entry_id[:2] / f"{entry_id}.pt"
        sample = torch.load(self.root / rel, weights_only=False, map_location="cpu")
        contact_map = sample["contact_map"].numpy()  # (Na, Nb) int64; -1 for missing Cβ
        sequences = list(sample["sequences"])
        # contact = (Cβ-Cβ < 8 Å); -1 entries are NOT contacts (mark as False)
        contact_bool = (contact_map >= 0) & (contact_map < self.contact_threshold_bin)
        return {"contact_map": contact_bool, "sequences": sequences, "raw_bins": contact_map}

    def score(self, entry: Entry, contact_path: Path) -> dict[str, float]:
        d = np.load(contact_path)
        probs_full = np.asarray(d["probs"], dtype=np.float32)  # (L, L)
        gt = self.gt_for(entry.id)
        contact_gt_rect = gt["contact_map"]  # (Na, Nb) bool
        seqs = gt["sequences"]
        la, lb = len(seqs[0]), len(seqs[1])
        L = la + lb
        if probs_full.shape[0] != L or probs_full.shape[1] != L:
            return {"_error": f"pred shape {probs_full.shape} != ({L}, {L})"}
        if contact_gt_rect.shape != (la, lb):
            return {"_error": f"gt shape {contact_gt_rect.shape} != ({la}, {lb})"}

        # Average the two off-diagonal blocks for symmetric scoring
        # (most models produce symmetric output; this is robust either way).
        probs_ab = probs_full[:la, la:la + lb]
        probs_ba = probs_full[la:la + lb, :la].T
        probs_inter = 0.5 * (probs_ab + probs_ba)

        return pak_inter_chain_rect(probs_inter, contact_gt_rect)
