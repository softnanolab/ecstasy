"""MENTOS square-GT dataset loader (seq_id_30 + the four deleaked val splits).

All MENTOS PDB-processed splits share one format: an ``index.parquet`` with a
``split`` column and a ``sequences`` array per row, and per-entry ground truth
at ``<gt_root>/<id[:2]>/<id>.pt`` holding a *square* (L, L) binned Cβ-Cβ distance
map (-1 marks unresolved Cβ). One class serves every such split; the split is
chosen by the registry row, not by subclassing.

Newer GT ``.pt`` files also carry full-atom coordinates (``atom37_positions`` &c),
which is what lets this loader render natives and score structures with DockQ. Older
distogram-only samples have them as ``None``; those entries skip structure scoring and
still score contacts normally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ecstasy.config import settings
from ecstasy.datasets.base import Dataset, Entry
from ecstasy.metrics.contact import pak_inter_chain
from ecstasy.metrics.structure import (
    monomer_metrics,
    random_placement_null,
    run_dockq,
)
from ecstasy.structure.pdb import (
    load_structure_npz,
    render_structure_npz,
    write_atom37_pdb,
)


class MentosSquareDataset(Dataset):
    kind = "mentos_square"
    has_structure_gt = True

    def __init__(self, name: str, index: str, gt_root: str, split: str = "val",
                 contact_bin: int = 5, swap_chains: bool = False):
        super().__init__(name)
        self.index = Path(index)
        self.gt_root = Path(gt_root)
        self.split = split
        self.contact_bin = int(contact_bin)
        # swap_chains: chain-order-permutation experiment. Reverse each dimer's chain
        # order (A,B)->(B,A) at input AND reindex the square GT to match, so the model
        # is scored on the same interface seen in flipped order. Monomers pass through.
        self.swap_chains = bool(swap_chains)

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
        sample = self._sample(entry_id)
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
        return {"contact_map": contact_map, "valid": valid, "sequences": seqs}

    def score(self, entry: Entry, contact_path: Path) -> dict[str, float]:
        d = np.load(contact_path)
        probs = np.asarray(d["probs"], dtype=np.float32)
        gt = self.gt_for(entry.id)
        contact_gt = gt["contact_map"]
        valid = gt["valid"]
        seqs = gt["sequences"]
        if len(seqs) != 2:
            return {"_skipped": "non-dimer"}
        la, lb = len(seqs[0]), len(seqs[1])
        L = la + lb
        if probs.shape[0] != L or contact_gt.shape[0] != L:
            return {"_error": f"shape mismatch: probs={probs.shape}, gt={contact_gt.shape}, L={L}"}
        chain_ids = np.array([0] * la + [1] * lb)
        return pak_inter_chain(probs, contact_gt, chain_ids, valid=valid)

    # --- structure path -------------------------------------------------------------

    def _sample(self, entry_id: str):
        """Load a GT sample, memoised on the last id.

        GT ``.pt`` files pickle a ``mentos.dataclasses.Sample``. The scoring env ships
        the ``mentos`` package (.venv-boltz / .venv-mentos both have it editable), so
        ``torch.load`` resolves the class natively — no rename shim. The torch import is
        lazy: only a scoring env reaches here, never the torch-less orchestrator. See
        the mentos_package_and_venvs memory.

        Memoised on one id because scoring a single entry touches its sample up to four
        times (contact GT, native PDB, native bundle, homodimer flag). The samples carry
        full-atom coordinates — megabytes each — and entries are visited in order, so one
        slot removes the repeat deserialisation without holding the split in memory.
        """
        import torch

        if getattr(self, "_sample_id", None) != entry_id:
            self._sample_cache = torch.load(
                self.gt_root / entry_id[:2] / f"{entry_id}.pt",
                weights_only=False, map_location="cpu")
            self._sample_id = entry_id
        return self._sample_cache

    def native_bundle(self, entry_id: str) -> dict[str, np.ndarray] | None:
        """Full-atom GT as an atom37 bundle, or ``None`` on a distogram-only sample."""
        s = self._sample(entry_id)
        if any(getattr(s, f, None) is None for f in
               ("aatype", "atom37_positions", "atom37_mask", "residue_index", "asym_id")):
            return None
        b = {
            "atom37_positions": s.atom37_positions.numpy(),
            "atom37_mask": s.atom37_mask.numpy(),
            "aatype": s.aatype.numpy(),
            "asym_id": s.asym_id.numpy(),
            "residue_index": s.residue_index.numpy(),
        }
        if self.swap_chains and len(s.sequences) == 2:
            # Match `gt_for`: the model saw (B, A), so the native must be reordered to
            # (B, A) too and its asym_id relabelled, or DockQ compares A-to-B.
            asym = b["asym_id"]
            perm = np.r_[np.flatnonzero(asym == 1), np.flatnonzero(asym == 0)]
            b = {k: v[perm] for k, v in b.items()}
            b["asym_id"] = 1 - b["asym_id"]
        return b

    def native_pdb(self, entry_id: str, cache_dir: Path | None = None) -> Path | None:
        """Render (and cache) the GT native PDB. ``None`` when there is no full-atom GT.

        Written by ``ecstasy.structure.pdb.write_atom37_pdb``, the same writer that
        renders predictions — and verified byte-identical to the natives the MENTOS
        DockQ checkpoint series was scored against.
        """
        cache_dir = Path(cache_dir) if cache_dir is not None else (
            settings().natives_root / self.name)
        out = cache_dir / f"{entry_id}_native.pdb"
        if out.exists():
            return out
        b = self.native_bundle(entry_id)
        if b is None:
            return None
        return write_atom37_pdb(
            out,
            positions=b["atom37_positions"], atom_mask=b["atom37_mask"],
            aatype=b["aatype"], asym_id=b["asym_id"],
            residue_index=b["residue_index"],
        )

    def score_structure(self, entry: Entry, structure_path: Path,
                        work_dir: Path | None = None, null_draws: int = 0,
                        dockq_bin: str | None = None,
                        natives_dir: Path | None = None) -> dict[str, float]:
        """DockQ + per-chain monomer quality for one predicted structure.

        ``structure_path`` is a runner's ``structure.npz`` (atom37 bundle). It is
        rendered to PDB with the native-identical writer, then scored with the fixed
        no-flag DockQ invocation. ``null_draws > 0`` additionally computes the
        random-placement floor for this target.
        """
        work_dir = Path(work_dir) if work_dir is not None else Path(structure_path).parent
        native = self.native_pdb(entry.id, cache_dir=natives_dir)
        if native is None:
            return {"_skipped": "no full-atom ground truth for this entry"}
        try:
            pred_bundle = load_structure_npz(structure_path)
        except (KeyError, OSError, ValueError) as e:
            return {"_error": f"unreadable structure.npz: {e}"}
        nat_bundle = self.native_bundle(entry.id)
        if nat_bundle is None:                       # pragma: no cover - native_pdb agrees
            return {"_skipped": "no full-atom ground truth for this entry"}
        if pred_bundle["asym_id"].shape != nat_bundle["asym_id"].shape:
            return {"_error": f"length mismatch: pred={pred_bundle['asym_id'].shape[0]} "
                              f"native={nat_bundle['asym_id'].shape[0]}"}

        pred_pdb = render_structure_npz(structure_path, work_dir / f"{entry.id}_pred.pdb")
        out: dict[str, float] = {}
        scores = run_dockq(pred_pdb, native, dockq_bin=dockq_bin)
        if scores is None:
            return {"_error": "DockQ produced no scores — is the `DockQ` CLI installed "
                              "(pip install DockQ)?"}
        out.update(scores)
        out.update({k: v for k, v in monomer_metrics(pred_bundle, nat_bundle).items()
                    if k != "per_chain"})
        # 40 of the 151 val dimers are true homodimers, where the linker hack feeds ESM2
        # one sequence duplicated around a poly-G run — something it has never seen. A
        # collapsed or domain-swapped prediction there is a property of the hack, not of
        # the model's docking, so the split must be reportable downstream.
        homo = getattr(self._sample(entry.id), "is_homodimer", None)
        if homo is not None:
            out["is_homodimer"] = float(bool(homo))
        if null_draws:
            null = random_placement_null(pred_pdb, native, entry.id,
                                         n_draws=null_draws, work_dir=work_dir,
                                         dockq_bin=dockq_bin)
            out["null_DockQ_mean"] = null["mean"]
            out["null_DockQ_max"] = null["max"]
        return out
