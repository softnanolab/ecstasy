"""ecstasy's own ground-truth format: per-entry, pickle-free, self-describing.

Replaces unpickling a ``mentos.dataclasses.Sample``, which was the **only** reason a
scoring environment needed MENTOS installed. With this, scoring needs numpy and pandas.

Two deliberate choices:

* **Coordinates are stored; contact bins are derived on load.** Bins are a pure function
  of the backbone (see :mod:`ecstasy.structure.geometry`), so storing both would create
  two sources of truth that can disagree — and the disagreement would be invisible.
  Deriving costs a cdist over L residues, milliseconds even at L≈1000.
* **No pickle.** ``np.load`` with ``allow_pickle=False`` cannot execute code, cannot
  depend on a class being importable, and cannot break when that class moves. The old
  format failed on all three counts.

Layout, one file per entry so sharding and resume work unchanged and a single entry can
be repaired in isolation:

    gt/<id[:2]>/<id>.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecstasy.structure.geometry import bins_from_atom37, contacts_from_bins

#: Bumped when the on-disk layout changes in a way older readers cannot handle.
FORMAT_VERSION = 1

_ARRAYS = ("atom37_positions", "atom37_mask", "aatype", "asym_id", "residue_index")


def write_entry(path: Path, *, sequences, atom37_positions, atom37_mask, aatype,
                asym_id, residue_index, chain_ids=None, is_homodimer=None,
                source: str | None = None) -> Path:
    """Write one entry's ground truth."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": FORMAT_VERSION,
        "sequences": list(sequences),
        "chain_ids": list(chain_ids) if chain_ids is not None else None,
        "is_homodimer": None if is_homodimer is None else bool(is_homodimer),
        "source": source,
    }
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(meta)),
        atom37_positions=np.asarray(atom37_positions, dtype=np.float32),
        atom37_mask=np.asarray(atom37_mask).astype(bool),
        aatype=np.asarray(aatype).astype(np.int8),
        asym_id=np.asarray(asym_id).astype(np.int8),
        residue_index=np.asarray(residue_index).astype(np.int32),
    )
    return path


def read_entry(path: Path, contact_bin: int = 19) -> dict:
    """Read one entry, deriving contacts from the stored coordinates.

    Returns the shape the scoring path expects: ``contact_map``, ``valid``,
    ``sequences``, plus the atom37 bundle for structure metrics.
    """
    with np.load(path, allow_pickle=False) as d:
        missing = [k for k in _ARRAYS if k not in d]
        if missing:
            raise KeyError(f"{path}: missing {missing}; has {sorted(d.files)}")
        meta = json.loads(str(d["meta"]))
        version = meta.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path}: format version {version}, this ecstasy reads {FORMAT_VERSION}. "
                f"Re-import the dataset rather than reading it with a mismatched reader.")
        bundle = {k: np.asarray(d[k]) for k in _ARRAYS}

    bins, dist = bins_from_atom37(bundle["atom37_positions"], bundle["atom37_mask"])
    contact_map, valid = contacts_from_bins(bins, contact_bin)
    return {
        "contact_map": contact_map,
        "valid": valid,
        "bins": bins,
        "distance": dist,
        "sequences": meta["sequences"],
        "chain_ids": meta.get("chain_ids"),
        "is_homodimer": meta.get("is_homodimer"),
        "source": meta.get("source"),
        **bundle,
    }


def entry_path(gt_root: Path, entry_id: str) -> Path:
    """``<gt_root>/<id[:2]>/<id>.npz`` — mirrors the MENTOS layout so the two-level fan
    out keeps directory sizes sane on a shared filesystem."""
    return Path(gt_root) / entry_id[:2] / f"{entry_id}.npz"
