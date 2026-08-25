"""The two gates that let ecstasy own its ground truth.

Gate 1 — **geometry**: recompute the Cβ-Cβ distance bins from the atom37 coordinates
already inside each MENTOS Sample and require them to equal the Sample's own contact_map.
This isolates the convention (virtual Cβ, AF2 edges, -1 for undefined) from the separate
problem of parsing mmCIF. If it does not hit 100%, ecstasy does not understand the
convention and no amount of correct structure parsing would rescue it.

Gate 2 — **the pickle-free path**: score the same predictions through the MENTOS loader
and the imported ecstasy loader and require identical numbers, while the ecstasy loader
imports neither mentos nor torch.

Anything short of exact here means the derived ground truth is a DIFFERENT ground truth
and must be versioned separately rather than mixed — see DESIGN.md D8. A 99% match is the
dangerous outcome: it looks fine while shifting a handful of published numbers.

Skipped when the MENTOS GT tree is not present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecstasy.structure.geometry import bins_from_atom37, contacts_from_bins

MENTOS_ROOT = Path("/rds/general/user/ha1822/ephemeral/MENTOS/DATA")
GT_ROOT = MENTOS_ROOT / "pdb" / "processed" / "data"
INDEX = MENTOS_ROOT / "pdb" / "processed" / "splits" / "val" / "index.parquet"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not GT_ROOT.exists(), reason="MENTOS GT tree not present"),
]


@pytest.fixture(scope="module")
def samples():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mentos", reason="reading the legacy .pt format needs mentos")
    out = []
    for p in sorted(GT_ROOT.rglob("*.pt")):
        s = torch.load(p, weights_only=False, map_location="cpu")
        if s.atom37_positions is not None and s.contact_map is not None:
            out.append((p.stem, s))
    if not out:
        pytest.skip("no full-atom samples found")
    return out


def test_geometry_reproduces_every_mentos_contact_map(samples):
    """GATE 1. Must be exact on all of them — see the module docstring."""
    mismatched = []
    for pid, s in samples:
        got, _ = bins_from_atom37(s.atom37_positions.numpy(), s.atom37_mask.numpy())
        ref = s.contact_map.numpy()
        if got.shape != ref.shape or not np.array_equal(got, ref):
            mismatched.append(pid)
    assert not mismatched, (
        f"{len(mismatched)}/{len(samples)} contact maps differ (e.g. {mismatched[:5]}). "
        f"The derived GT is therefore a DIFFERENT ground truth: version it separately "
        f"and do not average across the two.")


def test_thresholded_contacts_also_match(samples):
    """The bins feed a threshold; that is the number scoring actually consumes."""
    for pid, s in samples:
        bins, _ = bins_from_atom37(s.atom37_positions.numpy(), s.atom37_mask.numpy())
        got, _ = contacts_from_bins(bins, 19)
        ref_bins = s.contact_map.numpy()
        ref = (ref_bins >= 0) & (ref_bins < 19)
        assert np.array_equal(got, ref), pid


def test_undefined_pairs_are_preserved_exactly(samples):
    """Getting validity wrong changes the candidate pool, and so P@K, invisibly."""
    for pid, s in samples:
        bins, _ = bins_from_atom37(s.atom37_positions.numpy(), s.atom37_mask.numpy())
        assert np.array_equal(bins < 0, s.contact_map.numpy() < 0), pid


@pytest.mark.skipif(not INDEX.exists(), reason="val split index not present")
def test_imported_dataset_scores_identically_without_mentos(tmp_path):
    """GATE 2. Same predictions, both loaders, exact agreement — and no mentos/torch."""
    pytest.importorskip("torch")
    pytest.importorskip("mentos")
    from ecstasy.datasets.ecstasy_native import EcstasyDataset
    from ecstasy.datasets.importer import import_from_mentos
    from ecstasy.datasets.mentos import MentosSquareDataset

    preds = Path("/rds/general/user/ha1822/home/ecstasy/data/runs/mentos_val151/"
                 "minifold/full/predictions")
    if not preds.exists():
        pytest.skip("no real predictions available to score")

    src = MentosSquareDataset(
        name="val151", index=str(INDEX), gt_root=str(GT_ROOT), split="val",
        contact_bin=19, version=1, description="val", expected_entries=151)
    dest = tmp_path / "val151"
    report = import_from_mentos(src, dest, limit=25)
    assert report.n_written or report.n_already_present

    new = EcstasyDataset(name="val151", root=dest, split="val")
    metrics = ["AUC", "P@K", "P@K/2", "P@K(tol=2)"]
    entries = {e.id: e for e in src.entries()}

    compared = 0
    for d in sorted(preds.glob("*")):
        cp = d / "contact.npz"
        if not cp.exists() or d.name not in entries or not new.has_gt(d.name):
            continue
        a = src.score(entries[d.name], cp, metrics=metrics)
        b = new.score(entries[d.name], cp, metrics=metrics)
        if "_skipped" in a or "_error" in a:
            continue
        compared += 1
        for k in metrics:
            assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), (d.name, k)
    assert compared, "no targets were actually compared"


def test_the_ecstasy_loader_does_not_depend_on_mentos_or_torch():
    """The decoupling claim, asserted rather than assumed."""
    from ecstasy.datasets import ecstasy_native, store
    text = Path(ecstasy_native.__file__).read_text() + Path(store.__file__).read_text()
    assert "import mentos" not in text and "from mentos" not in text
    assert "import torch" not in text
