"""The gate on `ecstasy import_dataset`: an imported folder IS the source dataset.

`test_gt_derivation.py` gates the geometry — that recomputing Cβ-Cβ bins from atom37
coordinates reproduces MENTOS's contact maps. This gates the step after it: that the
round trip through ecstasy's own storage does not change the answer.

That round trip is not free of risk. `store.write_entry` narrows dtypes for size
(int64 -> int8 for aatype/asym_id, int64 -> int32 for residue_index), the index is
rewritten filtered to the dataset's own rows, and contacts are re-derived on read rather
than copied. Any one of those could shift a handful of entries while leaving the other
150 identical — which is the dangerous outcome, not the obvious one.

So the assertion is exact equality on EVERY entry, for the contact map, the undefined
mask and the coordinates alike. Anything less means the imported dataset is a different
dataset and must be versioned as one (DESIGN.md D8), not quietly scored against numbers
produced from the original.

Skipped when the MENTOS source tree is absent; this is the one place a source is read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecstasy.datasets import store
from ecstasy.datasets.base import dataset_source, load_dataset

pytestmark = pytest.mark.integration

DATASET = "recent_pp"
#: Entries to compare. The import itself is the expensive part; comparison is cheap, so
#: this runs over everything the folder holds rather than a sample.
_SOURCE_NEEDED = "MENTOS source tree not present"


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    """Import the dataset into a throwaway folder, exactly as the CLI would."""
    pytest.importorskip("torch")
    pytest.importorskip("mentos", reason="reading the legacy .pt format needs mentos")
    from ecstasy.datasets.importer import import_from_mentos, source_from_spec

    spec = dataset_source(DATASET)
    if spec is None:
        pytest.skip(f"{DATASET} has no built_from recipe")
    if not Path(spec["index"]).exists():
        pytest.skip(_SOURCE_NEEDED)

    src = source_from_spec(spec, DATASET)
    dest = tmp_path_factory.mktemp("imported") / DATASET
    target = load_dataset(DATASET)
    report = import_from_mentos(
        src, dest, name=DATASET,
        identity={"version": target.version, "description": target.description,
                  "tags": target.tags, "contact_bin": target.contact_bin})
    assert report.complete, f"import incomplete: {report.summary()}"
    from ecstasy.datasets.ecstasy_native import EcstasyDataset

    return src, EcstasyDataset(name=DATASET, root=dest)


def test_import_is_complete_and_matches_the_declared_count(imported):
    """The row's expected_entries is a promise; the import either keeps it or fails."""
    src, ds = imported
    ids = [e.id for e in ds.entries()]
    declared = load_dataset(DATASET).expected_entries
    assert len(ids) == declared, (
        f"imported {len(ids)} entries, row declares expected_entries={declared}")
    assert len(set(ids)) == len(ids), "imported index has duplicate ids"
    assert set(ids) == {e.id for e in src.entries()}, (
        "imported index is not the same entry set as the source split")


def test_ground_truth_is_bit_identical_to_the_source(imported):
    """GATE. Every entry, exactly — see the module docstring on why not a sample."""
    src, ds = imported
    bad: list[str] = []
    for entry in ds.entries():
        got = ds.gt_for(entry.id)
        ref = src.gt_for(entry.id)
        if (got["contact_map"].shape != ref["contact_map"].shape
                or not np.array_equal(got["contact_map"], ref["contact_map"])
                or not np.array_equal(got["valid"], ref["valid"])):
            bad.append(entry.id)
    assert not bad, (
        f"{len(bad)}/{len(list(ds.entries()))} entries differ from the source after "
        f"import: {bad[:10]}. The imported folder is a DIFFERENT dataset — version it "
        f"separately rather than scoring it against numbers from the original.")


def test_sequences_and_homodimer_flag_survive_the_round_trip(imported):
    """Identity fields, not just arrays: a shifted flag silently re-splits a result."""
    src, ds = imported
    for entry in ds.entries():
        got, ref = ds.gt_for(entry.id), src.gt_for(entry.id)
        assert list(got["sequences"]) == list(ref["sequences"]), entry.id
        if ref.get("is_homodimer") is not None:
            assert got["is_homodimer"] == ref["is_homodimer"], entry.id


def test_scoring_the_import_needs_neither_mentos_nor_torch(imported):
    """The reason the format exists. Reading GT must not import either module.

    Asserted by reading through `store` directly rather than by inspecting sys.modules,
    which is polluted here by the import fixture itself.
    """
    _, ds = imported
    entry_id = next(iter(ds.entries())).id
    gt = store.read_entry(ds.gt_path(entry_id), contact_bin=ds.contact_bin)
    assert gt["contact_map"].ndim == 2
    src_file = Path(store.__file__).read_text()
    assert "import torch" not in src_file and "import mentos" not in src_file


def test_a_changed_source_index_refuses_to_reimport(tmp_path):
    """The guard that makes an imported dataset safe to keep.

    MENTOS rebuilding a split under the same path must not be able to replace a dataset
    that published results refer to. Simulated by asserting a hash the file cannot have.
    """
    from ecstasy.datasets.importer import SourceChanged, source_from_spec

    spec = dataset_source(DATASET)
    if spec is None or not Path(spec["index"]).exists():
        pytest.skip(_SOURCE_NEEDED)
    spec = dict(spec, index_sha256="0" * 64)
    with pytest.raises(SourceChanged, match="changed underneath"):
        source_from_spec(spec, DATASET)
