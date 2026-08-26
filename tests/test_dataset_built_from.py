"""`built_from` is an import recipe and must never become a data path.

The whole point of importing a dataset is that afterwards nothing can reach the source.
A row that still carries the source's location is one accidental `getattr` away from
undoing that, so these tests pin the separation rather than trusting it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ecstasy.datasets import base
from ecstasy.datasets.importer import SourceChanged, source_from_spec

REGISTRY = Path(base.__file__).resolve().parents[1] / "registry" / "datasets.yaml"


@pytest.fixture(scope="module")
def rows() -> dict:
    return yaml.safe_load(REGISTRY.read_text())


def test_no_scorable_row_resolves_into_the_source_tree(rows):
    """The regression this file exists for.

    Every scorable row used to name ``${MENTOS_ROOT}``, and when MENTOS PR #266 deleted
    those splits every one of them broke at once. A scorable row now lives under
    ``${DATA_ROOT}`` and nowhere else; only ``built_from`` may name a source.
    """
    offenders = []
    for name, row in rows.items():
        if name.startswith("_"):
            continue
        scorable = {k: v for k, v in row.items() if k != "built_from"}
        if "MENTOS_ROOT" in yaml.safe_dump(scorable):
            offenders.append(name)
    assert not offenders, (
        f"rows resolve into the MENTOS tree outside built_from: {offenders}. "
        f"A dataset ecstasy scores against must not be able to change because another "
        f"project rebuilt a split.")


def test_every_row_is_ecstasy_owned_and_declares_identity(rows):
    for name, row in rows.items():
        if name.startswith("_"):
            continue
        assert row["kind"] == "ecstasy", f"{name}: kind={row['kind']!r}"
        assert row.get("root"), f"{name}: no root"
        assert row.get("description"), f"{name}: no description"
        assert row.get("expected_entries"), f"{name}: no expected_entries"


def test_built_from_is_dropped_when_a_dataset_is_loaded():
    """Load-bearing: the loader must not even receive the recipe."""
    name = "recent_pp"
    assert base.dataset_source(name) is not None, "test needs a row with a recipe"
    ds = base.load_dataset(name)
    assert not hasattr(ds, "built_from")
    assert "MENTOS" not in yaml.safe_dump(
        {k: str(v) for k, v in ds.source_paths().items()})


def test_dataset_source_returns_the_resolved_recipe():
    spec = base.dataset_source("recent_pp")
    assert spec["kind"] == "mentos_square"
    assert "${" not in spec["index"], "recipe should come back with ${VAR} expanded"
    assert spec["split"] == "val", (
        "recent_pp's source index also holds 23,463 train rows; the split is what "
        "selects the 151")


def test_dataset_source_is_none_for_a_row_without_one(tmp_path, monkeypatch):
    reg = {"owned": {"kind": "ecstasy", "root": str(tmp_path), "description": "x",
                     "expected_entries": 1}}
    monkeypatch.setattr(base, "_registry", lambda: reg)
    assert base.dataset_source("owned") is None


def test_source_from_spec_rejects_a_changed_index(tmp_path):
    index = tmp_path / "index.parquet"
    index.write_bytes(b"not the recorded bytes")
    spec = {"kind": "mentos_square", "index": str(index), "gt_root": str(tmp_path),
            "split": "val", "index_sha256": "0" * 64}
    with pytest.raises(SourceChanged) as e:
        source_from_spec(spec, "somedataset")
    # The message has to say it is a different split, not a corrupted file — the two
    # call for opposite responses.
    assert "DIFFERENT split" in str(e.value)


def test_source_from_spec_accepts_the_recorded_index(tmp_path):
    from ecstasy.provenance import sha256_file

    index = tmp_path / "index.parquet"
    index.write_bytes(b"some bytes")
    spec = {"kind": "mentos_square", "index": str(index), "gt_root": str(tmp_path),
            "split": "val", "index_sha256": sha256_file(index)}
    src = source_from_spec(spec, "somedataset")
    assert Path(src.index) == index


def test_source_from_spec_says_which_dataset_when_the_source_is_absent(tmp_path):
    spec = {"kind": "mentos_square", "index": str(tmp_path / "gone.parquet"),
            "gt_root": str(tmp_path), "split": "val"}
    with pytest.raises(FileNotFoundError, match="recent_pp"):
        source_from_spec(spec, "recent_pp")
