"""Unit tests for the declarative datasets/models layer (no GPU, no model deps)."""
from __future__ import annotations

import pytest

from ecstasy.config import resolve, settings
from ecstasy.datasets import dataset_names, load_dataset
from ecstasy.models import load_model, model_names, presets_for
from ecstasy.msa import store
from ecstasy import experiment


def test_var_resolution():
    s = settings()
    assert resolve("${MENTOS_ROOT}/x") == f"{s.MENTOS_ROOT}/x"
    assert resolve("${UNKNOWN}/x") == "${UNKNOWN}/x"   # unknown left intact
    assert resolve({"a": "${DATA_ROOT}"})["a"] == str(s.DATA_ROOT)


def test_datasets_registered():
    names = dataset_names()
    for n in ("mentos_seqid30", "val_seq_chain", "val_seq_pair",
              "val_pinder_chain", "val_pinder_pair"):
        assert n in names


def test_dataset_loads_and_resolves_paths():
    d = load_dataset("val_pinder_pair")
    assert d.name == "val_pinder_pair"
    assert "${" not in str(d.index) and str(d.index).endswith("val_pinder_pair/index.parquet")
    assert d.contact_bin == 5


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        load_dataset("does_not_exist")


def test_models_registered_with_presets():
    assert set(model_names()) == {"boltz2", "esmfold", "mentos", "colabfold", "msa_pairformer"}
    assert presets_for("boltz2") == ["fast", "full"]


def test_default_preset_and_variant():
    m = load_model("boltz2")
    assert m.preset == "full" and m.variant == "full"
    assert m.params["recycling_steps"] == 3
    assert m.infra["num_workers"] == 0          # infra separate from params
    assert m.msa == "boltz_csv" and m.needs_msa


def test_override_changes_variant_deterministically():
    a = load_model("boltz2", overrides={"recycling_steps": 5})
    b = load_model("boltz2", overrides={"recycling_steps": 5})
    assert a.variant == b.variant
    assert a.variant.startswith("full+") and a.variant != "full"
    assert a.params["recycling_steps"] == 5


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        load_model("boltz2", preset="nope")


def test_msa_store_hashing_is_stable():
    assert store.chain_hash("ACDE") == store.chain_hash("ACDE")
    assert len(store.chain_hash("ACDE")) == 16
    assert store.pair_hash(["AC", "DE"]) != store.pair_hash(["DE", "AC"])  # order matters


def test_store_lookup_arms():
    from ecstasy.datasets.base import Entry
    e = Entry(id="x", sequences=("ACDE", "FGHI"), chain_ids=("A", "B"))
    assert store.lookup(e, "none") is None
    # store is empty under the test DATA_ROOT, so all flavours miss -> None
    assert store.lookup(e, "per_chain") is None
    assert store.lookup(e, "complex") is None
    assert store.lookup(e, "boltz_csv") is None


def test_variant_distinguishes_different_overrides():
    a = load_model("boltz2", overrides={"recycling_steps": 5})
    b = load_model("boltz2", overrides={"recycling_steps": 6})
    assert a.variant != b.variant


def test_split_complex_hetero_homo_and_fallback():
    from ecstasy.msa.generate import _split_complex
    # heterodimer: explicit per-chain lengths
    assert _split_complex("AAABB", "#3,2\t1,1") == ["AAA", "BB"]
    # homodimer: one unique length, copies=2 -> must expand, not collapse
    homo = _split_complex("AAAAAA", "#3\t2")
    assert homo == ["AAA", "AAA"]
    # the regression: hash matches what prepare wrote from entry.sequences
    assert store.pair_hash(homo) == store.pair_hash(["AAA", "AAA"])
    # malformed header (lengths don't sum) and no header -> single chain
    assert _split_complex("ABCDE", "#9,9\t1,1") == ["ABCDE"]
    assert _split_complex("ABC", None) == ["ABC"]


def test_experiment_expands_matrix():
    m = {"name": "t", "datasets": ["val_seq_chain", "val_pinder_pair"],
         "runs": [{"model": "boltz2", "preset": "full"},
                  {"model": "boltz2", "preset": "full", "set": {"recycling_steps": 5}}]}
    runs = experiment.expand(m)
    assert len(runs) == 4                       # 2 datasets × 2 run specs
    variants = {(r.dataset.name, r.model.variant) for r in runs}
    assert ("val_seq_chain", "full") in variants
    assert any(v.startswith("full+") for _, v in variants)
