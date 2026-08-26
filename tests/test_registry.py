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
    for n in ("recent_pp", "foldbench_pp", "foldbench_abag", "foldbench"):
        assert n in names


def test_dataset_loads_and_resolves_paths():
    d = load_dataset("recent_pp")
    assert d.name == "recent_pp"
    assert "${" not in str(d.index) and str(d.index).endswith("recent_pp/index.parquet")
    assert d.contact_bin == 19


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        load_dataset("does_not_exist")


def test_models_registered_with_presets():
    assert set(model_names()) == {"boltz2", "boltz2_nomsa", "esmfold", "mentos",
                                  "colabfold", "msa_pairformer", "esm2",
                                  "plmgraph_inter", "deepinteract", "minifold"}
    assert presets_for("boltz2") == ["fast", "full", "r0", "r1", "r3", "r5"]
    # esm2 sweeps model size (no recycles); presets are the fair-esm size tiers.
    assert presets_for("esm2") == ["t12_35M", "t30_150M", "t33_650M", "t36_3B", "t6_8M"]


def test_esm2_default_preset_and_params():
    m = load_model("esm2")
    assert m.preset == "t33_650M" and m.variant == "t33_650M"
    assert m.params["model_name"] == "esm2_t33_650M_UR50D"
    assert m.params["chain_linker_length"] == 25
    assert m.msa == "none" and not m.needs_msa


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


def test_msa_backends_registered():
    from ecstasy.msa.backends import BACKENDS
    # boltz_csv (Boltz-2) + complex (MSA-Pairformer local, default) + complex_api (API fallback)
    assert set(BACKENDS) == {"boltz_csv", "complex", "complex_api"}
    for b in BACKENDS.values():
        assert all(hasattr(b, fn) for fn in ("prepare", "submit", "ingest"))


def test_experiment_expands_matrix():
    m = {"name": "t", "datasets": ["recent_pp", "foldbench_pp"],
         "runs": [{"model": "boltz2", "preset": "full"},
                  {"model": "boltz2", "preset": "full", "set": {"recycling_steps": 5}}]}
    runs = experiment.expand(m)
    assert len(runs) == 4                       # 2 datasets × 2 run specs
    variants = {(r.dataset.name, r.model.variant) for r in runs}
    assert ("recent_pp", "full") in variants
    assert any(v.startswith("full+") for _, v in variants)
