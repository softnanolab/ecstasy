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
              "val_pinder_chain", "val_pinder_pair", "mentos_val151"):
        assert n in names


def test_val151_is_the_dockq_split():
    """The 151-dimer set the MENTOS DockQ series was evaluated on. Verified by set
    equality against the natives in that evaluation's output dir — none of the deleaked
    splits substitutes for it, so it is its own row."""
    d = load_dataset("mentos_val151")
    assert str(d.index).endswith("splits/val/index.parquet")
    assert d.split == "val"
    assert d.contact_bin == 19              # inherits the MENTOS Cb-Cb threshold
    assert d.has_structure_gt


def test_dataset_loads_and_resolves_paths():
    d = load_dataset("val_pinder_pair")
    assert d.name == "val_pinder_pair"
    assert "${" not in str(d.index) and str(d.index).endswith("val_pinder_pair/index.parquet")
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


def test_minifold_cutoff_bin_is_17_not_19():
    """MiniFold bins over linspace(2, 25, 63); ESMFold over linspace(2.3125, 21.6875, 63).
    Bin 19 therefore means 8.68 A on MiniFold where it means 7.94 A on ESMFold. Bin 17
    reproduces 7.9355 A. This test exists to stop a well-meaning harmonisation to 19."""
    for preset in presets_for("minifold"):
        m = load_model("minifold", preset=preset)
        assert m.params["contact_cutoff_bin"] == 17, preset
        assert m.params["residue_index_offset"] == 512, preset

    esmfold_edges = _bin_edges(2.3125, 21.6875)
    minifold_edges = _bin_edges(2.0, 25.0)
    assert minifold_edges[17 - 1] == pytest.approx(esmfold_edges[19 - 1], abs=0.01)
    assert minifold_edges[19 - 1] > esmfold_edges[19 - 1] + 0.5   # the wrong choice


def _bin_edges(lo: float, hi: float, n_bounds: int = 63) -> list[float]:
    """`probs[..., :k].sum(-1)` is `P(d <= edges[k - 1])`."""
    step = (hi - lo) / (n_bounds - 1)
    return [lo + i * step for i in range(n_bounds)]


def test_minifold_shares_one_checkpoint_path_across_presets():
    """The YAML anchor is load-bearing: a checkpoint path duplicated per preset drifts."""
    paths = {load_model("minifold", preset=p).params["checkpoint"]
             for p in presets_for("minifold")}
    assert len(paths) == 1
    assert "${" not in paths.pop()


def test_minifold_default_preset_matches_the_settled_chain_break():
    m = load_model("minifold")
    assert m.preset == "full" and m.variant == "full"
    assert m.params["chain_linker_length"] == 25       # MiniFold's / esmfold `full`'s
    assert m.params["num_recycles"] == 3               # the checkpoint's trained setting
    assert m.msa == "none" and not m.needs_msa
    assert m.runner.name == "minifold_runner.py"


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
    m = {"name": "t", "datasets": ["val_seq_chain", "val_pinder_pair"],
         "runs": [{"model": "boltz2", "preset": "full"},
                  {"model": "boltz2", "preset": "full", "set": {"recycling_steps": 5}}]}
    runs = experiment.expand(m)
    assert len(runs) == 4                       # 2 datasets × 2 run specs
    variants = {(r.dataset.name, r.model.variant) for r in runs}
    assert ("val_seq_chain", "full") in variants
    assert any(v.startswith("full+") for _, v in variants)
