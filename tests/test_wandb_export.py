"""Tests for the wandb projection.

Deliberately exercise the *pure* projection and ``export(dry_run=True)`` only, so the
whole file runs in an environment with neither wandb nor a network — which is the only
environment the test gate has. A test that skips in the one place it could run is not
coverage, and this repo has been bitten by that three times.

The committed store is used as a fixture, but only for invariants that hold for *any*
row, so adding a published run cannot break these. The number-specific assertions use
synthetic rows.
"""
from __future__ import annotations

import copy

import pytest

from ecstasy import results, wandb_export


def _row(**over):
    """A minimal but realistically-shaped published row."""
    row = {
        "schema_version": 1,
        "published_utc": "2026-08-26T19:26:58+00:00",
        "dataset": {"name": "recent_pp", "version": 1, "expected_entries": 151},
        "model": {"name": "minifold", "variant": "full"},
        "n": {"evaluated": 151, "skipped": 0, "errors": 0},
        "coverage": {"complete": True, "fraction": 1.0},
        "metrics": {"names": ["AUC", "P@K"], "mean": {"P@K": 0.25, "AUC": 0.33},
                    "median": {"P@K": 0.14, "AUC": 0.17}},
        "structure": None,
        "flops": None,
        "fingerprints": {"prediction": "d8e0adcf3a", "scoring": "057f057954"},
        "provenance": {"ecstasy_sha": "5eb9164", "ecstasy_dirty": False,
                       "model_code": {}, "weights": {}, "host": "cx3-3-16"},
        "run_dir": "recent_pp/minifold/full",
    }
    row.update(over)
    return row


# --- identity ---------------------------------------------------------------

def test_the_same_row_always_gets_the_same_run_id():
    """Export must converge on re-run, not create a second copy of every result."""
    assert wandb_export.run_id(_row()) == wandb_export.run_id(_row())


def test_rescoring_produces_a_new_run_rather_than_overwriting_the_old_one():
    """A metric fix appends a JSONL row; it must likewise appear as a NEW wandb run,
    or the history of a number moving is destroyed at the point it becomes visible."""
    before = wandb_export.run_id(_row())
    after = wandb_export.run_id(
        _row(fingerprints={"prediction": "d8e0adcf3a", "scoring": "DIFFERENT"}))
    assert before != after


def test_run_id_is_a_legal_wandb_id():
    rid = wandb_export.run_id(_row())
    assert rid.isalnum() and rid.islower() and len(rid) == 16


def test_different_models_on_one_dataset_do_not_collide():
    a = wandb_export.run_id(_row(model={"name": "minifold", "variant": "full"}))
    b = wandb_export.run_id(_row(model={"name": "boltz2", "variant": "full"}))
    assert a != b


# --- the unmeasured-is-not-zero rule ---------------------------------------

def test_absent_flops_are_omitted_and_flagged_not_written_as_zero():
    """The guard that keeps a fabricated number off the compute axis.

    ESMFold2's FLOPs are currently refused outright because its ESMC-6B backbone is
    uncounted. Exporting 0 would put the strongest model at the origin of the very axis
    the FLOPs benchmark plan is built around, and it would look measured.
    """
    s = wandb_export.payload(_row(flops=None))["summary"]
    assert s["flops/measured"] is False
    assert not any(k.startswith("flops/") and k != "flops/measured" for k in s)


def test_present_flops_are_exported_with_a_tflops_convenience():
    s = wandb_export.payload(_row(flops={"mean_flops": 2.5e12,
                                         "median_flops": 2.0e12}))["summary"]
    assert s["flops/measured"] is True
    assert s["flops/mean_flops"] == 2.5e12
    assert s["flops/mean_tflops"] == pytest.approx(2.5)
    assert s["flops/median_tflops"] == pytest.approx(2.0)


def test_a_contact_only_run_reports_structure_as_unscored_not_missing():
    """"Not scored" and "scored badly" must not look the same. Only minifold emits
    structure.npz today, so most rows will legitimately have no structure block."""
    s = wandb_export.payload(_row(structure=None))["summary"]
    assert s["structure/scored"] is False
    assert not any(k.startswith("structure/") and k != "structure/scored" for k in s)


# --- caveats travel with the numbers ---------------------------------------

def test_a_dirty_model_tree_is_tagged():
    """The leaderboard marks this with a dagger; a wandb reader gets the same warning."""
    row = _row(provenance={**_row()["provenance"],
                           "model_code": {"minifold": {"dirty": True,
                                                       "dirty_files": ["x.py"]}}})
    p = wandb_export.payload(row)
    assert "dirty-model-tree" in p["tags"]
    assert p["config"]["dirty_model_trees"] == ["minifold"]


def test_a_dirty_ecstasy_tree_is_tagged():
    row = _row(provenance={**_row()["provenance"], "ecstasy_dirty": True})
    assert "dirty-ecstasy-tree" in wandb_export.payload(row)["tags"]


def test_partial_coverage_is_tagged():
    row = _row(coverage={"complete": False, "fraction": 0.5})
    assert "partial-coverage" in wandb_export.payload(row)["tags"]


def test_a_clean_complete_run_carries_no_caveat_tags():
    t = wandb_export.payload(_row())["tags"]
    assert not {"dirty-model-tree", "dirty-ecstasy-tree", "partial-coverage"} & set(t)


def test_tags_carry_the_grouping_axes():
    t = wandb_export.payload(_row())["tags"]
    assert {"dataset:recent_pp", "model:minifold", "variant:full"} <= set(t)


# --- structure projection --------------------------------------------------

def test_the_homo_hetero_split_survives_the_projection():
    """On recent_pp the two sides differ on DockQ *and* TM_min: heterodimer failure is
    partly a folding failure. Exporting only the pooled number loses that."""
    struct = {
        "n": 151, "mean": {"DockQ": 0.24}, "median": {"DockQ": 0.10},
        "acceptable_fraction": 0.39, "medium_fraction": 0.26, "high_fraction": 0.04,
        "homodimer_flag": {"n": 129, "mean": {"DockQ": 0.25, "TM_min": 0.76},
                           "median": {"DockQ": 0.117, "TM_min": 0.901}},
        "heterodimer_flag": {"n": 22, "mean": {"DockQ": 0.19, "TM_min": 0.50},
                             "median": {"DockQ": 0.048, "TM_min": 0.417}},
    }
    s = wandb_export.payload(_row(structure=struct))["summary"]
    assert s["structure/scored"] is True
    assert s["structure/homodimer/n"] == 129
    assert s["structure/heterodimer/median/TM_min"] == pytest.approx(0.417)
    assert s["structure/homodimer/median/TM_min"] == pytest.approx(0.901)
    assert s["structure/acceptable_fraction"] == pytest.approx(0.39)


def test_contact_means_and_medians_are_kept_apart():
    s = wandb_export.payload(_row())["summary"]
    assert s["contact/mean/P@K"] == pytest.approx(0.25)
    assert s["contact/median/P@K"] == pytest.approx(0.14)


# --- export shell ----------------------------------------------------------

def test_dry_run_needs_neither_wandb_nor_a_network():
    out = wandb_export.export([_row()], dry_run=True)
    assert len(out) == 1 and out[0]["id"] == wandb_export.run_id(_row())


def test_export_does_not_mutate_the_rows_it_is_given():
    """The store is the source of truth; a derived view must not touch it."""
    row = _row()
    before = copy.deepcopy(row)
    wandb_export.export([row], dry_run=True)
    assert row == before


# --- every committed row projects cleanly ----------------------------------

def test_every_committed_row_projects_without_error():
    """Property over the real store rather than a fixture copy of it, so the projection
    is exercised against whatever has actually been published."""
    rows = results.load()
    if not rows:
        pytest.skip("nothing published yet")
    for row in rows:
        p = wandb_export.payload(row)
        assert p["id"] and p["name"].count("/") == 2
        assert p["config"]["dataset"] and p["config"]["model"]
        for key, val in p["summary"].items():
            assert isinstance(val, (int, float, bool)) or val is None, (key, val)


def test_no_committed_row_exports_a_zero_flops_figure():
    """Guards the whole store against the fabricated-cost failure, not just one row."""
    rows = results.load()
    if not rows:
        pytest.skip("nothing published yet")
    for row in rows:
        s = wandb_export.payload(row)["summary"]
        if not s["flops/measured"]:
            assert "flops/mean_flops" not in s
        else:
            assert s["flops/mean_flops"] > 0


def test_committed_rows_have_unique_run_ids():
    """Two published rows collapsing to one wandb run would silently hide a result."""
    rows = results.load()
    if len(rows) < 2:
        pytest.skip("need at least two published rows")
    ids = [wandb_export.run_id(r) for r in rows]
    assert len(set(ids)) == len(ids)
