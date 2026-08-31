"""`contact_cutoff_bin` is 17 for MiniFold, and a comment cannot enforce that.

Bins are EDGES, and the models bin differently. Every other row here thresholds against
ecstasy's grid, ``linspace(2.3125, 21.6875, 63)``, where bin 19 means 7.9375 Å. MiniFold's
checkpoint has ``no_bins=64, max_dist=25``, so its boundaries are ``linspace(2, 25, 63)``
and the SAME index means a different distance — bin 19 there is 8.68 Å, 0.74 Å looser.

A looser threshold admits more true contacts, so copying 19 across would quietly inflate
MiniFold's P@K against every other model. Nothing about the run would look wrong.

These tests derive the right index from each model's own grid rather than asserting a
magic number, so they explain themselves and fail loudly if anyone "harmonises" the two.

The caveat no bin index can fix, restated because it belongs beside this: MiniFold's
distogram is **CA-CA** while ecstasy's ground truth is **Cb-Cb**. That is a real
remaining incomparability and must be stated wherever MiniFold's P@K is published.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import yaml

from ecstasy.models import load_model, model_names
from ecstasy.structure.geometry import DISTANCE_BIN_EDGES

MODELS_YAML = Path(__import__("ecstasy").__file__).parent / "registry" / "models.yaml"


def rows() -> dict:
    return yaml.safe_load(MODELS_YAML.read_text())

#: MiniFold's checkpoint config: no_bins=64, max_dist=25 -> 63 interior edges.
MINIFOLD_EDGES = np.linspace(2, 25, 63)

#: `probs[..., :k].sum(-1)` is P(d <= edges[k-1]), so bin k thresholds at edges[k-1].
def threshold_of(edges: np.ndarray, bin_index: int) -> float:
    return float(edges[bin_index - 1])


def test_minifold_presets_all_use_bin_17():
    presets = rows()["minifold"]["presets"]
    assert presets, "minifold row has no presets"
    for name, preset in presets.items():
        assert preset["contact_cutoff_bin"] == 17, (
            f"preset {name!r} uses contact_cutoff_bin="
            f"{preset['contact_cutoff_bin']}, not 17. Bin 19 on MiniFold's grid is "
            f"{threshold_of(MINIFOLD_EDGES, 19):.4f} Å, not the 7.9375 Å every other "
            f"row means by it — that inflates P@K rather than harmonising anything.")


def test_bin_17_on_minifolds_grid_matches_bin_19_on_ecstasys():
    """The derivation, not the number: 17 is what makes the two thresholds the same."""
    ecstasy_A = threshold_of(DISTANCE_BIN_EDGES, 19)
    minifold_A = threshold_of(MINIFOLD_EDGES, 17)
    assert ecstasy_A == pytest.approx(7.9375, abs=1e-4)
    assert minifold_A == pytest.approx(7.9355, abs=1e-3)
    assert abs(minifold_A - ecstasy_A) < 0.01, (
        f"bin 17 on MiniFold's grid is {minifold_A:.4f} Å against ecstasy's "
        f"{ecstasy_A:.4f} Å — no longer the matching index, so the constant must be "
        f"re-derived rather than kept.")


def test_bin_19_on_minifolds_grid_would_be_much_too_loose():
    """The failure that would otherwise be silent, made explicit."""
    wrong = threshold_of(MINIFOLD_EDGES, 19)
    right = threshold_of(DISTANCE_BIN_EDGES, 19)
    assert wrong == pytest.approx(8.6774, abs=1e-3)
    assert wrong - right > 0.7, (
        "bin 19 on MiniFold's grid should be ~0.74 Å looser than ecstasy's bin 19; if "
        "this no longer holds, one of the two grids changed and every published "
        "MiniFold P@K needs rechecking.")


def test_the_resolved_params_carry_bin_17():
    """Not just the file: what a run actually receives, after preset resolution."""
    for preset in ("full", "glinker32"):
        params = load_model("minifold", preset=preset).params
        assert params["contact_cutoff_bin"] == 17, f"{preset}: {params}"


def test_no_other_model_row_borrows_minifolds_bin():
    """17 is right for MiniFold *because of its grid*, so it must not spread."""
    for name in model_names():
        if name == "minifold":
            continue
        for preset_name, preset in (rows()[name].get("presets") or {}).items():
            bin_ = preset.get("contact_cutoff_bin")
            assert bin_ != 17, (
                f"{name}/{preset_name} uses contact_cutoff_bin=17, which is MiniFold's "
                f"value and is correct only for MiniFold's linspace(2, 25, 63) grid.")
