"""Structure metrics — the parts that do not need the DockQ binary.

The DockQ CLI is external and has its own correctness story; what is pinned here is
everything ecstasy wraps around it: the parsing contract copied from `mentos` (change it
and the numbers stop being comparable to the 23-checkpoint series), the geometry, and the
seeding of the placement floor. End-to-end behaviour against real ground truth, including
that seven registered metrics cost ONE subprocess, is covered by
tests/integration/test_structure_metrics.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecstasy.metrics import registry
from ecstasy.metrics.structure import (
    DOCKQ_BANDS,
    _DOCKQ_RE,
    ca_rmsd,
    dockq_bands,
    kabsch_superpose,
    per_chain_quality,
    random_rotation,
    stable_seed,
    tm_score,
)
from ecstasy.structure.pdb import ATOM_TYPES, CA_INDEX

# Real DockQ output: a banner and a legend precede the numbers, which is exactly why the
# parser uses .search() on loose patterns rather than reading fixed lines.
DOCKQ_STDOUT = """\
****************************************************************
*                       DockQ                                  *
*   Scoring function for protein-protein docking models        *
*   Definition of contact <5A (Fnat)                           *
*   Definition of interface <10A all heavy atoms (iRMSD)       *
****************************************************************
Model  : /tmp/21ie_pred.pdb
Native : /tmp/21ie_native.pdb
Fnat 0.234 12 correct of 51 native contacts
iRMSD 4.321
LRMSD 12.345
DockQ 0.318
"""


class TestDockqParsing:
    def test_all_four_components_are_recovered(self):
        got = {k: float(p.search(DOCKQ_STDOUT).group(1)) for k, p in _DOCKQ_RE.items()}
        assert got == {"DockQ": 0.318, "Fnat": 0.234, "iRMSD": 4.321, "LRMSD": 12.345}

    def test_dockq_pattern_skips_the_banner(self):
        """.search() must land on the score line, not the word DockQ in the header."""
        assert _DOCKQ_RE["DockQ"].search(DOCKQ_STDOUT).group(1) == "0.318"


class TestRegistration:
    def test_rmsd_metrics_are_marked_lower_is_better(self):
        """Ranking code must not need a hardcoded table of exceptions."""
        for name in ("iRMSD", "LRMSD", "CA_RMSD_mean"):
            assert registry.get(name).higher_is_better is False
        for name in ("DockQ", "Fnat", "TM_mean", "TM_min"):
            assert registry.get(name).higher_is_better is True

    def test_structure_metrics_are_a_distinct_kind(self):
        assert "DockQ" in registry.names("structure")
        assert "DockQ" not in registry.names("contact")

    def test_rmsd_is_reusable_as_asked(self):
        """P@K, P@K(tol=2), DockQ and RMSD all addressable from one registry."""
        for name in ("P@K", "P@K(tol=2)", "DockQ", "CA_RMSD_mean"):
            assert registry.get(name).description


class TestBands:
    def test_fractions_at_each_band(self):
        got = dockq_bands([0.10, 0.30, 0.55, 0.90])
        assert got["acceptable_fraction"] == 0.75
        assert got["medium_fraction"] == 0.50
        assert got["high_fraction"] == 0.25

    def test_thresholds_are_inclusive(self):
        got = dockq_bands(list(DOCKQ_BANDS.values()))
        assert got["acceptable_fraction"] == 1.0

    def test_empty_is_nan_not_zero(self):
        """Zero reads as 'nothing was acceptable'; nan reads as 'nothing scored'."""
        assert all(np.isnan(v) for v in dockq_bands([]).values())


class TestStableSeed:
    def test_is_stable_and_distinct(self):
        assert stable_seed("9zdi") == stable_seed("9zdi") == 1010912291
        assert stable_seed("9zdi") != stable_seed("21ie")

    def test_is_not_pythons_salted_hash(self):
        """hash() is salted per process unless PYTHONHASHSEED is set, so a hash-seeded
        floor drifts between runs — under the result it exists to anchor."""
        assert 0 <= stable_seed("9zdi") < 2 ** 32


class TestGeometry:
    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_random_rotation_is_proper(self, seed):
        r = random_rotation(np.random.default_rng(seed))
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(r) == pytest.approx(1.0)

    def test_kabsch_recovers_a_rigid_transform(self):
        rng = np.random.default_rng(0)
        target = rng.normal(size=(20, 3))
        moved = (random_rotation(rng) @ target.T).T + np.array([10.0, -3.0, 7.0])
        np.testing.assert_allclose(kabsch_superpose(moved, target), target, atol=1e-9)

    def test_kabsch_refuses_a_reflection(self):
        """A mirror-permitting fit would drive this chiral set to 0."""
        target = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert ca_rmsd(target * np.array([1.0, 1.0, -1.0]), target) > 0.1

    def test_tm_of_an_identical_trace_is_one(self):
        x = np.random.default_rng(2).normal(size=(60, 3)) * 5
        assert tm_score(x.copy(), x) == pytest.approx(1.0, abs=1e-9)

    def test_tm_falls_as_the_trace_degrades(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 3)) * 5
        near = tm_score(x + rng.normal(size=x.shape) * 0.3, x)
        far = tm_score(x + rng.normal(size=x.shape) * 6.0, x)
        assert 0.0 < far < near < 1.0

    def test_length_mismatch_is_nan_not_an_exception(self):
        assert np.isnan(tm_score(np.zeros((4, 3)), np.zeros((5, 3))))
        assert np.isnan(ca_rmsd(np.zeros((4, 3)), np.zeros((5, 3))))


def _bundle(chain_sizes=(4, 3), ca_present=True):
    n = sum(chain_sizes)
    pos = np.zeros((n, len(ATOM_TYPES), 3), dtype=np.float32)
    pos[:, CA_INDEX] = np.arange(n)[:, None] * np.array([1.0, 0.0, 0.0])
    mask = np.zeros((n, len(ATOM_TYPES)), dtype=bool)
    mask[:, CA_INDEX] = ca_present
    return {"atom37_positions": pos, "atom37_mask": mask,
            "asym_id": np.concatenate([np.full(s, i) for i, s in enumerate(chain_sizes)]),
            "aatype": np.zeros(n, dtype=np.int8),
            "residue_index": np.concatenate([np.arange(s) for s in chain_sizes])}


class TestPerChainQuality:
    """Takes two bundles, not an eval input — so it is testable without constructing one."""

    def test_a_perfect_prediction_scores_one_per_chain(self):
        got = per_chain_quality(_bundle(), _bundle())
        assert [c["chain"] for c in got] == [0, 1]
        assert all(c["TM"] == pytest.approx(1.0, abs=1e-6) for c in got)

    def test_each_chain_is_superposed_independently(self):
        """A correctly folded pair docked wrongly must still score TM 1.0 per chain —
        that separation is the entire reason these run beside DockQ."""
        pred = _bundle()
        pred["atom37_positions"][4:, CA_INDEX] += np.array([100.0, 100.0, 100.0])
        got = per_chain_quality(pred, _bundle())
        assert all(c["TM"] == pytest.approx(1.0, abs=1e-6) for c in got)

    def test_length_mismatch_is_flagged_not_scored(self):
        got = per_chain_quality(_bundle((4, 2)), _bundle((4, 3)))
        bad = [c for c in got if c["chain"] == 1][0]
        assert bad["n"] == 0 and "length mismatch" in bad["_note"]
        assert np.isnan(bad["TM"])

    def test_no_shared_ca_is_flagged(self):
        got = per_chain_quality(_bundle(ca_present=False), _bundle())
        assert all(c["_note"] == "no shared CA" for c in got)


class TestEvalCaching:
    """Caching lives on the eval input, not poked in from a metric module."""

    def test_per_chain_is_computed_once(self):
        from ecstasy.metrics import StructureEval
        ev = StructureEval(pred=_bundle(), native=_bundle())
        assert ev.per_chain() is ev.per_chain()

    def test_cache_is_kept_out_of_repr(self):
        """A debug print should not be a wall of cached numbers."""
        from ecstasy.metrics import StructureEval
        ev = StructureEval(pred=_bundle(), native=_bundle())
        ev.per_chain()
        assert "_per_chain" not in repr(ev)
