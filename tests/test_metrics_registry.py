"""The metric registry — one definition per name, reachable from everywhere.

The registry exists because tolerant P@K lived inside a plotting script where scoring
could not reach it. So the properties worth pinning are about *addressability and
uniqueness*, not arithmetic (that is `test_metrics_tolerance.py`).
"""
from __future__ import annotations

import numpy as np
import pytest

from ecstasy import metrics
from ecstasy.metrics import ContactEval, registry


def _ev():
    la, lb = 4, 4
    gt = np.zeros((8, 8), bool)
    gt[0, 4] = gt[1, 5] = True
    probs = np.zeros((8, 8))
    probs[0, 4] = 0.9
    probs[3, 7] = 0.8
    return ContactEval(probs=probs, gt=gt, valid=np.ones((8, 8), bool),
                       chain_lengths=(la, lb))


class TestCatalogue:
    def test_the_canonical_metrics_are_registered(self):
        for name in ("AUC", "P@K", "P@K/2", "P@K/5"):
            assert name in registry.names("contact")

    def test_tolerant_variants_are_registered(self):
        """The whole reason the registry exists — these used to be unreachable."""
        for name in ("P@K(tol=1)", "P@K(tol=2)", "P@K/5(tol=2)"):
            assert name in registry.names("contact")

    def test_names_are_reported_as_they_are_written(self):
        """Names land in committed results; snake_casing them would orphan history."""
        assert "P@K/2(tol=1)" in registry.names()

    def test_describe_is_machine_readable(self):
        rows = registry.describe("contact")
        assert rows and all({"name", "kind", "description", "higher_is_better"} <= set(r)
                            for r in rows)
        assert all(r["description"] for r in rows), "every metric must document itself"

    def test_unknown_metric_names_the_alternatives(self):
        with pytest.raises(KeyError, match="P@K"):
            registry.get("P@K(tol=99)")


class TestUniqueness:
    def test_duplicate_registration_is_refused(self):
        """Two definitions under one name is the ambiguity this registry prevents."""
        with pytest.raises(ValueError, match="already registered"):
            registry.register("P@K", "contact", lambda ev: 0.0, "a second P@K")

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="unknown kind"):
            registry.register("bogus", "vibes", lambda ev: 0.0, "d")


class TestCompute:
    def test_computes_requested_metrics_only(self):
        got = registry.compute(["P@K", "P@K(tol=2)"], _ev())
        assert set(got) == {"P@K", "P@K(tol=2)"}

    def test_values_are_floats(self):
        got = registry.compute(["P@K"], _ev())
        assert isinstance(got["P@K"], float)

    def test_kind_mismatch_is_a_clear_error(self):
        class _Fake:
            KIND = "structure"
        with pytest.raises(TypeError, match="contact"):
            registry.compute(["P@K"], _Fake())

    def test_skip_errors_preserves_the_other_metrics(self):
        """One broken metric must not cost a run every number it already earned."""
        registry.register("_boom", "contact", lambda ev: 1 / 0, "always raises")
        try:
            got = registry.compute(["P@K", "_boom"], _ev(), skip_errors=True)
            assert "P@K" in got and "_boom" not in got
            with pytest.raises(ZeroDivisionError):
                registry.compute(["_boom"], _ev())
        finally:
            registry._REGISTRY.pop("_boom", None)


class TestDefaults:
    def test_default_set_is_the_historical_one(self):
        """Adding a metric to the registry must never silently change a headline number."""
        assert metrics.DEFAULT_CONTACT_METRICS == ("AUC", "P@K", "P@K/2", "P@K/5")

    def test_defaults_are_all_registered(self):
        for name in metrics.DEFAULT_CONTACT_METRICS:
            assert registry.get(name).kind == "contact"
