"""Unit tests for the MENTOS square-GT dataset's pickle-alias shim.

``_alias_mentos_pickle_module`` registers the ``mentos`` -> ``mint`` alias that
lets the GT ``.pt`` files (pickled before the package rename) unpickle. It is
torch-free, so its error path is testable in the orchestrator env without GPU.
"""
from __future__ import annotations

import importlib.util

import pytest

from ecstasy.datasets.mentos import MentosSquareDataset


@pytest.mark.skipif(
    importlib.util.find_spec("mint") is not None,
    reason="mint installed in this env; the missing-mint raise path is not exercised here",
)
def test_alias_raises_clearly_without_mint():
    # gt_for is called unconditionally from score(), so an absent mint must fail
    # loudly with a pointer to .venv-mentos — not silently no-op into an opaque
    # ModuleNotFoundError deep in torch.load.
    with pytest.raises(ImportError, match="venv-mentos"):
        MentosSquareDataset._alias_mentos_pickle_module()
