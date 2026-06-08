"""Unit tests for the DRN-1D2D_Inter runner's pure logic.

The only genuinely new, env-free logic in the runner is ``_embed_block`` — the
mapping from DRN's ``(lenA, lenB)`` inter-chain block to the full symmetric
``(L, L)`` contact map the scorer consumes. It is numpy-only (torch is imported
lazily inside ``main()``), so it imports and runs in the orchestrator env.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecstasy.models._runners.drn_1d2d_inter_runner import _embed_block


def test_embed_block_shape_and_dtype():
    lenA, lenB = 5, 3
    block = np.random.default_rng(0).random((lenA, lenB)).astype(np.float32)
    probs = _embed_block(block, lenA, lenB)
    L = lenA + lenB
    assert probs.shape == (L, L)
    assert probs.dtype == np.float16


def test_embed_block_places_block_in_inter_chain_quadrant():
    lenA, lenB = 4, 6
    block = (np.arange(lenA * lenB).reshape(lenA, lenB) + 1).astype(np.float32) / 100.0
    probs = _embed_block(block, lenA, lenB)
    # Upper-right inter-chain quadrant holds the block (the strict-upper-tri pairs
    # the scorer actually reads); intra-chain quadrants stay zero.
    np.testing.assert_allclose(probs[:lenA, lenA:].astype(np.float32), block, rtol=0, atol=1e-3)
    assert np.all(probs[:lenA, :lenA] == 0)  # intra-A
    assert np.all(probs[lenA:, lenA:] == 0)  # intra-B


def test_embed_block_is_symmetric():
    lenA, lenB = 7, 2
    block = np.random.default_rng(1).random((lenA, lenB)).astype(np.float32)
    probs = _embed_block(block, lenA, lenB).astype(np.float32)
    np.testing.assert_array_equal(probs, probs.T)


def test_embed_block_rejects_wrong_shape():
    with pytest.raises(RuntimeError, match=r"DRN block .* != expected"):
        _embed_block(np.zeros((3, 4), dtype=np.float32), 4, 4)
