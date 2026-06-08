"""Unit tests for the DRN-1D2D_Inter runner's pure logic.

The only genuinely new, env-free logic in the runner is ``_embed_block`` — the
mapping from DRN's ``(lenA, lenB)`` inter-chain block to the full symmetric
``(L, L)`` contact map the scorer consumes. It is numpy-only (torch is imported
lazily inside ``main()``), so it imports and runs in the orchestrator env.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecstasy.models._runners.drn_1d2d_inter_runner import (
    _a3m_to_aln,
    _chain_a3m_from_rows,
    _embed_block,
    _paired_a3m_from_csvs,
    _read_boltz_csv,
)


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


# --- Boltz CSV -> DRN MSA conversion (the `msa: boltz_csv` reuse) -----------

def test_read_boltz_csv_parses_keys_and_seqs(tmp_path):
    csv = tmp_path / "0.csv"
    csv.write_text("key,sequence\n0,QUERYSEQ\n3,PAIREDSEQ\n-1,UNPAIRED\n")
    rows = _read_boltz_csv(csv)
    assert rows == [(0, "QUERYSEQ"), (3, "PAIREDSEQ"), (-1, "UNPAIRED")]


def test_chain_a3m_puts_query_first_with_all_rows():
    rows = [(0, "QUERY"), (2, "HOMOLOG"), (-1, "UNPAIR")]
    a3m = _chain_a3m_from_rows(rows, "A")
    lines = a3m.splitlines()
    assert lines[0] == ">A" and lines[1] == "QUERY"        # query first, clean header
    # homolog headers must NOT start with the query name (else LoadHHM's prefix match
    # shadows the query) -> ">h<i>", not ">A_<i>".
    assert lines[2:] == [">h1_k2", "HOMOLOG", ">h2_k-1", "UNPAIR"]
    assert not any(ln.startswith(">A") and ln != ">A" for ln in lines if ln.startswith(">"))


def test_paired_a3m_joins_on_shared_key_and_drops_unpaired():
    # Shared keys >= 0 are 0 and 3; key 5 (B-only) and -1 (unpaired) are excluded.
    rowsA = [(0, "QA"), (3, "XA"), (-1, "UA")]
    rowsB = [(0, "QB"), (3, "XB"), (5, "YB"), (-1, "UB")]
    paired = _paired_a3m_from_csvs(rowsA, rowsB, "paired")
    assert paired.splitlines() == [">paired", "QAQB", ">pair_k3", "XAXB"]


def test_paired_a3m_query_only_when_no_common_species():
    # Only the query (key 0) is shared -> a valid 1-row paired MSA, no coevolution.
    rowsA = [(0, "QA"), (1, "XA")]
    rowsB = [(0, "QB"), (2, "YB")]
    paired = _paired_a3m_from_csvs(rowsA, rowsB, "paired")
    assert paired.splitlines() == [">paired", "QAQB"]


def test_a3m_to_aln_strips_headers_and_insertions():
    # a3m insertions = lowercase + '.'; '-' is a match-state gap and is KEPT. Output is
    # headerless and every row is the query-length (4) match-state alignment.
    a3m = ">q\nABCD\n>h1\nABzzCD\n>h2\nA-cdCD\n"
    aln = _a3m_to_aln(a3m)
    assert aln.splitlines() == ["ABCD", "ABCD", "A-CD"]
    assert {len(r) for r in aln.splitlines()} == {4}
