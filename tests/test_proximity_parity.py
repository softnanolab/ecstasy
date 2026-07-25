"""Parity: ecstasy's proximity filter == colabfold-local's proximity.py.

The MSA-Pairformer paired-MSA method is implemented twice — in ecstasy's
``msa/colabfold.py`` (used by the API ``complex_api`` backend) and in the
``colabfold-local`` submodule's ``proximity.py`` (used by the local ``complex``
backend, which runs in the colabfold-local venv and cannot import ecstasy). This
test feeds identical ``A3mEntry`` inputs to both and asserts identical filter+stitch
output and identical accession encoding, so the shared **filter/encode/stitch logic**
cannot drift. It deliberately supplies coverage/identity directly and therefore does
NOT cover the one place the two paths genuinely differ — how each *derives* coverage/
identity (API: server span metadata; local: computed from the alignment). That
derivation gap can flip near-threshold keep/drop decisions for rows with internal gaps;
see ``msa/README.md`` ("Known parity gap") and the DB-gated integration cross-check.

If ``third_party/colabfold-local`` is not checked out, the test skips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ecstasy.msa import colabfold as E

_CFL_SRC = Path(__file__).resolve().parents[1] / "third_party" / "colabfold-local" / "src"
if not (_CFL_SRC / "proximity.py").exists():
    pytest.skip("colabfold-local submodule not checked out", allow_module_level=True)
sys.path.insert(0, str(_CFL_SRC))
import proximity as L  # noqa: E402


def test_defaults_match():
    e, l = E.SaveMsaFilters(), L.SaveMsaFilters()
    assert (e.min_coverage, e.min_identity, e.max_genomic_distance) == \
           (l.min_coverage, l.min_identity, l.max_genomic_distance) == (0.70, 0.15, 1)


@pytest.mark.parametrize("uid", ["A0A123", "Q1ABC7", "A0A0A0A0A1", "P12345"])
def test_accession_encoding_parity(uid):
    assert E._uniprot_to_number([uid]) == L._uniprot_to_number([uid])


def test_calc_distances_parity():
    nums = [0, 100, 101, 250, 250]
    assert E._calc_distances(nums) == L._calc_distances(nums)


def _pair(mod, rows):
    """Build a 2-chain per_chain structure of the module's own A3mEntry.

    rows: list of dicts with seqA, seqB, cov, ident, uidA, numA, uidB, numB, uniref.
    Row 0 is always the query.
    """
    chain_a, chain_b = [], []
    for i, r in enumerate(rows):
        q = i == 0
        chain_a.append(mod.A3mEntry(header="query" if q else r["uidA"], sequence=r["seqA"],
                                    coverage=r["cov"], identity=r["ident"], is_query=q,
                                    uid=r.get("uidA", ""), uniprot_num=r.get("numA", 0),
                                    has_uniref=r.get("uniref", False)))
        chain_b.append(mod.A3mEntry(header="query" if q else r["uidB"], sequence=r["seqB"],
                                    coverage=r["cov"], identity=r["ident"], is_query=q,
                                    uid=r.get("uidB", ""), uniprot_num=r.get("numB", 0),
                                    has_uniref=r.get("uniref", False)))
    return [chain_a, chain_b]


def _rows_only(result):
    # ecstasy returns (rows, stats); colabfold-local returns rows.
    return result[0] if isinstance(result, tuple) else result


def test_filter_and_stitch_parity():
    fixture = [
        {"seqA": "MKAA", "seqB": "MRDD", "cov": 1.0, "ident": 1.0},                       # query
        {"seqA": "MKAA", "seqB": "MRDD", "cov": 1.0, "ident": 1.0,                         # close pair -> keep
         "uidA": "A0A123", "numA": 10, "uidB": "B0B456", "numB": 11, "uniref": True},
        {"seqA": "MKAG", "seqB": "MRDE", "cov": 1.0, "ident": 1.0,                         # far pair -> drop
         "uidA": "A0A123", "numA": 10, "uidB": "B0B456", "numB": 90, "uniref": True},
        {"seqA": "MKAA", "seqB": "MRDD", "cov": 0.5, "ident": 1.0,                         # low cov -> drop
         "uidA": "A0A123", "numA": 10, "uidB": "B0B456", "numB": 11, "uniref": True},
        {"seqA": "MKAA", "seqB": "MRDD", "cov": 1.0, "ident": 1.0,                         # no UniRef -> keep
         "uidA": "", "numA": 0, "uidB": "", "numB": 0, "uniref": False},
    ]
    e_rows = _rows_only(E.apply_save_msa_filters(_pair(E, fixture), [4, 4], E.SaveMsaFilters()))
    l_rows = _rows_only(L.apply_save_msa_filters(_pair(L, fixture), [4, 4], L.SaveMsaFilters()))
    assert e_rows == l_rows
    # sanity: query + close pair + non-UniRef row survive; far + low-cov dropped
    assert [seq for _h, seq in l_rows] == ["MKAAMRDD", "MKAAMRDD", "MKAAMRDD"]
