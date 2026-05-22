"""Tests for ecstasy.msa.colabfold — parser + filters + UniProt encoding.

The HTTP client (submit_pair / poll_until_done / download_results) is
network-bound and tested via the build_ecstasy_v1 integration pipeline,
not here.
"""
from __future__ import annotations

import io
import tarfile

import pytest

from ecstasy.msa.colabfold import (
    SaveMsaFilters,
    apply_save_msa_filters,
    clean_sequence,
    make_query_fasta,
    parse_chain_a3m_chunk,
    parse_paired_a3m_bytes,
    stitch_paired_msa,
    _calc_distances,
    _extract_uniprot_id,
    _uniprot_to_number,
)


class TestSequenceHelpers:
    def test_clean_sequence_uppercases_and_strips(self):
        assert clean_sequence("acDEFg-1") == "ACDEFG"

    def test_make_query_fasta_starts_at_101(self):
        out = make_query_fasta(["ACD", "EF"])
        assert out == ">101\nACD\n>102\nEF\n"


class TestChainParser:
    def test_query_marked_as_is_query(self):
        text = ">101\nACDEF\n>UniRef100_X\nAAAAA\n"
        entries = parse_chain_a3m_chunk(text)
        assert entries[0].is_query is True
        assert entries[1].is_query is False

    def test_inserts_are_stripped(self):
        text = ">query\nACDef.GH\n"
        entries = parse_chain_a3m_chunk(text)
        # lowercase + dots are insertions, stripped
        assert entries[0].sequence == "ACGH"

    def test_metadata_filled_when_requested(self):
        # Real colabfold-style header:
        # >UniRef100_A0ACG9\t175\t0.266\t9.296E-44\t4\t322\t325\t1\t319\t323
        text = (
            ">101\nACDEFG\n"
            ">UniRef100_A0ACG9\t175\t0.266\t9.296E-44\t4\t322\t325\t1\t319\t323\nAAAAAA\n"
        )
        entries = parse_chain_a3m_chunk(text, extract_metadata=True)
        assert entries[0].is_query
        assert entries[0].coverage == 1.0
        assert entries[1].uid == "A0ACG9"
        assert entries[1].has_uniref is True
        # coverage = (q_end - q_start + 1) / q_len = (322 - 4 + 1) / 325
        assert entries[1].coverage == pytest.approx((322 - 4 + 1) / 325, rel=1e-6)
        assert entries[1].identity == pytest.approx(0.266)


def _make_tar_with_pair_a3m(content: bytes) -> bytes:
    """Build an in-memory tar.gz containing a single `pair.a3m` entry."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo(name="pair.a3m")
        ti.size = len(content)
        tf.addfile(ti, io.BytesIO(content))
    return buf.getvalue()


class TestParsePairedA3mBytes:
    def test_nul_byte_separates_chains(self):
        # Chain A: 1 query + 1 hit; chain B: 1 query + 1 hit
        # Server format = chain A FASTA, then \x00, then chain B FASTA
        a3m = (
            b">101\nACDE\n>UniRef100_X\tx\tx\tx\t1\t4\t4\tx\tx\tx\nAAAA\n"
            b"\x00"
            b">102\nGHIK\n>UniRef100_X\tx\tx\tx\t1\t4\t4\tx\tx\tx\nGGGG\n"
        )
        tar = _make_tar_with_pair_a3m(a3m)
        per_chain = parse_paired_a3m_bytes(tar, extract_metadata=False)
        assert len(per_chain) == 2
        # Chain A: 2 entries (query + 1 hit), same for chain B
        assert [len(c) for c in per_chain] == [2, 2]
        assert per_chain[0][0].sequence == "ACDE"
        assert per_chain[1][0].sequence == "GHIK"
        assert per_chain[0][1].sequence == "AAAA"
        assert per_chain[1][1].sequence == "GGGG"


class TestStitch:
    def test_stitch_concatenates_matched_rows(self):
        # 3 rows per chain; expected lens (4, 4) for both
        a3m = (
            b">101\nACDE\n>U1\tx\tx\tx\t1\t4\t4\tx\tx\tx\nAAAA\n>U2\tx\tx\tx\t1\t4\t4\tx\tx\tx\nBBBB\n"
            b"\x00"
            b">102\nGHIK\n>U1\tx\tx\tx\t1\t4\t4\tx\tx\tx\nGGGG\n>U2\tx\tx\tx\t1\t4\t4\tx\tx\tx\nHHHH\n"
        )
        per_chain = parse_paired_a3m_bytes(_make_tar_with_pair_a3m(a3m))
        stitched = stitch_paired_msa(per_chain, expected_chain_lens=[4, 4])
        # 3 rows in, 3 rows out (all matched-len OK)
        assert len(stitched) == 3
        assert stitched[0] == ("query", "ACDEGHIK")
        # Per-row pieces concatenate
        assert stitched[1][1] == "AAAAGGGG"
        assert stitched[2][1] == "BBBBHHHH"

    def test_stitch_drops_length_mismatch_rows(self):
        # Chain A row 1 has wrong matched length (3 instead of 4) — should be dropped
        a3m = (
            b">101\nACDE\n>U1\tx\tx\tx\t1\t4\t4\tx\tx\tx\nAAA\n"
            b"\x00"
            b">102\nGHIK\n>U1\tx\tx\tx\t1\t4\t4\tx\tx\tx\nGGGG\n"
        )
        per_chain = parse_paired_a3m_bytes(_make_tar_with_pair_a3m(a3m))
        stitched = stitch_paired_msa(per_chain, expected_chain_lens=[4, 4])
        # Only the query row survives (it has correct lengths)
        assert len(stitched) == 1
        assert stitched[0][1] == "ACDEGHIK"


class TestUniProtEncoding:
    def test_extract_uniprot_id_from_uniref100_header(self):
        h = ">UniRef100_A0ACG9\t175\t0.266"
        assert _extract_uniprot_id(h) == "A0ACG9"

    def test_extract_uniprot_id_rejects_short(self):
        assert _extract_uniprot_id(">UniRef100_AB\t1") == ""

    def test_extract_uniprot_id_rejects_non_uniref(self):
        assert _extract_uniprot_id(">some_other_header") == ""

    def test_extract_uniprot_id_handles_upi(self):
        # UPI IDs are valid identifiers; 13 chars total ("UPI" + 10 hex)
        h = ">UniRef100_UPI0001234567"
        assert _extract_uniprot_id(h) == "UPI0001234567"

    def test_uniprot_to_number_deterministic_and_distinct(self):
        # Two close UniProt IDs should produce different (deterministic) numbers
        a = _uniprot_to_number(["A0ACG9"])[0]
        b = _uniprot_to_number(["A0ACH0"])[0]
        assert a != 0
        assert b != 0
        assert a != b

    def test_calc_distances_marks_missing_with_minus_one(self):
        # Missing (=0) entries flagged as -1; present entries get abs diff
        assert _calc_distances([0, 5]) == [-1]
        assert _calc_distances([10, 7]) == [3]
        assert _calc_distances([10, 12, 20]) == [2, 8]


class TestApplySaveMsaFilters:
    def _build_paired_entries(self, *, depth: int, covs: list[float], ids: list[float], dists: list[int]):
        """Synthesize per-chain entry lists for filter testing."""
        from ecstasy.msa.colabfold import A3mEntry
        per_chain = []
        # chain A
        chain_a = [A3mEntry(header="query", sequence="AAAA",
                            coverage=1.0, identity=1.0, is_query=True,
                            has_uniref=False, uid="", uniprot_num=0)]
        chain_b = [A3mEntry(header="query", sequence="GGGG",
                            coverage=1.0, identity=1.0, is_query=True,
                            has_uniref=False, uid="", uniprot_num=0)]
        for i in range(depth):
            chain_a.append(A3mEntry(
                header=f"U{i}", sequence="A" * 4,
                coverage=covs[i], identity=ids[i], is_query=False,
                has_uniref=True, uid=f"A0AC{i:02d}", uniprot_num=100,
            ))
            chain_b.append(A3mEntry(
                header=f"U{i}", sequence="G" * 4,
                coverage=covs[i], identity=ids[i], is_query=False,
                has_uniref=True, uid=f"A0AC{i:02d}", uniprot_num=100 + dists[i],
            ))
        return [chain_a, chain_b]

    def test_query_row_passes_all_filters(self):
        per_chain = self._build_paired_entries(depth=0, covs=[], ids=[], dists=[])
        kept, stats = apply_save_msa_filters(per_chain, [4, 4])
        assert stats.kept == 1  # just the query
        assert kept[0] == ("query", "AAAAGGGG")

    def test_coverage_filter_drops_low_rows(self):
        per_chain = self._build_paired_entries(
            depth=2,
            covs=[0.9, 0.5],   # row 1 below 0.75
            ids=[0.5, 0.5],
            dists=[1, 1],
        )
        kept, stats = apply_save_msa_filters(per_chain, [4, 4])
        assert stats.kept == 2   # query + first hit
        assert stats.filtered_cov == 1

    def test_genomic_distance_filter_drops_far_rows(self):
        per_chain = self._build_paired_entries(
            depth=2,
            covs=[0.9, 0.9], ids=[0.5, 0.5],
            dists=[1, 5],  # row 2 has Δgene=5, default filter is 1
        )
        kept, stats = apply_save_msa_filters(per_chain, [4, 4])
        assert stats.kept == 2  # query + first hit
        assert stats.filtered_dist == 1

    def test_none_filters_disable(self):
        # All three filters None -> nothing dropped
        per_chain = self._build_paired_entries(
            depth=3,
            covs=[0.1, 0.1, 0.1], ids=[0.0, 0.0, 0.0], dists=[99, 99, 99],
        )
        kept, stats = apply_save_msa_filters(
            per_chain, [4, 4],
            filters=SaveMsaFilters(min_coverage=None, min_identity=None, max_genomic_distance=None),
        )
        assert stats.kept == 4  # query + 3 hits
        assert stats.filtered_cov == 0 and stats.filtered_id == 0 and stats.filtered_dist == 0
