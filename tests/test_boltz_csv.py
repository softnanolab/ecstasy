"""Unit tests for the boltz_csv MSA assembly (boltz compute_msa replica)."""
from __future__ import annotations

from ecstasy.msa import boltz_csv as b


def _a3m(seqs):
    # mmseqs-style a3m: alternating header/sequence lines (single-line seqs)
    return "".join(f">{i}\n{s}\n" for i, s in enumerate(seqs))


def test_pairing_keys_and_unpaired_minus_one():
    paired = _a3m(["QUERYAA", "PAIRED01", "PAIRED02"])      # query + 2 paired rows
    unpaired = _a3m(["QUERYAA", "UNP1", "UNP2"])
    rows = b.assemble_chain_csv(paired, unpaired)
    assert rows[0] == "key,sequence"
    body = rows[1:]
    # paired rows keyed by row index (0,1,2); unpaired keyed -1; query dropped from unpaired
    keys = [r.split(",", 1)[0] for r in body]
    assert keys[:3] == ["0", "1", "2"]
    assert keys[3:] == ["-1", "-1"]               # UNP1, UNP2 (query 'QUERYAA' dropped)
    assert "QUERYAA" in body[0]


def test_all_gap_paired_rows_dropped_but_indices_kept():
    # a fully-gapped paired row is dropped; surviving rows keep their original index key
    paired = _a3m(["QUERY", "----", "REALHIT"])
    rows = b.assemble_chain_csv(paired, _a3m(["QUERY"]))
    keys = [r.split(",", 1)[0] for r in rows[1:]]
    assert keys == ["0", "2"]                     # index 1 (all-gap) removed, 0 and 2 kept


def test_monomer_unpaired_only():
    rows = b.assemble_chain_csv("", _a3m(["QUERY", "H1", "H2"]))
    keys = [r.split(",", 1)[0] for r in rows[1:]]
    assert keys == ["-1", "-1", "-1"]             # no paired -> query NOT dropped, all -1
    assert "QUERY" in rows[1]


def test_caps_paired_at_8192():
    big = _a3m([f"P{i:05d}" for i in range(b.MAX_PAIRED_SEQS + 500)])
    rows = b.assemble_chain_csv(big, _a3m(["Q"]))
    n_paired = sum(1 for r in rows[1:] if not r.startswith("-1,"))
    assert n_paired == b.MAX_PAIRED_SEQS


def test_parse_qdb_lookup(tmp_path):
    p = tmp_path / "qdb.lookup"
    p.write_text("0\thashA\t0\n1\thashA\t0\n2\thashB\t1\n")
    m = b.parse_qdb_lookup(p)
    assert m == {"hashA": [0, 1], "hashB": [2]}


def test_write_chain_csv_returns_counts_and_file(tmp_path):
    dest = tmp_path / "sub" / "0.csv"
    n, n_paired = b.write_chain_csv(_a3m(["Q", "P1"]), _a3m(["Q", "U1", "U2"]), dest)
    assert dest.exists()
    text = dest.read_text()
    assert text.startswith("key,sequence\n") and text.endswith("\n")
    # 2 paired rows (query Q keyed 0, P1 keyed 1) + 2 unpaired (U1,U2; query dropped)
    assert (n, n_paired) == (4, 2)
    assert text.count("\n") == n + 1  # header + n rows + trailing newline


def _store_lookup(monkeypatch, tmp_path, sequences, write_indices):
    """Helper: point DATA_ROOT at tmp, write CSVs at given indices, run lookup."""
    from ecstasy import config
    from ecstasy.msa import store
    from ecstasy.datasets.base import Entry
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    config.settings.cache_clear()
    try:
        chain_ids = tuple(["A", "B", "C"][: len(sequences)])
        for i in write_indices:
            store.path_for_boltz_csv(sequences, i).parent.mkdir(parents=True, exist_ok=True)
            store.path_for_boltz_csv(sequences, i).write_text("key,sequence\n-1,ACDE\n")
        e = Entry(id="x", sequences=tuple(sequences), chain_ids=chain_ids)
        return store.lookup(e, "boltz_csv")
    finally:
        config.settings.cache_clear()


def test_store_lookup_heterodimer(monkeypatch, tmp_path):
    out = _store_lookup(monkeypatch, tmp_path, ["ACDE", "FGHI"], write_indices=[0, 1])
    assert out is not None and set(out) == {"A", "B"}
    assert out["A"].name == "0.csv" and out["B"].name == "1.csv"


def test_store_lookup_homodimer_resolves_both_chains_to_one_csv(monkeypatch, tmp_path):
    # colabfold dedups identical chains -> only 0.csv written; both chains must resolve
    out = _store_lookup(monkeypatch, tmp_path, ["ACDE", "ACDE"], write_indices=[0])
    assert out is not None and set(out) == {"A", "B"}
    assert out["A"].name == "0.csv" and out["B"].name == "0.csv"
