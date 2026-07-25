"""store.paired_depth — collapse visibility for complex MSAs."""
from __future__ import annotations

from ecstasy.msa import store


def test_paired_depth_counts_sequence_rows(tmp_path):
    a3m = tmp_path / "x.a3m"
    a3m.write_text("#4,4\t1,1\n>query\nMKAAMRDD\n>hit1\nMKAAMRDD\n>hit2\nMKAGMRDE\n")
    assert store.paired_depth(a3m) == 3  # header line excluded


def test_paired_depth_query_only_is_one(tmp_path):
    a3m = tmp_path / "collapsed.a3m"
    a3m.write_text("#4,4\t1,1\n>query\nMKAAMRDD\n")
    assert store.paired_depth(a3m) == 1  # collapsed


def test_paired_depth_missing_file_is_zero(tmp_path):
    assert store.paired_depth(tmp_path / "nope.a3m") == 0


def test_depth_report_counts_present_and_collapsed(tmp_path, monkeypatch):
    # deep complex (3 rows), collapsed complex (query-only), and a missing one.
    deep = tmp_path / "deep.a3m"
    deep.write_text("#4,4\t1,1\n>query\nMKAAMRDD\n>h1\nMKAAMRDD\n>h2\nMKAGMRDE\n")
    collapsed = tmp_path / "collapsed.a3m"
    collapsed.write_text("#4,4\t1,1\n>query\nMKAAMRDD\n")
    paths = {"deep": deep, "collapsed": collapsed, "missing": tmp_path / "absent.a3m"}
    monkeypatch.setattr(store, "path_for_pair", lambda seqs: paths[seqs[0]])
    items = {k: {"seqs": [k]} for k in paths}
    present, collapsed_n, depths = store.depth_report(items)
    assert present == 2               # missing excluded
    assert collapsed_n == 1           # the query-only one
    assert sorted(depths) == [1, 3]
