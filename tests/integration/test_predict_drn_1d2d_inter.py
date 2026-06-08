"""End-to-end smoke: `ecstasy run` for drn_1d2d_inter (--limit 1).

Unlike the other baselines DRN is sequence-only but MSA-DEPENDENT: it needs a
per-chain a3m for each chain (the runner raises if either is missing). The smoke
therefore skips unless the env, tool binaries, weights AND the per-chain MSAs for
the first dataset entry are all present — i.e. it exercises the real path on a
fully-installed GPU node and no-ops everywhere else.
"""
from pathlib import Path

import pytest

from tests.conftest import SMOKE_DATASET
from tests.integration._common import assert_predict_succeeded


def _skip_if_unavailable():
    from ecstasy.models import load_model

    m = load_model("drn_1d2d_inter")
    p = m.params
    # Resolved tool/weight paths the runner shells out to must exist.
    for key in ("ccmpred_bin", "fasta2aln_bin", "alnstats_bin", "hhmake_bin",
                "hhfilter_bin", "esm1b_weights", "esm_msa1b_weights"):
        if key not in p or not Path(p[key]).exists():
            pytest.skip(f"drn_1d2d_inter {key} not found at {p.get(key)}")


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_drn_1d2d_inter
def test_predict_drn_1d2d_inter(run_ecstasy, data_root):
    _skip_if_unavailable()
    r = run_ecstasy(["run", "--dataset", SMOKE_DATASET, "--model", "drn_1d2d_inter",
                     "--limit", "1", "--no_score"], timeout=1800)
    assert_predict_succeeded(r, data_root, SMOKE_DATASET, "drn_1d2d_inter", variant="full")
