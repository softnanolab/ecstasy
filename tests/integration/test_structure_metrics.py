"""Structure metrics end-to-end, against real ground truth and the real DockQ binary.

The synthetic tests in tests/test_metrics_structure.py cannot catch a whole class of
error — column alignment, chain labelling, residue numbering — because a fixture agrees
with the writer by construction. These use real GT and the actual CLI.

The load-bearing check feeds the native structure back in AS a prediction: a perfect
model must score DockQ 1.0. Anything less means ecstasy corrupted the structure somewhere
between the atom37 arrays and the scorer.

Skipped when the DockQ CLI or the MENTOS GT tree is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecstasy.metrics import StructureEval, registry
from ecstasy.metrics import structure as st
from ecstasy.structure.pdb import CA_INDEX, write_atom37_pdb

GT_ROOT = Path("/rds/general/user/ha1822/ephemeral/MENTOS/DATA/pdb/processed/data")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(st.dockq_binary() is None, reason="DockQ CLI not installed"),
    pytest.mark.skipif(not GT_ROOT.exists(), reason="MENTOS GT tree not present"),
]


@pytest.fixture(scope="module")
def bundle():
    """A real full-atom ground-truth dimer as an atom37 bundle."""
    torch = pytest.importorskip("torch")
    for pid in ("21ie", "9zdi", "10bl"):
        p = GT_ROOT / pid[:2] / f"{pid}.pt"
        if not p.exists():
            continue
        s = torch.load(p, weights_only=False, map_location="cpu")
        if s.atom37_positions is None or len(s.sequences) != 2:
            continue
        return pid, {
            "atom37_positions": s.atom37_positions.numpy(),
            "atom37_mask": s.atom37_mask.numpy(),
            "aatype": s.aatype.numpy(),
            "asym_id": s.asym_id.numpy(),
            "residue_index": s.residue_index.numpy(),
        }
    pytest.skip("no usable full-atom dimer in the GT tree")


def _render(b, path):
    return write_atom37_pdb(path, positions=b["atom37_positions"],
                            atom_mask=b["atom37_mask"], aatype=b["aatype"],
                            asym_id=b["asym_id"], residue_index=b["residue_index"])


def _ev(pred, native, tmp_path, entry_id):
    return StructureEval(pred=pred, native=native,
                         pred_pdb=_render(pred, tmp_path / "pred.pdb"),
                         native_pdb=_render(native, tmp_path / "native.pdb"),
                         entry_id=entry_id)


def test_native_against_itself_is_a_perfect_score(bundle, tmp_path):
    """The identity check: if anything corrupts the structure, this fails."""
    pid, b = bundle
    got = registry.compute(registry.names("structure"), _ev(b, b, tmp_path, pid))
    assert got["DockQ"] == pytest.approx(1.0, abs=1e-3)
    assert got["Fnat"] == pytest.approx(1.0, abs=1e-3)
    assert got["iRMSD"] == pytest.approx(0.0, abs=1e-2)
    assert got["LRMSD"] == pytest.approx(0.0, abs=1e-2)
    assert got["TM_mean"] == pytest.approx(1.0, abs=1e-3)
    assert got["CA_RMSD_mean"] == pytest.approx(0.0, abs=1e-3)


def test_all_dockq_metrics_cost_one_subprocess(bundle, tmp_path, monkeypatch):
    """Four registered DockQ names must not mean four invocations of the binary."""
    pid, b = bundle
    calls = []
    real = st.run_dockq
    monkeypatch.setattr(st, "run_dockq",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    registry.compute(["DockQ", "Fnat", "iRMSD", "LRMSD"], _ev(b, b, tmp_path, pid))
    assert len(calls) == 1, f"expected 1 DockQ call, got {len(calls)}"


def test_displacing_a_chain_destroys_docking_but_not_the_folds(bundle, tmp_path):
    """The shape a single-chain folder under a linker hack is expected to produce, and
    the reason per-chain metrics run beside DockQ."""
    pid, b = bundle
    moved = {k: v.copy() for k, v in b.items()}
    moved["atom37_positions"][moved["asym_id"] == 1] += np.array(
        [120.0, 0.0, 0.0], dtype=np.float32)
    got = registry.compute(registry.names("structure"), _ev(moved, b, tmp_path, pid))
    assert got["DockQ"] < 0.1
    assert got["LRMSD"] == pytest.approx(120.0, abs=1.0)   # the displacement itself
    assert got["TM_mean"] == pytest.approx(1.0, abs=1e-3)  # chains still perfect


def test_random_placement_floor_is_reproducible_and_below_the_truth(bundle, tmp_path):
    """A drifting floor is worse than none — it moves under the result it anchors."""
    pid, b = bundle
    ev = _ev(b, b, tmp_path, pid)
    first = st.random_placement_null(ev.pred_pdb, ev.native_pdb, pid, n_draws=3,
                                     work_dir=tmp_path)
    second = st.random_placement_null(ev.pred_pdb, ev.native_pdb, pid, n_draws=3,
                                      work_dir=tmp_path)
    assert first == second
    assert first["mean"] < 1.0          # cannot beat the native placement


def test_ca_only_prediction_does_not_crash(bundle, tmp_path):
    """Predictions mask unresolved atoms; a sparse bundle must still render and score."""
    pid, b = bundle
    sparse = {k: v.copy() for k, v in b.items()}
    mask = np.zeros_like(sparse["atom37_mask"])
    mask[:, CA_INDEX] = b["atom37_mask"][:, CA_INDEX]
    sparse["atom37_mask"] = mask
    got = registry.compute(registry.names("structure"), _ev(sparse, b, tmp_path, pid))
    # DockQ may decline to score a CA-only interface; what must not happen is a crash.
    assert "TM_mean" in got
