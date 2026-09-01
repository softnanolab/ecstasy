"""End-to-end structure scoring against real MENTOS ground truth.

Everything from a runner's ``structure.npz`` through PDB rendering, the DockQ CLI, its
output parsing, and the monomer metrics — driven with real GT rather than fixtures,
because the parts most likely to break (column alignment, chain labelling, residue
numbering) are exactly the parts a synthetic fixture would agree with by construction.

The trick is to feed the *native* coordinates back in as if they were a prediction:
a perfect model must score DockQ 1.0. Anything less means ecstasy corrupted the
structure somewhere between the npz and the scorer.

Skipped when the MENTOS GT tree or the DockQ CLI is unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecstasy.datasets import load_dataset
from ecstasy.datasets.base import Entry
from ecstasy.metrics.structure import dockq_binary
from ecstasy.structure.pdb import CA_INDEX, write_structure_npz

# Which split to score against is derived from the registry, not hardcoded. The
# original version pinned "mentos_val151", which was retired along with the other old
# splits -- and because this module only runs under the integration marker with a DockQ
# CLI present, nothing ever noticed it had gone stale.
PREFERRED = "recent_pp"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(dockq_binary() is None,
                       reason="DockQ CLI not installed in this env"),
]


def _first_full_atom_dimer():
    """The first (dataset, entry) pair in the registry carrying full-atom GT.

    Prefers PREFERRED when it is registered and built, so a normal run is
    deterministic, but falls back to any other registered split rather than
    skipping outright -- the point is to exercise the scoring path, and any
    dimer with full-atom ground truth does that equally well.
    """
    from ecstasy.datasets import dataset_names

    names = list(dataset_names())
    order = ([PREFERRED] if PREFERRED in names else []) + [n for n in names if n != PREFERRED]
    for name in order:
        d = load_dataset(name)
        if not d.index.exists():
            continue
        for e in d.entries():
            if len(e.sequences) == 2 and d.native_bundle(e.id) is not None:
                return d, e
    return None, None


@pytest.fixture(scope="module")
def _picked():
    d, e = _first_full_atom_dimer()
    if d is None:
        pytest.skip("no registered dataset carries full-atom ground truth")
    return d, e


@pytest.fixture(scope="module")
def dataset(_picked):
    return _picked[0]


@pytest.fixture(scope="module")
def entry(_picked) -> Entry:
    """The dimer chosen by `_picked` -- guaranteed to be from the same split as
    `dataset`, since both read the one selection rather than scanning separately."""
    return _picked[1]


def _npz_from(bundle, path):
    return write_structure_npz(path, **bundle)


def test_native_scored_against_itself_is_a_perfect_dockq(dataset, entry, tmp_path):
    """The end-to-end identity check: GT in as a prediction must come back DockQ 1.0."""
    bundle = dataset.native_bundle(entry.id)
    npz = _npz_from(bundle, tmp_path / "structure.npz")

    got = dataset.score_structure(entry, npz, work_dir=tmp_path,
                                  natives_dir=tmp_path / "natives")
    assert "_error" not in got and "_skipped" not in got, got
    assert got["DockQ"] == pytest.approx(1.0, abs=1e-3)
    assert got["Fnat"] == pytest.approx(1.0, abs=1e-3)
    assert got["iRMSD"] == pytest.approx(0.0, abs=1e-2)
    assert got["LRMSD"] == pytest.approx(0.0, abs=1e-2)
    assert got["TM_mean"] == pytest.approx(1.0, abs=1e-3)
    assert got["CA_RMSD_mean"] == pytest.approx(0.0, abs=1e-3)


def test_rendered_native_matches_the_datasets_own_native(dataset, entry, tmp_path):
    """The prediction writer and the native writer must be the same writer — otherwise
    a serialisation difference shows up inside every DockQ score."""
    bundle = dataset.native_bundle(entry.id)
    npz = _npz_from(bundle, tmp_path / "structure.npz")
    dataset.score_structure(entry, npz, work_dir=tmp_path,
                            natives_dir=tmp_path / "natives")

    pred = (tmp_path / f"{entry.id}_pred.pdb").read_text()
    native = (tmp_path / "natives" / f"{entry.id}_native.pdb").read_text()
    assert pred == native


def test_displacing_one_chain_destroys_dockq_but_not_the_fold(dataset, entry, tmp_path):
    """The separation the monomer metrics exist for: chains still folded, badly docked.

    This is the shape a single-chain folder run through the linker hack is expected to
    produce, and reading a low DockQ correctly depends on being able to see it.
    """
    bundle = dataset.native_bundle(entry.id)
    moved = {k: v.copy() for k, v in bundle.items()}
    chain_b = moved["asym_id"] == 1
    moved["atom37_positions"][chain_b] += np.array([120.0, 0.0, 0.0], dtype=np.float32)
    npz = _npz_from(moved, tmp_path / "structure.npz")

    got = dataset.score_structure(entry, npz, work_dir=tmp_path,
                                  natives_dir=tmp_path / "natives")
    assert "_error" not in got, got
    assert got["DockQ"] < 0.1
    assert got["TM_mean"] == pytest.approx(1.0, abs=1e-3)     # each chain still perfect
    assert got["CA_RMSD_mean"] == pytest.approx(0.0, abs=1e-3)


def test_length_mismatch_is_an_error_not_a_wrong_score(dataset, entry, tmp_path):
    """Silently scoring a truncated prediction would compare the wrong residues."""
    bundle = dataset.native_bundle(entry.id)
    short = {k: v[:-1] for k, v in bundle.items()}
    npz = _npz_from(short, tmp_path / "structure.npz")

    got = dataset.score_structure(entry, npz, work_dir=tmp_path,
                                  natives_dir=tmp_path / "natives")
    assert "_error" in got and "length mismatch" in got["_error"]


def test_random_placement_null_is_reproducible_and_below_the_truth(dataset, entry, tmp_path):
    """The floor must be stable across processes, or it moves under the result."""
    bundle = dataset.native_bundle(entry.id)
    npz = _npz_from(bundle, tmp_path / "structure.npz")

    first = dataset.score_structure(entry, npz, work_dir=tmp_path, null_draws=3,
                                    natives_dir=tmp_path / "natives")
    second = dataset.score_structure(entry, npz, work_dir=tmp_path, null_draws=3,
                                     natives_dir=tmp_path / "natives")
    assert first["null_DockQ_mean"] == pytest.approx(second["null_DockQ_mean"])
    # A randomly re-docked chain B cannot beat the native placement.
    assert first["null_DockQ_mean"] < first["DockQ"]


def test_ca_only_bundles_still_render(dataset, entry, tmp_path):
    """MiniFold masks unresolved atoms; a sparse bundle must not break the writer."""
    bundle = dataset.native_bundle(entry.id)
    sparse = {k: v.copy() for k, v in bundle.items()}
    mask = np.zeros_like(sparse["atom37_mask"])
    mask[:, CA_INDEX] = bundle["atom37_mask"][:, CA_INDEX]
    sparse["atom37_mask"] = mask
    npz = _npz_from(sparse, tmp_path / "structure.npz")

    got = dataset.score_structure(entry, npz, work_dir=tmp_path,
                                  natives_dir=tmp_path / "natives")
    # DockQ may decline to score a CA-only interface; what must not happen is a crash.
    assert "_error" not in got or "DockQ" in got["_error"]
