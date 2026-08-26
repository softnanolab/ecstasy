"""End-to-end smoke: `ecstasy run` for minifold (single-sequence, --limit 1).

Beyond the shared contact.npz checks, this asserts the structure.npz that makes the
DockQ axis possible — its presence, its shape agreement with the contact map, and its
two-chain labelling. Those are the parts a contact-only smoke would pass right through.
"""
from pathlib import Path

import pytest

from tests.conftest import SMOKE_DATASET, first_entry_id
from tests.integration._common import assert_predict_succeeded


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_minifold
def test_predict_minifold(run_ecstasy, data_root):
    r = run_ecstasy(["run", "--dataset", SMOKE_DATASET, "--model", "minifold",
                     "--limit", "1", "--no_score"], timeout=1800)
    npz = assert_predict_succeeded(r, data_root, SMOKE_DATASET, "minifold",
                                   variant="full")

    import numpy as np

    from ecstasy.structure.pdb import ATOM_TYPES, load_structure_npz

    structure_path = Path(npz).parent / "structure.npz"
    assert structure_path.exists(), (
        f"minifold produced no structure.npz beside {npz}; without it the DockQ axis "
        f"silently degrades to contacts only")

    b = load_structure_npz(structure_path)
    n = b["asym_id"].shape[0]
    with np.load(npz) as d:
        assert n == int(d["length"]), "structure and contact map disagree on length"
    assert b["atom37_positions"].shape == (n, len(ATOM_TYPES), 3)
    assert b["atom37_mask"].shape == (n, len(ATOM_TYPES))
    assert np.isfinite(b["atom37_positions"][b["atom37_mask"]]).all()

    # Two chains, and the linker is trimmed out rather than emitted as residues.
    entry_id = first_entry_id(SMOKE_DATASET)
    from ecstasy.datasets import load_dataset
    entry = next(e for e in load_dataset(SMOKE_DATASET).entries() if e.id == entry_id)
    assert sorted(set(b["asym_id"].tolist())) == list(range(len(entry.sequences)))
    assert n == sum(len(s) for s in entry.sequences)

    # residue_index restarts per chain and is 0-based: the PDB writer adds the +1 that
    # matches MENTOS's natives.
    for i, seq in enumerate(entry.sequences):
        chain = b["residue_index"][b["asym_id"] == i]
        assert chain.tolist() == list(range(len(seq)))
