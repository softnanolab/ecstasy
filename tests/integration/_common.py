"""Shared helpers for integration smoke tests (new run layout)."""
from __future__ import annotations

from pathlib import Path

import pytest


def assert_predict_succeeded(completed, data_root: Path, dataset: str, model: str,
                             variant: str = "full"):
    """After `ecstasy run ... --limit 1`, verify rc==0 and a valid contact.npz
    appeared for the dataset's first entry under runs/<dataset>/<model>/<variant>/."""
    from tests.conftest import find_contact_npz, assert_valid_contact_npz, first_entry_id

    if completed.returncode != 0:
        pytest.fail(
            f"ecstasy run failed (rc={completed.returncode})\n"
            f"--- STDOUT ---\n{completed.stdout}\n--- STDERR ---\n{completed.stderr}"
        )
    entry_id = first_entry_id(dataset)
    npz = find_contact_npz(data_root, dataset, model, variant, entry_id)
    assert npz is not None, (
        f"no contact.npz under {data_root}/ecstasy/runs/{dataset}/{model}/{variant}/ "
        f"for entry {entry_id}\n--- STDOUT ---\n{completed.stdout}\n--- STDERR ---\n{completed.stderr}"
    )
    assert_valid_contact_npz(npz)
    return npz
