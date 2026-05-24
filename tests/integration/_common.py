"""Shared helpers for integration tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def read_smoke_data_root(smoke_yaml_path: Path) -> Path:
    """Read the data_root that the smoke-config fixture wrote."""
    cfg = yaml.safe_load(smoke_yaml_path.read_text())
    return Path(cfg["data_root"])


def assert_predict_succeeded(completed, smoke_yaml_path: Path, model: str):
    """After `ecstasy bench predict`, verify the subprocess exited 0 and
    a valid contact.npz appeared at the expected location."""
    from tests.conftest import find_contact_npz, assert_valid_contact_npz, SMOKE_ENTRY_ID

    if completed.returncode != 0:
        pytest.fail(
            f"ecstasy bench predict failed (rc={completed.returncode})\n"
            f"--- STDOUT ---\n{completed.stdout}\n"
            f"--- STDERR ---\n{completed.stderr}"
        )

    data_root = read_smoke_data_root(smoke_yaml_path)
    npz = find_contact_npz(data_root, model, SMOKE_ENTRY_ID)
    assert npz is not None, (
        f"no contact.npz produced under {data_root}/ecstasy/benchmarks/mentos_seqid30/predictions/{model}/\n"
        f"--- STDOUT ---\n{completed.stdout}\n--- STDERR ---\n{completed.stderr}"
    )
    assert_valid_contact_npz(npz)
    return npz
