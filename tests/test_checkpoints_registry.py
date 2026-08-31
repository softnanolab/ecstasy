"""Unit tests for the committed checkpoint registry (src/ecstasy/registry/checkpoints.py).

Points `checkpoints._REGISTRY` at a temp file per test so these never touch the real,
committed src/ecstasy/registry/checkpoints.yaml (which ships empty) and never leak
_load()'s cache across tests.
"""
from __future__ import annotations

import yaml
import pytest

from ecstasy.config import settings
from ecstasy.registry import checkpoints


@pytest.fixture
def registry_file(tmp_path, monkeypatch):
    """Point the checkpoints registry loader at an isolated, writable temp file."""
    p = tmp_path / "checkpoints.yaml"
    monkeypatch.setattr(checkpoints, "_REGISTRY", p)
    checkpoints._load.cache_clear()
    yield p
    checkpoints._load.cache_clear()


def _write(path, data: dict) -> None:
    path.write_text(yaml.safe_dump({"checkpoints": data}, sort_keys=True))


def test_unknown_checkpoint_raises(registry_file):
    _write(registry_file, {})
    with pytest.raises(KeyError, match="not in registry"):
        checkpoints.checkpoint("does_not_exist")


def test_checkpoint_round_trips(registry_file):
    _write(registry_file, {
        "ck_a": {"abs_path": "/abs/weights.pt", "run_id": "run123", "num_recycles": 2},
    })
    row = checkpoints.checkpoint("ck_a")
    assert row["abs_path"] == "/abs/weights.pt"
    assert row["run_id"] == "run123"
    assert row["num_recycles"] == 2


def test_checkpoint_resolves_var_placeholders(registry_file):
    """New behavior vs. the old Notion-backed cache: abs_path/model_config_path may use
    ${VAR} placeholders, expanded the same way models.yaml/datasets.yaml are."""
    _write(registry_file, {
        "ck_var": {"abs_path": "${DATA_ROOT}/weights/ck_var.pt",
                   "model_config_path": "${DATA_ROOT}/weights/ck_var.yaml"},
    })
    row = checkpoints.checkpoint("ck_var")
    s = settings()
    assert row["abs_path"] == f"{s.DATA_ROOT}/weights/ck_var.pt"
    assert row["model_config_path"] == f"{s.DATA_ROOT}/weights/ck_var.yaml"
    assert "${" not in row["abs_path"]


def test_checkpoint_params_shape(registry_file):
    _write(registry_file, {
        "ck_b": {"abs_path": "/abs/b.pt", "run_id": "runB", "num_recycles": 3,
                 "model_config_path": "/abs/b.yaml"},
    })
    params = checkpoints.checkpoint_params("ck_b")
    assert params == {
        "model_weights_path": "/abs/b.pt",
        "run_id": "runB",
        "num_recycles": 3,
        "model_config_path": "/abs/b.yaml",
    }


def test_checkpoint_params_without_weights_raises(registry_file):
    """A registered-but-unrunnable checkpoint (e.g. a random-init baseline with no
    abs_path) resolves via checkpoint() but is refused by checkpoint_params()."""
    _write(registry_file, {"baseline": {"run_id": "init"}})
    assert checkpoints.checkpoint("baseline")["run_id"] == "init"
    with pytest.raises(ValueError, match="no abs_path"):
        checkpoints.checkpoint_params("baseline")


def test_missing_registry_file_raises_filenotfound(registry_file, monkeypatch):
    monkeypatch.setattr(checkpoints, "_REGISTRY", registry_file.with_name("nope.yaml"))
    checkpoints._load.cache_clear()
    with pytest.raises(FileNotFoundError, match="checkpoints.yaml"):
        checkpoints.checkpoint("anything")
