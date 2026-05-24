"""End-to-end smoke: `ecstasy bench predict` for mentos (single-sequence).

MENTOS requires a Hydra config + Lightning checkpoint. The smoke sbatch uses
the AFDD-pretrained 8M_35M run; we default to the same paths but allow
override via env vars MENTOS_CFG / MENTOS_WTS. Test skips if neither is reachable.
"""
import os
from pathlib import Path

import pytest

from tests.integration._common import assert_predict_succeeded

MENTOS_CFG_DEFAULT = "/projects/u6jv/harsh/MENTOS_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/config.yaml"
MENTOS_WTS_DEFAULT = "/projects/u6jv/harsh/MENTOS_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/checkpoints/last.ckpt"


def _mentos_paths():
    cfg = os.environ.get("MENTOS_CFG", MENTOS_CFG_DEFAULT)
    wts = os.environ.get("MENTOS_WTS", MENTOS_WTS_DEFAULT)
    if not Path(cfg).exists():
        pytest.skip(f"MENTOS config not found at {cfg} (override with MENTOS_CFG=...)")
    if not Path(wts).exists():
        pytest.skip(f"MENTOS weights not found at {wts} (override with MENTOS_WTS=...)")
    return cfg, wts


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_mentos
def test_predict_mentos(smoke_config, run_ecstasy):
    mentos_cfg, mentos_wts = _mentos_paths()
    cfg = smoke_config("mentos", extra={
        "model_config_path": mentos_cfg,
        "model_weights_path": mentos_wts,
    })
    r = run_ecstasy(["bench", "predict", "--config", str(cfg)], timeout=900)
    assert_predict_succeeded(r, cfg, "mentos")
