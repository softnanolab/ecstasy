"""End-to-end smoke: `ecstasy bench predict` for mint (single-sequence).

MINT requires a Hydra config + Lightning checkpoint. The smoke sbatch uses
the AFDD-pretrained 8M_35M run; we default to the same paths but allow
override via env vars MINT_CFG / MINT_WTS. Test skips if neither is reachable.
"""
import os
from pathlib import Path

import pytest

from tests.integration._common import assert_predict_succeeded

MINT_CFG_DEFAULT = "/projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/config.yaml"
MINT_WTS_DEFAULT = "/projects/u6jv/harsh/MINT_META/LOGS/MINT_AFDD_PRETRAIN_8M_35M/3khmvobe/checkpoints/last.ckpt"


def _mint_paths():
    cfg = os.environ.get("MINT_CFG", MINT_CFG_DEFAULT)
    wts = os.environ.get("MINT_WTS", MINT_WTS_DEFAULT)
    if not Path(cfg).exists():
        pytest.skip(f"MINT config not found at {cfg} (override with MINT_CFG=...)")
    if not Path(wts).exists():
        pytest.skip(f"MINT weights not found at {wts} (override with MINT_WTS=...)")
    return cfg, wts


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_mint
def test_predict_mint(smoke_config, run_ecstasy):
    mint_cfg, mint_wts = _mint_paths()
    cfg = smoke_config("mint", extra={
        "model_config_path": mint_cfg,
        "model_weights_path": mint_wts,
    })
    r = run_ecstasy(["bench", "predict", "--config", str(cfg)], timeout=900)
    assert_predict_succeeded(r, cfg, "mint")
