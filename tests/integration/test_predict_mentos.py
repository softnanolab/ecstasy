"""End-to-end smoke: `ecstasy run` for mentos (single-sequence, --limit 1).

MENTOS has **no committed presets**: its checkpoints are concrete (paths, steps, recycles)
and are resolved by *name* against the committed src/ecstasy/registry/checkpoints.yaml, so
a run needs `--checkpoint <name>` and at least one runnable row in that file.

This test therefore skips unless checkpoints.yaml names at least one runnable checkpoint.
It previously asserted a `pretrain_8m_35m` preset that no longer exists, and its own
skip-guard raised `KeyError` on the empty params dict — a guard that crashes instead of
skipping is worse than no guard, because it reports as a failure.
"""
from pathlib import Path

import pytest

from tests.conftest import SMOKE_DATASET
from tests.integration._common import assert_predict_succeeded


def _first_runnable_checkpoint() -> str:
    """A checkpoint name with weights on this machine, or skip."""
    from ecstasy.registry import checkpoints

    try:
        names = sorted(checkpoints._load().get("checkpoints", {}))
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if not names:
        pytest.skip("no checkpoints in checkpoints.yaml")
    for name in names:
        try:
            params = checkpoints.checkpoint_params(name)
        except (KeyError, ValueError):
            continue          # e.g. the random-init baseline, which has no weights
        if Path(params["model_weights_path"]).exists():
            return name
    pytest.skip(f"no checkpoint in checkpoints.yaml has reachable weights "
                f"(tried {len(names)})")


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_mentos
def test_predict_mentos(run_ecstasy, data_root):
    checkpoint = _first_runnable_checkpoint()
    r = run_ecstasy(["run", "--dataset", SMOKE_DATASET, "--model", "mentos",
                     "--checkpoint", checkpoint, "--limit", "1", "--no_score"],
                    timeout=900)
    assert_predict_succeeded(r, data_root, SMOKE_DATASET, "mentos", variant=checkpoint)
