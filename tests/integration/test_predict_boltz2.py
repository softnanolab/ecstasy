"""End-to-end smoke: `ecstasy bench predict` for boltz2 (single-sequence mode).

The smoke config sets predict_limit=1 and no-MSA fallback. Runs the boltz2
runner in .venv-boltz; the orchestrator (.venv-ecstasy) drives it.
"""
import pytest

from tests.integration._common import assert_predict_succeeded


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_boltz2
def test_predict_boltz2(smoke_config, run_ecstasy):
    cfg = smoke_config("boltz2")
    r = run_ecstasy(["bench", "predict", "--config", str(cfg)], timeout=900)
    assert_predict_succeeded(r, cfg, "boltz2")


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.model_boltz2
def test_score_boltz2(smoke_config, run_ecstasy, repo_root):
    """Predict (orchestrator) then score (.venv-boltz).

    `bench score` loads MENTOS-pickled .pt GT files via torch — needs torch and
    mentos installed. The orchestrator (.venv-ecstasy) intentionally has neither,
    so we drive `score` from .venv-boltz, matching the prod smoke sbatch usage.
    """
    import json
    import os
    import subprocess
    from tests.integration._common import read_smoke_data_root

    cfg = smoke_config("boltz2")
    r = run_ecstasy(["bench", "predict", "--config", str(cfg)], timeout=900)
    assert_predict_succeeded(r, cfg, "boltz2")

    boltz_ecstasy = repo_root / "envs" / ".venv-boltz" / "bin" / "ecstasy"
    if not boltz_ecstasy.exists():
        pytest.skip(f"ecstasy CLI not found at {boltz_ecstasy}")
    r = subprocess.run(
        [str(boltz_ecstasy), "bench", "score", "--config", str(cfg)],
        capture_output=True, text=True, timeout=300, cwd=str(repo_root),
        env={**os.environ},
    )
    assert r.returncode == 0, f"score failed\nstdout: {r.stdout}\nstderr: {r.stderr}"

    results_dir = read_smoke_data_root(cfg) / "ecstasy" / "benchmarks" / "mentos_seqid30" / "results"
    jsons = list(results_dir.glob("mentos_seqid30__boltz2__*.json"))
    assert jsons, f"no results JSON under {results_dir}"
    payload = json.loads(jsons[0].read_text())
    summary = payload["summary"]
    assert summary["n_evaluated"] >= 1, (
        f"no entries scored: summary={summary}  "
        f"errors_first_20={payload.get('errors_first_20')}  "
        f"skipped_first_20={payload.get('skipped_first_20')}"
    )
