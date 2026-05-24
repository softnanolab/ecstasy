"""Model registry: name -> runner spec.

Each entry binds a CLI-visible model name to a per-venv runner script and
declares whether the model consumes an MSA. The runner lives at
`models/_runners/<runner>` and is invoked by `ModelAdapter` via the venv's
python (see base.py). To add a model: drop a new `<name>_runner.py` into
`_runners/`, then add a row here.
"""
from __future__ import annotations

from pathlib import Path

from ecstasy.models.base import ModelAdapter

_RUNNERS_DIR = Path(__file__).resolve().parent / "_runners"

MODELS: dict[str, dict] = {
    "boltz2":         {"needs_msa": True,  "runner": "boltz2_runner.py"},
    "esmfold":        {"needs_msa": False, "runner": "esmfold_runner.py"},
    "mentos":         {"needs_msa": False, "runner": "mentos_runner.py"},
    "colabfold":      {"needs_msa": True,  "runner": "colabfold_runner.py"},
    "msa_pairformer": {"needs_msa": True,  "runner": "msa_pairformer_runner.py"},
}


def load_model(name: str, config: dict) -> ModelAdapter:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    spec = MODELS[name]
    return ModelAdapter(
        name=name,
        needs_msa=spec["needs_msa"],
        runner_script=_RUNNERS_DIR / spec["runner"],
        config=config,
    )


__all__ = ["ModelAdapter", "MODELS", "load_model"]
