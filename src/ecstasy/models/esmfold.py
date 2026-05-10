from __future__ import annotations

from pathlib import Path

from ecstasy.models.base import ModelAdapter, register_model

_HERE = Path(__file__).resolve().parent


@register_model
class ESMFoldAdapter(ModelAdapter):
    name = "esmfold"
    needs_msa = False
    runner_script = _HERE / "_runners" / "esmfold_runner.py"
