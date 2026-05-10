from __future__ import annotations

from pathlib import Path

from ecstasy.models.base import ModelAdapter, register_model

_HERE = Path(__file__).resolve().parent


@register_model
class Boltz2Adapter(ModelAdapter):
    name = "boltz2"
    needs_msa = True
    runner_script = _HERE / "_runners" / "boltz2_runner.py"
