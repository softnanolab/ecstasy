from __future__ import annotations

from pathlib import Path

from ecstasy.models.base import ModelAdapter, register_model

_HERE = Path(__file__).resolve().parent


@register_model
class MintAdapter(ModelAdapter):
    name = "mint"
    needs_msa = False
    runner_script = _HERE / "_runners" / "mint_runner.py"
