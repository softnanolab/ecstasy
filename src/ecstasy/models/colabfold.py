from __future__ import annotations

from pathlib import Path

from ecstasy.models.base import ModelAdapter, register_model

_HERE = Path(__file__).resolve().parent


@register_model
class ColabFoldAdapter(ModelAdapter):
    """AlphaFold-2 inference via colabfold_batch.

    Treated as `needs_msa = True` because the standard column we want to compare
    is AF2-with-MSA. To run no-MSA, set `msa_mode: single_sequence` in the config.
    """

    name = "colabfold"
    needs_msa = True
    runner_script = _HERE / "_runners" / "colabfold_runner.py"
