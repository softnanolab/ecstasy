"""Model registry + subprocess adapter.

Models are declared as rows (with named presets) in ``registry/models.yaml`` and
loaded via :func:`load_model`. Each carries a runner script under ``_runners/``
that runs inside the model's venv; :func:`predict_one` bridges to it.
"""
from ecstasy.models.registry import (
    ModelRun,
    load_model,
    model_names,
    presets_for,
)
from ecstasy.models.adapter import predict_one

__all__ = ["ModelRun", "load_model", "model_names", "presets_for", "predict_one"]
