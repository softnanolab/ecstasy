"""Model registry: rows + named presets -> a resolved :class:`ModelRun`.

``load_model(name, preset, overrides)`` reads ``registry/models.yaml``, resolves
``${VAR}`` paths, merges the chosen preset with any ``--set`` overrides, and
computes a stable, human-readable *variant* id used for the output directory:
``<preset>`` when there are no overrides, else ``<preset>+<sha8(overrides)>``.
Infra knobs (num_workers, devices, ...) are deliberately excluded from the
variant so machine-specific tweaks never fork a new run dir.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ecstasy.config import resolve

_REGISTRY = Path(__file__).resolve().parent.parent / "registry" / "models.yaml"
_RUNNERS_DIR = Path(__file__).resolve().parent / "_runners"


@dataclass(frozen=True)
class ModelRun:
    name: str
    runner: Path
    env: Path
    msa: str                      # none | per_chain | complex
    preset: str
    variant: str
    params: dict = field(default_factory=dict)
    infra: dict = field(default_factory=dict)

    @property
    def needs_msa(self) -> bool:
        return self.msa != "none"


def _registry() -> dict:
    return yaml.safe_load(_REGISTRY.read_text())


def model_names() -> list[str]:
    return sorted(_registry())


def presets_for(name: str) -> list[str]:
    return sorted(_registry()[name].get("presets", {}))


def _variant(preset: str, overrides: dict | None) -> str:
    if not overrides:
        return preset
    canonical = json.dumps(overrides, sort_keys=True, default=str)
    return f"{preset}+{hashlib.sha256(canonical.encode()).hexdigest()[:8]}"


def load_model(name: str, preset: str | None = None, overrides: dict | None = None,
               checkpoint: str | None = None) -> ModelRun:
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown model {name!r}; registered: {model_names()}")
    row = resolve(dict(reg[name]))
    presets = row.get("presets", {})

    def _make(preset_name: str, variant: str, params: dict) -> ModelRun:
        return ModelRun(
            name=name,
            runner=_RUNNERS_DIR / row["runner"],
            env=Path(row["env"]),
            msa=row.get("msa", "none"),
            preset=preset_name,
            variant=variant,
            params=params,
            infra=dict(row.get("infra", {})),
        )

    # Checkpoint models (e.g. mentos) have no committed presets: resolve a checkpoint *name*
    # to concrete params from the Notion-backed registry cache. The variant is the name.
    if checkpoint:
        from ecstasy.registry import local
        params = local.checkpoint_params(checkpoint)
        if overrides:
            params.update(overrides)
        return _make(checkpoint, _variant(checkpoint, overrides), params)
    if not presets:
        if preset:
            raise KeyError(f"model {name!r} has no presets; select a checkpoint with "
                           f"--checkpoint <name> (from the Notion registry)")
        return _make("", "", {})            # metadata-only (msa/env/needs_msa), not runnable

    preset = preset or row["default_preset"]
    if preset not in presets:
        raise KeyError(f"model {name!r} has no preset {preset!r}; have {sorted(presets)}")
    params = dict(presets[preset])
    if overrides:
        params.update(overrides)
    return _make(preset, _variant(preset, overrides), params)
