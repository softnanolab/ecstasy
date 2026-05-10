from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from ecstasy.benchmarks.base import Entry


class ModelAdapter(ABC):
    name: ClassVar[str]
    needs_msa: ClassVar[bool]
    runner_script: ClassVar[Path | None] = None

    def __init__(self, config: dict):
        self.config = config
        env_path = config.get("env_path") or config.get("model_config", {}).get("env_path")
        self.env_path = Path(env_path) if env_path else None

    def predict_one(self, entry: Entry, msa_paths: dict[str, Path] | None, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = self._build_bundle(entry, msa_paths, out_dir)
        if self.env_path is None:
            self._run_in_host(bundle)
        else:
            self._run_in_env(bundle)
        contact_path = out_dir / "contact.npz"
        if not contact_path.exists():
            raise RuntimeError(f"runner did not produce {contact_path}")
        return contact_path

    def _build_bundle(self, entry: Entry, msa_paths: dict[str, Path] | None, out_dir: Path) -> dict:
        return {
            "entry_id": entry.id,
            "sequences": list(entry.sequences),
            "chain_ids": list(entry.chain_ids),
            "msa_paths": {k: str(v) for k, v in (msa_paths or {}).items()},
            "out_dir": str(out_dir),
            "config": self.config,
        }

    def _run_in_env(self, bundle: dict) -> None:
        if self.runner_script is None:
            raise NotImplementedError(f"{self.name} adapter must set runner_script")
        python = self.env_path / "bin" / "python"
        if not python.exists():
            raise FileNotFoundError(f"no python at {python}")
        subprocess.run(
            [str(python), str(self.runner_script)],
            input=json.dumps(bundle), text=True, check=True,
        )

    def _run_in_host(self, bundle: dict) -> None:
        raise NotImplementedError(f"{self.name} runs in env; set env_path in the config")


MODELS: dict[str, type[ModelAdapter]] = {}


def register_model(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    MODELS[cls.name] = cls
    return cls


def load_model(name: str, config: dict) -> ModelAdapter:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](config=config)
