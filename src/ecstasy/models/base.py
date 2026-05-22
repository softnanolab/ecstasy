from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ecstasy.benchmarks.base import Entry


class ModelAdapter:
    """Subprocess-bridge to a model runner living in its own venv.

    Each registered model has a `<name>_runner.py` script under `_runners/`
    that reads a JSON bundle on stdin and writes `<out_dir>/contact.npz`.
    This adapter just packs the bundle and spawns the runner in `env_path`.
    """

    def __init__(self, *, name: str, needs_msa: bool, runner_script: Path, config: dict):
        self.name = name
        self.needs_msa = needs_msa
        self.runner_script = runner_script
        self.config = config
        env_path = config.get("env_path") or config.get("model_config", {}).get("env_path")
        if not env_path:
            raise ValueError(
                f"{name}: env_path required (top-level or under model_config)"
            )
        self.env_path = Path(env_path)

    def predict_one(
        self, entry: Entry, msa_paths: dict[str, Path] | None, out_dir: Path
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "entry_id": entry.id,
            "sequences": list(entry.sequences),
            "chain_ids": list(entry.chain_ids),
            "msa_paths": {k: str(v) for k, v in (msa_paths or {}).items()},
            "out_dir": str(out_dir),
            "config": self.config,
        }
        python = self.env_path / "bin" / "python"
        if not python.exists():
            raise FileNotFoundError(f"no python at {python}")
        subprocess.run(
            [str(python), str(self.runner_script)],
            input=json.dumps(bundle), text=True, check=True,
        )
        contact_path = out_dir / "contact.npz"
        if not contact_path.exists():
            raise RuntimeError(f"runner did not produce {contact_path}")
        return contact_path
