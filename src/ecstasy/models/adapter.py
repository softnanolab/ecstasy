"""Subprocess bridge: run a model's runner inside its own venv.

Each model has a ``_runners/<x>.py`` that reads a JSON bundle on stdin and writes
``<out_dir>/contact.npz``. This adapter packs the bundle (resolved params + infra,
plus whichever MSA form the model consumes) and spawns the runner via the
model's env python. Runners import no ecstasy code, so they work inside venvs
that don't have ecstasy installed.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ecstasy.datasets.base import Entry
from ecstasy.models.registry import ModelRun


def predict_one(model: ModelRun, entry: Entry, msa, out_dir: Path) -> Path:
    """Run `model` on `entry`. `msa` is a {chain_id: a3m} dict (per_chain),
    a single a3m path (complex), or None."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "entry_id": entry.id,
        "sequences": list(entry.sequences),
        "chain_ids": list(entry.chain_ids),
        "msa_paths": {k: str(v) for k, v in msa.items()} if isinstance(msa, dict) else {},
        "complex_a3m": str(msa) if (msa is not None and not isinstance(msa, dict)) else None,
        "out_dir": str(out_dir),
        "params": model.params,
        "infra": model.infra,
    }
    python = model.env / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(f"no python at {python} (env for model {model.name!r})")
    subprocess.run([str(python), str(model.runner)],
                   input=json.dumps(bundle), text=True, check=True)
    contact_path = out_dir / "contact.npz"
    if not contact_path.exists():
        raise RuntimeError(f"runner did not produce {contact_path}")
    return contact_path
