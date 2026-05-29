"""Single source of truth for filesystem roots, plus ``${VAR}`` interpolation.

Every absolute path in the codebase used to be hardcoded in benchmark classes
and in each per-run config. Now they live here once, as a small set of named
roots, and the registries (``registry/datasets.yaml``, ``registry/models.yaml``)
reference them with ``${VAR}`` placeholders that :func:`resolve` expands.

Resolution order for each root (first hit wins):
  1. an environment variable of the same name
  2. a ``KEY=VALUE`` line in the repo-root ``.env``
  3. the built-in default (the current cluster locations)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Built-in defaults = the current ISAMBARD/u6jv cluster layout. Override any of
# them via the environment or .env without touching code or the registries.
_DEFAULTS: dict[str, str] = {
    "DATA_ROOT": "/projects/u6jv/boltz_benchmarking/DATA",          # all run outputs
    "MENTOS_ROOT": "/projects/u6jv/public/MENTOS/DATA",             # MENTOS splits + GT
    "ECSTASY_ROOT": "/projects/u6jv/ecstasy",                       # model weights (e.g. msa_pairformer)
    "ENVS_ROOT": "/home/u6jv/harsh.u6jv/ecstasy/envs",              # per-model venvs (shared, absolute)
    "TOOLS_ROOT": "/home/u6jv/harsh.u6jv/ecstasy/tools",            # vendored binaries (shared, absolute)
}


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    env_path = _REPO_ROOT / ".env"
    out: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _value(key: str) -> str:
    return os.environ.get(key) or _dotenv().get(key) or _DEFAULTS[key]


@dataclass(frozen=True)
class Settings:
    DATA_ROOT: Path
    MENTOS_ROOT: Path
    ECSTASY_ROOT: Path
    ENVS_ROOT: Path
    TOOLS_ROOT: Path

    @property
    def runs_root(self) -> Path:
        return self.DATA_ROOT / "ecstasy" / "runs"

    @property
    def msa_store(self) -> Path:
        return self.DATA_ROOT / "ecstasy" / "msa_store"


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings(**{f.name: Path(_value(f.name)) for f in fields(Settings)})


_VAR = re.compile(r"\$\{(\w+)\}")


def resolve(value):
    """Expand ``${VAR}`` placeholders in a string (or recursively in a dict/list).

    Unknown vars are left untouched so typos surface as missing paths, not as
    silent empties. Non-string scalars pass through unchanged.
    """
    s = settings()
    table = {f.name: str(getattr(s, f.name)) for f in fields(Settings)}
    if isinstance(value, str):
        return _VAR.sub(lambda m: table.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v) for v in value]
    return value
