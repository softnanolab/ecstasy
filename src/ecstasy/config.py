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

# Filesystem roots are machine-specific absolute paths, so they are NOT committed.
# Set them in a repo-root `.env` (gitignored) or the environment; see `.env.example`
# for the keys. Defaults are intentionally empty so a missing root surfaces loudly
# rather than silently writing under the wrong tree.
#   DATA_ROOT     all run outputs (runs/, msa_store/) — ecstasy-only, clean
#   MENTOS_ROOT   MENTOS splits + ground truth
#   ECSTASY_ROOT  model weights (e.g. msa_pairformer)
#   ENVS_ROOT     per-model venvs        TOOLS_ROOT  vendored binaries (mmseqs, hhsuite)
_ROOTS = ("DATA_ROOT", "MENTOS_ROOT", "ECSTASY_ROOT", "ENVS_ROOT", "TOOLS_ROOT")
_DEFAULTS: dict[str, str] = {k: "" for k in _ROOTS}


@lru_cache(maxsize=1)
def _env_path() -> Path:
    """The ``.env`` to read: this checkout's, else the main checkout's.

    In a git worktree ``_REPO_ROOT`` is the worktree, which has no ``.env`` — it is
    gitignored, so it never travels with the branch. Every root then resolved to
    ``Path("")``, and a run wrote under the wrong tree instead of failing.

    A worktree's ``.git`` is a FILE holding ``gitdir: <main>/.git/worktrees/<name>``,
    so the main checkout is three parents up from that — findable without shelling out.
    """
    local = _REPO_ROOT / ".env"
    if local.exists():
        return local
    dot_git = _REPO_ROOT / ".git"
    if dot_git.is_file():
        text = dot_git.read_text().strip()
        if text.startswith("gitdir:"):
            main_root = Path(text.split(":", 1)[1].strip()).resolve().parents[2]
            if (main_root / ".env").exists():
                return main_root / ".env"
    return local


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    env_path = _env_path()
    out: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def env_value(key: str, default: str = "") -> str:
    """Read a machine-specific setting: env > .env > default.

    The single resolution chain for everything that must not be committed —
    the filesystem roots (via :func:`_value`) and ad-hoc keys like
    ``COLABFOLD_DBS`` / ``CUDA_MODULE``.
    """
    return os.environ.get(key) or _dotenv().get(key) or default


def _value(key: str) -> str:
    return env_value(key, _DEFAULTS[key])


@dataclass(frozen=True)
class Settings:
    DATA_ROOT: Path
    MENTOS_ROOT: Path
    ECSTASY_ROOT: Path
    ENVS_ROOT: Path
    TOOLS_ROOT: Path

    @property
    def runs_root(self) -> Path:
        return self.DATA_ROOT / "runs"

    @property
    def msa_store(self) -> Path:
        return self.DATA_ROOT / "msa_store"

    @property
    def natives_root(self) -> Path:
        """Ground-truth PDBs rendered from a dataset's own GT, cached per dataset.

        Derived, not transferred: written by ``ecstasy.structure.pdb`` from the
        registered ground truth, so a structure run needs no side-loaded native
        directory and cannot silently score against someone else's natives.
        """
        return self.DATA_ROOT / "natives"


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
