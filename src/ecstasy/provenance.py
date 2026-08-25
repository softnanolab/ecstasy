"""What code and what weights produced a result.

Without this, two *different experiments* serialise identically. The concrete case that
motivated the module: the MiniFold runner takes ``minifold_src`` as a **path**, and the
`residx` patch applied inside that tree is the entire difference between the intended
chain break and the linker-only variant. A run's ``params.json`` records the path, so a
patched and an unpatched tree produce byte-identical records. The runner refuses to start
unpatched — which protects the *run* and does nothing for the *record*.

The same hole is open everywhere else. ``modules/`` carries eight submodules, one of them
a fast-moving MENTOS, and nothing in a result names the commit behind any of them. And
because weight paths under ``${ECSTASY_ROOT}`` are frequently symlinks, a symlink target
can be swapped with no artifact changing.

So a provenance record answers three questions:

  * which ecstasy commit ran, and was the tree dirty
  * which commit of every vendored/third-party source it used — including source trees
    reached only through a *parameter path*, which is how the MiniFold hole is closed
  * which bytes the weights actually were, following symlinks

Capture is best-effort by construction: a missing ``git``, a detached checkout or an
unreadable file degrades to a recorded reason. Provenance must never be the thing that
fails a benchmark run.
"""
from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Bytes hashed from each end of a large file. A full sha256 of a 2.6 GB checkpoint costs
#: ~10 s per run for no practical gain: size + mtime + both ends already separate any two
#: checkpoints anyone would plausibly confuse. `sha256_full` is available when it matters.
_HASH_WINDOW = 1 << 20


def _git(*args: str, cwd: Path, strip: bool = True) -> str | None:
    """Run a git command, returning stdout or None if it fails for any reason.

    ``strip=False`` matters for ``status --porcelain``: its format is ``XY<space>path``
    where X is the index status and Y the work-tree status, so an unstaged modification
    begins with a literal space (``" M model.py"``). Stripping that leading space shifts
    every filename by one character.
    """
    try:
        out = subprocess.run(("git", *args), cwd=str(cwd), capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() if strip else out.stdout


def git_state(path: Path) -> dict | None:
    """``{sha, dirty, branch, describe}`` for the work tree containing `path`.

    ``dirty`` covers tracked modifications only — the question being answered is "was the
    code that ran the committed code", and untracked scratch files do not change that.
    Returns None when `path` is not inside a git work tree.
    """
    path = Path(path)
    cwd = path if path.is_dir() else path.parent
    if not cwd.exists():
        return None
    if _git("rev-parse", "--is-inside-work-tree", cwd=cwd) != "true":
        return None
    sha = _git("rev-parse", "HEAD", cwd=cwd)
    if sha is None:
        return None
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=cwd, strip=False)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    state = {
        "sha": sha,
        "dirty": bool(status and status.strip()),
        "branch": None if branch == "HEAD" else branch,
    }
    describe = _git("describe", "--tags", "--always", "--dirty", cwd=cwd)
    if describe:
        state["describe"] = describe
    if state["dirty"]:
        # The files that make it dirty, capped — enough to see what differed later.
        state["dirty_files"] = [ln[3:] for ln in status.splitlines() if ln.strip()][:20]
    return state


def submodule_state(repo_root: Path | None = None) -> dict:
    """``{submodule path: {sha, dirty}}`` for every registered submodule.

    ``git submodule status`` prefixes a line with '+' when the checked-out commit differs
    from the one the superproject records — i.e. the submodule is not at its pin. That is
    exactly the "MENTOS moved under me" case, so it is surfaced as ``at_pin: false``
    rather than folded into ``dirty``.
    """
    root = Path(repo_root) if repo_root else _REPO_ROOT
    raw = _git("submodule", "status", "--recursive", cwd=root)
    if not raw:
        return {}
    out: dict[str, dict] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        marker, rest = line[0], line[1:].strip()
        parts = rest.split()
        if len(parts) < 2:
            continue
        sha, name = parts[0], parts[1]
        out[name] = {
            "sha": sha,
            "at_pin": marker != "+",
            "uninitialised": marker == "-",
        }
    return out


def file_identity(path: Path, full_hash: bool = False) -> dict:
    """Identity of the bytes at `path`, following symlinks.

    ``${ECSTASY_ROOT}`` weight paths are routinely symlinks into another tree, and a
    symlink target can be repointed without any artifact changing — so the resolved
    target is recorded alongside the declared path.
    """
    p = Path(path)
    rec: dict = {"path": str(p)}
    try:
        resolved = p.resolve()
        if resolved != p:
            rec["resolved"] = str(resolved)
        st = resolved.stat()
    except OSError as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    rec["size"] = st.st_size
    rec["mtime_utc"] = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
    try:
        rec["sha256" if full_hash else "sha256_ends"] = (
            _sha256_full(resolved) if full_hash else _sha256_ends(resolved, st.st_size))
    except OSError as e:
        rec["hash_error"] = f"{type(e).__name__}: {e}"
    return rec


def _sha256_ends(path: Path, size: int) -> str:
    """sha256 over the first and last `_HASH_WINDOW` bytes, plus the size."""
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(_HASH_WINDOW))
        if size > 2 * _HASH_WINDOW:
            f.seek(-_HASH_WINDOW, os.SEEK_END)
            h.update(f.read(_HASH_WINDOW))
    return h.hexdigest()


def _sha256_full(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def params_provenance(params: dict, full_hash: bool = False) -> dict:
    """Provenance for every filesystem path appearing in a model's resolved params.

    This is what closes the MiniFold hole. ``minifold_src`` is a path to a git checkout,
    so it is recorded as a *commit plus dirty flag*; a patched and an unpatched tree stop
    being indistinguishable. Weight files are recorded as bytes. Values that are not
    paths, and paths that do not exist, are skipped rather than guessed at.
    """
    out: dict[str, dict] = {}
    for key, value in (params or {}).items():
        if not isinstance(value, (str, Path)):
            continue
        p = Path(value)
        if not p.is_absolute() or not p.exists():
            continue
        if p.is_dir():
            state = git_state(p)
            out[key] = {"path": str(p), "kind": "source_tree",
                        **({"git": state} if state else {"git": None})}
        else:
            out[key] = {"kind": "file", **file_identity(p, full_hash=full_hash)}
    return out


@lru_cache(maxsize=1)
def _static_env() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
    }


def capture(params: dict | None = None, repo_root: Path | None = None,
            full_hash: bool = False) -> dict:
    """The full provenance record for one run. Never raises."""
    root = Path(repo_root) if repo_root else _REPO_ROOT
    rec: dict = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ecstasy": git_state(root) or {"error": "not a git work tree"},
        "submodules": submodule_state(root),
        "env": _static_env(),
    }
    # Scheduler identity, when running under one — makes a result traceable back to the
    # job that produced it (and to its log).
    job = {k: os.environ[k] for k in ("PBS_JOBID", "SLURM_JOB_ID", "JOB_ID")
           if os.environ.get(k)}
    if job:
        rec["job"] = job
    if params:
        rec["params_provenance"] = params_provenance(params, full_hash=full_hash)
    return rec


def summarise(rec: dict) -> str:
    """One-line human summary, for run logs and the compare table."""
    e = rec.get("ecstasy") or {}
    sha = (e.get("sha") or "unknown")[:8]
    bits = [f"ecstasy {sha}{'-dirty' if e.get('dirty') else ''}"]
    off_pin = [n for n, s in (rec.get("submodules") or {}).items() if not s.get("at_pin")]
    if off_pin:
        bits.append(f"{len(off_pin)} submodule(s) off pin: {', '.join(sorted(off_pin))}")
    for key, p in (rec.get("params_provenance") or {}).items():
        g = p.get("git")
        if g and g.get("sha"):
            bits.append(f"{key} {g['sha'][:8]}{'-dirty' if g.get('dirty') else ''}")
    return " | ".join(bits)
