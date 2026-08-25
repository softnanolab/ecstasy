"""Fingerprints that decide when cached work may be reused.

The problem this exists to stop: a run directory is keyed
``<dataset>/<model>/<variant>`` and ``variant`` contains nothing about code, while
``run_predict`` skips any entry that already has a ``contact.npz``. So bumping a
dependency, editing a runner or repointing a weights symlink produces a run that reuses
old predictions while writing a *new* provenance record — a confidently false claim, which
is worse than recording nothing at all.

**Two fingerprints, not one.** Predictions never see ground truth: ``predict_one`` is handed
sequences, params and an MSA, and nothing else. Scoring is CPU-only and cheap. Conflating
them would mean a ground-truth regeneration or a metric bugfix discarded every prediction
and cost hours of GPU time to recompute results that were never affected.

  prediction  <- model code (from the venv), weights bytes, resolved params, MSA recipe,
                 the runner file itself, and the dataset index the sequences come from
  scoring     <- ground truth, metric implementations, the metric set, contact_bin

A fingerprint is a digest plus the inputs that produced it, so a mismatch can always be
explained in terms of *what changed* rather than just "differs".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ecstasy import provenance

_METRICS_DIR = Path(__file__).resolve().parent / "metrics"


def digest(inputs: dict) -> str:
    """Stable short digest of a fingerprint's inputs."""
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _code_identity(paths) -> dict:
    """sha256 over a set of source files, for 'did this implementation change'."""
    h = hashlib.sha256()
    listed: list[str] = []
    for p in sorted(Path(x) for x in paths):
        if not p.is_file():
            continue
        listed.append(p.name)
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return {"files": listed, "sha256": h.hexdigest()[:16]}


def _venv_code(env) -> dict:
    """Model code identity, reduced to the parts that can change an output."""
    rec = provenance.venv_packages(env)
    if "error" in rec:
        # Recorded, not raised: an absent venv is a real condition (scoring a run whose
        # model is not installed here), and it must not be silently treated as "same".
        return {"error": rec["error"]}
    out: dict[str, dict] = {}
    for name, pkg in (rec.get("packages") or {}).items():
        git = pkg.get("git") or {}
        out[name] = {
            "version": pkg.get("version"),
            "sha": git.get("sha"),
            # A dirty tree cannot be pinned to a commit, so it must never compare equal
            # to a clean one at the same sha — that is the patched-vs-unpatched case.
            "dirty": git.get("dirty"),
            "dirty_files": git.get("dirty_files"),
        }
    return out


def prediction_inputs(model, dataset, msa_recipe: str | None = None) -> dict:
    """Everything that can change a prediction."""
    return {
        "model": model.name,
        "variant": model.variant,
        "params": {k: str(v) for k, v in sorted((model.params or {}).items())},
        "params_provenance": provenance.params_provenance(model.params or {}),
        "runner": provenance.file_identity(model.runner),
        "venv": _venv_code(model.env),
        "msa_mode": model.msa,
        "msa_recipe": msa_recipe,
        "dataset": {
            "name": dataset.name,
            "version": dataset.version,
            # Sequences come from the index, so a changed index can change predictions.
            "index": dataset.fingerprint().get("index"),
        },
    }


def scoring_inputs(dataset, metrics) -> dict:
    """Everything that can change a score, given fixed predictions."""
    return {
        "dataset": {
            "name": dataset.name,
            "version": dataset.version,
            "contact_bin": getattr(dataset, "contact_bin", None),
            "gt": dataset.fingerprint().get("gt_root"),
        },
        "metrics": list(metrics),
        "metric_code": _code_identity(_METRICS_DIR.glob("*.py")),
    }


def make(kind: str, inputs: dict) -> dict:
    return {"kind": kind, "digest": digest(inputs), "inputs": inputs}


def compare(old: dict, new: dict) -> list[str]:
    """Human-readable account of what differs between two fingerprints.

    "The fingerprint changed" is useless at 3am; "minifold went 63db8b91 -> a1b2c3d4" is
    what tells you whether to force a re-run or fix your environment.
    """
    diffs: list[str] = []

    def walk(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), f"{path}.{key}" if path else key)
        elif a != b:
            diffs.append(f"{path}: {a!r} -> {b!r}")

    walk((old or {}).get("inputs", {}), (new or {}).get("inputs", {}))
    return diffs


class FingerprintMismatch(RuntimeError):
    """Raised when cached work was produced by different inputs than the current ones."""

    def __init__(self, kind: str, diffs: list[str], path: Path):
        self.kind = kind
        self.diffs = diffs
        shown = "\n".join(f"    {d}" for d in diffs[:12])
        more = f"\n    ... and {len(diffs) - 12} more" if len(diffs) > 12 else ""
        super().__init__(
            f"{kind} fingerprint does not match the cached run at {path}.\n"
            f"  What changed:\n{shown}{more}\n"
            f"  Reusing this directory would mix outputs from different code into one\n"
            f"  result that looks entirely normal. Either:\n"
            f"    - pass --force to recompute in place (discards the cached predictions), or\n"
            f"    - use a new --variant / --set so the new inputs get their own directory.")


def load(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save(path: Path, fp: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, indent=1, default=str))
    return path
