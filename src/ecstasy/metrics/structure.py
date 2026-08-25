"""Structure metrics: DockQ and its components, per-chain fold quality, a placement floor.

A contact map answers "which residue pairs touch". It cannot answer "is this complex
docked correctly", which is what a single-chain folder driven through a linker hack is
actually being asked. These are the metrics for that question, registered by name like
every other metric so scoring, plotting and manifests all reach the same implementation.

**Comparability rules — load-bearing, not stylistic.**

* The DockQ invocation is ``DockQ <model> <native>`` with no flags, parsed with
  :data:`_DOCKQ_RE`. Both are copied from ``mentos`` ``eval_structure_dockq``, which
  produced the 23-checkpoint MENTOS series. Change either and ecstasy's numbers stop
  being comparable to it. The ``DockQ[:\\s]+`` pattern with ``.search()`` skips the tool's
  banner and legend by an accident of output ordering; tidying it will break it.
* **Never read DockQ alone.** It averages fnat with two RMSD terms, so a prediction whose
  backbone has not formed still scores off fnat while both RMSD terms give near-zero
  credit — MENTOS step 2000 posts its *highest* median DockQ on its *worst* iRMSD for
  exactly this reason. iRMSD, LRMSD and the per-chain TM are registered alongside so a
  DockQ movement can always be read against them.

One DockQ subprocess serves every DockQ-derived metric: the components are parsed from a
single invocation and cached on the eval input, so registering four names does not mean
running the binary four times.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ecstasy.structure.pdb import CA_INDEX, atom_coords, chain_of, read_atom_lines, replace_coords

#: DockQ quality bands (Basu & Wallner 2016).
DOCKQ_BANDS = {"acceptable": 0.23, "medium": 0.49, "high": 0.80}

_DOCKQ_RE = {
    "DockQ": re.compile(r"DockQ[:\s]+([0-9.]+)", re.IGNORECASE),
    "Fnat": re.compile(r"Fnat[:\s]+([0-9.]+)", re.IGNORECASE),
    "iRMSD": re.compile(r"iRMS?D?[:\s]+([0-9.]+)", re.IGNORECASE),
    "LRMSD": re.compile(r"LRMS?D?[:\s]+([0-9.]+)", re.IGNORECASE),
}

_DOCKQ_TIMEOUT_S = 300


def dockq_binary() -> str | None:
    """Absolute path to the ``DockQ`` CLI, or None when it is not installed."""
    return shutil.which("DockQ")


def run_dockq(model_pdb: Path, native_pdb: Path,
              dockq_bin: str | None = None) -> dict[str, float] | None:
    """Score one model against one native. None if DockQ is absent or errored."""
    exe = dockq_bin or dockq_binary()
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, str(model_pdb), str(native_pdb)],
                             capture_output=True, text=True,
                             timeout=_DOCKQ_TIMEOUT_S).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    scores = {key: float(m.group(1))
              for key, pat in _DOCKQ_RE.items() if (m := pat.search(out))}
    return scores or None


def _dockq_components(ev) -> dict[str, float]:
    """All DockQ components from ONE subprocess, memoised on the eval input."""
    cached = getattr(ev, "_dockq_cache", None)
    if cached is None:
        cached = run_dockq(ev.pred_pdb, ev.native_pdb) or {}
        object.__setattr__(ev, "_dockq_cache", cached)
    return cached


def dockq_component(ev, key: str) -> float:
    """Registry adapter: one named DockQ component."""
    return float(_dockq_components(ev).get(key, float("nan")))


# --- monomer fold quality -------------------------------------------------------------

def kabsch_superpose(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """`mobile` rigidly superposed onto `target` (both (n, 3))."""
    mc, tc = mobile.mean(0), target.mean(0)
    m, t = mobile - mc, target - tc
    u, _, vt = np.linalg.svd(m.T @ t)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (r @ m.T).T + tc


def tm_score(mobile: np.ndarray, target: np.ndarray) -> float:
    """TM-score of two equal-length CA traces after Kabsch superposition.

    Uses the plain least-squares superposition rather than TM-align's iterative search,
    so this is a slight UNDER-estimate of the true TM-score. It answers "did the chain
    fold", which is a coarse question; do not quote it against published TM-align numbers.
    """
    n = len(target)
    if n == 0 or len(mobile) != n:
        return float("nan")
    d0 = 1.24 * (max(n, 19) - 15) ** (1 / 3) - 1.8
    d = np.linalg.norm(kabsch_superpose(mobile, target) - target, axis=1)
    return float((1.0 / (1.0 + (d / d0) ** 2)).mean())


def ca_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    if len(target) == 0 or len(mobile) != len(target):
        return float("nan")
    d = kabsch_superpose(mobile, target) - target
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def per_chain_quality(ev) -> list[dict]:
    """Fold quality per chain, each superposed independently and memoised.

    Superposing each chain on its own is the point: chains that folded well but docked
    badly is a completely different result from chains that never folded, and DockQ
    alone cannot distinguish them.
    """
    cached = getattr(ev, "_chain_cache", None)
    if cached is not None:
        return cached
    pred, native = ev.pred, ev.native
    p_asym, n_asym = np.asarray(pred["asym_id"]), np.asarray(native["asym_id"])
    out: list[dict] = []
    for chain in sorted(set(n_asym.tolist())):
        if chain < 0:
            continue
        pi, ni = np.flatnonzero(p_asym == chain), np.flatnonzero(n_asym == chain)
        if len(pi) != len(ni) or len(ni) == 0:
            # Length disagreement means prediction and GT disagree about the chain;
            # scoring it anyway would silently compare the wrong residues.
            out.append({"chain": int(chain), "n": 0, "TM": float("nan"),
                        "CA_RMSD": float("nan"),
                        "_note": f"length mismatch pred={len(pi)} native={len(ni)}"})
            continue
        keep = (np.asarray(pred["atom37_mask"])[pi, CA_INDEX]
                & np.asarray(native["atom37_mask"])[ni, CA_INDEX])
        if not keep.any():
            out.append({"chain": int(chain), "n": 0, "TM": float("nan"),
                        "CA_RMSD": float("nan"), "_note": "no shared CA"})
            continue
        pc = np.asarray(pred["atom37_positions"])[pi[keep], CA_INDEX]
        nc = np.asarray(native["atom37_positions"])[ni[keep], CA_INDEX]
        out.append({"chain": int(chain), "n": int(keep.sum()),
                    "TM": tm_score(pc, nc), "CA_RMSD": ca_rmsd(pc, nc)})
    object.__setattr__(ev, "_chain_cache", out)
    return out


def _chain_stat(ev, key: str, how: str) -> float:
    vals = [c[key] for c in per_chain_quality(ev) if c[key] == c[key]]
    if not vals:
        return float("nan")
    return float({"mean": np.mean, "min": np.min, "max": np.max}[how](vals))


# --- random-placement floor -----------------------------------------------------------

def stable_seed(key: str) -> int:
    """Reproducible 32-bit seed from a string.

    Deliberately not :func:`hash` — Python salts that per process unless PYTHONHASHSEED
    is set, so a hash-seeded floor silently moves between runs, under the very result it
    exists to anchor.
    """
    return int.from_bytes(hashlib.blake2s(key.encode(), digest_size=4).digest(), "big")


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform rotation via QR of a Gaussian matrix, sign-fixed for det=+1."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def random_placement_null(pred_pdb: Path, native_pdb: Path, entry_id: str,
                          n_draws: int = 10, work_dir: Path | None = None,
                          dockq_bin: str | None = None) -> dict:
    """DockQ floor: the model's own chains, chain B re-docked at random.

    Chain A is untouched; chain B gets a uniformly random rotation and is placed in a
    random direction at the NATIVE centre-of-mass separation. Fold quality is therefore
    identical to the real prediction and only the placement is destroyed — so whatever
    DockQ survives is what this target gives away for free, and it is the reference a low
    DockQ must be read against. DockQ superposes internally, so no global alignment is
    needed.
    """
    work_dir = Path(work_dir) if work_dir is not None else Path(pred_pdb).parent
    work_dir.mkdir(parents=True, exist_ok=True)

    lines = read_atom_lines(pred_pdb)
    a = [ln for ln in lines if chain_of(ln) == "A"]
    b = [ln for ln in lines if chain_of(ln) == "B"]
    nat = read_atom_lines(native_pdb)
    na = atom_coords([ln for ln in nat if chain_of(ln) == "A"])
    nb = atom_coords([ln for ln in nat if chain_of(ln) == "B"])
    if not a or not b or len(na) == 0 or len(nb) == 0:
        return {"n_draws": 0, "mean": float("nan"), "max": float("nan"), "scores": [],
                "_note": "prediction or native is not a two-chain A/B complex"}

    sep = float(np.linalg.norm(na.mean(0) - nb.mean(0)))
    xa, xb = atom_coords(a), atom_coords(b)
    rng = np.random.default_rng(stable_seed(entry_id))
    scores: list[float] = []
    for d in range(int(n_draws)):
        r = random_rotation(rng)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        xb2 = (xb - xb.mean(0)) @ r.T + xa.mean(0) + v * sep
        tmp = work_dir / f"{entry_id}_null{d}.pdb"
        tmp.write_text("\n".join(a + ["TER"] + replace_coords(b, xb2) + ["TER", "END"]) + "\n")
        try:
            s = run_dockq(tmp, native_pdb, dockq_bin=dockq_bin)
        finally:
            tmp.unlink(missing_ok=True)
        if s and "DockQ" in s:
            scores.append(s["DockQ"])
    return {
        "n_draws": len(scores),
        "mean": float(np.mean(scores)) if scores else float("nan"),
        "max": float(np.max(scores)) if scores else float("nan"),
        "scores": scores,
    }


def dockq_bands(scores: list[float]) -> dict[str, float]:
    """Fraction of targets at or above each DockQ quality band."""
    if not scores:
        return {f"{k}_fraction": float("nan") for k in DOCKQ_BANDS}
    a = np.asarray(scores, dtype=float)
    return {f"{k}_fraction": float((a >= thr).mean()) for k, thr in DOCKQ_BANDS.items()}
