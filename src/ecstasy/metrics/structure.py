"""Structure metrics: DockQ, per-chain monomer quality, and a random-placement floor.

The companion to :mod:`ecstasy.metrics.contact`. A contact map answers "which residue
pairs touch"; it cannot answer "is this complex docked correctly", which is what DockQ
measures and what a single-chain folder run through the linker hack is being asked.

**Comparability rules — these are load-bearing.**

* The DockQ invocation is ``DockQ <model> <native>`` with no flags, and the scores are
  parsed with :data:`_DOCKQ_RE`. Both are copied from ``mentos``
  ``eval_structure_dockq``. The 23-checkpoint MENTOS series was produced this way, so
  changing either makes ecstasy's numbers incomparable to it. The ``DockQ[:\\s]+`` pattern
  with ``.search()`` skips the tool's banner and legend and lands on the real value; that
  works by an accident of output ordering, and tidying it will break it.
* :func:`run_dockq` always reports ``iRMSD`` and ``LRMSD`` next to ``DockQ``. DockQ
  averages fnat with two RMSD terms, so a prediction whose backbone has not formed still
  scores off fnat alone while both RMSD terms give near-zero credit — MENTOS step 2000
  posts its *highest* median DockQ on its *worst* iRMSD for exactly this reason. A DockQ
  rise is not an improvement unless the RMSDs move with it.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ecstasy.structure.pdb import (
    CA_INDEX,
    atom_coords,
    chain_of,
    read_atom_lines,
    replace_coords,
)

#: DockQ quality bands (Basu & Wallner 2016).
DOCKQ_BANDS = {"acceptable": 0.23, "medium": 0.49, "high": 0.80}

#: Metric keys this module produces, in report order.
STRUCTURE_METRIC_KEYS = ("DockQ", "Fnat", "iRMSD", "LRMSD", "TM_mean", "TM_min", "CA_RMSD_mean")

_DOCKQ_RE = {
    "DockQ": re.compile(r"DockQ[:\s]+([0-9.]+)", re.IGNORECASE),
    "Fnat": re.compile(r"Fnat[:\s]+([0-9.]+)", re.IGNORECASE),
    "iRMSD": re.compile(r"iRMS?D?[:\s]+([0-9.]+)", re.IGNORECASE),
    "LRMSD": re.compile(r"LRMS?D?[:\s]+([0-9.]+)", re.IGNORECASE),
}

_DOCKQ_TIMEOUT_S = 300


def dockq_binary() -> str | None:
    """Absolute path to the ``DockQ`` CLI, or ``None`` when it is not installed."""
    return shutil.which("DockQ")


def run_dockq(model_pdb: Path, native_pdb: Path,
              dockq_bin: str | None = None) -> dict[str, float] | None:
    """Score one model against one native. ``None`` if DockQ is absent or errored.

    Invocation and parsing are fixed for comparability — see the module docstring.
    """
    exe = dockq_bin or dockq_binary()
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, str(model_pdb), str(native_pdb)],
                             capture_output=True, text=True,
                             timeout=_DOCKQ_TIMEOUT_S).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    scores: dict[str, float] = {}
    for key, pat in _DOCKQ_RE.items():
        m = pat.search(out)
        if m:
            scores[key] = float(m.group(1))
    return scores or None


# --- monomer quality ----------------------------------------------------------------

def kabsch_superpose(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``mobile`` rigidly superposed onto ``target`` (both (n, 3))."""
    mc, tc = mobile.mean(0), target.mean(0)
    m, t = mobile - mc, target - tc
    u, _, vt = np.linalg.svd(m.T @ t)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (r @ m.T).T + tc


def tm_score(mobile: np.ndarray, target: np.ndarray) -> float:
    """TM-score of two equal-length CA traces, after Kabsch superposition.

    Superposition is the plain least-squares one rather than TM-align's iterative
    search, so this is a slight *under*-estimate of the true TM-score. It is used only
    to say whether the individual chains folded, which is a coarse question; do not
    quote it against published TM-align numbers.
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


def monomer_metrics(pred: dict[str, np.ndarray], native: dict[str, np.ndarray]) -> dict:
    """Per-chain fold quality, each chain superposed on its own.

    The user asked for these alongside DockQ precisely so a low DockQ can be read
    correctly: chains that folded well but docked badly is a different result from
    chains that did not fold at all.

    Both arguments are atom37 bundles. Only residues whose CA is present in *both*
    structures enter a chain's score.
    """
    per_chain: list[dict] = []
    p_asym, n_asym = np.asarray(pred["asym_id"]), np.asarray(native["asym_id"])
    for chain in sorted(set(n_asym.tolist())):
        if chain < 0:
            continue
        pi = np.flatnonzero(p_asym == chain)
        ni = np.flatnonzero(n_asym == chain)
        if len(pi) != len(ni) or len(ni) == 0:
            # Length disagreement means the runner and the GT disagree about the
            # chain; scoring it would silently compare the wrong residues.
            per_chain.append({"chain": int(chain), "n": 0, "TM": float("nan"),
                              "CA_RMSD": float("nan"),
                              "_note": f"length mismatch pred={len(pi)} native={len(ni)}"})
            continue
        keep = (np.asarray(pred["atom37_mask"])[pi, CA_INDEX]
                & np.asarray(native["atom37_mask"])[ni, CA_INDEX])
        if not keep.any():
            per_chain.append({"chain": int(chain), "n": 0, "TM": float("nan"),
                              "CA_RMSD": float("nan"), "_note": "no shared CA"})
            continue
        pc = np.asarray(pred["atom37_positions"])[pi[keep], CA_INDEX]
        nc = np.asarray(native["atom37_positions"])[ni[keep], CA_INDEX]
        per_chain.append({"chain": int(chain), "n": int(keep.sum()),
                          "TM": tm_score(pc, nc), "CA_RMSD": ca_rmsd(pc, nc)})
    tms = [c["TM"] for c in per_chain if c["TM"] == c["TM"]]
    rms = [c["CA_RMSD"] for c in per_chain if c["CA_RMSD"] == c["CA_RMSD"]]
    return {
        "per_chain": per_chain,
        "TM_mean": float(np.mean(tms)) if tms else float("nan"),
        "TM_min": float(np.min(tms)) if tms else float("nan"),
        "CA_RMSD_mean": float(np.mean(rms)) if rms else float("nan"),
    }


# --- random-placement null ----------------------------------------------------------

def stable_seed(key: str) -> int:
    """Reproducible 32-bit seed from a string.

    Deliberately not :func:`hash` — Python salts ``hash`` per process unless
    ``PYTHONHASHSEED`` is set, so a ``hash``-seeded null silently changes between runs
    and the "floor" moves under the result it is supposed to anchor.
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

    Chain A is left alone and chain B is given a uniformly random rotation, then placed
    in a random direction from A at the *native* centre-of-mass separation. Fold quality
    is therefore identical to the real prediction and only the relative placement is
    destroyed, so whatever DockQ survives is what chain geometry and size give away for
    free. DockQ superposes internally, so no global alignment to the native is needed.

    Returns ``{n_draws, mean, max, scores}``; ``scores`` empty when DockQ is unavailable.
    """
    work_dir = Path(work_dir) if work_dir is not None else Path(pred_pdb).parent
    work_dir.mkdir(parents=True, exist_ok=True)

    lines = read_atom_lines(pred_pdb)
    a = [ln for ln in lines if chain_of(ln) == "A"]
    b = [ln for ln in lines if chain_of(ln) == "B"]
    if not a or not b:
        return {"n_draws": 0, "mean": float("nan"), "max": float("nan"), "scores": [],
                "_note": "prediction is not a two-chain A/B complex"}

    nat = read_atom_lines(native_pdb)
    na = atom_coords([ln for ln in nat if chain_of(ln) == "A"])
    nb = atom_coords([ln for ln in nat if chain_of(ln) == "B"])
    if len(na) == 0 or len(nb) == 0:
        return {"n_draws": 0, "mean": float("nan"), "max": float("nan"), "scores": [],
                "_note": "native is not a two-chain A/B complex"}
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


# --- aggregation --------------------------------------------------------------------

def dockq_bands(scores: list[float]) -> dict[str, float]:
    """Fraction of targets at or above each DockQ quality band."""
    if not scores:
        return {f"{k}_fraction": float("nan") for k in DOCKQ_BANDS}
    a = np.asarray(scores, dtype=float)
    return {f"{k}_fraction": float((a >= thr).mean()) for k, thr in DOCKQ_BANDS.items()}
