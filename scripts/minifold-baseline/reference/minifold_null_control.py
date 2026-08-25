"""Random-placement DockQ floor: MiniFold's own chains, chain B re-docked at random.

Chain A is left alone and chain B is given a uniformly random rotation and placed at a
random direction from A at the native centre-of-mass separation. Fold quality is
therefore identical to the real prediction and only the relative placement is
destroyed, so whatever DockQ survives is what geometry gives away for free. DockQ
superposes the complex internally, so no global alignment to the native is needed.

Usage: python minifold_null_control.py <pred_dir> <natives_dir> <work_dir> <out.json>
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

DOCKQ = "/rds/general/user/ha1822/home/code/nanolab/mentos/.venv/bin/DockQ"
N_DRAWS = 10


def parse_atoms(path):
    return [ln for ln in Path(path).read_text().splitlines() if ln.startswith("ATOM")]


def coords(lines):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines])


def set_coords(lines, xyz):
    out = []
    for ln, (x, y, z) in zip(lines, xyz):
        out.append(f"{ln[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{ln[54:]}")
    return out


def random_rotation(rng):
    """Uniform rotation via QR of a Gaussian matrix, sign-fixed for det=+1."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def dockq(model, native):
    try:
        out = subprocess.run([DOCKQ, str(model), str(native)],
                             capture_output=True, text=True, timeout=300).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    for ln in out.splitlines():
        if ln.strip().lower().startswith("dockq"):
            try:
                return float(ln.split()[-1])
            except ValueError:
                continue
    return None


def main():
    pred_dir, nat_dir, work_dir, out_json = (Path(a) for a in sys.argv[1:5])
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    preds = sorted(pred_dir.glob("*_pred.pdb"))
    for n, p in enumerate(preds, 1):
        pid = p.name.replace("_pred.pdb", "")
        native = nat_dir / f"{pid}_native.pdb"
        if not native.exists():
            continue

        lines = parse_atoms(p)
        a = [l for l in lines if l[21] == "A"]
        b = [l for l in lines if l[21] == "B"]
        if not a or not b:
            continue

        nat = parse_atoms(native)
        na = coords([l for l in nat if l[21] == "A"])
        nb = coords([l for l in nat if l[21] == "B"])
        sep = float(np.linalg.norm(na.mean(0) - nb.mean(0)))

        xa, xb = coords(a), coords(b)
        # Deterministic per target so the floor is reproducible across reruns.
        rng = np.random.default_rng(abs(hash(pid)) % (2**32))
        scores = []
        for d in range(N_DRAWS):
            r = random_rotation(rng)
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            xb2 = (xb - xb.mean(0)) @ r.T + xa.mean(0) + v * sep
            tmp = work_dir / f"{pid}_null{d}.pdb"
            tmp.write_text("\n".join(a + ["TER"] + set_coords(b, xb2) + ["TER", "END"]) + "\n")
            s = dockq(tmp, native)
            tmp.unlink(missing_ok=True)
            if s is not None:
                scores.append(s)

        if scores:
            rows.append({"id": pid, "n_draws": len(scores),
                         "mean": float(np.mean(scores)), "max": float(np.max(scores))})
        print(f"[{n}/{len(preds)}] {pid} null mean={np.mean(scores) if scores else float('nan'):.4f} "
              f"max={np.max(scores) if scores else float('nan'):.4f}", flush=True)

    summary = {
        "n_targets": len(rows),
        "n_draws_per_target": N_DRAWS,
        "mean_of_means": float(np.mean([r["mean"] for r in rows])) if rows else None,
        "mean_of_maxes": float(np.mean([r["max"] for r in rows])) if rows else None,
        "median_of_means": float(np.median([r["mean"] for r in rows])) if rows else None,
        "per_target": rows,
    }
    Path(out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_target"}, indent=2))


if __name__ == "__main__":
    main()
