"""Run DockQ over a directory of *_pred.pdb / *_native.pdb pairs.

Emits the same JSON schema as scripts/evals/eval_structure_dockq.py so the result
drops straight into the existing comparison tooling, and uses the identical regexes
and no-flag DockQ invocation so the numbers are comparable to the MENTOS series.

Usage: python score_dockq_dir.py <dir> <out.json> [label]
"""

import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

DOCKQ = "/rds/general/user/ha1822/home/code/nanolab/mentos/.venv/bin/DockQ"

_DOCKQ_RE = {
    "DockQ": re.compile(r"DockQ[:\s]+([0-9.]+)", re.IGNORECASE),
    "Fnat": re.compile(r"Fnat[:\s]+([0-9.]+)", re.IGNORECASE),
    "iRMSD": re.compile(r"iRMS?D?[:\s]+([0-9.]+)", re.IGNORECASE),
    "LRMSD": re.compile(r"LRMS?D?[:\s]+([0-9.]+)", re.IGNORECASE),
}


def run_dockq(model_pdb, native_pdb):
    try:
        out = subprocess.run([DOCKQ, str(model_pdb), str(native_pdb)],
                             capture_output=True, text=True, timeout=300).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    scores = {}
    for key, pat in _DOCKQ_RE.items():
        m = pat.search(out)
        if m:
            scores[key] = float(m.group(1))
    return scores or None


def main():
    d = Path(sys.argv[1])
    out_json = Path(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else d.name

    preds = sorted(d.glob("*_pred.pdb"))
    per_sample = []
    for n, p in enumerate(preds, 1):
        pid = p.name.replace("_pred.pdb", "")
        native = d / f"{pid}_native.pdb"
        if not native.exists():
            continue
        s = run_dockq(p, native)
        if s:
            per_sample.append({"id": pid, "scores": s})
        print(f"[{n}/{len(preds)}] {pid} {s}", flush=True)

    scored = [r["scores"]["DockQ"] for r in per_sample if "DockQ" in r["scores"]]
    res = {
        "label": label,
        "n_samples": len(preds),
        "n_scored": len(scored),
        "mean_dockq": (sum(scored) / len(scored)) if scored else None,
        "median_dockq": statistics.median(scored) if scored else None,
        "acceptable_fraction": (sum(s >= 0.23 for s in scored) / len(scored)) if scored else None,
        "medium_fraction": (sum(s >= 0.49 for s in scored) / len(scored)) if scored else None,
        "high_fraction": (sum(s >= 0.80 for s in scored) / len(scored)) if scored else None,
        "per_sample": per_sample,
    }
    # DockQ averages fnat with two RMSD terms, so a collapsed prediction scores a
    # non-trivial DockQ off fnat alone while both RMSDs sit at zero credit.
    for key in ("Fnat", "iRMSD", "LRMSD"):
        vals = [r["scores"][key] for r in per_sample if key in r["scores"]]
        res[f"mean_{key.lower()}"] = (sum(vals) / len(vals)) if vals else None
    out_json.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
