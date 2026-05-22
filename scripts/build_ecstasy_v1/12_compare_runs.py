"""Side-by-side comparison of all MSA Pairformer runs on ecstasy_v1.

Reads per-entry CSVs from results/ and emits:
  results/comparison.csv           — one row per (run, metric)
  results/comparison.md            — human-readable summary
  results/comparison.png           — 4-panel figure (P@K hist + cdf, per-entry scatter)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/results")

# All known run flavours
RUNS = [
    ("qid15",       "Cb (hhfilter qid=15, no save_msa filters)",            "P@K"),
    ("qid30_cb",    "Cb (notebook hhfilter defaults, no save_msa filters)", "P@K"),
    ("run0",        "Cb (notebook defaults + ConFind head saved)",          "P@K"),
    ("run0_cf",     "ConFind (notebook defaults, no save_msa filters)",     "P@K_confind"),
    ("filtered",    "Cb (notebook defaults + save_msa filters cov=75 qid=15 Δgene=1)", "P@K"),
    ("filtered_cf", "ConFind (notebook defaults + save_msa filters)",       "P@K_confind"),
]


def load(name):
    """Load a run's per-entry CSV; return None if missing."""
    if name in ("run0_cf", "filtered_cf"):
        base = name.replace("_cf", "")
    else:
        base = name
    csv = RESULTS / f"ecstasy_v1__msa_pairformer__{base}.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv)


def main() -> int:
    available = []
    summaries = []
    for run_id, label, metric in RUNS:
        df = load(run_id)
        if df is None:
            continue
        df = df[df["status"] == "ok"]
        if metric not in df.columns:
            continue
        s = df[metric].dropna()
        summaries.append({
            "run_id": run_id,
            "label": label,
            "metric": metric,
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "max": s.max(),
            "above_20pct": (s > 0.20).sum(),
            "above_10pct": (s > 0.10).sum(),
            "above_05pct": (s > 0.05).sum(),
            "above_01pct": (s > 0.01).sum(),
        })
        available.append((run_id, label, metric, df))

    if not summaries:
        print("no results found")
        return 1

    out_csv = RESULTS / "comparison.csv"
    pd.DataFrame(summaries).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    # Markdown
    lines = ["# MSA Pairformer runs on ecstasy_v1 — side-by-side", ""]
    lines.append("| Run | Head | Filters | N | mean P@K | median | max | >20% | >10% | >5% |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| `{s['run_id']}` | {s['metric']} | "
            f"{s['label']} | {s['n']} | "
            f"**{s['mean']:.4f}** | {s['median']:.4f} | {s['max']:.4f} | "
            f"{int(s['above_20pct'])} | {int(s['above_10pct'])} | {int(s['above_05pct'])} |"
        )
    (RESULTS / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS / 'comparison.md'}")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
