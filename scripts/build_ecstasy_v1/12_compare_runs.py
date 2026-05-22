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

    # === 4-panel comparison figure ===
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.tab10.colors

    # (a) histogram of P@K per run
    ax = axes[0, 0]
    for (run_id, label, metric, df), c in zip(available, colors):
        s = df[metric].dropna().values
        ax.hist(s, bins=np.linspace(0, max(s.max() * 1.05, 0.4), 25), histtype="step",
                linewidth=1.6, color=c, label=run_id)
    ax.set_xlabel("P@K")
    ax.set_ylabel("# entries")
    ax.set_title("P@K distribution across runs")
    ax.legend(fontsize=8, loc="upper right")

    # (b) CDF of P@K
    ax = axes[0, 1]
    for (run_id, label, metric, df), c in zip(available, colors):
        s = np.sort(df[metric].dropna().values)
        cdf = np.arange(1, len(s) + 1) / max(len(s), 1)
        ax.plot(s, cdf, drawstyle="steps-post", color=c, label=run_id)
    ax.set_xlabel("P@K")
    ax.set_ylabel("CDF over entries")
    ax.set_title("P@K CDF (rightward = better)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # (c) per-entry scatter: best Cb run vs best ConFind run
    ax = axes[1, 0]
    cb_runs = [(rid, lbl, m, df) for rid, lbl, m, df in available if m == "P@K"]
    cf_runs = [(rid, lbl, m, df) for rid, lbl, m, df in available if m == "P@K_confind"]
    if cb_runs and cf_runs:
        cb = max(cb_runs, key=lambda x: x[3][x[2]].dropna().mean())
        cf = max(cf_runs, key=lambda x: x[3][x[2]].dropna().mean())
        merged = cb[3][["id", cb[2]]].merge(
            cf[3][["id", cf[2]]], on="id", how="inner", suffixes=("_cb", "_cf")
        )
        ax.scatter(merged[cb[2]], merged[cf[2]], s=12, alpha=0.6)
        m_max = max(merged[cb[2]].max(), merged[cf[2]].max(), 0.4)
        ax.plot([0, m_max], [0, m_max], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f"P@K  ({cb[0]})")
        ax.set_ylabel(f"P@K_confind  ({cf[0]})")
        ax.set_title(f"Per-entry: {cb[0]} vs {cf[0]}")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # (d) bar chart of headline aggregates
    ax = axes[1, 1]
    labels = [s["run_id"] for s in summaries]
    means = [s["mean"] for s in summaries]
    medians = [s["median"] for s in summaries]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, means, 0.4, label="mean P@K")
    ax.bar(x + 0.2, medians, 0.4, label="median P@K")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("P@K")
    ax.set_title("Headline aggregates per run")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"MSA Pairformer on ecstasy_v1 — {len(summaries)} runs side-by-side",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out_png = RESULTS / "comparison.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
