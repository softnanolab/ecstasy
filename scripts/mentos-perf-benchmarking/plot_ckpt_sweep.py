"""Plot MENTOS checkpoint-sweep metrics (inter/intra AUC + P@K) vs. training step and
pick the best overall checkpoint by mean rank across the four metrics.

  python scripts/plot_ckpt_sweep.py --results-dir <dir of step_*.json> --out <png>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _plotstyle import use_cmu_concrete  # noqa: E402

use_cmu_concrete()

_METRICS = [("inter_AUC", "inter AUC", "C0", "-"), ("intra_AUC", "intra AUC", "C0", "--"),
            ("inter_P@K", "inter P@K", "C1", "-"), ("intra_P@K", "intra P@K", "C1", "--")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = sorted((json.loads(Path(f).read_text()) for f in glob.glob(f"{args.results_dir}/step_*.json")),
                  key=lambda r: r["step"])
    steps = np.array([r["step"] for r in rows])
    vals = {k: np.array([r.get(k) if r.get(k) is not None else np.nan for r in rows])
            for k, *_ in _METRICS}

    # best overall = lowest mean rank across the 4 metrics (rank 1 = best per metric)
    keys = [k for k, *_ in _METRICS]
    ranks = np.zeros((len(rows), len(keys)))
    for j, k in enumerate(keys):
        order = np.argsort(-vals[k])          # descending: higher metric = better = rank 1
        ranks[order, j] = np.arange(1, len(rows) + 1)
    mean_rank = ranks.mean(axis=1)
    best_i = int(np.argmin(mean_rank))
    best_step = int(steps[best_i])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for k, lab, c, ls in _METRICS:
        ax = a1 if "AUC" in k else a2
        ax.plot(steps, vals[k], ls, color=c, marker="o", ms=4, lw=1.6, label=lab)
    for ax, ttl in ((a1, "AUC"), (a2, "P@K")):
        ax.axvline(best_step, color="0.5", ls=":", lw=1, zorder=0)
        ax.set_xlabel("training step"); ax.set_ylabel(f"mean {ttl} (val_seq_pair)")
        ax.set_title(ttl); ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=9)
    fig.suptitle(f"MENTOS checkpoint sweep — val_seq_pair (best overall: step {best_step:,})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160); fig.savefig(str(Path(args.out).with_suffix(".pdf")))

    print(f"\n{'step':>8}{'inter_AUC':>11}{'inter_P@K':>11}{'intra_AUC':>11}{'intra_P@K':>11}{'mean_rank':>11}")
    for i, r in enumerate(rows):
        mark = "  <-- BEST" if i == best_i else ""
        print(f"{r['step']:>8}{vals['inter_AUC'][i]:>11.4f}{vals['inter_P@K'][i]:>11.4f}"
              f"{vals['intra_AUC'][i]:>11.4f}{vals['intra_P@K'][i]:>11.4f}{mean_rank[i]:>11.2f}{mark}")
    print(f"\nper-metric best: " + ", ".join(f"{k}=step {int(steps[np.nanargmax(vals[k])]):,}" for k in keys))
    print(f"BEST OVERALL (mean rank): step {best_step:,}")
    print(f"wrote {args.out} (+pdf)")


if __name__ == "__main__":
    main()
