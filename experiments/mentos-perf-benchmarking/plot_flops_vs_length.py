"""Per-protein inference FLOPs vs. sequence length, one series per preset.

Reads the ``flops.json`` sidecars (written by ``ecstasy run --profile``) under
``$DATA_ROOT/runs/<dataset>/<model>/<preset>/predictions/<id>/`` — each records
``L`` (the model's input length) and ``flops`` — and scatters FLOPs (y) against
L (x), one colored series per preset. For a recycle sweep the presets share an
identical L per protein, so the series stack into parallel compute bands that
show both the length-scaling of the architecture and the per-recycle multiplier.

Usage:
  python scripts/plot_flops_vs_length.py --dataset val_seq_chain --model esmfold \
      --presets r0,r1,r3,r5 [--out plot.png]
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _plotstyle import use_cmu_concrete  # noqa: E402

use_cmu_concrete()  # noqa: E402

from ecstasy.config import settings  # noqa: E402


def _series(dataset: str, model: str, preset: str) -> tuple[np.ndarray, np.ndarray]:
    root = settings().runs_root / dataset / model / preset / "predictions"
    Ls, Fs = [], []
    for fp in glob.glob(str(root / "*" / "flops.json")):
        try:
            d = json.loads(open(fp).read())
            Ls.append(float(d["L"]))
            Fs.append(float(d["flops"]))
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    order = np.argsort(Ls)
    return np.array(Ls)[order], np.array(Fs)[order]


def _binned_line(L, F, edges):
    """Per-bin (center, median, p10, p90) over shared bin edges; bins with <3 points dropped.
    Median is robust to the MSA-depth FLOPs spread; the p10-p90 band shows that spread."""
    idx = np.digitize(L, edges)
    cen, med, lo, hi = [], [], [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() < 3:
            continue
        cen.append(float(L[m].mean()))
        med.append(float(np.median(F[m])))
        lo.append(float(np.percentile(F[m], 10)))
        hi.append(float(np.percentile(F[m], 90)))
    return np.array(cen), np.array(med), np.array(lo), np.array(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="esmfold")
    ap.add_argument("--presets", default="r0,r1,r3,r5",
                    help="comma-separated preset names, low->high compute (one model)")
    ap.add_argument("--series", default="",
                    help="comma-separated model:preset tokens for a CROSS-MODEL plot "
                         "(e.g. boltz2:r0,esmfold:r0,msa_pairformer:full); overrides --model/--presets")
    ap.add_argument("--out", default=None)
    ap.add_argument("--xlog", action="store_true", help="log-scale the x (length) axis too")
    ap.add_argument("--style", choices=["scatter", "line"], default="scatter",
                    help="line = binned-median curve per series + faint p10-p90 spread band")
    ap.add_argument("--nbins", type=int, default=22, help="number of length bins for --style line")
    args = ap.parse_args()

    # Build the (model, preset, label) list and pick a palette. Recycle sweep of one
    # model -> ordered plasma; cross-model -> categorical tab10.
    if args.series:
        seq = []
        for tok in args.series.split(","):
            tok = tok.strip()
            if not tok:
                continue
            model, _, preset = tok.partition(":")
            seq.append((model, preset, f"{model} ({preset})"))
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(seq))]
        legend_title, title_sub = "model (preset)", "one series per model at a fixed preset"
    else:
        presets = [p.strip() for p in args.presets.split(",") if p.strip()]
        seq = [(args.model, p, p) for p in presets]
        cmap = plt.get_cmap("plasma")
        colors = [cmap(t) for t in np.linspace(0.08, 0.82, len(seq))]
        legend_title, title_sub = "recycles", "one series per recycle preset (r = recycling steps)"

    # Collect each series' (L, FLOPs) first so line mode can share one set of bin edges.
    data = []
    for (model, preset, label), c in zip(seq, colors):
        L, F = _series(args.dataset, model, preset)
        if not L.size:
            print(f"  (skip {model}:{preset}: no flops.json sidecars)")
            continue
        data.append((label, c, L, F))
    if not data:
        raise SystemExit(f"no flops.json under {settings().runs_root / args.dataset}")

    fig, ax = plt.subplots(figsize=(8, 6))
    if args.style == "line":
        allL = np.concatenate([L for _, _, L, _ in data])
        edges = np.linspace(allL.min(), allL.max(), args.nbins + 1)
        for label, c, L, F in data:
            cen, med, lo, hi = _binned_line(L, F, edges)
            ax.fill_between(cen, lo, hi, color=c, alpha=0.15, lw=0, zorder=2)
            ax.plot(cen, med, "-", color=c, lw=1.8, zorder=3, label=f"{label}  (n={L.size})")
    else:
        for label, c, L, F in data:
            ax.scatter(L, F, s=14, color=c, alpha=0.55, edgecolors="none",
                       label=f"{label}  (n={L.size})", zorder=3)
    drew = len(data)

    ax.set_yscale("log")
    if args.xlog:
        ax.set_xscale("log")
    ax.set_xlabel("total dimer sequence length L  (chainA + chainB residues = contact-map size)")
    ax.set_ylabel("inference FLOPs  (true = 2×MACs, contact-dependency subgraph, log)")
    head = "models" if args.series else args.model
    if args.style == "line":
        title_sub += "  ·  binned-median line, p10–p90 band"
    ax.set_title(f"{head} — inference FLOPs vs. sequence length  ({args.dataset})\n{title_sub}")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(title=legend_title, fontsize=9, loc="best")

    fig.tight_layout()
    out = args.out or str(settings().runs_root / args.dataset
                          / f"flops_vs_length_{args.model}.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}  ({drew} series)")


if __name__ == "__main__":
    main()
