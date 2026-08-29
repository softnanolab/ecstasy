"""P@K vs. inference-FLOPs — original chain order (A,B) vs swapped (B,A), val_seq_pair.

Same axes/encoding as plot_pak_vs_flops.py but overlays the two chain orders for the
three swap-experiment models (boltz2, boltz2_nomsa, esmfold) across r0/r1/r3/r5:
filled markers + solid line = original; open markers + dashed line = swapped. The two
curves coincide -> the compute->quality tradeoff is order-symmetric in the mean (the
order effect is per-protein, see the scatter figures).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _plotstyle import use_cmu_concrete  # noqa: E402
from ecstasy.config import settings  # noqa: E402

use_cmu_concrete()

R = settings().runs_root
MODELS = [("boltz2", "Boltz2 (with MSAs)"), ("boltz2_nomsa", "Boltz2 (no MSAs)"),
          ("esmfold", "ESMFold")]
PRESETS = ["r0", "r1", "r3", "r5"]
_RNG, _NB = 0, 2000


def _ci(v: np.ndarray):
    if v.size < 2:
        return float(v.mean()), float(v.mean())
    rng = np.random.default_rng(_RNG)
    m = v[rng.integers(0, v.size, size=(_NB, v.size))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _cell(split, model, preset):
    base = R / split / model / preset
    rj = base / "result.json"
    if not rj.exists():
        return None
    per = json.loads(rj.read_text()).get("per_protein", {})
    pak = np.array([v["P@K"] for v in per.values()
                    if isinstance(v.get("P@K"), (int, float)) and v["P@K"] == v["P@K"]])
    fl = np.array([json.loads(fp.read_text())["flops"]
                   for fp in base.glob("predictions/*/flops.json")])
    if not pak.size or not fl.size:
        return None
    lo, hi = _ci(pak)
    return float(fl.mean()), float(pak.mean()), lo, hi


def main():
    cmap = plt.get_cmap("tab10")
    color = {m: cmap(i) for i, (m, _) in enumerate(MODELS)}
    fig, ax = plt.subplots(figsize=(8, 6))
    for split, fld, ls, mk in [("val_seq_pair", True, "-", "o"),
                               ("val_seq_pair_swapped", False, "--", "s")]:
        for m, _ in MODELS:
            pts = [(_cell(split, m, p)) for p in PRESETS]
            pts = [(p, c) for p, c in zip(PRESETS, pts) if c]
            if not pts:
                continue
            xs = [c[0] for _, c in pts]; ys = [c[1] for _, c in pts]
            ax.plot(xs, ys, ls, color=color[m], alpha=0.45, lw=1.2, zorder=1)
            for p, c in pts:
                f, pak, lo, hi = c
                ax.errorbar(f, pak, yerr=[[pak - lo], [hi - pak]], fmt=mk, color=color[m],
                            ecolor=color[m], elinewidth=0.8, capsize=2, ms=7,
                            mfc=(color[m] if fld else "white"), mec=color[m], mew=1.4, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("inference FLOPs (true = 2×MACs, contact-dependency subgraph, log scale)")
    ax.set_ylabel("mean inter-chain P@K  (val_seq_pair)")
    ax.set_title("Contact-prediction quality vs. inference compute\n"
                 "chain order (A,B) vs (B,A) — the mean curve is order-symmetric")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    handles = [plt.Line2D([], [], marker="o", ls="", color=color[m], mfc=color[m], label=lbl)
               for m, lbl in MODELS]
    handles += [
        plt.Line2D([], [], marker="o", ls="-", color="0.3", mfc="0.3", label="filled, solid = original (A,B)"),
        plt.Line2D([], [], marker="s", ls="--", color="0.3", mfc="white", mec="0.3", label="open, dashed = swapped (B,A)"),
        plt.Line2D([], [], color="0.4", marker="|", markersize=10, mew=1.5, ls="", label="bar = 95% bootstrap CI of mean P@K"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="best")
    fig.tight_layout()
    out = R / "val_seq_pair_swapped" / "swap_pak_vs_flops.png"
    fig.savefig(out, dpi=160); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} (+pdf)")


if __name__ == "__main__":
    main()
