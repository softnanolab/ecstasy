"""Mean inter-chain P@K vs. paired-MSA depth, per model.

x = paired MSA depth of the complex (number of *paired* homologs in the Boltz-2
boltz_csv MSA, key != -1 — the sequences that carry inter-chain coevolution; uncapped,
a per-complex property). y = mean P@K, one line per model with a 95% bootstrap-CI band.
Pools the four original val splits. MSA-using models (boltz2, msa_pairformer) should
climb with depth; single-seq models (boltz2_nomsa, esmfold, mentos) should stay flat.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _plotstyle import use_cmu_concrete  # noqa: E402

use_cmu_concrete()

from ecstasy.config import settings  # noqa: E402
from ecstasy.datasets import load_dataset  # noqa: E402
from ecstasy.msa import store  # noqa: E402

R = settings().runs_root
SPLITS = ["val_seq_chain", "val_seq_pair", "val_pinder_chain", "val_pinder_pair"]
# model -> (preset, display, color-index)
MODELS = [("boltz2", "r3", "Boltz2 (with MSAs)"),
          ("msa_pairformer", "full", "MSA-Pairformer"),
          ("esmfold", "r3", "ESMFold"),
          ("mentos", "a5sgd6ul_latest", "MENTOS"),
          ("boltz2_nomsa", "r3", "Boltz2 (no MSAs)")]
_BINS = {  # well-populated bin edges per depth mode (distributions differ a lot)
    "paired": [0, 1, 8, 128, 2048, 9000],          # paired depth 1..8192
    "total":  [0, 512, 4096, 9000, 13000, 17000],  # total depth 2..16383
}
_MINBIN = 20


def _depth(seqs, mode) -> int:
    """MSA depth from the boltz_csv chain-0 CSV (0 if absent). mode='paired' counts only
    paired rows (key != -1, inter-chain coevolution); 'total' counts all sequences."""
    p = store.path_for_boltz_csv(seqs, 0)
    if not p.exists():
        return 0
    n = 0
    with open(p) as f:
        next(f, None)  # header
        for line in f:
            if mode == "total" or not line.startswith("-1,"):
                n += 1
    return n


def _paks(split, model, preset):
    rj = R / split / model / preset / "result.json"
    if not rj.exists():
        return {}
    per = json.loads(rj.read_text()).get("per_protein", {})
    return {k: v["P@K"] for k, v in per.items()
            if isinstance(v.get("P@K"), (int, float)) and v["P@K"] == v["P@K"]}


def _ci(v):
    if v.size < 2:
        return float(v.mean()), float(v.mean())
    rng = np.random.default_rng(0)
    m = v[rng.integers(0, v.size, size=(1000, v.size))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", choices=["paired", "total"], default="paired",
                    help="paired = inter-chain homologs (key!=-1); total = all sequences")
    mode = ap.parse_args().depth
    # depth per (split, id), pak per (split, id, model)
    depth, pak = {}, {m: {} for m, _, _ in MODELS}
    for s in SPLITS:
        ds = load_dataset(s)
        seqs_by_id = {e.id: e.sequences for e in ds.entries() if len(e.sequences) == 2}
        for eid, seqs in seqs_by_id.items():
            depth[(s, eid)] = _depth(seqs, mode)
        for m, preset, _ in MODELS:
            for eid, v in _paks(s, m, preset).items():
                if eid in seqs_by_id:
                    pak[m][(s, eid)] = v
    keys = list(depth)
    d_all = np.array([depth[k] for k in keys])
    print(f"pooled complexes: {len(keys)}  paired-depth: "
          f"min={d_all.min()} median={int(np.median(d_all))} max={d_all.max()}  "
          f"zero-depth(single-seq fallback)={(d_all==0).sum()}")

    bins = _BINS[mode]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    centers = [np.sqrt(max(bins[i], 1) * bins[i + 1]) for i in range(len(bins) - 1)]
    for ci_, (m, _, disp) in enumerate(MODELS):
        xs, ys, los, his = [], [], [], []
        for i in range(len(bins) - 1):
            lo_e, hi_e = bins[i], bins[i + 1]
            vals = np.array([pak[m][k] for k in keys
                             if k in pak[m] and lo_e <= depth[k] < hi_e])
            if vals.size < _MINBIN:
                continue
            lo, hi = _ci(vals)
            xs.append(centers[i]); ys.append(vals.mean()); los.append(lo); his.append(hi)
        if not xs:
            continue
        c = cmap(ci_)
        ax.fill_between(xs, los, his, color=c, alpha=0.15, lw=0)
        ax.plot(xs, ys, "-o", color=c, lw=1.8, ms=5, label=disp)

    lab = "paired" if mode == "paired" else "total"
    extra = "# paired homologs, key!=-1" if mode == "paired" else "# sequences, paired+unpaired"
    ax.set_xscale("log")
    ax.set_xlabel(f"{lab} MSA depth  ({extra} in the Boltz-2 MSA, log scale)")
    ax.set_ylabel("mean inter-chain P@K")
    ax.set_title(f"Inter-chain P@K vs. {lab}-MSA depth\n"
                 "(pooled val splits; MSA models sit above single-seq baselines)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = R / f"pak_vs_msadepth_{mode}.png"
    fig.savefig(out, dpi=160); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out} (+pdf)")


if __name__ == "__main__":
    main()
