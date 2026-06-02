"""P@K vs. interface size (per model), two x-modes.

  --xmode contacts : x = K = number of true inter-chain contacts (log scale)
  --xmode percent  : x = interface residues as a % of total residues (linear)

y = per-protein inter-chain P@K. One representative preset per model. Each model
is a **binned-mean line wrapped in a translucent 95% bootstrap-CI band** (no
per-point scatter). Per-protein (K, %, P@K) come from each run's result.json when
present, else are scored in-memory from available predictions (PREVIEW of an
in-flight sweep), WITHOUT writing anything.

Interface residue = a residue with >=1 true inter-chain contact (Cβ-Cβ, valid);
% = 100 * interface_residues / (La + Lb). Both x-stats come from the ground
truth (model-independent), cached across models.

Usage:
  python scripts/plot_pak_vs_interface.py --dataset val_seq_pair --xmode percent \
      [--presets boltz2=r1,esmfold=r1,msa_pairformer=full] [--cap 250]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _plotstyle import use_cmu_concrete  # noqa: E402

use_cmu_concrete()

from ecstasy.config import settings  # noqa: E402

_DEFAULT_PRESETS = {"boltz2": "r1", "esmfold": "r1", "msa_pairformer": "full"}
_RNG = np.random.default_rng(0)
_N_BOOT = 1000
_MIN_BIN = 3                      # drop bins with fewer proteins than this


def _gt_stats(ds, cache: dict, entry):
    """(K, interface_pct) from the GT for one entry; cached. None if unusable."""
    if entry.id in cache:
        return cache[entry.id]
    out = None
    try:
        gt = ds.gt_for(entry.id)
        cmap = np.asarray(gt["contact_map"]).astype(bool)
        valid = np.asarray(gt["valid"]).astype(bool)
        seqs = gt["sequences"]
        if len(seqs) == 2:
            la, lb = len(seqs[0]), len(seqs[1])
            L = la + lb
            if cmap.shape == (L, L):
                cid = np.array([0] * la + [1] * lb)
                inter = cid[:, None] != cid[None, :]
                ti = cmap & valid & inter                     # true inter-chain contacts
                K = int(np.triu(ti, 1).sum())
                iface_res = int(ti.any(axis=1).sum())
                if K > 0:
                    out = (K, 100.0 * iface_res / L)
    except Exception:        # noqa: BLE001
        out = None
    cache[entry.id] = out
    return out


def _collect(ds, entries, gt_cache, run_dir: Path, xmode: str, cap: int):
    """Return (x, pak) arrays for a run. P@K from result.json if present, else
    scored in-memory (capped). x from the GT stats (cached)."""
    xs, paks = [], []
    result = run_dir / "result.json"
    have_result = result.exists()
    per = json.loads(result.read_text()).get("per_protein", {}) if have_result else {}
    pred_dir = run_dir / "predictions"

    ids = list(per) if have_result else [p.name for p in sorted(pred_dir.glob("*"))]
    for pid in ids:
        if len(paks) >= cap:
            break
        entry = entries.get(pid)
        if entry is None:
            continue
        if have_result:
            p = per[pid].get("P@K")
        else:
            cpath = pred_dir / pid / "contact.npz"
            if not cpath.exists():
                continue
            try:
                p = ds.score(entry, cpath).get("P@K")
            except Exception:        # noqa: BLE001
                continue
        if not isinstance(p, (int, float)) or p != p:
            continue
        st = _gt_stats(ds, gt_cache, entry)
        if st is None:
            continue
        xs.append(st[0] if xmode == "contacts" else st[1])
        paks.append(float(p))
    return np.array(xs, float), np.array(paks, float)


def _binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray):
    """Per-bin (center, mean, ci_lo, ci_hi); bins with < _MIN_BIN points dropped."""
    idx = np.digitize(x, edges) - 1
    cen, mean, lo, hi = [], [], [], []
    log = (edges[0] > 0) and (edges[-1] / max(edges[0], 1e-9) > 50)
    for b in range(len(edges) - 1):
        yb = y[idx == b]
        if yb.size < _MIN_BIN:
            continue
        c = np.sqrt(edges[b] * edges[b + 1]) if log else 0.5 * (edges[b] + edges[b + 1])
        boots = yb[_RNG.integers(0, yb.size, size=(_N_BOOT, yb.size))].mean(axis=1)
        cen.append(c); mean.append(float(yb.mean()))
        lo.append(float(np.percentile(boots, 2.5))); hi.append(float(np.percentile(boots, 97.5)))
    return np.array(cen), np.array(mean), np.array(lo), np.array(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--xmode", choices=["contacts", "percent"], default="contacts")
    ap.add_argument("--presets", default=",".join(f"{k}={v}" for k, v in _DEFAULT_PRESETS.items()))
    ap.add_argument("--cap", type=int, default=250)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    presets = dict(p.split("=", 1) for p in args.presets.split(","))
    root = settings().runs_root / args.dataset

    from ecstasy.datasets import load_dataset
    ds = load_dataset(args.dataset)
    entries = {e.id: e for e in ds.entries()}
    gt_cache: dict = {}

    cmap = plt.get_cmap("tab10")
    color = {m: cmap(i % 10) for i, m in enumerate(presets)}
    fig, ax = plt.subplots(figsize=(8.5, 6))

    series, allx, n_partial = [], [], 0
    for model, preset in presets.items():
        run_dir = root / model / preset
        if not (run_dir / "predictions").exists():
            continue
        x, pak = _collect(ds, entries, gt_cache, run_dir, args.xmode, args.cap)
        if x.size:
            partial = not (run_dir / "result.json").exists()
            n_partial += int(partial)
            series.append((model, preset, x, pak, partial)); allx.append(x)
    if not series:
        raise SystemExit(f"no scorable runs under {root} for {presets}")

    allx = np.concatenate(allx)
    if args.xmode == "contacts":
        edges = np.logspace(np.log10(max(1, allx.min())), np.log10(allx.max() + 1), 9)
        ax.set_xscale("log")
        ax.set_xlabel("interface size  K = # true inter-chain contacts  (log scale)")
    else:
        edges = np.linspace(0, np.ceil(allx.max() / 5) * 5, 11)
        ax.set_xlabel("interface size  (% of residues at the inter-chain interface)")

    for model, preset, x, pak, partial in series:
        c = color[model]
        cen, mean, lo, hi = _binned(x, pak, edges)
        if not cen.size:
            continue
        ax.fill_between(cen, lo, hi, color=c, alpha=0.18, zorder=1)
        ax.plot(cen, mean, "-o", color=c, lw=2, ms=4, zorder=3,
                label=f"{model}/{preset} (n={x.size}{', partial' if partial else ''})")

    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(f"per-protein inter-chain P@K  ({args.dataset})")
    title = "Contact-prediction quality vs. interface size"
    title += "  (% of length)" if args.xmode == "percent" else ""
    if n_partial:
        title += f"\n(PREVIEW — {n_partial} run(s) partially scored, in-flight sweep)"
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="line = binned mean · band = 95% CI", frameon=False)

    fig.tight_layout()
    out = args.out or str(root / f"pak_vs_interface_{args.xmode}.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}  ({len(series)} models, {allx.size} proteins, xmode={args.xmode})")


if __name__ == "__main__":
    main()
