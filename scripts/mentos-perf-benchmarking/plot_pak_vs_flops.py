"""P@K vs. inference-FLOPs plot (the deliverable; see FLOPS_BENCHMARK_PLAN.md §4).

Reads every run under ``$DATA_ROOT/runs/<dataset>/<model>/<variant>/`` that has
both a ``result.json`` (per-protein P@K) and per-protein ``flops.json`` sidecars
(from ``ecstasy run --profile``), and draws:

  * x = log10(mean inference FLOPs), y = mean P@K
  * one marker per (model, variant); same-model presets joined by a faint line
    (the architecture's own compute->quality curve — the whole point of the FLOPs
    reframing over params)
  * color = model family; marker fill = MSA dependency (filled = uses MSA)
  * vertical bars = 95% bootstrap CI of P@K over proteins
  * horizontal whisker = 10-90th percentile of per-protein FLOPs
  * dashed Pareto staircase = max P@K achievable at <= given FLOPs

Usage:
  python scripts/plot_pak_vs_flops.py --dataset val_pinder_pair [--out plot.png]
  (run from an env with ecstasy + matplotlib, e.g. .venv-boltz)
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

use_cmu_concrete()  # noqa: E402

# ~2x larger type than matplotlib defaults so the figures read at thumbnail size.
plt.rcParams.update({
    "font.size": 20,          # base (was ~10)
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
})

from ecstasy.config import settings  # noqa: E402
from ecstasy.models.registry import load_model, model_names  # noqa: E402

_RNG_SEED = 0          # deterministic bootstrap
_N_BOOT = 2000


def _msa_dependency() -> dict[str, bool]:
    out = {}
    for m in model_names():
        try:
            out[m] = load_model(m).needs_msa
        except Exception:        # noqa: BLE001
            out[m] = False
    return out


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        v = float(values.mean()) if values.size else float("nan")
        return v, v
    rng = np.random.default_rng(_RNG_SEED)
    means = values[rng.integers(0, values.size, size=(_N_BOOT, values.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _collect(dataset: str, partial_cap: int = 0, exclude: set[str] | None = None,
             tolerance: int = 0, topk_divisor: int = 1,
             include: set[str] | None = None) -> list[dict]:
    root = settings().runs_root / dataset
    # Iterate every model/variant dir that has FLOPs sidecars (not just finished runs).
    pts = []
    entries_by_id = None     # lazily built only if partial scoring is needed
    dataset_obj = None
    gt_cache: dict = {}      # id -> gt_for(...) cache, reused across runs (tolerance mode)
    for pred_dir in sorted(root.glob("*/*/predictions")):
        run_dir = pred_dir.parent
        model, variant = run_dir.parts[-2], run_dir.parts[-1]
        if model == "mentos" and variant not in _MENTOS_LABEL:
            continue        # show only labelled mentos variants (headline + named sweeps)
        # Split a named recycle-sweep checkpoint into its own series so its r0..r5 ladder
        # draws as one clean line instead of tangling with the headline mentos points.
        if model == "mentos" and variant.startswith("s0xlqidn"):
            model = "mentos_s0xlqidn"
        if include and model not in include:
            continue
        if exclude and model in exclude:
            continue

        flops = []
        for fp in pred_dir.glob("*/flops.json"):
            try:
                flops.append(float(json.loads(fp.read_text())["flops"]))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        if not flops:
            continue

        result = run_dir / "result.json"
        partial = False
        if tolerance > 0 or topk_divisor > 1:
            # Rescore saved predictions (no inference, no result.json) when a spatial
            # tolerance and/or a non-unit top-K divisor (P@K/n) is requested — neither
            # is precomputed in result.json.
            if dataset_obj is None:
                from ecstasy.datasets import load_dataset
                dataset_obj = load_dataset(dataset)
            paks = _tolerant_paks(dataset_obj, gt_cache, pred_dir, tolerance, topk_divisor)
        elif result.exists():
            per = json.loads(result.read_text()).get("per_protein", {})
            paks = np.array([v["P@K"] for v in per.values()
                             if isinstance(v.get("P@K"), (int, float)) and v["P@K"] == v["P@K"]])
        elif partial_cap:
            # No result.json yet (run still in flight): score available predictions
            # in-memory, capped, WITHOUT writing anything (don't disturb the live job).
            if dataset_obj is None:
                from ecstasy.datasets import load_dataset
                dataset_obj = load_dataset(dataset)
                entries_by_id = {e.id: e for e in dataset_obj.entries()}
            paks, partial = _score_partial(dataset_obj, entries_by_id, pred_dir, partial_cap), True
        else:
            continue
        if not paks.size or not flops:
            continue

        flops = np.array(flops)
        lo, hi = _bootstrap_ci(paks)
        # x-position is the MEDIAN per-protein FLOPs (the distribution is right-skewed in
        # complex length; median + IQR is a matched, skew-robust summary so the marker sits
        # centred in its whisker). The horizontal bar is the IQR (p25-p75) spread across
        # complexes — dispersion, NOT a sampling CI (per-protein FLOPs is deterministic in L).
        pts.append({
            "model": model, "variant": variant,
            "pak": float(paks.mean()), "pak_lo": lo, "pak_hi": hi, "n_pak": int(paks.size),
            "partial": partial,
            "flops": float(np.median(flops)), "n_flops": len(flops),
            "flops_lo": float(np.percentile(flops, 25)),
            "flops_hi": float(np.percentile(flops, 75)),
        })
    return pts


def _score_partial(dataset_obj, entries_by_id, pred_dir, cap: int) -> np.ndarray:
    """Score up to ``cap`` available predictions in-memory; return P@K array."""
    paks = []
    for entry_dir in sorted(pred_dir.glob("*")):
        if len(paks) >= cap:
            break
        cpath = entry_dir / "contact.npz"
        entry = entries_by_id.get(entry_dir.name)
        if entry is None or not cpath.exists():
            continue
        try:
            res = dataset_obj.score(entry, cpath)
        except Exception:        # noqa: BLE001
            continue
        v = res.get("P@K")
        if isinstance(v, (int, float)) and v == v:
            paks.append(float(v))
    return np.array(paks)


from scipy.ndimage import binary_dilation as _binary_dilation  # noqa: E402

_TOL_ST = np.ones((3, 3), bool)  # Chebyshev neighbourhood for spatial tolerance


def _tol_inter_pak(cp_full: np.ndarray, gt: dict, tol: int, divisor: int = 1) -> float | None:
    """Tolerant inter-chain P@(K/divisor) from a saved (L,L) contact-prob map.

    K = #true inter contacts; the metric scores the top ``max(1, round(K/divisor))``
    predicted inter pairs (divisor=1 → P@K, divisor=5 → P@K/5, the stricter
    high-confidence precision). A top prediction counts as correct if a true inter
    contact lies within Chebyshev-``tol`` in (chainA-res, chainB-res) space (GT
    dilated by ``tol``); ``tol=0`` is exact. Returns None when undefined
    (non-dimer / shape mismatch / no true inter contacts).
    """
    seqs = gt["sequences"]
    if len(seqs) != 2:
        return None
    la, lb = len(seqs[0]), len(seqs[1])
    if cp_full.shape[0] != la + lb:
        return None
    cp = cp_full[:la, la:]
    gti = gt["contact_map"][:la, la:]
    vi = gt["valid"][:la, la:]
    K = int((gti & vi).sum())
    if K == 0:
        return None
    topk = max(1, int(round(K / divisor)))
    dil = (_binary_dilation(gti, _TOL_ST, iterations=tol) if tol > 0 else gti) & vi
    order = np.argsort(-np.where(vi, cp, -1.0).ravel())[:topk]
    return float(dil.ravel()[order].sum()) / topk


def _tolerant_paks(dataset_obj, gt_cache: dict, pred_dir: Path, tol: int,
                   divisor: int = 1) -> np.ndarray:
    """Per-protein tolerant inter P@(K/divisor) for one run, scored from saved
    contact.npz (no inference). GT is cached across runs since it's shared."""
    paks = []
    for entry_dir in sorted(pred_dir.glob("*")):
        cpath = entry_dir / "contact.npz"
        if not cpath.exists():
            continue
        gt = gt_cache.get(entry_dir.name)
        if gt is None:
            try:
                gt = dataset_obj.gt_for(entry_dir.name)
            except Exception:        # noqa: BLE001
                continue
            gt_cache[entry_dir.name] = gt
        try:
            cp_full = np.load(cpath)["probs"].astype(np.float64)
        except Exception:        # noqa: BLE001
            continue
        v = _tol_inter_pak(cp_full, gt, tol, divisor)
        if v is not None:
            paks.append(v)
    return np.array(paks)


# Human-readable model names for the legend (MSA dependency spelled out).
_DISPLAY_NAME = {
    "boltz2": "Boltz2 (with MSAs)",
    "boltz2_nomsa": "Boltz2 (no MSAs)",
    "esmfold": "ESMFold",
    "mentos": "MENTOS-188M",
    "mentos_s0xlqidn": "MENTOS-150M (recycle 0→5)",
    "mentos_35m": "MENTOS-43M",
    "mentos_43m": "MENTOS-43M",
    "msa_pairformer": "MSA-Pairformer",
    "esm2_650m": "ESM2-650M",
    "esm2_150m": "ESM2-150M",
    "plmgraph_inter": "PLMGraph-Inter",
    "colabfold": "ColabFold",
}


def _display_name(model: str) -> str:
    return _DISPLAY_NAME.get(model, model)


# MSA-Pairformer is a genuine single forward (no recycle loop) -> its one preset is r=0.
# MENTOS recycles pair_stack.num_recycles passes. We display only the headline step-90000
# checkpoint (best by the val_seq_pair sweep): s90k = num_recycles 1 (r=1), s90k_r0 = 0 (r=0).
# Other mentos variants (older checkpoints) are skipped in _collect.
_R0_FULL_MODELS = {"msa_pairformer"}
_MENTOS_LABEL = {
    # 188M was trained with recycling=1, so r0 is off-distribution — show only r1
    # (its trained operating point). The 43M baseline (trained at r0) comes via --extra-points.
    "a5sgd6ul_s90k": "r1",
    # s0xlqidn = 150M recycle_0to5 run (step-55000), evaluated as a recycle ladder.
    "s0xlqidn_r0": "r0", "s0xlqidn_r1": "r1", "s0xlqidn_r3": "r3", "s0xlqidn_r5": "r5",
}

# Friendly dataset titles (the plot title is just the dataset name).
_DATASET_TITLE = {
    "val_seq_pair": "V1: Sequence Deleak Dataset",
    "val_pinder_pair": "V2: Interface Deleak Dataset",
}


def _disp_variant(model: str, variant: str) -> str:
    if model.startswith("mentos"):
        return _MENTOS_LABEL.get(variant, variant)
    if model in _R0_FULL_MODELS:
        return "r=0 full"
    return variant


def _is_r0_rep(model: str, variant: str) -> bool:
    """The r=0 representative of a model: its r0 sweep point (incl. MENTOS s90k_r0),
    or (for single-forward models) its only preset."""
    return variant in ("r0", "a5sgd6ul_s90k_r0") or model in _R0_FULL_MODELS


def _pareto(pts: list[dict]) -> list[dict]:
    """Upper-left frontier: lowest-FLOPs points whose P@K is not beaten by a cheaper one."""
    best = -np.inf
    out = []
    for p in sorted(pts, key=lambda d: d["flops"]):
        if p["pak"] > best:
            out.append(p)
            best = p["pak"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--partial-cap", type=int, default=0,
                    help="if >0, score up to N available predictions in-memory for runs whose "
                         "result.json isn't written yet (PREVIEW of an in-flight sweep)")
    ap.add_argument("--exclude-models", default="",
                    help="comma-separated model names to omit (e.g. an in-flight series)")
    ap.add_argument("--models", default="",
                    help="comma-separated whitelist of model names to include (others dropped); "
                         "empty = all discovered models")
    ap.add_argument("--annotate-r0", action="store_true",
                    help="label each model's r=0 point with its mean P@K and mean T-FLOPs")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="spatial tolerance (residues) for inter P@K: a top-K prediction counts if a "
                         "true inter contact is within this Chebyshev distance. 0 = exact (reads result.json); "
                         ">0 rescores saved predictions (e.g. 2 = ±2 nearby pairs)")
    ap.add_argument("--topk-divisor", type=int, default=1,
                    help="score the top K/divisor predictions instead of top K (K = #true inter "
                         "contacts). 1 = P@K (default); 5 = P@K/5, the stricter high-confidence "
                         "precision. >1 rescores saved predictions (not precomputed in result.json)")
    ap.add_argument("--extra-points", default=None,
                    help="path to a JSON list/dict of precomputed marker dicts (model, variant, pak, "
                         "pak_lo, pak_hi, flops, flops_lo, flops_hi) to inject — e.g. a model scored "
                         "outside this script's GT path (subject to --models filtering)")
    args = ap.parse_args()
    if args.topk_divisor < 1:
        raise SystemExit("--topk-divisor must be >= 1")

    exclude = {m.strip() for m in args.exclude_models.split(",") if m.strip()}
    include = {m.strip() for m in args.models.split(",") if m.strip()}
    pts = _collect(args.dataset, partial_cap=args.partial_cap, exclude=exclude,
                   tolerance=args.tolerance, topk_divisor=args.topk_divisor, include=include)
    metric = "P@K" if args.topk_divisor == 1 else f"P@K/{args.topk_divisor}"

    if args.extra_points:
        # one or more JSON files (comma-separated). Each is a list of marker dicts, or a
        # {dataset: marker} map (self-filtering: a per-split map only injects on its split).
        # Injected points are explicit, so they bypass --models/--exclude-models filtering;
        # curate one file per plot (or use per-split maps) to control where they appear.
        for path in (p.strip() for p in args.extra_points.split(",") if p.strip()):
            extra = json.loads(Path(path).read_text())
            if isinstance(extra, dict):
                extra = [extra[args.dataset]] if args.dataset in extra else []
            for pt in extra:
                pts.append({**pt, "partial": pt.get("partial", False)})
    if not pts:
        raise SystemExit(f"no runs with flops.json under "
                         f"{settings().runs_root / args.dataset} — run `ecstasy run --profile` first")
    n_partial = sum(p.get("partial", False) for p in pts)

    needs_msa = _msa_dependency()
    # injected (extra-points) markers can declare their own MSA dependency for fill semantics
    for p in pts:
        if "needs_msa" in p:
            needs_msa[p["model"]] = p["needs_msa"]
    models = sorted({p["model"] for p in pts})
    cmap = plt.get_cmap("tab10")
    color = {m: cmap(i % 10) for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(11, 8.25))

    for m in models:
        mpts = sorted((p for p in pts if p["model"] == m), key=lambda d: d["flops"])
        xs = [p["flops"] for p in mpts]
        ys = [p["pak"] for p in mpts]
        # connecting line across this model's presets (the compute->quality curve)
        if len(mpts) > 1:
            ax.plot(xs, ys, "-", color=color[m], alpha=0.35, lw=2.6, zorder=1)
        filled = needs_msa.get(m, False)
        for p in mpts:
            ax.errorbar(
                p["flops"], p["pak"],
                yerr=[[p["pak"] - p["pak_lo"]], [p["pak_hi"] - p["pak"]]],
                xerr=[[max(0.0, p["flops"] - p["flops_lo"])], [max(0.0, p["flops_hi"] - p["flops"])]],
                fmt="o", color=color[m], ecolor=color[m], elinewidth=1.8, capsize=5,
                ms=16, mfc=(color[m] if filled else "white"), mec=color[m], mew=2.8, zorder=3,
            )
            vlabel = _disp_variant(m, p["variant"])
            if args.annotate_r0 and _is_r0_rep(m, p["variant"]):
                txt = f"{vlabel}\nP@K {p['pak']:.2f} · {p['flops'] / 1e12:.1f} TFLOP"
                ax.annotate(txt, (p["flops"], p["pak"]), textcoords="offset points",
                            xytext=(10, 7), fontsize=13, color=color[m], zorder=5,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                      ec=color[m], lw=1.0, alpha=0.9))
            else:
                ax.annotate(vlabel, (p["flops"], p["pak"]), textcoords="offset points",
                            xytext=(9, 6), fontsize=14, color=color[m])

    # Pareto staircase (upper-left envelope): max P@K achievable at <= given FLOPs
    pf = _pareto(pts)
    if len(pf) > 1:
        ax.step([p["flops"] for p in pf], [p["pak"] for p in pf], where="post",
                color="0.4", ls="--", lw=2.2, zorder=2)

    tol_tag = f" (±{args.tolerance} Residue Tolerance)" if args.tolerance > 0 else ""
    ax.set_xscale("log")
    ax.set_xlabel("Median Inference FLOPs")
    ax.set_ylabel(f"Mean Inter-chain {metric}{tol_tag}")
    title = _DATASET_TITLE.get(args.dataset, args.dataset)
    if n_partial:
        title += f"  (PREVIEW — {n_partial} run(s) partially scored)"
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # legend: model colors (human-readable) + fill semantics + error-bar meaning
    handles = [plt.Line2D([], [], marker="o", ls="", color=color[m], mfc=color[m],
                          label=_display_name(m)) for m in models]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color="0.3", mfc="0.3", label="filled = uses MSA"),
        plt.Line2D([], [], marker="o", ls="", color="0.3", mfc="white", mec="0.3", label="hollow = single-seq"),
    ]
    ax.legend(handles=handles, loc="best", markerscale=1.6)

    fig.tight_layout()
    _suffix = (f"_pakdiv{args.topk_divisor}" if args.topk_divisor > 1 else "") + \
              (f"_tol{args.tolerance}" if args.tolerance > 0 else "")
    _fname = f"pak_vs_flops{_suffix}.png"
    out = args.out or str(settings().runs_root / args.dataset / _fname)
    fig.savefig(out, dpi=160)
    fig.savefig(str(Path(out).with_suffix(".pdf")))   # vector copy for the report
    print(f"wrote {out} (+pdf)  ({len(pts)} runs, {len(models)} models)")


if __name__ == "__main__":
    main()
