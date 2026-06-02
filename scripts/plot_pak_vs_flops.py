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


def _collect(dataset: str, partial_cap: int = 0, exclude: set[str] | None = None) -> list[dict]:
    root = settings().runs_root / dataset
    # Iterate every model/variant dir that has FLOPs sidecars (not just finished runs).
    pts = []
    entries_by_id = None     # lazily built only if partial scoring is needed
    dataset_obj = None
    for pred_dir in sorted(root.glob("*/*/predictions")):
        run_dir = pred_dir.parent
        model, variant = run_dir.parts[-2], run_dir.parts[-1]
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
        if result.exists():
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
        pts.append({
            "model": model, "variant": variant,
            "pak": float(paks.mean()), "pak_lo": lo, "pak_hi": hi, "n_pak": int(paks.size),
            "partial": partial,
            "flops": float(flops.mean()), "n_flops": len(flops),
            "flops_p10": float(np.percentile(flops, 10)),
            "flops_p90": float(np.percentile(flops, 90)),
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


# Human-readable model names for the legend (MSA dependency spelled out).
_DISPLAY_NAME = {
    "boltz2": "Boltz2 (with MSAs)",
    "boltz2_nomsa": "Boltz2 (no MSAs)",
    "esmfold": "ESMFold",
    "mentos": "MENTOS",
    "msa_pairformer": "MSA-Pairformer",
    "colabfold": "ColabFold",
}


def _display_name(model: str) -> str:
    return _DISPLAY_NAME.get(model, model)


# MSA-Pairformer is a genuine single forward (no recycle loop) -> its one preset is r=0.
# MENTOS recycles pair_stack.num_recycles passes: a5sgd6ul_latest = num_recycles 1 (r=1),
# a5sgd6ul_r0 = overridden to 0 (r=0). Map each variant to its true recycle label.
_R0_FULL_MODELS = {"msa_pairformer"}
_MENTOS_LABEL = {"a5sgd6ul_latest": "r=1 full", "a5sgd6ul_r0": "r=0 full"}


def _disp_variant(model: str, variant: str) -> str:
    if model == "mentos":
        return _MENTOS_LABEL.get(variant, variant)
    if model in _R0_FULL_MODELS:
        return "r=0 full"
    return variant


def _is_r0_rep(model: str, variant: str) -> bool:
    """The r=0 representative of a model: its r0 sweep point (incl. MENTOS a5sgd6ul_r0),
    or (for single-forward models) its only preset."""
    return variant in ("r0", "a5sgd6ul_r0") or model in _R0_FULL_MODELS


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
    ap.add_argument("--annotate-r0", action="store_true",
                    help="label each model's r=0 point with its mean P@K and mean T-FLOPs")
    args = ap.parse_args()

    exclude = {m.strip() for m in args.exclude_models.split(",") if m.strip()}
    pts = _collect(args.dataset, partial_cap=args.partial_cap, exclude=exclude)
    if not pts:
        raise SystemExit(f"no runs with flops.json under "
                         f"{settings().runs_root / args.dataset} — run `ecstasy run --profile` first")
    n_partial = sum(p.get("partial", False) for p in pts)

    needs_msa = _msa_dependency()
    models = sorted({p["model"] for p in pts})
    cmap = plt.get_cmap("tab10")
    color = {m: cmap(i % 10) for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for m in models:
        mpts = sorted((p for p in pts if p["model"] == m), key=lambda d: d["flops"])
        xs = [p["flops"] for p in mpts]
        ys = [p["pak"] for p in mpts]
        # connecting line across this model's presets (the compute->quality curve)
        if len(mpts) > 1:
            ax.plot(xs, ys, "-", color=color[m], alpha=0.35, lw=1.2, zorder=1)
        filled = needs_msa.get(m, False)
        for p in mpts:
            ax.errorbar(
                p["flops"], p["pak"],
                yerr=[[p["pak"] - p["pak_lo"]], [p["pak_hi"] - p["pak"]]],
                xerr=[[max(0.0, p["flops"] - p["flops_p10"])], [max(0.0, p["flops_p90"] - p["flops"])]],
                fmt="o", color=color[m], ecolor=color[m], elinewidth=0.8, capsize=2,
                ms=8, mfc=(color[m] if filled else "white"), mec=color[m], mew=1.5, zorder=3,
            )
            vlabel = _disp_variant(m, p["variant"])
            if args.annotate_r0 and _is_r0_rep(m, p["variant"]):
                txt = f"{vlabel}\nP@K {p['pak']:.2f} · {p['flops'] / 1e12:.1f} TFLOP"
                ax.annotate(txt, (p["flops"], p["pak"]), textcoords="offset points",
                            xytext=(7, 5), fontsize=6.5, color=color[m], zorder=5,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                      ec=color[m], lw=0.6, alpha=0.9))
            else:
                ax.annotate(vlabel, (p["flops"], p["pak"]), textcoords="offset points",
                            xytext=(6, 4), fontsize=7, color=color[m])

    # Pareto staircase (upper-left envelope): max P@K achievable at <= given FLOPs
    pf = _pareto(pts)
    if len(pf) > 1:
        ax.step([p["flops"] for p in pf], [p["pak"] for p in pf], where="post",
                color="0.4", ls="--", lw=1.0, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("inference FLOPs (true = 2×MACs, contact-dependency subgraph, log scale)")
    ax.set_ylabel(f"mean inter-chain P@K  ({args.dataset})")
    title = "Contact-prediction quality vs. inference compute"
    if n_partial:
        title += f"\n(PREVIEW — {n_partial} run(s) partially scored, in-flight sweep)"
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # legend: model colors (human-readable) + fill semantics + error-bar meaning
    handles = [plt.Line2D([], [], marker="o", ls="", color=color[m], mfc=color[m],
                          label=_display_name(m)) for m in models]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color="0.3", mfc="0.3", label="filled = uses MSA"),
        plt.Line2D([], [], marker="o", ls="", color="0.3", mfc="white", mec="0.3", label="hollow = single-seq"),
        plt.Line2D([], [], color="0.4", marker="|", markersize=10, mew=1.5, ls="",
                   label="vertical bar = 95% bootstrap CI of mean P@K"),
        plt.Line2D([], [], color="0.4", marker="_", markersize=10, mew=1.5, ls="",
                   label="horizontal bar = 10–90th pct of per-protein FLOPs"),
        plt.Line2D([], [], ls="--", color="0.4", label="Pareto frontier"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="best")

    fig.tight_layout()
    out = args.out or str(settings().runs_root / args.dataset / "pak_vs_flops.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}  ({len(pts)} runs, {len(models)} models)")


if __name__ == "__main__":
    main()
