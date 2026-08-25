"""Predict / score / compare over a (dataset, model, variant).

Output layout — stable and human-readable (no opaque whole-config hash):

  $DATA_ROOT/ecstasy/runs/<dataset>/<model>/<variant>/
      params.json                       # provenance: preset, params, infra, msa
      predictions/<entry_id>/contact.npz
      predictions/<entry_id>/structure.npz   # optional, structure-emitting models only
      result.json                       # scoring summary + per-protein metrics

`variant` is the preset name (e.g. ``full``), or ``<preset>+<sha8>`` when --set
overrides were given. Trivial infra changes never fork the dir.

``contact.npz`` is the required output of every runner; ``structure.npz`` is optional
and additive. When a model emits one AND the dataset carries full-atom ground truth,
scoring also produces DockQ / iRMSD / LRMSD and per-chain monomer metrics. Nothing about
the contact path changes when it is absent, so contact-only models and structure-less
datasets are unaffected.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecstasy.config import settings
from ecstasy.datasets import Dataset, load_dataset
from ecstasy.models import ModelRun, load_model, predict_one
from ecstasy.msa import store

_METRIC_KEYS = ["AUC", "P@K", "P@K/2", "P@K/5"]
#: Structure metrics, aggregated only for runs that produced them.
_STRUCTURE_KEYS = ["DockQ", "Fnat", "iRMSD", "LRMSD", "TM_mean", "TM_min",
                   "CA_RMSD_mean", "null_DockQ_mean", "null_DockQ_max"]


@dataclass(frozen=True)
class Run:
    dataset: Dataset
    model: ModelRun

    @property
    def out_dir(self) -> Path:
        return settings().runs_root / self.dataset.name / self.model.name / self.model.variant

    @property
    def predictions_dir(self) -> Path:
        return self.out_dir / "predictions"

    @property
    def params_path(self) -> Path:
        return self.out_dir / "params.json"

    @property
    def result_path(self) -> Path:
        return self.out_dir / "result.json"

    def write_params(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.params_path.write_text(json.dumps({
            "dataset": self.dataset.name,
            "model": self.model.name,
            "preset": self.model.preset,
            "variant": self.model.variant,
            "msa": self.model.msa,
            "params": self.model.params,
            "infra": self.model.infra,
        }, indent=1, default=str))


def make_run(dataset: str, model: str, preset: str | None = None,
             overrides: dict | None = None, checkpoint: str | None = None) -> Run:
    return Run(dataset=load_dataset(dataset),
               model=load_model(model, preset=preset, overrides=overrides, checkpoint=checkpoint))


def run_predict(run: Run, limit: int | None = None, profile: bool = False,
                shard: str | None = None) -> None:
    run.write_params()
    # shard = "i/N": process only entries with index % N == i (for parallel jobs;
    # combined with the contact.npz skip, shards never collide and are resumable).
    si, sn = 0, 1
    if shard:
        si, sn = (int(x) for x in str(shard).split("/"))
        if not (sn > 0 and 0 <= si < sn):
            raise ValueError(f"--shard must be 'i/N' with 0 <= i < N, got {shard!r}")
    n = 0
    for idx, entry in enumerate(run.dataset.entries()):
        if idx % sn != si:
            continue
        if limit is not None and n >= int(limit):
            break
        n += 1
        out_dir = run.predictions_dir / entry.id
        # In profile mode a prior non-profile run may have left contact.npz without
        # flops.json — don't skip until both exist.
        done = (out_dir / "contact.npz").exists() and (
            not profile or (out_dir / "flops.json").exists())
        if done:
            print(f"[skip] {entry.id} ({'contact.npz+flops.json' if profile else 'contact.npz'} exists)")
            continue
        msa = store.lookup(entry, run.model.msa) if run.model.needs_msa else None
        if run.model.needs_msa and not msa:
            print(f"[warn] {entry.id}: no {run.model.msa} MSA in store, single-sequence fallback",
                  file=sys.stderr)
        print(f"[predict] {entry.id} -> {out_dir/'contact.npz'}")
        try:
            predict_one(run.model, entry, msa, out_dir, profile=profile)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {entry.id}: {e}", file=sys.stderr)
    print(f"\nDone. processed {n} entries -> {run.predictions_dir}")


def run_score(run: Run, limit: int | None = None, null_draws: int = 0) -> None:
    per_protein: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    struct_skipped: list[tuple[str, str]] = []
    n = 0
    for entry in run.dataset.entries():
        if limit is not None and n >= int(limit):
            break
        n += 1
        entry_dir = run.predictions_dir / entry.id
        contact_path = entry_dir / "contact.npz"
        if not contact_path.exists():
            skipped.append((entry.id, "no contact.npz"))
            continue
        try:
            res = run.dataset.score(entry, contact_path)
        except FileNotFoundError as e:
            skipped.append((entry.id, str(e)))
            continue
        except Exception as e:  # noqa: BLE001
            errors.append((entry.id, str(e)))
            continue
        if "_skipped" in res:
            skipped.append((entry.id, res["_skipped"]))
            continue
        if "_error" in res:
            errors.append((entry.id, res["_error"]))
            continue
        res = dict(res)
        # Structure scoring is additive: it runs only when the model emitted a
        # structure AND the dataset has full-atom GT, and a failure there never
        # discards the contact metrics that already succeeded.
        structure_path = entry_dir / "structure.npz"
        if structure_path.exists() and run.dataset.has_structure_gt:
            try:
                s = run.dataset.score_structure(
                    entry, structure_path, work_dir=entry_dir, null_draws=null_draws)
            except Exception as e:  # noqa: BLE001
                struct_skipped.append((entry.id, f"{type(e).__name__}: {e}"))
            else:
                if "_skipped" in s or "_error" in s:
                    struct_skipped.append((entry.id, s.get("_skipped") or s["_error"]))
                else:
                    res.update(s)
        per_protein[entry.id] = {k: float(v) for k, v in res.items()}

    aggregate: dict = {
        "dataset": run.dataset.name, "model": run.model.name, "variant": run.model.variant,
        "summary": {"n_evaluated": len(per_protein), "n_skipped": len(skipped),
                    "n_errors": len(errors)},
        "per_protein": per_protein,
        "skipped_first_20": skipped[:20], "errors_first_20": errors[:20],
    }
    if struct_skipped:
        aggregate["summary"]["n_structure_skipped"] = len(struct_skipped)
        aggregate["structure_skipped_first_20"] = struct_skipped[:20]
    if per_protein:
        aggregate["summary"].update(_aggregate(per_protein.values(), _METRIC_KEYS))
        struct = [v for v in per_protein.values() if "DockQ" in v]
        if struct:
            aggregate["summary"]["structure"] = _structure_summary(struct)

    run.out_dir.mkdir(parents=True, exist_ok=True)
    run.result_path.write_text(json.dumps(aggregate, indent=1))
    s = aggregate["summary"]
    if "mean" in s:
        print(f"[{run.dataset.name}/{run.model.name}/{run.model.variant}] "
              f"n={s['n_evaluated']} mean P@K={s['mean']['P@K']:.3f} "
              f"AUC={s['mean']['AUC']:.3f} skipped={s['n_skipped']} errors={s['n_errors']}")
    if "structure" in s:
        st = s["structure"]
        print(f"  structure n={st['n']} mean DockQ={st['mean']['DockQ']:.3f} "
              f"median={st['median']['DockQ']:.3f} "
              f"iRMSD={st['mean']['iRMSD']:.2f}A LRMSD={st['mean']['LRMSD']:.2f}A "
              f"acceptable={st['acceptable_fraction']:.3f}")
    print(f"result -> {run.result_path}")


def _nan_safe(value: float, fmt: str = ".4f") -> str:
    """Format a float, rendering NaN as an em dash rather than the literal 'nan'."""
    return f"{value:{fmt}}" if value == value else "—"


def _aggregate(rows, keys: list[str]) -> dict:
    """Mean + median per key, over the rows where that key is present and finite."""
    arrs = {k: np.array([v[k] for v in rows
                         if not np.isnan(v.get(k, np.nan))]) for k in keys}
    return {
        "mean": {k: float(arrs[k].mean()) if arrs[k].size else float("nan") for k in keys},
        "median": {k: float(np.median(arrs[k])) if arrs[k].size else float("nan")
                   for k in keys},
    }


def _structure_summary(rows: list[dict]) -> dict:
    """DockQ summary: central tendency, quality bands, and the homo/hetero split.

    The split is not decoration. Under the poly-G-linker multimer hack a homodimer is
    one sequence duplicated around a glycine run, which no language model has seen in
    training; a collapsed or domain-swapped prediction there says something about the
    hack, not about the model's ability to dock. Reporting the pooled mean alone hides
    that.
    """
    from ecstasy.metrics.structure import dockq_bands

    out: dict = {"n": len(rows)}
    out.update(_aggregate(rows, _STRUCTURE_KEYS))
    out.update(dockq_bands([v["DockQ"] for v in rows if not np.isnan(v.get("DockQ", np.nan))]))
    # Only rows that actually carry the flag are split. Defaulting a missing flag to
    # "heterodimer" would report a split that was never measured.
    labelled = [v for v in rows if "is_homodimer" in v]
    homo = [v for v in labelled if v["is_homodimer"] >= 0.5]
    hetero = [v for v in labelled if v["is_homodimer"] < 0.5]
    for label, subset in (("homodimer", homo), ("heterodimer", hetero)):
        if not subset:
            continue
        scores = [v["DockQ"] for v in subset if not np.isnan(v.get("DockQ", np.nan))]
        out[label] = {"n": len(subset), **_aggregate(subset, _STRUCTURE_KEYS),
                      **dockq_bands(scores)}
    return out


def flops_summary(run_dir: Path) -> dict | None:
    """Aggregate per-protein ``flops.json`` sidecars under ``run_dir/predictions``.

    Returns the dataset-mean FLOPs (the x-coordinate paired with mean-P@K) plus
    median and the 10/90 percentiles for the horizontal whisker (plan §3, §4), or
    ``None`` if no sidecars exist. FLOPs are true FLOPs = 2*MACs.
    """
    vals = []
    for fp in (run_dir / "predictions").glob("*/flops.json"):
        try:
            vals.append(float(json.loads(fp.read_text())["flops"]))
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    if not vals:
        return None
    a = np.array(vals)
    return {
        "n_flops": int(a.size),
        "mean_flops": float(a.mean()),
        "median_flops": float(np.median(a)),
        "p10_flops": float(np.percentile(a, 10)),
        "p90_flops": float(np.percentile(a, 90)),
    }


def run_compare(dataset: str) -> None:
    """Aggregate every run's result.json for a dataset into a CSV + Markdown table.

    When a run also has per-protein ``flops.json`` sidecars (from ``run --profile``),
    its dataset-mean inference FLOPs are folded in as the x-axis for the
    P@K-vs-FLOPs plot.
    """
    root = settings().runs_root / dataset
    files = sorted(root.glob("*/*/result.json"))
    if not files:
        print(f"no result.json under {root}", file=sys.stderr)
        return
    rows: list[dict] = []
    for p in files:
        data = json.loads(p.read_text())
        summary = data.get("summary", {})
        if "mean" not in summary:
            continue
        mean, median = summary["mean"], summary.get("median", {})
        fl = flops_summary(p.parent) or {}
        st = summary.get("structure", {})
        st_mean, st_median = st.get("mean", {}), st.get("median", {})
        rows.append({
            "model": data.get("model", p.parts[-3]),
            "variant": data.get("variant", p.parts[-2]),
            "n": summary.get("n_evaluated", 0),
            "skipped": summary.get("n_skipped", 0),
            "errors": summary.get("n_errors", 0),
            "mean_P@K": mean.get("P@K", float("nan")),
            "median_P@K": median.get("P@K", float("nan")),
            "mean_P@K/2": mean.get("P@K/2", float("nan")),
            "mean_P@K/5": mean.get("P@K/5", float("nan")),
            "mean_AUC": mean.get("AUC", float("nan")),
            "mean_flops": fl.get("mean_flops", float("nan")),
            "median_flops": fl.get("median_flops", float("nan")),
            "n_flops": fl.get("n_flops", 0),
            "n_structure": st.get("n", 0),
            "mean_DockQ": st_mean.get("DockQ", float("nan")),
            "median_DockQ": st_median.get("DockQ", float("nan")),
            "mean_iRMSD": st_mean.get("iRMSD", float("nan")),
            "mean_LRMSD": st_mean.get("LRMSD", float("nan")),
            "mean_TM": st_mean.get("TM_mean", float("nan")),
            "acceptable_fraction": st.get("acceptable_fraction", float("nan")),
        })
    if not rows:
        print(f"no result.json with summary metrics under {root}", file=sys.stderr)
        return
    rows.sort(key=lambda r: r["mean_P@K"], reverse=True)

    cols = ["model", "variant", "n", "skipped", "errors",
            "mean_P@K", "median_P@K", "mean_P@K/2", "mean_P@K/5", "mean_AUC",
            "mean_flops", "median_flops", "n_flops",
            "n_structure", "mean_DockQ", "median_DockQ", "mean_iRMSD", "mean_LRMSD",
            "mean_TM", "acceptable_fraction"]
    csv_path = root / "comparison.csv"
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])
                             for c in cols) + "\n")
    md_path = root / "comparison.md"
    with md_path.open("w") as f:
        f.write(f"# {dataset} — interchain contact prediction\n\n")
        f.write(f"Aggregated from {len(rows)} run(s) under `{root}`.\n\n")
        f.write("| model | variant | n | mean P@K | median P@K | P@K/2 | P@K/5 | AUC | "
                "mean GFLOPs | n_flops |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            gflops = r["mean_flops"] / 1e9
            gf = f"{gflops:.1f}" if gflops == gflops else "—"   # NaN-safe
            f.write(f"| {r['model']} | {r['variant']} | {r['n']} | {r['mean_P@K']:.4f} | "
                    f"{r['median_P@K']:.4f} | {r['mean_P@K/2']:.4f} | {r['mean_P@K/5']:.4f} | "
                    f"{r['mean_AUC']:.4f} | {gf} | {r['n_flops']} |\n")

        struct_rows = [r for r in rows if r["n_structure"]]
        if struct_rows:
            struct_rows.sort(key=lambda r: r["mean_DockQ"], reverse=True)
            f.write("\n## Structure (DockQ)\n\n")
            # iRMSD/LRMSD are printed beside DockQ deliberately: DockQ averages fnat
            # with two RMSD terms, so an unformed backbone still scores off fnat alone.
            # A DockQ column read on its own is misleading.
            f.write("| model | variant | n | mean DockQ | median DockQ | iRMSD (Å) | "
                    "LRMSD (Å) | mean TM | acceptable |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in struct_rows:
                f.write(f"| {r['model']} | {r['variant']} | {r['n_structure']} | "
                        f"{_nan_safe(r['mean_DockQ'])} | {_nan_safe(r['median_DockQ'])} | "
                        f"{_nan_safe(r['mean_iRMSD'], '.2f')} | "
                        f"{_nan_safe(r['mean_LRMSD'], '.2f')} | "
                        f"{_nan_safe(r['mean_TM'])} | "
                        f"{_nan_safe(r['acceptable_fraction'], '.3f')} |\n")
    print(f"wrote {csv_path}\nwrote {md_path}\n")
    print(md_path.read_text())
