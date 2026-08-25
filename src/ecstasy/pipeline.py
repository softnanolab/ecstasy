"""Predict / score / compare over a (dataset, model, variant).

Output layout — stable and human-readable (no opaque whole-config hash):

  $DATA_ROOT/ecstasy/runs/<dataset>/<model>/<variant>/
      params.json                       # provenance: preset, params, infra, msa
      predictions/<entry_id>/contact.npz
      result.json                       # scoring summary + per-protein metrics

`variant` is the preset name (e.g. ``full``), or ``<preset>+<sha8>`` when --set
overrides were given. Trivial infra changes never fork the dir.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecstasy.config import settings
from ecstasy.datasets import Dataset, load_dataset
from ecstasy.metrics import DEFAULT_CONTACT_METRICS
from ecstasy.models import ModelRun, load_model, predict_one
from ecstasy.msa import store

#: Columns the comparison table reports. The scored metric set is per-run and lives in
#: result.json["metrics"]; this is only what `compare` puts in its fixed columns.
_METRIC_KEYS = ["AUC", "P@K", "P@K/2", "P@K/5"]


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

    @property
    def provenance_path(self) -> Path:
        return self.out_dir / "provenance.json"

    def write_params(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.params_path.write_text(json.dumps({
            "dataset": self.dataset.name,
            "dataset_version": self.dataset.version,
            "model": self.model.name,
            "preset": self.model.preset,
            "variant": self.model.variant,
            "msa": self.model.msa,
            "params": self.model.params,
            "infra": self.model.infra,
        }, indent=1, default=str))

    @property
    def prediction_fp_path(self) -> Path:
        return self.out_dir / "prediction_fingerprint.json"

    @property
    def scoring_fp_path(self) -> Path:
        return self.out_dir / "scoring_fingerprint.json"

    def check_prediction_fingerprint(self, force: bool = False) -> dict:
        """Refuse to reuse a run directory whose predictions came from different inputs.

        This is what makes the ``contact.npz`` skip safe. Without it, bumping a
        dependency silently mixes old predictions into a new run — the persisted version
        of the same hazard as editing ``src/`` mid-sweep.

        ``force`` recomputes in place and re-stamps the fingerprint; it is the deliberate
        escape hatch, never the default.
        """
        from ecstasy import fingerprint as fp

        current = fp.make("prediction", fp.prediction_inputs(self.model, self.dataset))
        previous = fp.load(self.prediction_fp_path)
        if previous and previous.get("digest") != current["digest"]:
            diffs = fp.compare(previous, current)
            if not force:
                raise fp.FingerprintMismatch("prediction", diffs, self.out_dir)
            print(f"[force] prediction inputs changed ({len(diffs)} difference(s)); "
                  f"recomputing in place")
            for d in diffs[:8]:
                print(f"    {d}")
            # Stale predictions must go, or --force would silently keep exactly the
            # mixture it was invoked to resolve.
            for old in self.predictions_dir.glob("*/contact.npz"):
                old.unlink()
        fp.save(self.prediction_fp_path, current)
        return current

    def check_scoring_fingerprint(self, metrics) -> dict:
        """Record what the scores were computed from. Scoring is cheap, so a change here
        never blocks — it simply re-scores and re-stamps."""
        from ecstasy import fingerprint as fp

        current = fp.make("scoring", fp.scoring_inputs(self.dataset, metrics))
        previous = fp.load(self.scoring_fp_path)
        if previous and previous.get("digest") != current["digest"]:
            print(f"[rescore] scoring inputs changed: "
                  f"{'; '.join(fp.compare(previous, current)[:4])}")
        fp.save(self.scoring_fp_path, current)
        return current

    def write_provenance(self) -> dict:
        """Record which code and which bytes produced this run.

        Without it two different experiments serialise identically: the MiniFold runner
        takes ``minifold_src`` as a *path*, and whether the residx patch is applied inside
        that tree is the whole difference between the intended chain break and the
        linker-only variant. ``params.json`` records only the path. This records the
        commit and the dirty flag, so the two stop being indistinguishable.
        """
        from ecstasy import provenance

        rec = provenance.capture(params=self.model.params, env=self.model.env)
        rec["dataset"] = {**self.dataset.manifest(), "fingerprint": self.dataset.fingerprint()}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_path.write_text(json.dumps(rec, indent=1, default=str))
        return rec


def make_run(dataset: str, model: str, preset: str | None = None,
             overrides: dict | None = None, checkpoint: str | None = None) -> Run:
    return Run(dataset=load_dataset(dataset),
               model=load_model(model, preset=preset, overrides=overrides, checkpoint=checkpoint))


def run_predict(run: Run, limit: int | None = None, profile: bool = False,
                shard: str | None = None, force: bool = False) -> None:
    # Gate BEFORE anything is written. The cached-prediction skip below is only safe if
    # the cache was produced by the same inputs; otherwise this run would reuse old
    # predictions under a new provenance record and look entirely normal.
    run.check_prediction_fingerprint(force=force)
    run.write_params()
    prov = run.write_provenance()
    from ecstasy import provenance as _prov
    print(f"[provenance] {_prov.summarise(prov)}")
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


def run_score(run: Run, limit: int | None = None,
              metrics: tuple[str, ...] | None = None,
              allow_partial: bool = False) -> None:
    metrics = tuple(metrics) if metrics else DEFAULT_CONTACT_METRICS
    scoring_fp = run.check_scoring_fingerprint(metrics)
    per_protein: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    n = 0
    for entry in run.dataset.entries():
        if limit is not None and n >= int(limit):
            break
        n += 1
        contact_path = run.predictions_dir / entry.id / "contact.npz"
        if not contact_path.exists():
            skipped.append((entry.id, "no contact.npz"))
            continue
        try:
            res = run.dataset.score(entry, contact_path, metrics=metrics)
        except FileNotFoundError as e:
            skipped.append((entry.id, str(e)))
            continue
        except Exception as e:  # noqa: BLE001
            errors.append((entry.id, str(e)))
            continue
        if "_skipped" in res:
            skipped.append((entry.id, res["_skipped"]))
        elif "_error" in res:
            errors.append((entry.id, res["_error"]))
        else:
            per_protein[entry.id] = {k: float(v) for k, v in res.items()}

    # Coverage is scoped to what was actually asked for: with --limit N the run only ever
    # intended N targets, so completeness is measured against that, not the whole split.
    n_intended = int(limit) if limit is not None else run.dataset.coverage()["n_entries"]
    covered = len(per_protein) / n_intended if n_intended else 0.0
    complete = len(per_protein) == n_intended

    aggregate: dict = {
        "dataset": run.dataset.name, "model": run.model.name, "variant": run.model.variant,
        "metrics": list(metrics),
        "scoring_fingerprint": scoring_fp["digest"],
        "prediction_fingerprint": (fingerprint_load(run) or {}).get("digest"),
        "coverage": {"n_evaluated": len(per_protein), "n_intended": n_intended,
                     "fraction": covered, "complete": complete,
                     "limit": limit},
        "summary": {"n_evaluated": len(per_protein), "n_skipped": len(skipped),
                    "n_errors": len(errors)},
        "per_protein": per_protein,
        "skipped_first_20": skipped[:20], "errors_first_20": errors[:20],
    }
    if not complete and not allow_partial:
        # A mean over 8% of a split prints identically to a mean over all of it. Refusing
        # to compute the headline is the only reliable way to stop that being quoted; the
        # per-protein numbers are still written, so nothing is lost.
        aggregate["summary"]["partial"] = True
        aggregate["summary"]["partial_reason"] = (
            f"scored {len(per_protein)}/{n_intended} targets ({covered:.1%}). No headline "
            f"mean was computed. Re-run with allow_partial=True to publish it as partial, "
            f"or supply the missing predictions/ground truth.")
        run.out_dir.mkdir(parents=True, exist_ok=True)
        run.result_path.write_text(json.dumps(aggregate, indent=1))
        print(f"[partial] {aggregate['summary']['partial_reason']}")
        print(f"result -> {run.result_path}")
        return
    if not complete:
        aggregate["summary"]["partial"] = True
        print(f"[warn] partial result accepted: {len(per_protein)}/{n_intended} "
              f"({covered:.1%}) — summary means are over the scored subset only")
    # Provenance travels WITH the result, not only beside it. A result.json that outlives
    # its run directory still names the code and the split that produced it.
    if run.provenance_path.exists():
        aggregate["provenance"] = json.loads(run.provenance_path.read_text())
    else:
        aggregate["provenance"] = run.write_provenance()
    if per_protein:
        # Aggregate whatever was actually computed, rather than a fixed key list — that is
        # what lets a run request P@K(tol=2) and have it summarised without a code change.
        keys = [k for k in metrics if any(k in v for v in per_protein.values())]
        arrs = {k: np.array([v[k] for v in per_protein.values()
                             if not np.isnan(v.get(k, np.nan))]) for k in keys}
        aggregate["summary"]["mean"] = {k: float(arrs[k].mean()) if arrs[k].size else float("nan")
                                        for k in keys}
        aggregate["summary"]["median"] = {k: float(np.median(arrs[k])) if arrs[k].size else float("nan")
                                          for k in keys}

    run.out_dir.mkdir(parents=True, exist_ok=True)
    run.result_path.write_text(json.dumps(aggregate, indent=1))
    s = aggregate["summary"]
    if "mean" in s:
        headline = " ".join(f"{k}={s['mean'][k]:.3f}" for k in list(s["mean"])[:4])
        print(f"[{run.dataset.name}/{run.model.name}/{run.model.variant}] "
              f"n={s['n_evaluated']} {headline} "
              f"skipped={s['n_skipped']} errors={s['n_errors']}")
    print(f"result -> {run.result_path}")


def fingerprint_load(run: Run) -> dict | None:
    """The prediction fingerprint stamped on this run, if it has one."""
    from ecstasy import fingerprint as fp
    return fp.load(run.prediction_fp_path)


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
        })
    if not rows:
        print(f"no result.json with summary metrics under {root}", file=sys.stderr)
        return
    rows.sort(key=lambda r: r["mean_P@K"], reverse=True)

    cols = ["model", "variant", "n", "skipped", "errors",
            "mean_P@K", "median_P@K", "mean_P@K/2", "mean_P@K/5", "mean_AUC",
            "mean_flops", "median_flops", "n_flops"]
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
    print(f"wrote {csv_path}\nwrote {md_path}\n")
    print(md_path.read_text())
