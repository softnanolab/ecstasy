"""End-to-end contact-prediction benchmark pipeline.

Stages:
  msa     — materialize per-chain a3m files (cached by sha256(seq)[:16]; not yet implemented)
  predict — for each entry, call adapter.predict_one(); writes contact.npz
  score   — for each entry, load contact.npz + GT, compute P@K; aggregate to results JSON
  compare — aggregate results JSONs across models into a CSV/markdown table (TBD)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from ecstasy.benchmarks.base import Benchmark
from ecstasy.models.base import ModelAdapter


def _run_id(cfg: dict) -> str:
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _predict_dir(bench: Benchmark, model: ModelAdapter, run_id: str) -> Path:
    return bench.out_root / "predictions" / model.name / run_id


def _msa_paths_for(entry, msa_root: Path) -> dict[str, Path] | None:
    if not msa_root.exists():
        return None
    out = {}
    for cid, seq in zip(entry.chain_ids, entry.sequences):
        h = hashlib.sha256(seq.encode()).hexdigest()[:16]
        p = msa_root / f"{h}.a3m"
        if p.exists():
            out[cid] = p
    return out or None


def run_msa(cfg: dict, bench: Benchmark, model: ModelAdapter) -> None:
    raise NotImplementedError(
        "MSA pipeline pending. Generate complex MSAs with softnanolab/colabfold-local "
        "and pass `complex_a3m_dir` in the adapter's model_config."
    )


def run_predict(cfg: dict, bench: Benchmark, model: ModelAdapter, submit: bool = False) -> None:
    if submit:
        raise NotImplementedError("--submit is not implemented yet; run inline for the smoke test")
    rid = _run_id(cfg)
    pred_root = _predict_dir(bench, model, rid)
    pred_root.mkdir(parents=True, exist_ok=True)
    msa_root = bench.out_root / "msas"
    limit = (cfg.get("model_config") or {}).get("predict_limit")
    n = 0
    for entry in bench.entries():
        if limit is not None and n >= int(limit):
            break
        out_dir = pred_root / entry.id
        contact_path = out_dir / "contact.npz"
        if contact_path.exists():
            print(f"[skip] {entry.id} (contact.npz exists)")
            n += 1
            continue
        msa_paths = _msa_paths_for(entry, msa_root) if model.needs_msa else None
        if model.needs_msa and not msa_paths:
            print(f"[warn] {entry.id}: no MSAs found, falling back to single-sequence", file=sys.stderr)
        print(f"[predict] {entry.id} -> {contact_path}")
        try:
            model.predict_one(entry, msa_paths, out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {entry.id}: {e}", file=sys.stderr)
        n += 1
    print(f"\nDone. processed {n} entries -> {pred_root}")


def run_score(cfg: dict, bench: Benchmark, model: ModelAdapter) -> None:
    rid = _run_id(cfg)
    pred_root = _predict_dir(bench, model, rid)
    results_root = bench.out_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    per_protein: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []

    limit = (cfg.get("model_config") or {}).get("predict_limit")
    n = 0
    for entry in bench.entries():
        if limit is not None and n >= int(limit):
            break
        n += 1
        contact_path = pred_root / entry.id / "contact.npz"
        if not contact_path.exists():
            skipped.append((entry.id, "no contact.npz"))
            continue
        try:
            res = bench.score(entry, contact_path)
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
        per_protein[entry.id] = {k: float(v) for k, v in res.items()}

    keys = ["AUC", "P@K", "P@K/2", "P@K/5"]
    aggregate: dict = {
        "summary": {
            "n_evaluated": len(per_protein),
            "n_skipped": len(skipped),
            "n_errors": len(errors),
        },
        "per_protein": per_protein,
        "skipped_first_20": skipped[:20],
        "errors_first_20": errors[:20],
    }
    if per_protein:
        arrs = {k: np.array([v[k] for v in per_protein.values() if not np.isnan(v.get(k, np.nan))]) for k in keys}
        aggregate["summary"]["mean"] = {k: float(arrs[k].mean()) if arrs[k].size else float("nan") for k in keys}
        aggregate["summary"]["median"] = {k: float(np.median(arrs[k])) if arrs[k].size else float("nan") for k in keys}

    out = results_root / f"{bench.name}__{model.name}__{rid}.json"
    out.write_text(json.dumps(aggregate, indent=1))
    s = aggregate["summary"]
    if "mean" in s:
        print(f"[{out.name}]  n={s['n_evaluated']}  mean P@K={s['mean']['P@K']:.3f}  "
              f"median P@K={s['mean'].get('P@K', float('nan')):.3f}  "
              f"AUC={s['mean']['AUC']:.3f}  skipped={s['n_skipped']}  errors={s['n_errors']}")
    print(f"results -> {out}")


def run_compare(task: str, data_root: Path) -> None:
    """Aggregate results/<task>__<model>__<run_id>.json into a single table.

    Writes both CSV and a markdown summary into the same results/ dir.
    """
    results_root = Path(data_root) / "ecstasy" / "benchmarks" / task / "results"
    files = sorted(results_root.glob(f"{task}__*.json"))
    if not files:
        print(f"no result files under {results_root}", file=sys.stderr)
        return

    rows: list[dict] = []
    for p in files:
        data = json.loads(p.read_text())
        summary = data.get("summary", {})
        if not summary or "mean" not in summary:
            continue
        # filename: <task>__<model>__<run_id>.json
        parts = p.stem.split("__")
        if len(parts) < 3:
            continue
        model, run_id = parts[1], parts[2]
        mean = summary["mean"]
        median = summary.get("median", {})
        rows.append({
            "model": model,
            "run_id": run_id,
            "n": summary.get("n_evaluated", 0),
            "skipped": summary.get("n_skipped", 0),
            "errors": summary.get("n_errors", 0),
            "mean_P@K": mean.get("P@K", float("nan")),
            "median_P@K": median.get("P@K", float("nan")),
            "mean_P@K/2": mean.get("P@K/2", float("nan")),
            "mean_P@K/5": mean.get("P@K/5", float("nan")),
            "mean_AUC": mean.get("AUC", float("nan")),
        })

    if not rows:
        print(f"no result files with summary metrics under {results_root}", file=sys.stderr)
        return

    rows.sort(key=lambda r: r["mean_P@K"], reverse=True)

    csv_path = results_root / f"{task}__comparison.csv"
    cols = ["model", "run_id", "n", "skipped", "errors",
            "mean_P@K", "median_P@K", "mean_P@K/2", "mean_P@K/5", "mean_AUC"]
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_fmt_csv(r[c]) for c in cols) + "\n")

    md_path = results_root / f"{task}__comparison.md"
    with md_path.open("w") as f:
        f.write(f"# {task} — interchain contact prediction\n\n")
        f.write(f"Aggregated from {len(rows)} result file(s) under `{results_root}`.\n\n")
        f.write("| model | n | mean P@K | median P@K | P@K/2 | P@K/5 | AUC |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['model']} ({r['run_id']}) | {r['n']} | "
                f"{r['mean_P@K']:.4f} | {r['median_P@K']:.4f} | "
                f"{r['mean_P@K/2']:.4f} | {r['mean_P@K/5']:.4f} | {r['mean_AUC']:.4f} |\n"
            )
    print(f"wrote {csv_path}\nwrote {md_path}")
    print()
    print(md_path.read_text())


def _fmt_csv(v) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _load_config_path(p: str) -> dict:
    return yaml.safe_load(Path(p).read_text())
