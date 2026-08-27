"""The committed benchmark record: one JSONL line per published run.

Results lived only in ``$DATA_ROOT`` — machine-local, gitignored, and copied into Notion
by hand. So there was no versioned record of a benchmark number, no way to see in a PR
that a change had moved one, and no way for an agent with no token and no network to
find out what had already been benchmarked.

``results/runs.jsonl`` is that record. One line per published run, appended, never
rewritten. A dependency bump writes a **new line** rather than editing the old one, so
``git log -p results/runs.jsonl`` shows a number moving *and* what changed underneath it.

Summaries only. Per-protein detail stays in ``$DATA_ROOT`` beside the predictions — the
repo keeps numbers you can diff, not blobs.

Publishing is deliberate (:func:`publish` is only ever called by ``ecstasy publish``).
A ``--limit 1`` smoke and an abandoned experiment must not silently become the record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Committed, and deliberately in the repo rather than $DATA_ROOT: the point is that it
#: is versioned and readable without a network.
STORE = _REPO_ROOT / "results" / "runs.jsonl"

#: Bumped when a line's shape changes in a way a reader cannot handle.
SCHEMA_VERSION = 1


class PublishRefused(RuntimeError):
    """A run was not fit to publish. The message says what and how to override."""


@dataclass(frozen=True)
class Key:
    """What makes a published row distinct.

    Both fingerprints, because they answer different questions: two runs of the same
    model on the same dataset are the *same result* only if the inputs to prediction AND
    the inputs to scoring were both unchanged. Re-scoring after a metric fix is a new
    row against identical predictions — which is exactly the history worth keeping.
    """

    dataset: str
    model: str
    variant: str
    prediction_fp: str
    scoring_fp: str

    @classmethod
    def from_record(cls, rec: dict) -> "Key":
        fps = rec.get("fingerprints", {})
        return cls(rec["dataset"]["name"], rec["model"]["name"], rec["model"]["variant"],
                   fps.get("prediction", ""), fps.get("scoring", ""))

    def short(self) -> str:
        return (f"{self.dataset} x {self.model}/{self.variant} "
                f"[pred {self.prediction_fp[:8]} score {self.scoring_fp[:8]}]")


def load(store: Path | None = None) -> list[dict]:
    """Every published row, oldest first. Missing store is empty, not an error."""
    store = Path(store or STORE)
    if not store.exists():
        return []
    rows = []
    for n, line in enumerate(store.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{store}:{n} is not valid JSON: {e}") from e
    return rows


def _digest(fp_path: Path) -> str:
    if not fp_path.exists():
        return ""
    return json.loads(fp_path.read_text()).get("digest", "")


def _flops(run_dir: Path) -> dict | None:
    """Measured FLOPs for a profiled run, from the ONE canonical aggregator.

    This delegates to :func:`ecstasy.pipeline.flops_summary` rather than reading the
    sidecars again. An earlier version of this module re-implemented that aggregation
    and took ``sorted(vals)[len(vals) // 2]`` as the median, which is the upper-middle
    element rather than the mean of the two middle values: on an even number of targets
    it disagreed with `ecstasy compare` on the same run (30.0 vs 25.0 for 10/20/30/40).
    Two different numbers both called "median FLOPs" is precisely what a published
    record cannot afford. There is one aggregator; everything reads it.
    """
    from ecstasy.pipeline import flops_summary
    return flops_summary(run_dir)


def build_record(result_path: Path) -> dict:
    """Project a run's result.json into one publishable row.

    Everything here already exists in the run directory; nothing is recomputed, so a
    published number cannot disagree with the run it claims to describe.
    """
    result_path = Path(result_path)
    run_dir = result_path.parent
    res = json.loads(result_path.read_text())
    summary = res.get("summary", {})
    prov = res.get("provenance") or {}
    eco = prov.get("ecstasy", {})

    model_code = {}
    for name, pkg in (prov.get("venv", {}).get("packages") or {}).items():
        git = pkg.get("git") or {}
        if git:
            model_code[name] = {"sha": git.get("sha"), "dirty": bool(git.get("dirty")),
                                "dirty_files": git.get("dirty_files") or []}

    weights = {}
    for key, val in (prov.get("params_provenance") or {}).items():
        if isinstance(val, dict) and val.get("kind") == "file":
            weights[key] = {"path": val.get("path"), "resolved": val.get("resolved"),
                            "size": val.get("size"),
                            "sha256_ends": val.get("sha256_ends")}

    from ecstasy.datasets.base import load_dataset
    try:
        ds = load_dataset(res["dataset"])
        ds_meta = {"name": ds.name, "version": ds.version,
                   "expected_entries": ds.expected_entries}
    except Exception:  # a dataset row can be removed after a run was scored
        ds_meta = {"name": res["dataset"], "version": None, "expected_entries": None}

    return {
        "schema_version": SCHEMA_VERSION,
        "published_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": ds_meta,
        "model": {"name": res["model"], "variant": res["variant"]},
        "n": {"evaluated": summary.get("n_evaluated"),
              "skipped": summary.get("n_skipped"),
              "errors": summary.get("n_errors")},
        "coverage": res.get("coverage"),
        "metrics": {"names": res.get("metrics"),
                    "mean": summary.get("mean"), "median": summary.get("median")},
        "structure": summary.get("structure"),
        "flops": _flops(run_dir),
        "fingerprints": {
            "prediction": _digest(run_dir / "prediction_fingerprint.json"),
            "scoring": _digest(run_dir / "scoring_fingerprint.json"),
        },
        "provenance": {
            "ecstasy_sha": eco.get("sha"),
            "ecstasy_dirty": bool(eco.get("dirty")),
            "model_code": model_code,
            "weights": weights,
            "captured_utc": prov.get("captured_utc"),
            "host": (prov.get("env") or {}).get("host"),
        },
        # Relative to DATA_ROOT, never absolute: this file is committed and read on
        # other machines, where /rds/general/user/<someone>/... means nothing.
        "run_dir": _relative_run_dir(run_dir),
    }


def _relative_run_dir(run_dir: Path) -> str:
    """``<dataset>/<model>/<variant>``, never an absolute path.

    The fallback keeps three components rather than one. A run directory is always
    ``<runs_root>/<dataset>/<model>/<variant>``, so ``.name`` alone would record
    ``full`` — which names no run and is indistinguishable between models. That is
    also what hid a bug once: a test pointed DATA_ROOT elsewhere, this fell back
    silently, and the row recorded ``full`` instead of failing loudly.
    """
    from ecstasy.config import settings
    run_dir = Path(run_dir)
    try:
        return str(run_dir.relative_to(settings().runs_root))
    except (ValueError, TypeError):
        return str(Path(*run_dir.parts[-3:]))


def check_publishable(rec: dict, allow_partial: bool = False,
                      allow_dirty: bool = False) -> list[str]:
    """Reasons this row should not be published. Empty means go ahead.

    Two refusals, and deliberately only two.

    **Incomplete coverage.** A mean over part of a split prints identically to a mean
    over all of it; publishing one makes that permanent and quotable.

    **A dirty ecstasy tree.** ``ecstasy_sha`` would name a commit that does not contain
    the code that produced the number, which is worse than no provenance at all.

    A dirty *model* tree is NOT refused, and that is a considered choice rather than an
    oversight. MiniFold is benchmarked with the ``residx`` patch applied to its working
    tree — that is the intended experiment, and it is permanent. A gate that every
    MiniFold publish must override is not a gate, it is a habit of typing --allow_dirty.
    The dirty flag and the file list are recorded on the row instead, and
    ``ecstasy report`` marks such rows, so the condition is visible where it is read.
    """
    problems = []
    cov = rec.get("coverage") or {}
    if not cov.get("complete", False) and not allow_partial:
        frac = cov.get("fraction")
        problems.append(
            f"incomplete coverage ({frac:.1%} of the split)" if isinstance(frac, float)
            else "incomplete coverage")
    if rec.get("n", {}).get("errors"):
        problems.append(f"{rec['n']['errors']} targets errored")
    if rec["provenance"].get("ecstasy_dirty") and not allow_dirty:
        problems.append(
            "the ecstasy tree was dirty, so ecstasy_sha names a commit that does not "
            "contain the code that produced this number")
    return problems


def publish(result_path: Path, store: Path | None = None, allow_partial: bool = False,
            allow_dirty: bool = False, again: bool = False) -> tuple[dict, str]:
    """Append one row. Returns (record, note).

    Re-publishing an identical (dataset, model, variant, both fingerprints) is refused
    by default: it is not a new measurement, it is the same one twice, and a duplicated
    row would make a leaderboard count it twice.
    """
    store = Path(store or STORE)
    rec = build_record(result_path)

    problems = check_publishable(rec, allow_partial, allow_dirty)
    if problems:
        raise PublishRefused(
            "refusing to publish " + Key.from_record(rec).short() + ":\n  - "
            + "\n  - ".join(problems)
            + "\n\nThis is the record other people will quote. Fix the run, or pass "
              "--allow_partial / --allow_dirty if the limitation is the point of the "
              "experiment — it is recorded on the row either way.")

    key = Key.from_record(rec)
    if not key.prediction_fp or not key.scoring_fp:
        raise PublishRefused(
            f"{key.short()}: missing a fingerprint, so this row could not be told apart "
            f"from a later one with different inputs. Re-run `ecstasy score` on current "
            f"code, which writes both.")

    existing = {Key.from_record(r) for r in load(store)}
    note = ""
    if key in existing:
        if not again:
            raise PublishRefused(
                f"already published: {key.short()}\n\nIdentical fingerprints mean "
                f"identical inputs to both prediction and scoring — this is the same "
                f"measurement, not a new one. Pass --again only to record a deliberate "
                f"repeat (a determinism check, say); it will appear as a second row.")
        note = "duplicate fingerprints, recorded deliberately (--again)"

    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec, note
