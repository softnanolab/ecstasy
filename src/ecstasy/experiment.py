"""Experiment manifests: a recorded dataset×model sweep.

```yaml
name: recent_pp_nomsa_ladder
datasets: [recent_pp]
runs:
  - {model: esmfold, preset: r1}
  - {model: boltz2, preset: full, set: {recycling_steps: 5}}   # per-cell override
```

`expand` takes the cartesian product (datasets × runs) into `Run`s through the
*same* `pipeline.make_run` the CLI uses, so the two entry points can't diverge.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ecstasy.pipeline import Run, make_run, run_predict, run_score


def load_manifest(path: str) -> dict:
    m = yaml.safe_load(Path(path).read_text())
    if "datasets" not in m or "runs" not in m:
        raise ValueError(f"manifest {path} needs `datasets:` and `runs:` keys")
    return m


def expand(manifest: dict) -> list[Run]:
    runs: list[Run] = []
    for dataset in manifest["datasets"]:
        for spec in manifest["runs"]:
            runs.append(make_run(
                dataset=dataset,
                model=spec["model"],
                preset=spec.get("preset"),
                overrides=spec.get("set"),
            ))
    return runs


def run_experiment(path: str, limit: int | None = None, score: bool = True,
                   profile: bool = False, shard: str | None = None) -> None:
    """Run every dataset×model combination in `manifest`.

    `shard` ("i/N") is forwarded to run_predict so one manifest can be spread over N
    concurrent jobs. Every runner is a fresh subprocess that reloads its weights per
    entry, so the model-load cost is paid 151× per run either way; sharding is what
    keeps that off the critical path, and short jobs also backfill far better than one
    long one on a contested queue. Shards skip entries whose contact.npz (and, under
    --profile, flops.json) already exists, so they never collide and are resumable.

    Scoring is suppressed while sharding: each shard only predicts its own slice, so
    letting it score would race the other shards writing the same result.json and
    persist a summary over a partial set. Score once after the shards finish, with the
    same manifest and no --shard.
    """
    manifest = load_manifest(path)
    runs = expand(manifest)
    print(f"experiment {manifest.get('name', Path(path).stem)}: {len(runs)} run(s)")
    for r in runs:
        print(f"  - {r.dataset.name} × {r.model.name}/{r.model.variant}")
    if limit == 0:
        print("\n(dry run: --limit 0, nothing executed)")
        return
    if shard and score:
        print(f"[shard {shard}] scoring suppressed; re-run without --shard to score")
        score = False
    for r in runs:
        print(f"\n=== {r.dataset.name} × {r.model.name}/{r.model.variant} ===")
        run_predict(r, limit=limit, profile=profile, shard=shard)
        if score:
            run_score(r, limit=limit)
