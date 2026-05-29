"""Experiment manifests: a recorded dataset×model sweep.

```yaml
name: boltz2_val_splits
datasets: [val_seq_chain, val_seq_pair, val_pinder_chain, val_pinder_pair]
runs:
  - {model: boltz2, preset: full}
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


def run_experiment(path: str, limit: int | None = None, score: bool = True) -> None:
    manifest = load_manifest(path)
    runs = expand(manifest)
    print(f"experiment {manifest.get('name', Path(path).stem)}: {len(runs)} run(s)")
    for r in runs:
        print(f"  - {r.dataset.name} × {r.model.name}/{r.model.variant}")
    if limit == 0:
        print("\n(dry run: --limit 0, nothing executed)")
        return
    for r in runs:
        print(f"\n=== {r.dataset.name} × {r.model.name}/{r.model.variant} ===")
        run_predict(r, limit=limit)
        if score:
            run_score(r, limit=limit)
