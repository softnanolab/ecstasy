"""ecstasy CLI — datasets × models, driven by registries, CLI matrix, or a manifest.

  ecstasy list
  ecstasy msa     --datasets D[,D] --kind per_chain|complex [--phase prepare|submit|ingest] [--a3m_dir DIR]
  ecstasy run     --dataset D[,D] --model M[,M] [--preset P] [--set '{k: v}'] [--limit N] [--no_score]
  ecstasy score   --dataset D[,D] --model M[,M] [--preset P] [--set '{k: v}'] [--limit N]
  ecstasy compare --dataset D
  ecstasy experiment <manifest.yaml> [--limit N] [--no_score]

`--set` takes a dict, e.g. `--set '{recycling_steps: 5}'`. `--limit 1` is the smoke;
`--limit 0` with `experiment` is a dry materialization (lists runs, executes nothing).
"""
from __future__ import annotations

import json

import fire

from ecstasy.datasets import dataset_names
from ecstasy.models import model_names, presets_for, load_model
from ecstasy import pipeline, experiment


def _as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x]
    return [s for s in str(x).split(",") if s]


def _matrix(dataset, model, preset, overrides, checkpoint=None):
    for d in _as_list(dataset):
        for m in _as_list(model):
            yield pipeline.make_run(d, m, preset=preset, overrides=overrides, checkpoint=checkpoint)


class Ecstasy:
    def list(self):
        """Show registered datasets, models, and their presets."""
        print("datasets:")
        for d in dataset_names():
            print(f"  {d}")
        print("models:")
        for m in model_names():
            mr = load_model(m)
            print(f"  {m:16} msa={mr.msa:9} env={mr.env.name:14} presets={presets_for(m)}")

    def msa(self, datasets, kind, phase="prepare", a3m_dir=None):
        """Populate the shared MSA store for a given `kind`.

        kind: boltz_csv (Boltz-2 per-chain CSVs, local colabfold_search) |
              complex (MSA-Pairformer stitched a3m, local colabfold-local) |
              complex_api (MSA-Pairformer via the ColabFold API — fallback only).
        Each model uses a DIFFERENT pipeline; see src/ecstasy/msa/README.md.

        phase: prepare (write missing-complex FASTA/manifest), submit (sbatch the
               local search; complex_api fetches inline), or ingest (assemble/copy
               results into the store).
        """
        from ecstasy.msa import generate
        ds = _as_list(datasets)
        if phase == "prepare":
            generate.prepare(ds, kind)
        elif phase == "submit":
            generate.submit(ds, kind)
        elif phase == "ingest":
            generate.ingest(ds, kind, a3m_dir=a3m_dir)
        else:
            raise ValueError(f"--phase must be prepare|submit|ingest, got {phase!r}")

    def run(self, dataset, model, preset=None, set=None, limit=None, no_score=False,
            profile=False, checkpoint=None, shard=None):
        """Predict (and score, unless --no_score) over the dataset×model matrix.

        --checkpoint <name> selects a checkpoint from the Notion benchmarking Registry
        (for models without committed presets, e.g. mentos): the name resolves to concrete
        weights/recycles via registry.local.yaml (run notion_pull.py first).
        --profile additionally measures inference FLOPs and writes a flops.json
        sidecar next to each contact.npz (see FLOPS_BENCHMARK_PLAN.md).
        --shard 'i/N' processes only every N-th entry (offset i) for parallel jobs;
        combined with the contact.npz skip the shards never collide and are resumable.
        """
        for r in _matrix(dataset, model, preset, set, checkpoint):
            print(f"\n=== {r.dataset.name} × {r.model.name}/{r.model.variant} (predict) ===")
            pipeline.run_predict(r, limit=limit, profile=profile, shard=shard)
            if not no_score:
                pipeline.run_score(r, limit=limit)

    def score(self, dataset, model, preset=None, set=None, limit=None, checkpoint=None,
              metrics=None):
        """Score existing predictions over the dataset×model matrix.

        --metrics 'P@K,P@K(tol=2)' selects registered metrics by name; see
        `ecstasy metrics`. Defaults to the canonical set, so adding a metric to the
        registry never silently changes a headline number.
        """
        for r in _matrix(dataset, model, preset, set, checkpoint):
            pipeline.run_score(r, limit=limit, metrics=_as_list(metrics) or None)

    def metrics(self, kind=None, json_out=False):
        """List registered metrics — the reusable set available to any run or plot."""
        from ecstasy.metrics import registry
        rows = registry.describe(kind=kind)
        if json_out:
            print(json.dumps(rows, indent=1))
            return
        for m in rows:
            params = f"  {m['params']}" if m["params"] else ""
            arrow = "higher is better" if m["higher_is_better"] else "lower is better"
            print(f"  {m['name']:18} [{m['kind']}, {arrow}]{params}\n"
                  f"      {m['description']}")

    def datasets(self, verify=False, json_out=False):
        """Describe registered datasets; --verify checks each split against its row.

        `verify` walks each index and reports entry-count drift — a split is a file that
        nothing stops from changing under a published result, so the declared
        `expected_entries` is asserted rather than trusted.
        """
        from ecstasy.datasets.base import dataset_manifests, dataset_names, load_dataset
        if verify:
            reports = [load_dataset(n).verify() for n in dataset_names()]
            if json_out:
                print(json.dumps(reports, indent=1))
            else:
                for r in reports:
                    mark = "ok  " if r["ok"] else "FAIL"
                    print(f"  [{mark}] {r['name']:24} n={r['n_entries']} "
                          f"expected={r['expected_entries']}")
                    for p in r["problems"]:
                        print(f"         - {p}")
            if any(not r["ok"] for r in reports):
                raise SystemExit(1)
            return
        manifests = dataset_manifests()
        if json_out:
            print(json.dumps(manifests, indent=1))
            return
        for m in manifests:
            print(f"  {m['name']:24} v{m['version']}  n={m['expected_entries']}  "
                  f"tags={','.join(m['tags'])}")
            print(f"      {' '.join((m['description'] or '(no description)').split())}")

    def compare(self, dataset):
        """Aggregate all runs for a dataset into comparison.{csv,md}."""
        pipeline.run_compare(dataset)

    def experiment(self, manifest, limit=None, no_score=False):
        """Run a dataset×model sweep from a manifest YAML."""
        experiment.run_experiment(manifest, limit=limit, score=not no_score)


def main():
    fire.Fire(Ecstasy())


if __name__ == "__main__":
    main()
