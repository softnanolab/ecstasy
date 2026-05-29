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


def _matrix(dataset, model, preset, overrides):
    for d in _as_list(dataset):
        for m in _as_list(model):
            yield pipeline.make_run(d, m, preset=preset, overrides=overrides)


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
        """Populate the shared MSA store via colabfold-local.

        phase: prepare (write missing-chains FASTA), submit (sbatch colabfold),
               or ingest (copy resulting a3ms into the store).
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

    def run(self, dataset, model, preset=None, set=None, limit=None, no_score=False):
        """Predict (and score, unless --no_score) over the dataset×model matrix."""
        for r in _matrix(dataset, model, preset, set):
            print(f"\n=== {r.dataset.name} × {r.model.name}/{r.model.variant} (predict) ===")
            pipeline.run_predict(r, limit=limit)
            if not no_score:
                pipeline.run_score(r, limit=limit)

    def score(self, dataset, model, preset=None, set=None, limit=None):
        """Score existing predictions over the dataset×model matrix."""
        for r in _matrix(dataset, model, preset, set):
            pipeline.run_score(r, limit=limit)

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
