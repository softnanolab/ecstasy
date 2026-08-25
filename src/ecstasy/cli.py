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
            profile=False, checkpoint=None, shard=None, null_draws=0):
        """Predict (and score, unless --no_score) over the dataset×model matrix.

        --checkpoint <name> selects a checkpoint from the Notion benchmarking Registry
        (for models without committed presets, e.g. mentos): the name resolves to concrete
        weights/recycles via registry.local.yaml (run notion_pull.py first).
        --profile additionally measures inference FLOPs and writes a flops.json
        sidecar next to each contact.npz (see FLOPS_BENCHMARK_PLAN.md).
        --shard 'i/N' processes only every N-th entry (offset i) for parallel jobs;
        combined with the contact.npz skip the shards never collide and are resumable.
        --null_draws N adds the random-placement DockQ floor (see `score`).
        """
        for r in _matrix(dataset, model, preset, set, checkpoint):
            print(f"\n=== {r.dataset.name} × {r.model.name}/{r.model.variant} (predict) ===")
            pipeline.run_predict(r, limit=limit, profile=profile, shard=shard)
            if not no_score:
                pipeline.run_score(r, limit=limit, null_draws=null_draws)

    def score(self, dataset, model, preset=None, set=None, limit=None, checkpoint=None,
              null_draws=0):
        """Score existing predictions over the dataset×model matrix.

        Models that emitted a structure.npz are additionally scored with DockQ, iRMSD,
        LRMSD and per-chain monomer metrics when the dataset carries full-atom GT.

        --null_draws N computes the random-placement floor per target: the model's own
        chains with chain B randomly re-docked, N draws, DockQ'd. Fold quality is held
        fixed and only the placement is destroyed, so the result is what DockQ gives away
        for free on this target and it is the reference a low DockQ must be read against.
        Costs N extra DockQ invocations per target; off by default.
        """
        for r in _matrix(dataset, model, preset, set, checkpoint):
            pipeline.run_score(r, limit=limit, null_draws=null_draws)

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
