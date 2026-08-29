# scripts/ — shared infrastructure

This directory holds tooling that every benchmarking campaign depends on, not any one
campaign's own scripts. Campaign-specific manifests, plotting scripts, and sweeps live
under [`../experiments/`](../experiments/README.md) — one subdirectory per experiment.

- **`install/`** — per-model environment setup (`boltz.sh`, `esmfold.sh`, `mentos.sh`, …).
  Each script builds the dedicated venv a model runs in (`$ENVS_ROOT/.venv-<model>`).
- **`run_experiment.sbatch`** — a generic SLURM wrapper around `ecstasy experiment
  <manifest.yaml>`:

  ```bash
  sbatch scripts/run_experiment.sbatch experiments/boltz2_headline/manifest.yaml
  ```

  Runs in `.venv-boltz` (has `ecstasy` + `torch` + `mentos`, so predict subprocesses to
  each model's env and scoring can load the MENTOS GT here).

> Per-cluster/per-job SLURM `*.sbatch` wrappers you write yourself for a specific
> campaign are **not committed** — keep them local (they're git-ignored). See each
> experiment's own README for portable, non-cluster-specific commands.
