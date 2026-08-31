# experiments/ — everything experiment-related, one folder each

Every ecstasy experiment — a dataset×model sweep manifest, or a whole benchmarking
campaign with its own plotting/sweep scripts — lives under its own subdirectory here.
This is the one place to look for anything experiment-related; shared infrastructure
that every experiment depends on (per-model environment setup, the generic SLURM
runner) stays outside, in [`../scripts/`](../scripts/README.md).

## Convention

- **One experiment = one folder** under `experiments/`, always — even a
  manifest with nothing else in it gets a folder, so "find the experiment" always
  means "find the folder," with no exception to remember for the simple case.
- **A manifest-driven experiment** (a dataset×model sweep run via `ecstasy experiment`)
  keeps its manifest at a fixed filename, `manifest.yaml`, inside its folder:

  ```
  experiments/boltz2_headline/
      manifest.yaml
  ```

  Run it with:

  ```bash
  ecstasy experiment experiments/boltz2_headline/manifest.yaml [--limit N] [--no_score]
  ```

  The fixed filename means renaming the experiment folder never requires renaming (or
  risks desyncing) a file inside it — the folder name is the only place the
  experiment's identity lives.

- **A script-driven campaign** (one that needs reusable plotting/sweep scripts beyond
  a single manifest, e.g. a whole benchmarking project with several sub-experiments)
  gets a folder with its scripts and its own `README.md` documenting them:

  ```
  experiments/mentos-perf-benchmarking/
      README.md
      plot_pak_vs_flops.py
      swap_compare.py
      …
  ```

  Such a campaign may or may not also have a `manifest.yaml` — use one if the sweep is
  cleanly expressible as a dataset×model matrix; if not, plain `ecstasy …` CLI calls
  documented in the campaign's own README are fine (see
  `mentos-perf-benchmarking/README.md` for an example).

## What stays outside experiments/

Shared infrastructure that isn't specific to any one experiment lives in
[`../scripts/`](../scripts/README.md):

- `scripts/install/` — per-model venv setup, used by every experiment that runs that
  model.
- `scripts/run_experiment.sbatch` — a generic SLURM wrapper around
  `ecstasy experiment <manifest.yaml>`, usable by any manifest-driven experiment.

## Naming note

`DESIGN.md`'s Phase 2 (not yet built) design also mentions an `experiments/` path —
but that one is a subtree of `$DATA_ROOT` (the run-output layout: `params.json`,
`result.json`, `predictions/`), not this repo-level directory. The two are unrelated
besides the shared name; see the note in `DESIGN.md` where it's introduced.
