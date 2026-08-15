# Benchmarking (datasets × models)

Datasets and models are **declarative data**, not code. Adding a split or a
parameter variant is a YAML edit, not a new class or a new config file.

- **Datasets** — `src/ecstasy/registry/datasets.yaml`. Each row: `kind`
  (`mentos_square`), `index` parquet, `gt_root`, `contact_bin`.
- **Models** — `src/ecstasy/registry/models.yaml`. Each row: `runner`, `env`,
  `msa` (`none`|`per_chain`|`complex`), `infra`, and named `presets`.
- **Paths** — every `${VAR}` resolves via `src/ecstasy/config.py`
  (`DATA_ROOT`, `MENTOS_ROOT`, `ECSTASY_ROOT`, `ENVS_ROOT`, `TOOLS_ROOT`),
  overridable through the environment or repo-root `.env`.

## CLI

```bash
ecstasy list                                   # datasets, models, presets
ecstasy run     --dataset D[,D] --model M[,M] [--preset P] [--set '{k: v}'] [--limit N] [--no_score]
ecstasy score   --dataset D[,D] --model M[,M] [--preset P] [--set '{k: v}'] [--limit N]
ecstasy compare --dataset D                    # comparison.{csv,md} across all runs
ecstasy msa     --datasets D[,D] --kind per_chain|complex [--phase prepare|submit|ingest]
ecstasy experiment experiments/<name>.yaml [--limit N] [--no_score]
```

`--limit 1` is the smoke (no separate `_smoke` configs). `--set` takes a dict.

## Outputs (stable, human-readable)

```
$DATA_ROOT/ecstasy/runs/<dataset>/<model>/<variant>/
    params.json                       # preset + params + infra + msa (provenance)
    predictions/<entry_id>/contact.npz
    result.json                       # scoring summary + per-protein metrics
    ../comparison.{csv,md}            # from `ecstasy compare`
```

`<variant>` is the preset name (e.g. `full`), or `<preset>+<sha8>` when `--set`
overrides are given. Infra knobs (num_workers/devices) are *not* part of the
variant, so machine tweaks never fork a run dir or trigger a re-run.

## Datasets

`recent_pp` is the **only** registered dataset. MENTOS now ships one split
(`pdb/processed/splits/val/`), which replaced `seq_id_30` and the four deleaked
`val_*` sets — their parquets no longer exist, so those rows were removed rather
than left dangling. See the header comment in `registry/datasets.yaml` for the
provenance (it replicates §A.2.10 of the ESMFold2 paper) and for why the paper's
DockQ pass rates are *not* comparable to our inter-chain P@K.

## Running the driver

The per-model runners are subprocessed into their own venvs by
`models/adapter.py`, so the driver env only needs `ecstasy` + `torch` + `mentos`
(scoring unpickles a `mentos.dataclasses.Sample`). **The editable installs of
`ecstasy` and `mentos` are broken in every venv on this machine** — they point at
a deleted worktree — so put both source trees on `PYTHONPATH` and invoke
`python -m ecstasy.cli` rather than the `ecstasy` console script:

```bash
export PYTHONPATH=<mentos-checkout>/src:<ecstasy-checkout>/src
PY=$ENVS_ROOT/.venv-mentos/bin/python                 # torch 2.12, pandas, fire, yaml

$PY -m ecstasy.cli list
```

## ESMFold + single-sequence Boltz-2 on recent_pp (the headline run)

Both models are MSA-free, so there is no MSA phase at all.

```bash
# 1. Smoke one entry per model — gate on a non-degenerate P@K and a contact.npz
#    of shape len(chainA)+len(chainB). A shape mismatch is reported as `_error`
#    in result.json rather than raised, so check it before burning GPU hours.
$PY -m ecstasy.cli run --dataset recent_pp --model esmfold      --limit 1
$PY -m ecstasy.cli run --dataset recent_pp --model boltz2_nomsa --limit 1

# 2. The recycle ladders, with FLOPs (8 runs x 151 dimers)
sbatch scripts/run_experiment.sbatch experiments/recent_pp_nomsa_ladder.yaml --profile

# 3. Table
$PY -m ecstasy.cli compare --dataset recent_pp
```

`--profile` is what produces the FLOPs axis, and it must be passed to
`experiment` too, not just `run` — FLOPs are length-dependent, so measurements
from the old splits do not transfer to this one.

## Notes

- **Legacy predictions** under `$DATA_ROOT/ecstasy/benchmarks/<name>/` (the old
  `run_id`-hashed layout) are **not** migrated; new runs land under
  `ecstasy/runs/`. Re-run, or symlink if you want to keep them.
- Other models reuse the same flow: `esmfold`/`mentos` are single-sequence;
  `msa_pairformer` needs `--kind complex` MSAs (paired), not the per-chain ones.
- The chain-order-permutation experiment (`swap_chains: true`) is still supported
  by `MentosSquareDataset`; only the old `val_seq_pair_swapped` *row* went away
  with its parquet. Add a swapped row for `recent_pp` if you want it back — give
  it its own name, since it needs its own run dir and pair-hash MSAs.
