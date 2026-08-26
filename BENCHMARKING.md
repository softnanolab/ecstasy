# Benchmarking (datasets × models)

Datasets and models are **declarative data**, not code. Adding a split or a
parameter variant is a YAML edit, not a new class or a new config file.

- **Datasets** — `src/ecstasy/registry/datasets.yaml`. Each row: `kind`
  (`ecstasy`), `root` (the dataset folder), `contact_bin`, plus the identity
  fields (`version`, `description`, `expected_entries`, `tags`) and a
  `built_from` recipe saying how the folder was originally built.
- **Models** — `src/ecstasy/registry/models.yaml`. Each row: `runner`, `env`,
  `msa` (`none`|`per_chain`|`complex`), `infra`, and named `presets`.
- **Paths** — every `${VAR}` resolves via `src/ecstasy/config.py`
  (`DATA_ROOT`, `MENTOS_ROOT`, `ECSTASY_ROOT`, `ENVS_ROOT`, `TOOLS_ROOT`),
  overridable through the environment or repo-root `.env`.

## Datasets are built, not committed

ecstasy owns its evaluation data. Every scorable row lives under
`$DATA_ROOT/datasets/<name>/` as one self-contained folder:

```
$DATA_ROOT/datasets/<name>/
    dataset.yaml        identity, contact-bin convention, import provenance, coverage
    index.parquet       exactly this dataset's rows — no split column to re-apply
    composition.json    what the split is made of, computed once
    gt/<xx>/<id>.npz    pickle-free ground truth: atom37 coords + masks + sequences
```

Nothing on the scoring path resolves into another project's tree, so MENTOS can
rebuild a split, rename it, or lose its ephemeral storage without touching a
published ecstasy result. That is not hypothetical — MENTOS PR #266 retired the
five splits every row here used to name, and `seq_id_30`'s parquet is gone.

A folder is compressed binary — 11 MB for `recent_pp`, 45 MB for all four — so it is
**built on each machine**, not committed:

```bash
ecstasy datasets --verify              # says "not built yet" and names the command
ecstasy import_dataset --dataset recent_pp
```

`import_dataset` reads the row's `built_from` recipe — the only place a source
path appears, and one `load_dataset` deliberately drops so no scoring path can
follow it. The recipe pins `index_sha256`: if the source is rebuilt under the
same path, the import **stops** rather than replacing your dataset with a
different one of the same name.

Ground truth is stored as coordinates and the contact bins are **derived on
read**, so there is one source of truth rather than two that can disagree.
Reading it needs numpy and pandas — no torch, no `mentos`, no unpickling a class
that has to stay importable.

| dataset | n | what it is |
|---|---|---|
| `recent_pp` | 151 | MENTOS's primary validation split, temporally held out. Full-atom GT, so it scores contacts **and** structure (DockQ/iRMSD/LRMSD/TM). The headline set. |
| `foldbench_pp` | 193 | FoldBench protein-protein interfaces — the PPI headline |
| `foldbench_pp_post2024` | 96 | the 193 restricted to releases on/after 2024-01-01 — FoldBench's post-2024 leaderboard. The only set that is **both** homology-controlled and temporally held out |
| `foldbench_abag` | 137 | FoldBench antibody-antigen. Report beside `foldbench_pp`, never averaged in |
| `foldbench` | 330 | the union of `foldbench_pp` and `foldbench_abag` |

Two of these **overlap**: `foldbench_pp_post2024` is a strict subset of `foldbench_pp`,
and `foldbench` contains both `foldbench_pp` and `foldbench_abag`. Overlapping sets are
not independent evidence — report them side by side, never pooled, and never as if a
model had been tested on 619 complexes.

A dataset's name and its `built_from` split value need not match, and here deliberately
do not: the source calls the post-2024 set `foldbench_pp_2024`, which names a year
without saying on which side of it the entries fall. The recipe records the source's
name; the row uses the one that says what the split *is*.

## CLI

```bash
ecstasy list                                   # datasets, models, presets
ecstasy datasets [--verify]                    # what exists, and does it still match
ecstasy import_dataset --dataset D             # build a dataset folder from built_from
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

## Boltz-2 on the headline splits

Boltz-2 needs **per-chain unpaired** MSAs (shared store, keyed by sequence hash,
so overlapping chains are never searched twice).

```bash
cd <repo>; source envs/.venv-boltz/bin/activate

# 0. Once per machine: build the dataset folders
ecstasy import_dataset --dataset recent_pp
ecstasy import_dataset --dataset foldbench_pp

# 1. MSAs: write the missing-chains FASTA, run colabfold-local, ingest into the store
ecstasy msa --datasets recent_pp,foldbench_pp --kind per_chain --phase submit
#   (or run the printed sbatch yourself, then:)
ecstasy msa --datasets recent_pp,foldbench_pp --kind per_chain --phase ingest

# 2. Smoke one entry end-to-end, then the full sweep
ecstasy run --dataset recent_pp --model boltz2 --limit 1
sbatch scripts/run_experiment.sbatch experiments/boltz2_headline.yaml

# 3. Table
ecstasy compare --dataset recent_pp
```

Scoring an imported dataset needs only numpy and pandas — the ground truth is
ecstasy's own `.npz`, not a MENTOS pickle. MENTOS is needed **once**, by
`import_dataset`, to read the source samples; after that no scoring environment
requires it.

## Notes

- **Legacy predictions** under `$DATA_ROOT/ecstasy/benchmarks/<name>/` (the old
  `run_id`-hashed layout) are **not** migrated; new runs land under
  `ecstasy/runs/`. Re-run, or symlink if you want to keep them.
- Other models reuse the same flow: `esmfold`/`mentos` are single-sequence;
  `msa_pairformer` needs `--kind complex` MSAs (paired), not the per-chain ones.
```
