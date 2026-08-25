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
    predictions/<entry_id>/structure.npz   # optional — structure-emitting models only
    result.json                       # scoring summary + per-protein metrics
    ../comparison.{csv,md}            # from `ecstasy compare`
$DATA_ROOT/natives/<dataset>/<entry_id>_native.pdb   # rendered GT, cached
```

`<variant>` is the preset name (e.g. `full`), or `<preset>+<sha8>` when `--set`
overrides are given. Infra knobs (num_workers/devices) are *not* part of the
variant, so machine tweaks never fork a run dir or trigger a re-run.

## Structure metrics (DockQ)

`contact.npz` is required of every runner. `structure.npz` — an atom37 bundle, see
`src/ecstasy/structure/pdb.py` — is **optional and additive**: when a model emits one
and the dataset carries full-atom ground truth (`has_structure_gt`), scoring also
produces DockQ, Fnat, iRMSD, LRMSD and per-chain monomer metrics (TM, CA-RMSD), split
by homo/heterodimer. Contact-only models and structure-less datasets are unaffected,
and a DockQ failure on one target never discards that target's contact metrics.

```bash
ecstasy score --dataset mentos_val151 --model minifold --null_draws 10
```

Three rules keep the numbers meaningful, and all three are enforced in code:

- **Predictions and natives go through one writer.** `write_atom37_pdb` is a port of
  the MENTOS `eval_structure_dockq` writer, verified byte-identical on all 151 natives
  of `mentos_val151`, so DockQ compares structures rather than serialisations — and
  ecstasy's numbers stay comparable to the 23-checkpoint MENTOS series.
- **The DockQ invocation is fixed**: `DockQ <model> <native>`, no flags, parsed with the
  regexes copied from `mentos`. Changing either breaks that comparability.
- **Never read DockQ alone.** It averages fnat with two RMSD terms, so a prediction whose
  backbone has not formed still scores off fnat while both RMSDs give near-zero credit.
  iRMSD and LRMSD are reported beside it everywhere, and `--null_draws N` gives the
  random-placement floor: the model's own chains with chain B re-docked at random, which
  is what the target gives away for free.

## Boltz-2 on the four val splits (the headline run)

Boltz-2 needs **per-chain unpaired** MSAs (shared store, keyed by sequence hash,
so the four splits never regenerate overlapping chains — 1,327 unique total).

```bash
cd <repo>; source envs/.venv-boltz/bin/activate      # has ecstasy + torch + mentos

# 1. MSAs: write the missing-chains FASTA, run colabfold-local, ingest into the store
ecstasy msa --datasets val_seq_chain,val_seq_pair,val_pinder_chain,val_pinder_pair --kind per_chain --phase submit
#   (or run the printed sbatch yourself, then:)
ecstasy msa --datasets val_seq_chain,val_seq_pair,val_pinder_chain,val_pinder_pair --kind per_chain --phase ingest

# 2. Smoke one entry end-to-end, then the full sweep
ecstasy run --dataset val_pinder_pair --model boltz2 --limit 1
sbatch scripts/run_experiment.sbatch experiments/boltz2_val_splits.yaml

# 3. Table
ecstasy compare --dataset val_pinder_pair
```

Scoring loads the MENTOS-pickled GT, so run `score`/`run` from an env with
`torch` + `mentos` (`.venv-boltz` has both; `.venv-esmfold` needs
`pip install -e /home/u6jv/harsh.u6jv/mentos` for the no-MSA models).

## Notes

- **Legacy predictions** under `$DATA_ROOT/ecstasy/benchmarks/<name>/` (the old
  `run_id`-hashed layout) are **not** migrated; new runs land under
  `ecstasy/runs/`. Re-run, or symlink if you want to keep them.
- Other models reuse the same flow: `esmfold`/`mentos` are single-sequence;
  `msa_pairformer` needs `--kind complex` MSAs (paired), not the per-chain ones.
```
