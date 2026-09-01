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

### Publishing to Weights & Biases

`results/runs.jsonl` is the source of truth. wandb is a **derived view** of it, in exactly
the sense `results/LEADERBOARD.md` is — so it is regenerated, not maintained:

```bash
uv pip install -e '.[wandb]'    # opt-in extra
wandb login                     # or a ~/.netrc entry for api.wandb.ai
ecstasy wandb --dry_run         # inspect the projection, no network
ecstasy wandb                   # one wandb run per published row
```

**Export is idempotent.** A run's wandb id is a digest of the row's dataset, model,
variant and *both* fingerprints, so re-running converges instead of creating duplicates.
Re-scoring after a metric fix changes the scoring fingerprint and therefore appears as a
**new** wandb run — the same "the number moved, and here is what changed underneath it"
history that the append-only JSONL gives you in `git log -p`.

`ecstasy publish --wandb` mirrors as it publishes, but the export runs strictly after the
row is appended and a failure is a warning, not an error. `publish` and `report` are
deliberately usable with no network and no token; that property is what lets someone read
the benchmark state from the repo alone, and it is not traded away for convenience.

Two conventions worth knowing before reading the runs table:

- **An unmeasured quantity is absent, not zero.** A row with no FLOPs sets
  `flops/measured = False` and omits the number entirely. This is live today: ESMFold2's
  FLOPs are refused because its ESMC-6B backbone is uncounted (#62), and a zero would put
  the strongest model at the origin of the compute axis looking measured.
- **Caveats are tags.** The leaderboard's `†` (dirty model tree), `‡` (dirty ecstasy tree)
  and `*` (partial coverage) become `dirty-model-tree`, `dirty-ecstasy-tree` and
  `partial-coverage`. Someone filtering the runs table is as entitled to know that a
  recorded commit does not describe what ran as someone reading the markdown.

Compute-node egress to `api.wandb.ai` is direct on Isambard, so a sweep can log live;
`WANDB_MODE=offline` plus `wandb sync` remains available for hosts where it is not.

### Structure scoring prerequisites

Structure metrics (DockQ, iRMSD, LRMSD, Fnat, TM) need the **`DockQ` CLI on `PATH`** --
`ecstasy.metrics.structure.dockq_binary()` is `shutil.which("DockQ")`, and every structure
metric returns *skipped* without it. That is a quiet failure mode: a sweep will complete and
publish contact metrics while silently reporting no DockQ at all.

Install it as an opt-in extra:

```bash
uv pip install -e '.[structure]'
```

**It needs Python development headers.** DockQ compiles a C extension and has no wheel for
every platform+interpreter pair here, so on an interpreter without `Python.h` the build fails
with `compilation terminated`. Neither cluster's *system* python carries headers; a
uv-managed interpreter does:

```bash
uv python install 3.11
uv venv <path> --python 3.11 --python-preference only-managed
```

Verified end-to-end by `tests/integration/test_score_structure.py`, which feeds a split's own
native coordinates back in as if they were a prediction and requires DockQ 1.0 -- anything
less means the structure was corrupted between the `.npz` and the scorer. It is skipped
automatically when the CLI is absent, so check it actually *ran* before trusting a green run.

Note the separate gap: full-atom ground truth is present for **every** registered split, but
only the `minifold` runner currently writes `structure.npz`, so DockQ is not yet obtainable
for the other models regardless of the CLI being installed.

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
ecstasy publish --dataset D --model M          # append to the committed record
ecstasy report  [--show]                       # regenerate results/LEADERBOARD.md
ecstasy msa     --datasets D[,D] --kind per_chain|complex [--phase prepare|submit|ingest]
ecstasy experiment experiments/<name>/manifest.yaml [--limit N] [--no_score]
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
sbatch scripts/run_experiment.sbatch experiments/boltz2_headline/manifest.yaml

# 3. Table
ecstasy compare --dataset recent_pp
```

Scoring an imported dataset needs only numpy and pandas — the ground truth is
ecstasy's own `.npz`, not a MENTOS pickle. MENTOS is needed **once**, by
`import_dataset`, to read the source samples; after that no scoring environment
requires it.

## Publishing a result

Results used to live only in `$DATA_ROOT` — machine-local, gitignored, copied into
Notion by hand. So no benchmark number was versioned, a PR could not show that a
change had moved one, and an agent with no token could not find out what had already
been measured.

`results/runs.jsonl` is the committed record; `results/LEADERBOARD.md` is generated
from it. Commit both together.

```bash
ecstasy publish --dataset recent_pp --model minifold --preset full
```

**Publishing is deliberate.** Nothing publishes itself, so a `--limit 1` smoke or an
abandoned experiment never becomes the number someone quotes. It refuses:

| refusal | why | override |
|---|---|---|
| incomplete coverage | a mean over part of a split prints identically to a mean over all of it | `--allow_partial` |
| any errored target | | — (fix the run) |
| a dirty ecstasy tree | `ecstasy_sha` would name a commit that does not contain the code that produced the number | `--allow_dirty` |
| identical fingerprints to an existing row | same inputs to prediction *and* scoring means the same measurement, not a new one | `--again` |

A dirty **model** tree is *not* refused. MiniFold is benchmarked with the `residx`
patch applied to its working tree — that is the intended experiment and it is
permanent, so a gate every MiniFold publish had to override would just be a habit of
typing `--allow_dirty`. The dirty flag and file list go on the row, and the leaderboard
marks it `†`.

Rows are keyed by dataset, model, variant and **both** fingerprints. Re-scoring after a
metric fix therefore appends a new row against identical predictions rather than
editing the old one, so `git log -p results/runs.jsonl` shows a number moving *and*
what changed underneath it.

Summaries only — per-protein detail stays in `$DATA_ROOT`. The repo keeps numbers you
can diff, not blobs.

### Results with ESMFold2 (2026-08-18, 151 dimers, all three models complete)

Inter-chain P@K at exact match and at Chebyshev ±2 (GT dilated by a 5×5 L∞ ball).

| rung | ESMFold2 | | ESMFold | | Boltz-2 (no MSA) | |
|------|---------:|---------:|--------:|--------:|-----:|-----:|
|      | tol=0 | ±2 | tol=0 | ±2 | tol=0 | ±2 |
| r0   | 0.450 | 0.562 | 0.243 | 0.353 | 0.047 | 0.113 |
| r1   | 0.473 | 0.574 | 0.263 | 0.370 | 0.058 | 0.127 |
| r3   | 0.488 | 0.582 | **0.288** | **0.397** | 0.081 | 0.162 |
| r5   | **0.512** | **0.608** | 0.285 | 0.391 | **0.094** | **0.173** |

**ESMFold2 changes the qualitative picture, not just the ranking.** The means understate
it, because ESMFold's mean is propped up by a minority of successes:

| | median | ==0 | >0.5 | >0.8 |
|---|---:|---:|---:|---:|
| ESMFold2 r5 | **0.698** | 26.5% | 58.9% | 42.4% |
| ESMFold2 r0 | 0.447 | 26.5% | 49.0% | 36.4% |
| ESMFold r3   | 0.042 | 43.0% | 32.5% | 10.6% |
| Boltz-2 r5   | 0.000 | 57.0% | 6.0% | 1.3% |

ESMFold2 is 1.8× ESMFold on the mean but **17× on the median** (0.698 vs 0.042), and
substantially solves (>0.8) 42% of dimers against ESMFold's 11%. ESMFold gets *nothing*
on 43% of the split; ESMFold2 on 26.5%. Even ESMFold2's cheapest rung beats ESMFold's
best by a wide margin.

Also note ESMFold2 has **not saturated** at r5 (0.450 → 0.512, still climbing), whereas
ESMFold peaks at r3 and is flat-to-down at r5.

The ±2 tolerance gain is inversely ordered with accuracy — ESMFold2 +19–25%, ESMFold
+37–45%, Boltz-2 +84–142% — i.e. the stronger the model, the more of its hits are
already exact rather than near-misses. Read tolerant numbers against the random baseline
(0.0026 exact → 0.0234 at ±2; the positive set inflates 9.05×).

ESMFold2 runs deterministic single-pass here (see ESMFOLD2_INTEGRATION.md §7.1), so it
is if anything a floor relative to the paper's ensembled folding-eval protocol.

**ESMFold2 FLOPs are not reported**: the measurement omits the ESMC-6B backbone and
understates cost by roughly a third; the runner now refuses to emit such a sidecar. Its
accuracy numbers are unaffected — the ladder runs without `--profile`.

### Results, ESMFold and Boltz-2 only (2026-08-17, 151 dimers)

P@K is inter-chain precision at K = #true inter contacts. FLOPs are true FLOPs
(2*MACs) over the contact-map dependency subgraph, measured on a length-representative
26-entry subset (`--shard 0/6`: mean L 608 vs 611 for the full split).

| rung | ESMFold P@K | Boltz-2 (no MSA) P@K | ESMFold TFLOPs | Boltz-2 TFLOPs |
|------|-------------|----------------------|----------------|----------------|
| r0   | 0.243       | 0.047                | 56.5           | 70.6           |
| r1   | 0.263       | 0.058                | 109.3          | 141.1          |
| r3   | **0.288**   | 0.081                | 214.9          | 282.2          |
| r5   | 0.285       | **0.094**            | 332.2          | 423.2          |

- **ESMFold dominates under single-sequence conditions**: ~5x Boltz-2 at matched
  compute (r0 vs r0, within 20% on FLOPs), and its *cheapest* rung beats Boltz-2's
  *most expensive* — 0.243 at 56.5 TFLOPs vs 0.094 at 423.2. Read this as a statement
  about single-sequence priors, not about Boltz-2 in general: Boltz-2 is an MSA model
  being run in its deliberate worst case, and it says so itself in the log ("Found
  explicit empty MSA … predictions will be suboptimal").
- **Recycling saturates.** ESMFold peaks at r3 and is flat-to-down at r5 (their
  bootstrap CIs overlap); Boltz-2 is still climbing at r5.
- **The mean hides a bimodal distribution — do not quote it alone.** ESMFold r3 scores
  *exactly* 0 on 43% of dimers and >0.8 on 10.6%; Boltz-2 r5 is 0 on 57% and >0.5 on
  only 6%. These models either largely solve an interface or get nothing. Recycling
  mostly converts total failures into successes (ESMFold zeros: 51.7% at r0 -> 43.0%
  at r3) rather than sharpening already-good predictions.
- **Chebyshev tolerance** (`--tolerance 2`, GT dilated by a 5x5 L-inf ball) lifts
  ESMFold r3 to 0.397 and Boltz-2 r5 to 0.146, but it inflates the positive set 9.05x,
  so a random predictor also rises from 0.0026 to 0.0234. In enrichment-over-random
  terms both models look *worse* tolerant than exact (ESMFold r3: 111x -> 17x), i.e.
  their hits are mostly exact rather than near-misses. The ranking is unchanged.
- **Decision 8c holds** (per-entry, common subset): Boltz-2 FLOPs ratios are
  1.9988 +/- 0.0006 / 3.9963 +/- 0.0017 / 5.9938 +/- 0.0029 against 2/4/6, with an
  intercept of 0.1% of a single pass. ESMFold sits below (1.894/3.681/5.468) with a
  6.7% intercept, which is exactly right — ESM-2 runs once and only the folding trunk
  recycles.

Figure: `$DATA_ROOT/runs/recent_pp/pak_vs_flops.png` (+pdf), table:
`comparison.csv`/`.md` in the same directory.

## ESMFold2 on recent_pp

Separate venv and separate manifest — ESMFold2 needs py3.12 (ESMFold-v1's env is pinned
to py3.7 by openfold's cp37 CUDA extension) and pulls a large ESMC-6B checkpoint.

```bash
# once
bash scripts/install/esmfold2.sh          # builds .venv-esmfold2, self-checks the bin grid

# smoke, then the ladder
$PY -m ecstasy.cli run --dataset recent_pp --model esmfold2 --preset r0 --limit 1
sbatch --mem=96G scripts/run_experiment.sbatch \
       experiments/recent_pp_esmfold2_ladder.yaml --profile
```

`num_loops` is ESMFold2's recycle knob, so its r0/r1/r3/r5 ladder is directly comparable
with esmfold's `num_recycles` and boltz2's `recycling_steps`.

**Its contact threshold is an Ångström distance, not a bin index** — because the two
checkpoint families bin differently. The release checkpoint `biohub/ESMFold2` is 64 bins
on a uniform 2–22 Å grid, the same grid Boltz-2 and the MENTOS ground truth use, so
7.9375 Å is bin 19 there; the `-Experimental` checkpoints use a 128-bin ~1.5–54.5 Å grid
where the same distance is bin 16. Specifying Ångström is what lets one preset serve both.
The grid is not recoverable from the shipped code, so it was established empirically
(median GT Cβ–Cβ distance per predicted argmax bin, anchored on backbone `i, i+1` pairs);
the runner refuses any bin count it has no calibrated grid for. See
`ESMFOLD2_INTEGRATION.md`.

Two other traps, both silent rather than loud:

- Do **not** use the packaged `fold()`. It runs the full 200-step diffusion sampler and
  defaults to `lm_dropout=0.3`, which is stochastic by design — it would return plausible
  contact maps that differ between runs.
- Do **not** use a `-Cutoff2025` checkpoint; its training data overlaps the recent_pp
  holdout. The runner refuses them and checks the loaded config's cutoff.

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
