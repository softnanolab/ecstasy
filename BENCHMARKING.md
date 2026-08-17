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

**Its contact threshold is an Ångström distance, not a bin index.** ESMFold2's output
distogram is 128 bins over ~1.5–54.5 Å, *not* the 64-bin 2–22 Å grid that Boltz-2 and the
MENTOS ground truth share — that 64-bin grid is ESMFold2's input *conditioning*
distogram. `contact_cutoff_bin: 19` would score at ~8.9 Å instead of 7.94 Å. The runner
derives the bin index from the model's own grid and refuses to run if the head is not 128
bins. See `ESMFOLD2_INTEGRATION.md`.

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
