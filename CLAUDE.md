# ecstasy — agent notes

## Before starting ANY benchmarking task, look at what already exists

Do not add a dataset, a metric, or a model without first checking whether it is already
there. These commands are the source of truth and are machine-readable with `--json_out`:

```bash
ecstasy datasets            # every evaluation set: version, size, tags, what it IS
ecstasy datasets --verify   # does each split on disk still match its declared row?
ecstasy metrics             # every reusable metric, what it means, higher/lower better
ecstasy list                # models, their presets, their venvs
```

**And read `results/LEADERBOARD.md` — it says what has already been benchmarked.**
It is generated from `results/runs.jsonl` and committed, so it needs no network, no
token and no `$DATA_ROOT`. Check it before running anything: the number may already
exist. `git log -p results/runs.jsonl` shows every number that has ever moved and what
changed underneath it.

Three rules that follow from how this repo is built:

- **A result is published deliberately, and only when it is fit to be quoted.**
  `ecstasy publish` appends one line to `results/runs.jsonl`; nothing publishes itself,
  so a `--limit 1` smoke never becomes the record. It refuses incomplete coverage, any
  errored target, and a dirty ecstasy tree. It does *not* refuse a dirty model tree —
  MiniFold's `residx` patch is exactly that, and it is the intended experiment — but the
  row is flagged `†` in the leaderboard. Rows are keyed by BOTH fingerprints, so
  re-scoring after a metric fix appends a row rather than editing one.
- **A metric belongs in the registry, never in a script.** Tolerant P@K spent its life
  inside a plotting script where `ecstasy score` could not reach it, so anything wanting
  tolerance had to copy it. Add metrics in `src/ecstasy/metrics/builtins.py`; a name is
  registered once and reachable from scoring, plotting, manifests and the CLI alike.
- **A dataset row carries identity.** `version`, `description`, `expected_entries` are
  required, and `ecstasy datasets --verify` asserts the count. A split is a file nothing
  stops from changing under a published number.
- **ecstasy owns its evaluation data — never register a row pointing into MENTOS.**
  Every scorable row is `kind: ecstasy` under `${DATA_ROOT}/datasets/<name>`. The only
  place a foreign path may appear is a row's `built_from` recipe, which `load_dataset`
  drops so no scoring path can follow it, and which `ecstasy import_dataset` reads once.
  This is not tidiness: MENTOS PR #266 retired the five splits every row used to name,
  and `seq_id_30`'s parquet is already gone. A dataset folder is BUILT per machine
  (`ecstasy import_dataset --dataset recent_pp`), not committed — `--verify` says "not
  built yet" and names the command, which is the normal state on a new cluster.
- **Every model gets its own venv.** `models.yaml` names it; `adapter.py` spawns the
  runner with that venv's python; runners import no ecstasy code. This is deliberate —
  it is what stops one model's dependency tree breaking another's.

Contact-prediction benchmark: inter-chain P@K vs. empirically-measured inference FLOPs
across structure models (boltz2, esmfold, msa_pairformer, mentos). Declarative
`datasets × models`; per-model runners under `src/ecstasy/models/_runners/`.

## MSA generation — DO NOT conflate the two pipelines

There are **two distinct MSA pipelines for two different models**. Same local search
engine, different output and purpose. Getting these mixed up silently corrupts results.

- **Boltz-2** → `msa: boltz_csv` → local `colabfold_search` → **paired+unpaired per-chain
  CSVs** (reproduces `boltz --use_msa_server`; a plain `.a3m` would drop ALL pairing).
- **MSA-Pairformer** → `msa: complex` → external **softnanolab/colabfold-local** (pinned
  `third_party/colabfold-local`) → **stitched complex a3m**; paired-filter + chain-aware
  select + depth-512 happen at model load, not at generation.
- `complex_api` (ColabFold API) is a **fallback only** — it is NOT how the eval MSAs were
  generated (despite what the code's recency might suggest).

Before changing anything MSA-related, read **`src/ecstasy/msa/README.md`** (the full
model→pipeline map, the differences table, exact `ecstasy msa …` commands, and the
colabfold-local pin). Store keys are order-dependent (`pair_hash`).

## DockQ: never read it without iRMSD, and never trust an early checkpoint

DockQ averages fnat with two RMSD terms. When nothing is actually docked, both RMSD terms
give near-zero credit and **fnat carries the score** — so a model that has docked nothing
lands on a non-trivial DockQ, and lands *highest* exactly where its geometry is worst.

This is not theoretical. In one MENTOS-vs-MiniFold campaign it produced four wrong
conclusions before it was caught:

1. a comparison run against MENTOS step 2000 (iRMSD 21.2 Å) instead of the converged model
2. "MENTOS never clears medium" — true only of steps 2000-12000
3. a median inversion that was an artefact of step 2000 being the best-median checkpoint
4. "on a typical target the two are indistinguishable" — true at steps 2000/4000, false
   against the other 21 checkpoints

Verified per-target head-to-head vs MiniFold, full 151 each:

| MENTOS checkpoint | MENTOS wins | MiniFold wins | MENTOS iRMSD |
|---|---|---|---|
| 4000 | 75 | 76 | 21.38 Å |
| 14000 | 58 | 89 | 13.13 Å |
| 22000 | 59 | 87 | 13.99 Å |
| 50000 | 57 | 92 | 13.71 Å |

MENTOS looks competitive precisely where its interfaces are worst. **Rules:** report iRMSD
and LRMSD beside every DockQ; never lead on median (the statistic most corrupted by the
fnat floor); and treat any early-checkpoint DockQ as suspect until its iRMSD is checked.

Related: selecting the best of N checkpoints **on the evaluation set** is test-set
selection. Such a number is an upper bound on that model, not its performance, and makes
the opposing model's margin a lower bound. Say which, and name the selection criterion —
"best checkpoint" means six different checkpoints depending on the metric.

## Provenance — a result names the code that produced it

Every run writes `provenance.json` beside `result.json`, and the record is embedded in
`result.json` too so it survives the run directory. It carries the ecstasy commit + dirty
flag, every submodule SHA (and whether each is at its pin), and — for any *path* in the
model's params — the git state of that source tree and the byte identity of that weight
file, following symlinks.

This is not reproducibility hygiene, it is correctness. The MiniFold runner takes
`minifold_src` as a path, and whether the `residx` patch is applied inside that tree is
the entire difference between the intended chain break and the linker-only variant the
user rejected. Before provenance, those two *different experiments* wrote byte-identical
records. Same failure mode: a sweep whose `src/` is edited mid-run silently mixes code
versions and looks completely normal.

If you add a runner that reaches outside its params for code or weights, add it to the
params so provenance can see it.

## FLOPs profiling scope (when reasoning about the numbers)

Measured inference FLOPs are the **contact-map dependency subgraph only**: Boltz-2 =
trunk + distogram (diffusion/structure module skipped); ESMFold = includes its structure
module. True FLOPs = 2×MACs via `torch.utils.flop_counter`. See `FLOPS_BENCHMARK_PLAN.md`.
