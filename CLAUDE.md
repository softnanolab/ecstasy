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

Three rules that follow from how this repo is built:

- **A metric belongs in the registry, never in a script.** Tolerant P@K spent its life
  inside a plotting script where `ecstasy score` could not reach it, so anything wanting
  tolerance had to copy it. Add metrics in `src/ecstasy/metrics/builtins.py`; a name is
  registered once and reachable from scoring, plotting, manifests and the CLI alike.
- **A dataset row carries identity.** `version`, `description`, `expected_entries` are
  required, and `ecstasy datasets --verify` asserts the count. A split is a file nothing
  stops from changing under a published number.
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
