# ecstasy — agent notes

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

## FLOPs profiling scope (when reasoning about the numbers)

Measured inference FLOPs are the **contact-map dependency subgraph only**: Boltz-2 =
trunk + distogram (diffusion/structure module skipped); ESMFold = includes its structure
module. True FLOPs = 2×MACs via `torch.utils.flop_counter`. See `FLOPS_BENCHMARK_PLAN.md`.
