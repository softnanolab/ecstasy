# ecstasy — agent notes

Contact-prediction benchmark: inter-chain P@K vs. empirically-measured inference FLOPs
across structure models (boltz2, esmfold, msa_pairformer, mentos, minifold). Declarative
`datasets × models`; per-model runners under `src/ecstasy/models/_runners/`.

A **second, optional axis** measures docking, not contacts: a runner may additionally
emit `structure.npz` (atom37), and datasets with full-atom GT then score it with DockQ /
iRMSD / LRMSD plus per-chain monomer metrics. It is strictly additive — `contact.npz` is
still required of every runner and the contact path is unchanged. See BENCHMARKING.md
"Structure metrics"; the comparability rules there (one PDB writer, a fixed no-flag DockQ
invocation, never reading DockQ without the RMSDs) are not stylistic.

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

**MiniFold is the exception** and records it: unlike ESMFold, its structure module is
*not* on the contact path (the distogram leaves `self.fold` before the structure module
runs), but it must run for the DockQ axis. Its `flops.json` therefore carries
`scope: full_model_incl_structure_module` — right for DockQ, an over-count for P@K.
Do not pool it with the other rows on the P@K-vs-FLOPs plot without saying so.

## Distogram bins are per-model — `contact_cutoff_bin` does NOT transfer

`contact_cutoff_bin: 19` means ~8 Å only under ESMFold's/Boltz-2's binning. It is an
*index into that model's bin edges*, and `probs[..., :k].sum(-1)` is `P(d ≤ edges[k-1])`.
MiniFold bins over `linspace(2, 25, 63)`, so its matching value is **17** (7.9355 Å);
19 would silently mean 8.68 Å and inflate its P@K. Before adding any distogram model,
read its bin edges out of the checkpoint and derive the index — never copy 19.

Related: MiniFold's distogram is **CA–CA**, while ecstasy's GT and the other heads are
**Cβ–Cβ**. No bin index fixes that; it must be stated wherever MiniFold's P@K is published.
