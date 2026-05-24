# `build_ecstasy_v1` — leakage-free dimer benchmark construction + MSA Pairformer runner

End-to-end pipeline that:

1. Builds the `ecstasy_v1` master test set (222 dimers) from Boltz-2's
   `validation_ids_v2.txt`, Foldseek-deleaked against MENTOS-softnano train chains
   at Pinder defaults (coverage ≥ 0.5, LDDT ≥ 0.7).
2. Runs MSA Pairformer on it in a notebook-faithful way, using the colabfold
   MMseqs2 API for paired MSAs.

All bulk data lives under `/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/`.
Only code + small configs live in this repo.

## Pipeline

Each script is a plain Python entrypoint with hard-coded input/output paths
under `/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/`. Wrap in your own
scheduler (SLURM, Kubernetes, local cron, …) as appropriate.

### Master set construction

| Step | Script |
|---|---|
| 1. Enumerate dimers from Boltz-2 `validation_ids_v2.txt` | `02_enumerate_dimers.py` |
| 2. Extract MENTOS-train chain PDBs | `03_extract_train_chains.py` |
| 3. Foldseek all-vs-all candidates vs MENTOS-train | `04_run_foldseek.py` |
| 4. Compute Pinder-style interface-coverage edges | `05_compute_interface_edges.py` |
| 5. Apply default cut (cov ≥ 0.5, LDDT ≥ 0.7) + build master `.pt` GT files | `07_build_master.py` |

### MSA Pairformer evaluation

| Step | Script |
|---|---|
| 6. Fetch paired MSAs from `api.colabfold.com/ticket/pair` | `08_fetch_msas_colabfold.py` |
| 7. (Optional) Apply notebook `save_msa` filters (cov=75, qid=15, Δgene=1) | `11_apply_notebook_filters.py` |
| 8. Run MSA Pairformer inference (notebook-faithful) | `09_run_msa_pairformer.py` (raw MSAs by default; pass `--msas-dir .../msas_filtered` for the notebook-`save_msa`-filtered variant) |
| 9. Score interchain P@K (Cb + ConFind heads) | `10_score_msa_pairformer.py` |


## Prerequisites

- aarch64-built foldseek binary at `tools/foldseek/bin/foldseek` (install via `scripts/install/foldseek.sh`)
- aarch64-built hh-suite (hhfilter) at `tools/hhsuite/bin/hhfilter` (install via `scripts/install/hhsuite.sh`)
- Existing ecstasy `.venv-boltz` (for data wrangling) and `.venv-esmfold` (for MSA Pairformer inference)
- MENTOS-softnano raw CIFs at `/projects/u6jv/public/MENTOS/DATA/pdb/raw/cif_unzipped/`

## Headline results

- **ecstasy_v1 master set**: 222 dimers (83 homo / 139 hetero), post-Boltz-2 cutoff (2023-06-07 → 2023-12-27), Foldseek-deleaked vs MENTOS-train.
- **MSA Pairformer (notebook-faithful, ConFind head, all filters)**:
  mean P@K = 0.046, median 0.012, max 0.35, 15/194 entries > 20% P@K.

See `docs/` (or the Notion pages cited there) for the full audit + verification examples.

## Analysis / one-off figures

One-off analysis scripts (deleak report figure, all-runs comparison figure, 1B70
and 8dq2 debug-reproduction runners) are intentionally **not versioned**. They
are regenerable from the parquets on `/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/`
and the rendered outputs live on the corresponding Notion pages.
