# `build_ecstasy_v1` — leakage-free dimer benchmark construction + MSA Pairformer runner

End-to-end pipeline that:

1. Builds the `ecstasy_v1` master test set (222 dimers) from Boltz-2's
   `validation_ids_v2.txt`, Foldseek-deleaked against MINT-softnano train chains
   at Pinder defaults (coverage ≥ 0.5, LDDT ≥ 0.7).
2. Runs MSA Pairformer on it in a notebook-faithful way, using the colabfold
   MMseqs2 API for paired MSAs.

All bulk data lives under `/projects/u6jv/ecstasy/benchmarks/ecstasy_v1/`.
Only code + small configs live in this repo.

## Pipeline (in order; SLURM-only for compute)

### Master set construction

| Step | Script | Job |
|---|---|---|
| 1. Enumerate dimers from Boltz-2 `validation_ids_v2.txt` | `02_enumerate_dimers.py` | runs in-process |
| 2. Extract MINT-train chain PDBs | `03_extract_train_chains.py` | runs in-process |
| 3. Foldseek all-vs-all candidates vs MINT-train | `04_run_foldseek.py` | `foldseek.sbatch` |
| 4. Compute Pinder-style interface-coverage edges | `05_compute_interface_edges.py` | (chained in `foldseek.sbatch`) |
| 5. Deleak report (threshold sweep) | `06_deleak_report.py` | (chained) |
| 6. Apply default cut + build master `.pt` GT files | `07_build_master.py` | `build_master.sbatch` |

### MSA Pairformer evaluation

| Step | Script | Job |
|---|---|---|
| 7. Fetch paired MSAs from `api.colabfold.com/ticket/pair` | `08_fetch_msas_colabfold.py` | `fetch_msas.sbatch` |
| 8. (Optional) Apply notebook `save_msa` filters (cov=75, qid=15, Δgene=1) | `11_apply_notebook_filters.py` | `apply_filters.sbatch` |
| 9. Run MSA Pairformer inference (notebook-faithful) | `09_run_msa_pairformer.py` (raw MSAs) <br> `09b_run_msa_pairformer_filtered.py` (filtered MSAs) | `inference.sbatch` / `inference_filtered.sbatch` |
| 10. Score interchain P@K (Cb + ConFind heads) | `10_score_msa_pairformer.py` | (chained) |
| 11. Compare all runs side-by-side | `12_compare_runs.py` | runs in-process |

### Debug / verification utilities

| File | Purpose |
|---|---|
| `debug_1b70.py` + `.sbatch` | Sanity-check our runner on the notebook's bundled `data/1B70_A_1B70_B.fas` |
| `debug_8dq2_monomer.py` + `.sbatch` | Reproduce notebook output on an external `msa.a3m` to verify code path |
| `common.py` | Shared utilities for CIF → bio-assembly chain extraction, Cβ-Cβ distance, interface residues |

## Prerequisites

- aarch64-built foldseek binary at `tools/foldseek/bin/foldseek` (install via `scripts/install/foldseek.sh`)
- aarch64-built hh-suite (hhfilter) at `tools/hhsuite/bin/hhfilter` (install via `scripts/install/hhsuite.sbatch`)
- Existing ecstasy `.venv-boltz` (for data wrangling) and `.venv-esmfold` (for MSA Pairformer inference)
- MINT-softnano raw CIFs at `/projects/u6jv/public/MINT/DATA/pdb/raw/cif_unzipped/`

## Headline results

- **ecstasy_v1 master set**: 222 dimers (83 homo / 139 hetero), post-Boltz-2 cutoff (2023-06-07 → 2023-12-27), Foldseek-deleaked vs MINT-train.
- **MSA Pairformer (notebook-faithful, ConFind head, all filters)**:
  mean P@K = 0.046, median 0.012, max 0.35, 15/194 entries > 20% P@K.

See `docs/` (or the Notion pages cited there) for the full audit + verification examples.
