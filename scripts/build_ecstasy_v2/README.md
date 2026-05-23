# `build_ecstasy_v2` — leakage-free dimer benchmark (post-Boltz-2 cutoff)

End-to-end pipeline that builds **`ecstasy_v2`**: all post-Boltz-2-cutoff
real PDB dimers, Foldseek-deleaked against **both** the Mentos training
chain set and a conservative approximation of the Boltz-2 training chain
set, at Pinder Level-2 defaults (coverage ≥ 0.5, LDDT ≥ 0.7).

Compared to `ecstasy_v1` (222 dimers, deleaked vs Mentos only, sourced
from Boltz-2's `validation_ids_v2.txt`):

| Aspect | v1 | v2 |
|---|---|---|
| Candidate source | `validation_ids_v2.txt` (1611 IDs) | RCSB Search API: `initial_release_date ≥ 2023-06-01`, X-ray, ≥2 protein chains |
| Train-chain DB(s) | Mentos train (≤ 2021-09-30) | **Mentos train + Boltz-2 train** (≤ 2023-06-01) |
| Deleak rule | Either chain hits Mentos at cov ≥ 0.5 ∧ LDDT ≥ 0.7 | Either chain hits **either** DB at the same thresholds |
| Master schema | `index.parquet` + `data/<id[:2]>/<id>.pt` | identical (drop-in for the existing benchmark loader) |

All bulk data lives under `/projects/u6jv/ecstasy/benchmarks/ecstasy_v2/`.
Only code + small configs live in this repo.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. RCSB candidate search + metadata | `01_query_rcsb_candidates.py` | `candidates/pdb_ids.txt`, `candidates/rcsb_metadata.parquet` |
| 2. Enumerate candidate dimers | `02_enumerate_dimers.py` | `candidates/dimers.parquet`, `candidates/chains/*.pdb` |
| 3a. Mentos-train chains | `03a_extract_mentos_chains.py` | `train_db/mentos_train_chains/*.pdb` (symlinks v1 if available) |
| 3b. Boltz-2 delta chains | `03b_extract_boltz2_delta_chains.py` | `train_db/boltz2_delta_chains/*.pdb` (PDB released 2021-10-01 .. 2023-05-31) |
| 4. Foldseek easy-search (both DBs) | `04_run_foldseek.py` | `foldseek_hits_mentos.m8`, `foldseek_hits_boltz2.m8` |
| 5. Interface-coverage edges | `05_compute_interface_edges.py` | `interface_edges.parquet`, `all_edges_with_coverage.parquet` |
| 7. Apply cut + build master `.pt` set | `07_build_master.py` | `master/index.parquet`, `master/data/<id[:2]>/<id>.pt`, `master/master_README.md` |

(Step numbers skip 06 to match v1's layout: 06 was reserved for a deleak
report in v1; v2's per-source attribution lives in `dropped_dimers.parquet`.)

## Design choices (vs the v2 issue's open questions)

1. **Boltz-2 training chain list.** No public explicit list, so v2 approximates
   it as `Mentos chains ∪ all PDB chains released 2021-10-01 .. 2023-05-31`.
   This is a strict superset and therefore **conservative** for deleaking —
   it will not produce false negatives (missed leaks), only false positives
   (over-strict drops). Acceptable for a test-set construction.
2. **Scale.** RCSB returns ~30k post-cutoff X-ray protein-multimer entries.
   v1's ~14% candidate-hit-rate suggests ~4k candidate dimers pre-deleak;
   after dual-DB deleak we expect a few hundred. Headline counts land in
   `master/master_README.md`.
3. **Foldseek strategy.** Two separate `easy-search` runs (Mentos DB and
   Boltz-2 DB) — preserves per-source attribution in the dropped-dimer
   manifest. The Boltz-2 DB is built by `createdb`'ing both chain
   directories together.
4. **Mentos rename.** v2 scripts use `mentos` in identifiers and on-disk
   layout under `ecstasy_v2/train_db/mentos_train_chains/`. On-disk data
   paths under `/projects/u6jv/public/MINT/...` are unchanged pending #11.

## Prerequisites

- aarch64-built foldseek at `tools/foldseek/bin/foldseek` (install via
  `scripts/install/foldseek.sh`)
- Existing ecstasy `.venv-boltz` for data wrangling
- Mentos raw CIFs at `/projects/u6jv/public/MINT/DATA/pdb/raw/cif_unzipped/`
  (only needed if `ecstasy_v1/train_db/mint_train_chains/` is missing —
  otherwise step 3a reuses v1's outputs via symlink)
- Outbound HTTPS to `data.rcsb.org`, `search.rcsb.org`, `files.rcsb.org`
  (~50k delta CIF downloads in step 3b; cache them with care)
