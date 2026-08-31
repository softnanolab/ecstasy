# MSA generation in ecstasy

**Read this before touching MSAs.** There are two distinct MSA pipelines for two
different models. They use the *same* local search engine but produce *different*
outputs for *different* purposes — conflating them silently corrupts results.

## Model → pipeline map

| Model (`models.yaml`) | `msa:` kind | What it gets |
|---|---|---|
| `boltz2` | `boltz_csv` | Boltz-2 paired+unpaired **per-chain CSVs** |
| `msa_pairformer` | `complex` | MSA-Pairformer **stitched complex a3m** |
| `boltz2_nomsa`, `esmfold`, `mentos` | `none` | single-sequence (no MSA) |
| `colabfold` | `none` | colabfold_batch fetches its own |

Both `boltz_csv` and `complex` are **local** (GPU `mmseqs2` against `COLABFOLD_DBS`).
`complex_api` is a network fallback (api.colabfold.com) and was **NOT** used for the
benchmark data — don't assume it was.

## Boltz-2 (`boltz_csv`) vs MSA-Pairformer (`complex`) — the differences

| | `boltz_csv` (Boltz-2) | `complex` (MSA-Pairformer) |
|---|---|---|
| Generator | in-repo `backends/boltz_csv.py` (sbatch) | external **softnanolab/colabfold-local** via `backends/complex.py` |
| Engine / DBs | local `colabfold_search` (mmseqs-gpu) / `COLABFOLD_DBS` | *same engine + DBs* |
| Pairing | `--pair-mode unpaired_paired`, taxonomy/greedy; keeps **paired + unpaired** | drops unpaired via **paired-sequence filter** (both chains <50% gaps) |
| Selection / depth | none — **full depth** (server-like) | **chain-aware** diversity select + cap **512** |
| Output | **per-chain CSVs** (paired/unpaired columns), Boltz format | **single complex a3m**, `#L1,L2⇥1,1` header |
| Where filtering happens | at assembly (`msa/boltz_csv.py`) | at **model load** (`msa_pairformer_runner.py`), not at generation |
| Goal | reproduce `boltz --use_msa_server` exactly | maximize inter-chain coevolution signal |
| Failure if conflated | custom a3m → `taxonomy=None` → **0 pairing** (breaks Boltz) | naive concat → hhfilter keeps one-chain-diverse rows (weak inter-chain signal) |

Store keys are **order-dependent** (`store.pair_hash = sha256("seqA\|seqB")`), so a
chain-swap experiment needs MSAs regenerated for the flipped order — not reused.

## How to (re)generate

All via `ecstasy msa --datasets <D[,D]> --kind <kind> --phase <prepare|submit|ingest>`.

```bash
# Boltz-2 MSAs (per-chain CSVs)
ecstasy msa --datasets recent_pp --kind boltz_csv --phase submit   # sbatch GPU search
ecstasy msa --datasets recent_pp --kind boltz_csv --phase ingest   # assemble CSVs into store

# MSA-Pairformer MSAs (local colabfold-local; how the eval data was made)
ecstasy msa --datasets recent_pp --kind complex --phase submit     # sbatch colabfold-local
# (writes straight to the store; `--phase ingest` then just verifies coverage)

# Manual colabfold-local run already done elsewhere? ingest its a3ms (named <pair_hash>.a3m):
ecstasy msa --datasets recent_pp --kind complex --phase ingest --a3m_dir <dir>

# Network fallback only (NOT the eval path):
ecstasy msa --datasets recent_pp --kind complex_api --phase submit
```

Store layout: `$DATA_ROOT/msa_store/{boltz_csv,complex,...}/` keyed by hash.

## colabfold-local dependency (the `complex` route)

- Repo: `git@github.com:softnanolab/colabfold-local.git`
- **Pinned commit: `38088c9`** ("chain-aware MSA selection + paired filtering for complexes")
- Vendored as the git submodule `third_party/colabfold-local`. Override with
  `COLABFOLD_LOCAL_DIR` (checkout) and `COLABFOLD_LOCAL_VENV` (its venv) if installed
  elsewhere (e.g. `~/colabfold-local`).
- The submit job exports `DATA_DIR=$COLABFOLD_DBS` and `MMSEQS_BIN` (ecstasy's
  mmseqs-gpu) so the engine/DBs match `boltz_csv`. colabfold-local's
  `get_paired_msa_local()` is the exact function the eval used.

## Gotcha (learned the hard way)

The in-repo `complex_api.py` (API) postdates the actual data (added 2026-05-30; the
a3ms were written 2026-05-29). The eval MSA-Pairformer MSAs came from **colabfold-local**,
not the API. Don't read the API backend and conclude the data is API-sourced.
