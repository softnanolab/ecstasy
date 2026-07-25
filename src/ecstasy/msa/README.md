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
| Pairing | `--pair-mode unpaired_paired`, taxonomy/greedy; keeps **paired + unpaired** | **proximity save_msa**: UniRef accession genomic-distance ≤ **Δgene=1** + cov ≥ 0.70 + query-id ≥ 0.15, then stitch (see below) |
| Selection / depth | none — **full depth** (server-like) | **chain-aware** diversity select + cap **512** |
| Output | **per-chain CSVs** (paired/unpaired columns), Boltz format | **single complex a3m**, `#L1,L2⇥1,1` header |
| Where filtering happens | at assembly (`msa/boltz_csv.py`) | at **model load** (`msa_pairformer_runner.py`), not at generation |
| Goal | reproduce `boltz --use_msa_server` exactly | maximize inter-chain coevolution signal |
| Failure if conflated | custom a3m → `taxonomy=None` → **0 pairing** (breaks Boltz) | naive concat → hhfilter keeps one-chain-diverse rows (weak inter-chain signal) |

Store keys are **order-dependent** (`store.pair_hash = sha256("seqA\|seqB")`), so a
chain-swap experiment needs MSAs regenerated for the flipped order — not reused.

## Proximity method — `complex` (local) and `complex_api` are now identical

Both MSA-Pairformer backends apply the **same** proximity `save_msa` post-processing,
matching the *current* upstream generator `MSA_Pairformer_with_MMseqs2.ipynb`
@ yoakiyama/MSA_Pairformer main (restructured 2026-07-24 — `get_paired_msa` now
lives only in that notebook). Exact params: **neighbor_stitching=True, Δgene=1,
qid=15 → min_identity 0.15, cov=70 → min_coverage 0.70**. The server call fetches
broad (`paircomplete-pairfilterprox_20`, for cache reuse) and the operon-proximity
narrowing to Δgene≤1 is applied client-side. The 512-seq hhfilter cap stays at
model load (`msa_pairformer_runner.py`).

- **Single source of the method, two implementations:** `complex_api` uses
  `msa/colabfold.py::apply_save_msa_filters` (reads the server's per-hit
  coverage/identity metadata); `complex` (local) uses the colabfold-local submodule's
  `proximity.py` (derives coverage/identity from the alignment so as to *reproduce*
  the server's definitions — local `colabfold_search` output carries no metadata
  header). `tests/test_proximity_parity.py`
  feeds an identical fixture to both and asserts identical filter+stitch output, so they
  cannot drift.
- **Coverage/identity derivation is now server-equivalent (was a real parity gap).**
  `complex_api` reads the server's span coverage `(qend-qstart+1)/qlen` and `fident`.
  The local path previously computed coverage as non-gap/qlen and identity over
  comparable columns, so a hit with internal gaps scored *lower* locally by the
  internal-gap fraction — at `min_coverage=0.70` the local path silently dropped rows
  the API keeps, biasing local MSAs shallower (and depth is what this benchmark
  correlates against). Fixed in the **local** direction, not the notebook direction:
  `proximity._coverage_vs_query` now measures the aligned span and
  `_identity_vs_query` divides by alignment length to mirror mmseqs `fident`. Upstream
  is server/metadata-based, so the API path is the faithful one to converge on.
  `tests/integration/` cross-checks the real local-vs-API paired set where DBs +
  network are available.
- Previously `complex_api` used the wider `prox_20 / id 0.30 / cov 0.75` and `complex`
  applied **no** proximity filter at all (it ingested the raw `colabfold_search` a3m) —
  both superseded by the above.

## How to (re)generate

All via `ecstasy msa --datasets <D[,D]> --kind <kind> --phase <prepare|submit|ingest>`.

```bash
# Boltz-2 MSAs (per-chain CSVs)
ecstasy msa --datasets val_seq_chain --kind boltz_csv --phase submit   # sbatch GPU search
ecstasy msa --datasets val_seq_chain --kind boltz_csv --phase ingest   # assemble CSVs into store

# MSA-Pairformer MSAs (local colabfold-local; how the eval data was made)
ecstasy msa --datasets val_seq_chain --kind complex --phase submit     # sbatch colabfold-local
# (writes straight to the store; `--phase ingest` then just verifies coverage)

# Manual colabfold-local run already done elsewhere? ingest its a3ms (named <pair_hash>.a3m):
ecstasy msa --datasets val_seq_chain --kind complex --phase ingest --a3m_dir <dir>

# Network fallback only (NOT the eval path):
ecstasy msa --datasets val_seq_chain --kind complex_api --phase submit
```

Store layout: `$DATA_ROOT/msa_store/{boltz_csv,complex,...}/` keyed by hash.

## colabfold-local dependency (the `complex` route)

- Repo: `git@github.com:softnanolab/colabfold-local.git`
- **Pinned commit: `1817916`** ("proximity save_msa post-processing for local paired MSAs";
  colabfold-local PR #1). Bump to the merge commit once that PR lands.
- Vendored as the git submodule `third_party/colabfold-local`. Override with
  `COLABFOLD_LOCAL_DIR` (checkout) and `COLABFOLD_LOCAL_VENV` (its venv) if installed
  elsewhere (e.g. `~/colabfold-local`).
- The submit job exports `DATA_DIR=$COLABFOLD_DBS` and `MMSEQS_BIN` (ecstasy's
  mmseqs-gpu) so the engine/DBs match `boltz_csv`. colabfold-local's
  `get_paired_msa_local()` runs the local search and (for complexes) applies the
  proximity `save_msa` post-processing via colabfold-local's `proximity.py` — see
  "Proximity method" above.

## Gotcha (learned the hard way)

The in-repo `complex_api.py` (API) postdates the actual data (added 2026-05-30; the
a3ms were written 2026-05-29). The eval MSA-Pairformer MSAs came from **colabfold-local**,
not the API. Don't read the API backend and conclude the data is API-sourced.
