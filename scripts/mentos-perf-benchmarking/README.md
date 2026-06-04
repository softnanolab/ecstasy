# mentos-perf-benchmarking

Figure + analysis scripts for the contact-prediction benchmark (inter-chain P@K vs.
empirically-measured inference FLOPs across boltz2 / esmfold / msa_pairformer / mentos).
They read the run artifacts written by `ecstasy run --profile`
(`$DATA_ROOT/runs/<dataset>/<model>/<variant>/{predictions/*/contact.npz,flops.json,result.json}`)
and render the report figures.

## ⚠️ Notion is the source of truth AND the publishing target — keep it in sync

The benchmark's inputs *and* outputs live in Notion. **If you change what these scripts
produce, the matching Notion page is part of the deliverable — update it.** Do not leave
the repo and Notion telling different stories.

- **Registry (inputs) — pull, don't hand-edit.** Checkpoint/dataset *names* → concrete
  paths/params resolve from the Notion benchmarking **Registry** (`ECSTASY_REGISTRY_PAGE_ID`,
  child DBs `ECSTASY_REGISTRY_DB_{CHECKPOINTS,DATASETS,RESULTS,FIGURES}`). `notion_pull.py`
  mirrors it into a **gitignored** `registry.local.yaml` at the repo root. Re-run it whenever
  the Registry changes; never commit the cache. (Datasets also resolve from the committed
  `src/ecstasy/registry/datasets.yaml`.)

- **Figures (outputs) — push when you regenerate.** The headline figures are embedded on the
  **"P@K vs FLOPs (interface size / length / MSA depth)"** page:
  <https://www.notion.so/agrawalh/P-K-vs-FLOPs-interface-size-length-MSA-depth-374702c0f7408101b8e6d48a828c9421>
  (page id `374702c0f7408101b8e6d48a828c9421`). When you regenerate a figure, **replace the
  image on that page** (and its vector-PDF `[file]` sibling, where present) — Notion keeps its
  own copy, so a stale figure will silently persist until you re-upload. Match the figure to
  the dataset: each `P@K vs FLOPs — <split>` block is split-specific; only touch the splits you
  actually re-ran.

### Recipe — replace a figure on a Notion page (REST API)

There is no committed push helper (uploads are done ad-hoc). The reliable flow, using the
internal integration token in `.env` (`NOTION_API_TOKEN` — **secret, never commit/echo it**):

1. `POST /v1/file_uploads` with `{"filename","content_type"}` → returns `{id, upload_url}`.
2. `POST {upload_url}` as `multipart/form-data` with the file bytes → status must be `"uploaded"`
   (send only `Authorization` + `Notion-Version` headers; let the client set the multipart
   `Content-Type`).
3. `PATCH /v1/blocks/{block_id}` with `{"image":{"file_upload":{"id":<id>},"caption":[…]}}`
   for an image block, or `{"file":{…}}` for the PDF `[file]` block.

   **Gotcha:** on an *update* PATCH, do **not** include `"type":"file_upload"` inside the
   `image`/`file` object — Notion rejects it (`body.image.type should be not present`). The
   `"type"` key is only for *creating* a new block.

Use `Notion-Version: 2022-06-28`. To find the block ids, `GET /v1/blocks/{page_id}/children`
and match on the existing captions; the vector PDF is the `file` block immediately after each
image.

## Running the scripts

Run with `ecstasy` importable. These are model-agnostic (they read sidecars, not weights), so
any venv with `ecstasy` + matplotlib/scipy/requests works — e.g.:

```bash
PYTHONPATH=src DATA_ROOT=/projects/u6jv/ecstasy_data \
  <ENVS_ROOT>/.venv-boltz/bin/python scripts/mentos-perf-benchmarking/plot_pak_vs_flops.py \
    --dataset val_seq_pair [--tolerance 2] [--out plot.png]
```

(Each model has its own venv now — `.venv-mentos`, `.venv-msa_pairformer`, `.venv-boltz`, … —
needed only to *run* a model; plotting/analysis does not.) Fonts: figures use the CMU Concrete
typeface via `_plotstyle.use_cmu_concrete()` (`_plotstyle.py`).

## Scripts

| script | what it does |
|---|---|
| `plot_pak_vs_flops.py` | **The deliverable.** P@K vs. inference FLOPs, one line per model/recycle ladder. `--tolerance N` rescores saved predictions with ±N-residue (Chebyshev) spatial tolerance → `pak_vs_flops_tol{N}.png`; default is exact (`pak_vs_flops.png`). MENTOS variants rendered are whitelisted in `_MENTOS_LABEL`. |
| `plot_flops_vs_length.py` | Per-protein inference FLOPs vs. sequence length, one series per preset. |
| `plot_pak_vs_interface.py` | P@K vs. interface size — `--xmode contacts` (K) or `--xmode frac` (% of length). |
| `plot_pak_vs_msadepth.py` | Mean inter-chain P@K vs. paired-MSA depth, per model. |
| `mentos_ckpt_sweep.py` / `plot_ckpt_sweep.py` | Evaluate a MENTOS checkpoint (inter+intra AUC/P@K) across training steps and pick the best. |
| `distogram_evolution.py` / `plot_distogram_evolution.py` | Dump + faithfully render MENTOS contact-prob maps across training checkpoints (`--overlay` = 3-class GT/FP/missed colours). |
| `swap_compare.py` / `plot_swap_flops.py` | Chain-permutation experiment: original (A,B) vs swapped (B,A) order on val_seq_pair. |
| `notion_pull.py` | Mirror the Notion Registry → gitignored `registry.local.yaml` (run before resolving names offline). |
| `_plotstyle.py` | Shared CMU Concrete matplotlib style. |

## Conventions (don't re-derive — see also `../../CLAUDE.md`, `FLOPS_BENCHMARK_PLAN.md`)

- **Contact** = Cβ–Cβ distogram bin `< 19` (≤ 7.94 Å). **Inter-chain P@K** with K = #true
  inter-chain contacts. **FLOPs** = true (2×MACs), contact-map dependency subgraph only.
- **MENTOS recycling** = `model.pair_stack.num_recycles` (runs `num_recycles+1` pair-stack
  passes); FLOPs depend only on `(L, num_recycles)` + architecture (weight-independent). The
  headline checkpoint is run `a5sgd6ul` step-90000. On val_seq_pair the recycle line saturates
  after r1 (r3/r5 add ~3× FLOPs for ~0 P@K).
- **MSA pipelines are model-specific and easy to conflate** — read `src/ecstasy/msa/README.md`
  before touching anything MSA-related.
