# mentos-perf-benchmarking

Figure + analysis scripts for the contact-prediction benchmark (inter-chain P@K vs.
empirically-measured inference FLOPs across boltz2 / esmfold / msa_pairformer / mentos).
They read the run artifacts written by `ecstasy run --profile`
(`$DATA_ROOT/runs/<dataset>/<model>/<variant>/{predictions/*/contact.npz,flops.json,result.json}`)
and render the report figures.

## Environment

Every script (and CLI call) needs the package on `PYTHONPATH` and the data-root env
vars resolved by `ecstasy.config.settings`. Run with the project venv:

```bash
export PYTHONPATH=src                # plots resolve _plotstyle from their own dir
export DATA_ROOT=…   MENTOS_ROOT=…   ECSTASY_ROOT=…   ENVS_ROOT=…   TOOLS_ROOT=…
export COLABFOLD_DBS=…            # local ColabFold DBs (MSA generation)
export LOGS_DIR=…                 # W&B run files (mentos config lookup)
PYB="$ENVS_ROOT/.venv-boltz/bin/python"   # 3.12; has ecstasy + matplotlib
```

Paths come from `settings()` (e.g. `settings().runs_root`), never hardcoded. CMU
Concrete figures look for the font in `$CMU_FONT_DIR` (default `~/.fonts`); they fall
back to the default font if it's not installed.

## Registry (inputs) — committed, hand-edited

Checkpoint/dataset *names* → concrete paths/params resolve from committed registry files:
checkpoints from `src/ecstasy/registry/checkpoints.yaml`, datasets from
`src/ecstasy/registry/datasets.yaml`. Add or edit a row directly and commit it — there is
no external registry to pull from or keep in sync.

```bash
# checkpoint names resolve from the committed src/ecstasy/registry/checkpoints.yaml
$PYB -m ecstasy.cli run --dataset val_seq_pair --model mentos --checkpoint a5sgd6ul_s90k --profile
```

`mentos_ckpt_sweep.py` and `distogram_evolution.py` likewise take `--checkpoint <name>` and
derive the run's checkpoint directory from the registry.

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
  <ENVS_ROOT>/.venv-boltz/bin/python experiments/mentos-perf-benchmarking/plot_pak_vs_flops.py \
    --dataset val_seq_pair [--tolerance 2] [--out plot.png]
```

(Each model has its own venv now — `.venv-mentos`, `.venv-msa_pairformer`, `.venv-boltz`, … —
needed only to *run* a model; plotting/analysis does not.) Fonts: figures use the CMU Concrete
typeface via `_plotstyle.use_cmu_concrete()` (`_plotstyle.py`).

## Scripts (read `runs/<split>/…`, write PNG + PDF)

| Script | Example |
|---|---|
| `plot_pak_vs_flops.py` | **The deliverable.** P@K vs. inference FLOPs, one line per model/recycle ladder. `--tolerance N` rescores saved predictions with ±N-residue (Chebyshev) spatial tolerance → `pak_vs_flops_tol{N}.png`; default is exact (`pak_vs_flops.png`). MENTOS variants rendered are whitelisted in `_MENTOS_LABEL`. `experiments/mentos-perf-benchmarking/plot_pak_vs_flops.py --dataset val_seq_pair [--annotate-r0] [--exclude-models boltz2_nomsa]` |
| `plot_flops_vs_length.py` | Per-protein inference FLOPs vs. sequence length, one series per preset. `experiments/mentos-perf-benchmarking/plot_flops_vs_length.py --dataset val_seq_chain --model esmfold --presets r0,r1,r3,r5 [--style line]` |
| `plot_pak_vs_interface.py` | P@K vs. interface size — `--xmode contacts` (K) or `--xmode frac` (% of length). `experiments/mentos-perf-benchmarking/plot_pak_vs_interface.py --dataset val_seq_pair --xmode {contacts,percent} --cap 800` |
| `plot_pak_vs_msadepth.py` | Mean inter-chain P@K vs. paired-MSA depth, per model. `experiments/mentos-perf-benchmarking/plot_pak_vs_msadepth.py --depth {paired,total}` |
| `mentos_ckpt_sweep.py` / `plot_ckpt_sweep.py` | Evaluate a MENTOS checkpoint (inter+intra AUC/P@K) across training steps and pick the best. `experiments/mentos-perf-benchmarking/plot_ckpt_sweep.py --results-dir <dir> --out ckpt_sweep.png` |
| `distogram_evolution.py` / `plot_distogram_evolution.py` | Dump + faithfully render MENTOS contact-prob maps across training checkpoints (`--overlay` = 3-class GT/FP/missed colours). `experiments/mentos-perf-benchmarking/distogram_evolution.py --ids 8pdc,9uc5 --checkpoint a5sgd6ul_s90k --out-dir <dir>` (GPU) |
| `swap_compare.py` / `plot_swap_flops.py` | Chain-permutation experiment: original (A,B) vs swapped (B,A) order on val_seq_pair. `experiments/mentos-perf-benchmarking/plot_swap_flops.py` (original vs swapped overlay) |
| `_plotstyle.py` | Shared CMU Concrete matplotlib style. |

## Eval / sweep workflows (CLI)

**FLOPs recycle sweep** — one GPU run per (model, preset); `--profile` writes the
`flops.json` sidecars, then it auto-scores P@K:

```bash
for split in val_seq_chain val_seq_pair val_pinder_chain val_pinder_pair; do
  for preset in r0 r1 r3 r5; do
    $PYB -m ecstasy.cli run --dataset $split --model boltz2       --preset $preset --profile
    $PYB -m ecstasy.cli run --dataset $split --model boltz2_nomsa --preset $preset --profile
    $PYB -m ecstasy.cli run --dataset $split --model esmfold      --preset $preset --profile
  done
done
$PYB -m ecstasy.cli run --dataset $split --model mentos         --checkpoint a5sgd6ul_s90k --profile  # name from registry/checkpoints.yaml
$PYB -m ecstasy.cli run --dataset $split --model msa_pairformer --preset full               --profile
```

(In practice each `(split, model, preset)` cell is one SLURM array task — see the
template below.)

**Chain-order (swap) experiment** — isolated under the `val_seq_pair_swapped` dataset
(chains flipped + GT reindexed; registered in `registry/datasets.yaml`):

```bash
# 1. regenerate Boltz MSAs from scratch for the flipped order (new pair-hashes)
$PYB -m ecstasy.cli msa --datasets val_seq_pair_swapped --kind boltz_csv --phase submit   # GPU
$PYB -m ecstasy.cli msa --datasets val_seq_pair_swapped --kind boltz_csv --phase ingest

# 2. eval the order-sensitive models across recycles
for m in boltz2 boltz2_nomsa esmfold; do for p in r0 r1 r3 r5; do
  $PYB -m ecstasy.cli run --dataset val_seq_pair_swapped --model $m --preset $p --profile
done; done

# 3. compare original vs swapped
$PYB experiments/mentos-perf-benchmarking/swap_compare.py     # ΔP@K table + swap_scatter_<model>.png
$PYB experiments/mentos-perf-benchmarking/plot_swap_flops.py  # P@K-vs-FLOPs overlay (A,B) vs (B,A)
```

MSA generation details (Boltz `boltz_csv` vs MSA-Pairformer `complex`/`complex_api`) are
documented in [`../../src/ecstasy/msa/README.md`](../../src/ecstasy/msa/README.md) — **do not
conflate the two pipelines.**

## SLURM (local wrappers, not committed)

Generate a wrapper locally; resolve paths from the submit dir, don't hardcode them:

```bash
#!/bin/bash
#SBATCH --job-name=ecstasy --gpus-per-node=1 --ntasks-per-node=1 --time=24:00:00
#SBATCH --array=0-15%16
set -uo pipefail
WT="${SLURM_SUBMIT_DIR:?}"                 # repo root — not a hardcoded path
export PYTHONPATH="$WT/src"
export DATA_ROOT=…  MENTOS_ROOT=…  ECSTASY_ROOT=…  ENVS_ROOT=…  TOOLS_ROOT=…  COLABFOLD_DBS=…  LOGS_DIR=…
PYB="$ENVS_ROOT/.venv-boltz/bin/python"
CELLS=( "val_seq_pair boltz2 r0" "val_seq_pair boltz2 r1" … )   # one (split model preset) per index
read -r split model preset <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
$PYB -m ecstasy.cli run --dataset "$split" --model "$model" --preset "$preset" --profile
```

## Conventions (don't re-derive — see also `../../CLAUDE.md`, `FLOPS_BENCHMARK_PLAN.md`)

- **Contact** = Cβ–Cβ distogram bin `< 19` (≤ 7.94 Å). **Inter-chain P@K** with K = #true
  inter-chain contacts. **FLOPs** = true (2×MACs), contact-map dependency subgraph only.
- **MENTOS recycling** = `model.pair_stack.num_recycles` (runs `num_recycles+1` pair-stack
  passes); FLOPs depend only on `(L, num_recycles)` + architecture (weight-independent). The
  headline checkpoint is run `a5sgd6ul` step-90000. On val_seq_pair the recycle line saturates
  after r1 (r3/r5 add ~3× FLOPs for ~0 P@K).
- **MSA pipelines are model-specific and easy to conflate** — read `src/ecstasy/msa/README.md`
  before touching anything MSA-related.
