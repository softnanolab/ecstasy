# scripts/ — benchmarking projects

Each benchmarking campaign gets its own subdirectory. The current one is
**`mentos-perf-benchmarking/`** (inter-chain P@K vs. inference-FLOPs across structure models,
the chain-order swap experiment, the checkpoint sweep, distogram-evolution, the figures, and
the Notion-registry sync). `install/` holds environment setup. The **eval/sweep work is plain
`ecstasy …` CLI calls**; the scripts only plot, sweep, and sync.

Concrete pointers (checkpoint & dataset paths) are **not committed** — they live in the Notion
benchmarking Registry and are resolved by name (see "Checkpoints & datasets" below and the
top-level README → Configuration).

> SLURM `*.sbatch` wrappers are **not committed** — they hardcode per-cluster/per-job
> paths. Keep them local (they're git-ignored) and use the portable commands below.
> A minimal, parameterized template is at the end of this file.

## Environment

Every script (and CLI call) needs the package on `PYTHONPATH` and the data-root env
vars resolved by `ecstasy.config.settings`. Run with the project venv:

```bash
# The editable installs of `ecstasy` and `mentos` are broken in every venv on this
# machine (they point at a deleted worktree), so BOTH source trees go on PYTHONPATH and
# you invoke `python -m ecstasy.cli`, never the `ecstasy` console script. `mentos` is
# needed because scoring unpickles a `mentos.dataclasses.Sample`.
export PYTHONPATH=src:<mentos-checkout>/src   # plots resolve _plotstyle from their own dir
export DATA_ROOT=…   MENTOS_ROOT=…   ECSTASY_ROOT=…   ENVS_ROOT=…   TOOLS_ROOT=…
export COLABFOLD_DBS=…            # local ColabFold DBs (MSA generation)
export LOGS_DIR=…                 # W&B run files (mentos config lookup)
PYB="$ENVS_ROOT/.venv-mentos/bin/python"  # 3.12; torch 2.12 + pandas + fire + yaml
```

Paths come from `settings()` (e.g. `settings().runs_root`), never hardcoded. CMU
Concrete figures look for the font in `$CMU_FONT_DIR` (default `~/.fonts`); they fall
back to the default font if it's not installed.

## Checkpoints & datasets (Notion registry)

MENTOS checkpoints and validation datasets are referenced by **name**, never path; the
name → concrete-path mapping lives in the Notion benchmarking Registry. Pull it into a
gitignored local cache once (re-run whenever the Registry changes), then select by name:

```bash
$PYB scripts/mentos-perf-benchmarking/notion_pull.py        # writes registry.local.yaml (gitignored)
$PYB -m ecstasy.cli run --dataset recent_pp --model mentos --checkpoint a5sgd6ul_s90k --profile
```

`mentos_ckpt_sweep.py` and `distogram_evolution.py` likewise take `--checkpoint <name>` and
derive the run's checkpoint directory from the registry.

## Figures (read `runs/<split>/…`, write PNG + PDF)

All scripts live in `scripts/mentos-perf-benchmarking/` (abbreviated `MPB/` below).

| Script | Example |
|---|---|
| `MPB/plot_pak_vs_flops.py` | `$PYB scripts/mentos-perf-benchmarking/plot_pak_vs_flops.py --dataset recent_pp [--annotate-r0] [--exclude-models boltz2_nomsa]` |
| `MPB/plot_flops_vs_length.py` | `$PYB scripts/mentos-perf-benchmarking/plot_flops_vs_length.py --dataset recent_pp --model esmfold --presets r0,r1,r3,r5 [--style line]` |
| `MPB/plot_pak_vs_interface.py` | `$PYB scripts/mentos-perf-benchmarking/plot_pak_vs_interface.py --dataset recent_pp --xmode {contacts,percent} --cap 800` |
| `MPB/plot_pak_vs_msadepth.py` | `$PYB scripts/mentos-perf-benchmarking/plot_pak_vs_msadepth.py --depth {paired,total}` |
| `MPB/plot_swap_flops.py` | `$PYB scripts/mentos-perf-benchmarking/plot_swap_flops.py` (original vs swapped overlay) |
| `MPB/swap_compare.py` | `$PYB scripts/mentos-perf-benchmarking/swap_compare.py` (ΔP@K table + scatters, orig vs swapped) |
| `MPB/plot_ckpt_sweep.py` | `$PYB scripts/mentos-perf-benchmarking/plot_ckpt_sweep.py --results-dir <dir> --out ckpt_sweep.png` |
| `MPB/distogram_evolution.py` | `$PYB scripts/mentos-perf-benchmarking/distogram_evolution.py --ids 8pdc,9uc5 --checkpoint a5sgd6ul_s90k --out-dir <dir>` (GPU) |

## Eval / sweep workflows (CLI)

**FLOPs recycle sweep** — one GPU run per (model, preset); `--profile` writes the
`flops.json` sidecars, then it auto-scores P@K:

```bash
for split in recent_pp; do            # recent_pp is the only registered split now
  for preset in r0 r1 r3 r5; do
    $PYB -m ecstasy.cli run --dataset $split --model boltz2       --preset $preset --profile
    $PYB -m ecstasy.cli run --dataset $split --model boltz2_nomsa --preset $preset --profile
    $PYB -m ecstasy.cli run --dataset $split --model esmfold      --preset $preset --profile
  done
done
$PYB -m ecstasy.cli run --dataset $split --model mentos         --checkpoint a5sgd6ul_s90k --profile  # name from the Notion registry
$PYB -m ecstasy.cli run --dataset $split --model msa_pairformer --preset full               --profile
```

(In practice each `(split, model, preset)` cell is one SLURM array task — see the
template below.)

**Sharded sweep, then finish it.** A whole manifest can be spread over N concurrent jobs
with `--shard i/N`; shards skip entries whose `contact.npz` (plus `flops.json` under
`--profile`) already exists, so they never collide and are resumable. Short jobs also
backfill far better than one long one on a contested queue.

```bash
N=12
for i in $(seq 0 $((N-1))); do
  sbatch --time=08:00:00 --cpus-per-task=8 --job-name="ecst-s${i}" \
         scripts/run_experiment.sbatch experiments/recent_pp_nomsa_ladder.yaml --shard "${i}/${N}"
done
```

Scoring is deliberately suppressed while sharding — each shard sees only its own slice,
so letting it score would race the other shards on the same `result.json` and persist a
summary over a partial set. Finish with an unsharded pass over the same manifest, which
re-uses every prediction on disk and only scores:

```bash
# 0. anything still missing? (expect 8 runs x 151 entries = 1208)
find "$DATA_ROOT/runs/recent_pp" -name contact.npz | wc -l
# resubmit the same shard command to pick up stragglers — it is idempotent

# 1. score (predictions are skipped, so this is cheap and CPU-only)
$PYB -m ecstasy.cli experiment experiments/recent_pp_nomsa_ladder.yaml

# 2. comparison table -> runs/recent_pp/comparison.csv + .md
$PYB -m ecstasy.cli compare --dataset recent_pp

# 3. figure (95% bootstrap CI over proteins, deterministic seed)
$PYB scripts/mentos-perf-benchmarking/plot_pak_vs_flops.py --dataset recent_pp
```

Step 3 needs `flops.json` sidecars; without them there is no x-axis. See
`FLOPS_BENCHMARK_PLAN.md` §2.1 before trusting any FLOPs number.

**Chain-order (swap) experiment** — NOT currently registered. `swap_chains: true` is
still implemented by `MentosSquareDataset`, but the `val_seq_pair_swapped` row went away
with its parquet. To run it again, add a swapped row for `recent_pp` under its own name
(it needs its own run dir, and pair-hash MSA keys are order-dependent), then substitute
that name below:

```bash
# 1. regenerate Boltz MSAs from scratch for the flipped order (new pair-hashes)
$PYB -m ecstasy.cli msa --datasets val_seq_pair_swapped --kind boltz_csv --phase submit   # GPU
$PYB -m ecstasy.cli msa --datasets val_seq_pair_swapped --kind boltz_csv --phase ingest

# 2. eval the order-sensitive models across recycles
for m in boltz2 boltz2_nomsa esmfold; do for p in r0 r1 r3 r5; do
  $PYB -m ecstasy.cli run --dataset val_seq_pair_swapped --model $m --preset $p --profile
done; done

# 3. compare original vs swapped
$PYB scripts/mentos-perf-benchmarking/swap_compare.py     # ΔP@K table + swap_scatter_<model>.png
$PYB scripts/mentos-perf-benchmarking/plot_swap_flops.py  # P@K-vs-FLOPs overlay (A,B) vs (B,A)
```

MSA generation details (Boltz `boltz_csv` vs MSA-Pairformer `complex`/`complex_api`) are
documented in [`../src/ecstasy/msa/README.md`](../src/ecstasy/msa/README.md) — **do not
conflate the two pipelines.**

## SLURM (local wrappers, not committed)

Generate a wrapper locally; resolve paths from the submit dir, don't hardcode them:

```bash
#!/bin/bash
#SBATCH --job-name=ecstasy --gpus-per-node=1 --ntasks-per-node=1 --time=24:00:00
#SBATCH --array=0-15%16
#SBATCH --partition=gpu --qos=freegpu      # no `workq` on this cluster
set -uo pipefail
WT="${SLURM_SUBMIT_DIR:?}"                 # repo root — not a hardcoded path
export PYTHONPATH="$WT/src:<mentos-checkout>/src"
export DATA_ROOT=…  MENTOS_ROOT=…  ECSTASY_ROOT=…  ENVS_ROOT=…  TOOLS_ROOT=…  COLABFOLD_DBS=…  LOGS_DIR=…
PYB="$ENVS_ROOT/.venv-mentos/bin/python"
CELLS=( "recent_pp boltz2 r0" "recent_pp boltz2 r1" … )   # one (split model preset) per index
read -r split model preset <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
$PYB -m ecstasy.cli run --dataset "$split" --model "$model" --preset "$preset" --profile
```
