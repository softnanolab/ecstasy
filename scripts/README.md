# scripts/ — benchmark sweeps & figures

Helper scripts for the inter-chain P@K vs. inference-FLOPs benchmark, the chain-order
(swap) experiment, and all the figures. The **eval/sweep work is plain `ecstasy …` CLI
calls**; the scripts here only generate plots and aggregate results.

> SLURM `*.sbatch` wrappers are **not committed** — they hardcode per-cluster/per-job
> paths. Keep them local (they're git-ignored) and use the portable commands below.
> A minimal, parameterized template is at the end of this file.

## Environment

Every script (and CLI call) needs the package on `PYTHONPATH` and the data-root env
vars resolved by `ecstasy.config.settings`. Run with the project venv:

```bash
export PYTHONPATH=src:scripts
export DATA_ROOT=…   MENTOS_ROOT=…   ECSTASY_ROOT=…   ENVS_ROOT=…   TOOLS_ROOT=…
export COLABFOLD_DBS=…            # local ColabFold DBs (MSA generation)
export LOGS_DIR=…                 # W&B run files (mentos config lookup)
PYB="$ENVS_ROOT/.venv-boltz/bin/python"   # 3.12; has ecstasy + matplotlib
```

Paths come from `settings()` (e.g. `settings().runs_root`), never hardcoded. CMU
Concrete figures look for the font in `$CMU_FONT_DIR` (default `~/.fonts`); they fall
back to the default font if it's not installed.

## Figures (read `runs/<split>/…`, write PNG + PDF)

| Script | Example |
|---|---|
| `plot_pak_vs_flops.py` | `$PYB scripts/plot_pak_vs_flops.py --dataset val_seq_pair [--annotate-r0] [--exclude-models boltz2_nomsa]` |
| `plot_flops_vs_length.py` | `$PYB scripts/plot_flops_vs_length.py --dataset val_seq_chain --model esmfold --presets r0,r1,r3,r5 [--style line]` |
| `plot_pak_vs_interface.py` | `$PYB scripts/plot_pak_vs_interface.py --dataset val_seq_pair --xmode {contacts,percent} --cap 800` |
| `plot_pak_vs_msadepth.py` | `$PYB scripts/plot_pak_vs_msadepth.py --depth {paired,total}` |
| `plot_swap_flops.py` | `$PYB scripts/plot_swap_flops.py` (original vs swapped overlay) |
| `swap_compare.py` | `$PYB scripts/swap_compare.py` (ΔP@K table + scatters, orig vs swapped) |

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
$PYB -m ecstasy.cli run --dataset $split --model mentos         --preset a5sgd6ul_latest --profile
$PYB -m ecstasy.cli run --dataset $split --model msa_pairformer --preset full            --profile
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
$PYB scripts/swap_compare.py          # ΔP@K table + swap_scatter_<model>.png
$PYB scripts/plot_swap_flops.py       # P@K-vs-FLOPs overlay (A,B) vs (B,A)
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
set -uo pipefail
WT="${SLURM_SUBMIT_DIR:?}"                 # repo root — not a hardcoded path
export PYTHONPATH="$WT/src"
export DATA_ROOT=…  MENTOS_ROOT=…  ECSTASY_ROOT=…  ENVS_ROOT=…  TOOLS_ROOT=…  COLABFOLD_DBS=…  LOGS_DIR=…
PYB="$ENVS_ROOT/.venv-boltz/bin/python"
CELLS=( "val_seq_pair boltz2 r0" "val_seq_pair boltz2 r1" … )   # one (split model preset) per index
read -r split model preset <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
$PYB -m ecstasy.cli run --dataset "$split" --model "$model" --preset "$preset" --profile
```
