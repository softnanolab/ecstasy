#!/usr/bin/env bash
#SBATCH --job-name=ecstasy-exp
#SBATCH --output=logs/ecstasy_exp_%j.out
#SBATCH --error=logs/ecstasy_exp_%j.err
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --qos=freegpu
# Generic experiment runner. Usage:
#   sbatch scripts/run_experiment.sbatch experiments/recent_pp_nomsa_ladder/manifest.yaml --profile
#
# --mem is NOT optional. This cluster defaults to ~1G per CPU, and ESMFold silently
# livelocks under that: loading it needs the 2.7G esmfold_3B_v1 state dict plus the 5.6G
# esm2_t36_3B_UR50D one, then materialises ~3.5B fp32 params on the CPU before .to(cuda)
# — a ~25-30G peak. Against a small cgroup limit the loader does not get OOM-killed, it
# just thrashes page-cache reclaim forever: job 62424 sat at MaxRSS 8.07G/8G for its full
# 45min walltime and never got past `esm.pretrained.esmfold_v1()`. Diagnose this with
#   sacct -j <id> --format=JobID,ReqMem,MaxRSS,State
# (MaxRSS pinned at ReqMem == starved, not hung). GPU nodes have 960G across 64 cores, so
# 64G is well inside a fair share and costs nothing in scheduling priority.
#
# Partition/QOS: this cluster has cpu|highmem|hbm|gpu|intergpu (there is no `workq`).
# For a BATCH gpu job, `gpu` + `freegpu` is the only accepted combination — submitting to
# `intergpu` is rejected outright ("only partition gpu with the qos freegpu is valid for a
# batch gpu job"); despite its higher QOS priority it is reserved for interactive
# salloc/srun. Check your associations with
#   sacctmgr -nP show assoc user=$USER format=Account,Partition,QOS
#
# The gpu partition is heavily contested (routinely 20+ running, 40+ pending), and
# `freegpu` is low priority, so walltime is the practical scheduling lever: a short
# --time backfills into gaps that a 24h request cannot. Request what the sweep actually
# needs rather than the 24h default below when you can bound it.
#
# Driver env: the per-model runners are subprocessed into their own venvs by
# models/adapter.py, so this env only has to import ecstasy + torch + mentos (scoring
# unpickles a `mentos.dataclasses.Sample`). The editable installs of `ecstasy` and
# `mentos` are broken in every venv on this machine — they point at a deleted worktree —
# so both source trees go on PYTHONPATH and we invoke `python -m ecstasy.cli` rather than
# the `ecstasy` console script (which does not exist there either).
#   sbatch scripts/run_experiment.sbatch experiments/boltz2_headline/manifest.yaml [extra ecstasy args]
# Runs in .venv-boltz (has ecstasy + torch + mentos, so predict subprocesses to
# each model's env and scoring can load the MENTOS GT here).
set -euo pipefail
# Do NOT derive REPO_ROOT from ${BASH_SOURCE[0]}: sbatch copies this file to a spool dir
# and runs the copy, so under SLURM that resolves to /var/spool/slurm/d and every path
# below lands outside the repo (job 62930 died at 0:00 on `mkdir /var/spool/slurm/d/logs:
# Permission denied`). SLURM_SUBMIT_DIR is the directory sbatch was invoked from, which
# is the repo root; the BASH_SOURCE form is kept only as the non-SLURM fallback for
# running this script directly. ECSTASY_REPO_ROOT overrides both.
REPO_ROOT="${ECSTASY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [[ ! -d "${REPO_ROOT}/src/ecstasy" ]]; then
  echo "ERROR: REPO_ROOT=${REPO_ROOT} has no src/ecstasy — submit from the repo root," \
       "or set ECSTASY_REPO_ROOT." >&2
  exit 2
fi
mkdir -p "${REPO_ROOT}/logs"

# Filesystem roots come from the repo-root .env, but a variable already present in the
# environment wins over the file — that is what lets you point a one-off run at a
# scratch tree, e.g. `DATA_ROOT=/…/data_debug sbatch scripts/run_experiment.sbatch …`.
# `set -a; source .env` alone does the opposite (the file clobbers the caller), so
# snapshot the exported environment first and re-apply it afterwards. Using bash's own
# `export -p` output rather than hand-parsing .env keeps quoting/escaping correct.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  _ecstasy_pre_env="$(export -p)"
  set -a; source "${REPO_ROOT}/.env"; set +a
  eval "${_ecstasy_pre_env}"
  unset _ecstasy_pre_env
fi

VENV="${ECSTASY_VENV:-${ENVS_ROOT:?set ECSTASY_VENV or ENVS_ROOT}/.venv-mentos}"
MENTOS_SRC="${MENTOS_SRC:-/lustre/home/mmm1486/projects/harsh/mentos/src}"

export PYTHONPATH="${REPO_ROOT}/src:${MENTOS_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${VENV}/bin/python" -m ecstasy.cli experiment "$@"
