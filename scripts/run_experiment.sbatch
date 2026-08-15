#!/usr/bin/env bash
#SBATCH --job-name=ecstasy-exp
#SBATCH --output=logs/ecstasy_exp_%j.out
#SBATCH --error=logs/ecstasy_exp_%j.err
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --qos=freegpu
# Generic experiment runner. Usage:
#   sbatch scripts/run_experiment.sbatch experiments/recent_pp_nomsa_ladder.yaml --profile
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
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO_ROOT}/logs"

# Filesystem roots come from the repo-root .env (env > .env). A git worktree has no
# .env of its own, so fall back to the main checkout's copy.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a; source "${REPO_ROOT}/.env"; set +a
fi

VENV="${ECSTASY_VENV:-${ENVS_ROOT:?set ECSTASY_VENV or ENVS_ROOT}/.venv-mentos}"
MENTOS_SRC="${MENTOS_SRC:-/lustre/home/mmm1486/projects/harsh/mentos/src}"

export PYTHONPATH="${REPO_ROOT}/src:${MENTOS_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${VENV}/bin/python" -m ecstasy.cli experiment "$@"
