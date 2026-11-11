#!/usr/bin/env bash
#SBATCH --job-name=ConR_PostAdpt
#SBATCH --partition=gpu-mig,defq
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH --output=postadapt/log/output/%x_%j.log
#SBATCH --error=postadapt/log/error/%x_%j.log

set -euo pipefail
# Single source checkpoint; same positional interface as RankSim.
# bash postadapt/run_postadapt.sh [--dry-run] agedb /path/to/ckpt.best.pth.tar [h loss l2sp batch lr] [key=value ...]
# For direct sbatch, submit from the repo root after creating the log directories,
# or set CONR_ROOT explicitly. Batch submission does this automatically.
if [[ -n "${CONR_ROOT:-}" ]]; then
    repo=$CONR_ROOT
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    repo=${SLURM_SUBMIT_DIR:-$PWD}
else
    repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [[ ! -f "$repo/postadapt.py" && -f "$repo/../postadapt.py" ]]; then
    repo=$(cd -- "$repo/.." && pwd)
fi
if [[ ! -f "$repo/postadapt.py" || ! -f "$repo/postadapt/launch_postadapt.py" ]]; then
    echo "ERROR: Cannot locate ConR; set CONR_ROOT=/absolute/path/to/ConR." >&2
    exit 2
fi
export CONR_ROOT=$repo
exec "${PYTHON_BIN:-${PYTHON:-python}}" "$repo/postadapt/launch_postadapt.py" run "$@"
