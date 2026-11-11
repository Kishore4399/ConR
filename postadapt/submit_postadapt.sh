#!/usr/bin/env bash
set -euo pipefail
# Run from any directory. Default: submit finished AgeDB DIR-aligned checkpoints.
# bash postadapt/submit_postadapt.sh --dry-run
# bash postadapt/submit_postadapt.sh
# bash postadapt/submit_postadapt.sh local
# DATASETS="agedb imdb_wiki" RUNS='*' bash postadapt/submit_postadapt.sh
# Grid environment variables: BANDWIDTHS, COND_LOSSES, L2SP_WEIGHTS, BATCH_SIZES, LRS.
repo=${CONR_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
if [[ ! -f "$repo/postadapt.py" ]]; then
    echo "ERROR: Cannot locate ConR; set CONR_ROOT=/absolute/path/to/ConR." >&2
    exit 2
fi
export CONR_ROOT=$repo
exec "${PYTHON_BIN:-${PYTHON:-python}}" "$repo/postadapt/launch_postadapt.py" submit "$@"
