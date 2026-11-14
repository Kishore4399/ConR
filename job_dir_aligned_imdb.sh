#!/usr/bin/env bash
#SBATCH --job-name=ConR_DIR_Aligned_IMDB
#SBATCH --partition=gpu-h100,gpu-mig
#SBATCH --time=999:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=100G
#SBATCH --output=./log/output/%x_%j.log
#SBATCH --error=./log/error/%x_%j.log

set -euo pipefail

# IMDB-WIKI reproduction of the ConR paper (Keramati et al., ICLR 2024), from scratch.
# Submit from the ConR repository root:
#   sbatch --export=ALL job_dir_aligned_imdb.sh vanilla 0
#   sbatch --export=ALL job_dir_aligned_imdb.sh conr 0
#   sbatch --export=ALL job_dir_aligned_imdb.sh conr_fds 0
#   sbatch --export=ALL job_dir_aligned_imdb.sh conr_lds_fds 0
# Override Python with PYTHON_BIN=/absolute/path/to/env/bin/python.
# If submitting outside the repo, set CONR_ROOT=/absolute/path/to/ConR.
METHOD=${1:-conr}
SEED=${2:-0}
case "$METHOD" in
  vanilla) EXTRA_FLAGS=() ;;
  conr) EXTRA_FLAGS=(--conr) ;;
  conr_fds)
    EXTRA_FLAGS=(--conr --fds)
    ;;
  conr_lds_fds)
    EXTRA_FLAGS=(--conr --fds --lds --reweight sqrt_inv)
    ;;
  *)
    echo "ERROR: Unknown method '$METHOD' (vanilla | conr | conr_fds | conr_lds_fds)" >&2
    exit 2
    ;;
esac
# Shared smoothing hyperparameters; flags are only activated for requested modes.
FDS_LDS_PARAMS=(
  --lds_kernel gaussian --lds_ks 5 --lds_sigma 2
  --fds_kernel gaussian --fds_ks 5 --fds_sigma 2
  --start_update 0 --start_smooth 1 --fds_mmt 0.9
)
echo "Method=$METHOD Seed=$SEED Extra flags: ${EXTRA_FLAGS[*]:-none}"
REPO_ROOT=${CONR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$REPO_ROOT/imdb-wiki-dir/train.py" || ! -f "$REPO_ROOT/imdb-wiki-dir/resnet.py" ]]; then
  echo "ERROR: Expected imdb-wiki-dir/train.py under '$REPO_ROOT'." >&2
  echo "Submit from ConR repo root or provide CONR_ROOT=/absolute/path/to/ConR." >&2
  exit 2
fi
cd "$REPO_ROOT"
mkdir -p log/output log/error
PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable '$PYTHON_BIN' not found" >&2
  exit 2
fi
ISILON=/isilon/datalake/dma_research/scratch/damalab
DATA_DIR=${IMDB_DATA_DIR:-$ISILON/Tools/imbalanced-regression/imdb-wiki-dir/data}
CSV=${IMDB_CSV:-$ISILON/Tools/Rank-N-Contrast/data/imdb_wiki.csv}
if [[ ! -f "$CSV" ]]; then
  echo "ERROR: CSV not found: $CSV; set IMDB_CSV to the DIR IMDB-WIKI CSV." >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: image directory not found: $DATA_DIR" >&2
  exit 2
fi
echo "Repo: $PWD"
echo "Python: $(command -v "$PYTHON_BIN" || printf '%s' "$PYTHON_BIN")"
"$PYTHON_BIN" -c 'import torch, torchvision; print("PyTorch:", torch.__version__, "Torchvision:", torchvision.__version__, "CUDA:", torch.cuda.is_available())'

# ConR paper, App. A.4 (age estimation), as stated in the text:
#   ResNet50 encoder from scratch + one linear regression layer (DIR normalisation mean/std 0.5),
#   batch 64, Adam LR 2.5e-4, weight decay 1e-4, LR x0.1 at epochs 60 and 80, 90 epochs,
#   L1 regression loss (alpha 1), random crop + horizontal flip, two augmented views for ConR,
#   window w 1, temperature 0.2, beta 4, eta 0.01. Full precision, as in the paper.
#   Best epoch by validation MAE over all epochs, then tested.
# Where the public code differs from the text (see DIR_ALIGNED_README.md): its Adam gets no
#   weight decay, its README command leaves -t at 0.07, and it adds beta * L_ConR twice.
#   Override WD=0 or TEMP=0.07 to try the code's behaviour.
WD=${WD:-1e-4}
TEMP=${TEMP:-0.2}
"$PYTHON_BIN" imdb-wiki-dir/train.py \
  --dataset imdb_wiki --data_dir "$DATA_DIR" --data_csv "$CSV" \
  --model resnet50 \
  --epoch 90 --batch_size 64 --lr 2.5e-4 \
  --optimizer adam --weight_decay "$WD" \
  --schedule 60 80 --lr_gamma 0.1 --save_start 1 \
  --img_size 224 --workers 8 --seed "$SEED" \
  --loss l1 -w 1 -t "$TEMP" --beta 4 -e 0.01 \
  "${FDS_LDS_PARAMS[@]}" "${EXTRA_FLAGS[@]}" \
  --store_root checkpoint/imdb_wiki/paper \
  --store_name "resnet50_${METHOD}_paper_wd${WD}_t${TEMP}_seed${SEED}"
