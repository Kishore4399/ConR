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

# IMDB-WIKI with the shared DIR recipe (same as SLR's and RankSim's dirrecipe runs), from scratch.
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

# Shared DIR recipe (Yang et al., ICML 2021), identical to SLR's and RankSim's IMDB-WIKI jobs:
#   ResNet50 from scratch (no --imagenet_pretrained, so DIR normalisation mean/std 0.5),
#   90 epochs, batch 256, Adam LR 1e-3 with no weight decay (the public ConR, RankSim and DIR
#   code pass weight decay only to SGD), LR x0.1 at epochs 60 and 80, L1 loss, best val MAE
#   over all epochs, plain shuffled batches. The ConR paper itself used batch 64 and LR 2.5e-4.
# ConR on top (paper App. A.4): two extra augmented views per image, window w 1,
#   temperature 0.2 (the public README command leaves -t at its 0.07 default), beta 4, eta 0.01.
#   ConR runs three forward passes per step (original + 2 views), so batch 256 needs a full
#   GPU; it will not fit a MIG slice.
# Kept from this pipeline: mixed precision (losses and ConR stay float32).
"$PYTHON_BIN" imdb-wiki-dir/train.py \
  --dataset imdb_wiki --data_dir "$DATA_DIR" --data_csv "$CSV" \
  --model resnet50 \
  --epoch 90 --batch_size 256 --lr 1e-3 \
  --optimizer adam --weight_decay 0 \
  --schedule 60 80 --lr_gamma 0.1 --save_start 1 --amp \
  --img_size 224 --workers 8 --seed "$SEED" \
  --loss l1 -w 1 -t 0.2 --beta 4 -e 0.01 \
  "${FDS_LDS_PARAMS[@]}" "${EXTRA_FLAGS[@]}" \
  --store_root checkpoint/imdb_wiki/dirrecipe \
  --store_name "resnet50_${METHOD}_dirrecipe_seed${SEED}"
