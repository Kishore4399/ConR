#!/usr/bin/env bash
#SBATCH --job-name=ConR_DIR_Aligned
#SBATCH --partition=gpu-mig,defq
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH --output=./log/output/%x_%j.log
#SBATCH --error=./log/error/%x_%j.log

set -euo pipefail

# Submit from the ConR repository root:
#   sbatch --export=ALL job_dir_aligned.sh 0
#   sbatch --export=ALL job_dir_aligned.sh 1
# Override Python with PYTHON_BIN=/absolute/path/to/env/bin/python.
# If submitting outside the repo, set CONR_ROOT=/absolute/path/to/ConR.
SEED=${1:-0}
REPO_ROOT=${CONR_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$REPO_ROOT/agedb-dir/train.py" || ! -f "$REPO_ROOT/agedb-dir/resnet.py" ]]; then
  echo "ERROR: Expected agedb-dir/train.py under '$REPO_ROOT'." >&2
  echo "Submit from ConR repo root or provide CONR_ROOT=/absolute/path/to/ConR." >&2
  exit 2
fi
cd "$REPO_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable '$PYTHON_BIN' not found" >&2
  exit 2
fi
ISILON=/isilon/datalake/dma_research/scratch/damalab
DATA_DIR=${AGEDB_DATA_DIR:-$ISILON/Data/Shihua/82bd6-main}
CSV=${AGEDB_CSV:-$ISILON/Tools/Rank-N-Contrast/data/agedb.csv}
if [[ ! -f "$CSV" ]]; then
  echo "ERROR: CSV not found: $CSV; set AGEDB_CSV to the DIR AgeDB CSV." >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: image directory not found: $DATA_DIR" >&2
  exit 2
fi
echo "Repo: $PWD"
echo "Python: $(command -v "$PYTHON_BIN" || printf '%s' "$PYTHON_BIN")"
"$PYTHON_BIN" -c 'import torch, torchvision; print("PyTorch:", torch.__version__, "Torchvision:", torchvision.__version__, "CUDA:", torch.cuda.is_available())'

# This entrypoint is AgeDB-only; IMDB-WIKI still uses submission.sh.
# Match ImbalanceRegression@dir-pipeline: ImageNet ResNet18, 300 epochs, Adam,
# batch 64, LR 1e-3, WD 1e-4, step LR x0.5 every 50 epochs, best val MAE >= epoch 80.
# Preserve ConR original three-view augmentation and regularization settings.
"$PYTHON_BIN" agedb-dir/train.py \
  --dataset agedb --data_dir "$DATA_DIR" --data_csv "$CSV" \
  --model resnet18 --imagenet_pretrained \
  --epoch 300 --batch_size 64 --lr 1e-3 \
  --optimizer adam --weight_decay 1e-4 \
  --lr_step 50 --lr_gamma 0.5 --save_start 80 \
  --img_size 224 --workers 8 --seed "$SEED" \
  --loss l1 --conr -w 1 -t 0.2 --beta 4 -e 0.01 \
  --store_root checkpoint/agedb/diraligned \
  --store_name "resnet18_conr_diraligned_seed${SEED}"
