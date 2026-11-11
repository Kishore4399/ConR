#!/usr/bin/env bash
#SBATCH --job-name=conr
#SBATCH --partition=gpu-mig,defq
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH --output=conr-%j.log

set -euo pipefail

usage() {
    echo "Usage: bash submission.sh [agedb|imdb_wiki|both] [--dry-run|--check-data] [train.py options]"
}

case "${1:-both}" in
    agedb|imdb_wiki) datasets=("$1"); shift ;;
    both) datasets=(agedb imdb_wiki); if (( $# )); then shift; fi ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

mode="${1:-}"
if [[ "$mode" == --dry-run || "$mode" == --check-data ]]; then
    shift
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Slurm runs a copied script. Submit from the checkout or set CONR_ROOT.
    repo="${CONR_ROOT:-$SLURM_SUBMIT_DIR}"
else
    repo="${CONR_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
fi
cd -- "$repo"
python_bin="${PYTHON:-python}"

for dataset in "${datasets[@]}"; do
    if [[ "$mode" == --check-data ]]; then
        "$python_bin" dir_data.py "$dataset" --check
        continue
    fi
    case "$dataset" in
        agedb) folder=agedb-dir ;;
        imdb_wiki) folder=imdb-wiki-dir ;;
    esac
    prepare=()
    if [[ "$mode" != --dry-run ]]; then
        prepare=(--prepare)
    fi
    paths=$("$python_bin" dir_data.py "$dataset" "${prepare[@]}")
    data_dir=${paths%%$'\n'*}
    data_csv=${paths#*$'\n'}
    cmd=("$python_bin" "$folder/train.py" --dataset "$dataset"
         --data_dir "$data_dir" --data_csv "$data_csv"
         --store_root "${CONR_OUTPUT_DIR:-checkpoint}/$dataset"
         --workers "${SLURM_CPUS_PER_GPU:-8}" --seed "${CONR_SEED:-0}"
         --conr --model resnet50 --optimizer adam --loss l1
         --batch_size 64 --lr 0.00025 --weight_decay 0.0001
         --epoch 90 --schedule 60 80 -w 1 -t 0.2 --beta 4 -e 0.01 "$@")
    if [[ "$mode" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
done
