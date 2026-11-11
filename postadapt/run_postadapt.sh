#!/bin/bash

#SBATCH --job-name=ConR_PostAdpt
#SBATCH --partition=defq,gpu-mig,gpu,gpu-h100
#SBATCH --time=999:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=50G
#SBATCH --output=./log/output/%x_%j.log
#SBATCH --error=./log/error/%x_%j.log

# Post-training adaptation of one finished ConR run.
#
# Usage (run from postadapt/):
#   sbatch run_postadapt.sh <dataset> <conr_ckpt> [bandwidth] [condLoss] [l2spWeight] [batchSize] [lr]
#   dataset    : agedb | imdb_wiki
#   conr_ckpt  : ../checkpoint/agedb/diraligned/<run>/ckpt.best.pth.tar (DIR-aligned, job_dir_aligned.sh)
#                or ../<dataset>-dir/checkpoint/<run>/ckpt.best.pth.tar (original ConR runs)
#   bandwidth  : Gaussian label kernel width in years (default 2)
#   condLoss   : w2 (default) | l1 | kl | skl
#   l2spWeight : L2-SP weight (default 1.0)
#   batchSize  : adaptation batch size (default 64)
#   lr         : learning rate (default 1e-5)
# Any further arguments are passed to postadapt.py as key=value overrides, e.g. dataset.balance.enabled=true
#
# Results: <ConR run>/PostTrainAdpt_<condLoss>_h:<bw>_l2spWeight:<w>_lr:<lr>_bsz:<b>_all_sgd_sourceAligned_*/

dataset=$1
ckpt=$2
bandwidth=${3:-2}
condLoss=${4:-w2}
l2spWeight=${5:-1.0}
batchSize=${6:-64}
lr=${7:-1e-5}
shift $(( $# < 7 ? $# : 7 ))

if [ -z "$dataset" ] || [ -z "$ckpt" ]; then
    echo "usage: sbatch run_postadapt.sh <agedb|imdb_wiki> <conr ckpt.best.pth.tar> [bandwidth] [condLoss] [l2spWeight] [batchSize] [lr] [key=value ...]"
    exit 1
fi

case $dataset in
    agedb)
        DATA="trainer.max_steps=2000" ;;
    imdb_wiki)
        DATA="trainer.max_steps=10000 \
              logger.save_train_predictions=false" ;;
    *) echo "unknown dataset: $dataset (agedb | imdb_wiki)"; exit 1 ;;
esac

mkdir -p log/output log/error

# postadapt.py and PostTrainAdpt.yaml live in the ConR repo root. ConR's YAML defaults to
# adam / backbone, so SGD on all parameters (RankSim's defaults) is set explicitly here.
python ../postadapt.py -c ../PostTrainAdpt.yaml ${DATA} \
    model.pretrained="${ckpt}" \
    model.bandwidth=${bandwidth} \
    model.condLoss=${condLoss} \
    model.l2spWeight=${l2spWeight} \
    model.optim.name=sgd \
    model.optim.param=all \
    model.optim.lr=${lr} \
    dataset.batch_size=${batchSize} \
    "$@"
