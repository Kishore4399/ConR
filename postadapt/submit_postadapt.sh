#!/bin/bash
# Post-adapt every ConR run, over a grid of settings (one SLURM job each).
#   bash submit_postadapt.sh                  # AgeDB DIR-aligned runs
#   DATASETS="agedb imdb_wiki" bash submit_postadapt.sh
#   RUNS='*conr_lds_fds*seed0*' bash submit_postadapt.sh   # only runs whose folder matches this glob
#   RUNS='*' bash submit_postadapt.sh         # also original (non DIR-aligned) ConR runs
#   bash submit_postadapt.sh local            # run one after another in this shell instead of sbatch
#   bash submit_postadapt.sh list             # only print which runs match; submit nothing
# Every run folder with ckpt.best.pth.tar is used; data CSV/root/normalization come from its training.log. Edit the grids below.
# Runs are looked up in ../checkpoint/<ds>/diraligned (job_dir_aligned.sh) and ../<ds>-dir/checkpoint.

# (0.1 0.5 1.0 2.0 3.0 5.0)
bandwidths=(0.1)
condLosses=(w2)
l2spWeights=(10.0)
batchSizes=(64)
lrs=(1e-5)

datasets=(${DATASETS:-agedb})
runs_glob=${RUNS:-"*diraligned*"}

cd "$(dirname "$0")" && mkdir -p log/output log/error

found=0; skipped=0; jobs=0
for ds in "${datasets[@]}"; do
    # agedb -> agedb-dir, imdb_wiki -> imdb-wiki-dir
    for root in "../checkpoint/${ds}/diraligned" "../${ds//_/-}-dir/checkpoint"; do
    echo "Searching $(realpath -m "$root")/${runs_glob}"
    for run in ${root}/${runs_glob}; do
        [ -d "$run" ] || continue                     # glob matched nothing
        name=$(basename "$run")
        ckpt="${run}/ckpt.best.pth.tar"
        if [ ! -f "$ckpt" ]; then
            echo "  SKIP  $name (no ckpt.best.pth.tar)"; skipped=$((skipped + 1)); continue
        fi
        echo "  FOUND $name"; found=$((found + 1))
        [ "$1" == "list" ] && continue
        for bw in "${bandwidths[@]}"; do for cl in "${condLosses[@]}"; do for w in "${l2spWeights[@]}"; do
        for b in "${batchSizes[@]}"; do for lr in "${lrs[@]}"; do
            jobs=$((jobs + 1))
            if [ "$1" == "local" ]; then
                bash run_postadapt.sh "$ds" "$ckpt" "$bw" "$cl" "$w" "$b" "$lr"
            else
                sbatch --job-name="PostAdpt_${ds}_h${bw}_${cl}_w${w}" run_postadapt.sh "$ds" "$ckpt" "$bw" "$cl" "$w" "$b" "$lr"
            fi
        done; done; done; done; done
    done
    done
done
echo "Found ${found} run(s), skipped ${skipped}; launched ${jobs} job(s)."
