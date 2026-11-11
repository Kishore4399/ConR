# ConR postadapt submission

SLURM scripts for post-training adaptation of finished ConR runs. They mirror
`postadapt/run_postadapt.sh` and `postadapt/submit_postadapt.sh` in RankSim
(`dir-pipeline-aligned`); `postadapt.py` and `PostTrainAdpt.yaml` stay in the ConR repo root.

Run from this folder. The scripts use the currently active Python environment.

```bash
cd postadapt

# one ConR run: <dataset> <ckpt> [bandwidth] [condLoss] [l2spWeight] [batchSize] [lr] [key=value ...]
sbatch run_postadapt.sh agedb ../checkpoint/agedb/diraligned/<run>/ckpt.best.pth.tar
sbatch run_postadapt.sh agedb ../checkpoint/agedb/diraligned/<run>/ckpt.best.pth.tar 0.1 w2 10.0 64 1e-5 dataset.balance.enabled=true

# every finished run (edit the grids at the top of the script)
bash submit_postadapt.sh
RUNS='*conr_lds_fds*seed0*' bash submit_postadapt.sh
DATASETS="agedb imdb_wiki" RUNS='*' bash submit_postadapt.sh   # include original ConR runs
bash submit_postadapt.sh local                                 # sequentially, without SLURM
```

`submit_postadapt.sh` looks for runs in `../checkpoint/<dataset>/diraligned` (written by
`job_dir_aligned.sh`) and `../<dataset>-dir/checkpoint`, filters them by `RUNS`
(default `*diraligned*`), and only picks up runs whose `training.log` has the final test block.

Defaults in `run_postadapt.sh` match RankSim: SGD on all parameters, `lr 1e-5`, `bandwidth 2`,
`w2`, `l2spWeight 1.0`, batch 64; 2,000 steps for AgeDB and 10,000 for IMDB-WIKI (train
predictions off). The submit grid is `bandwidth 0.1`, `w2`, `l2spWeight 10.0`, batch 64,
`lr 1e-5`. ConR's `PostTrainAdpt.yaml` defaults to Adam / backbone, so `run_postadapt.sh`
sets `model.optim.name=sgd model.optim.param=all` explicitly; pass different values as
`key=value` overrides to change them.

Image normalization, image root, split CSV, image size and backbone stay at the YAML's `auto`
values, resolved from the source run's `training.log` (keep it beside the checkpoint). Outputs
go to `<ConR run>/PostTrainAdpt_<settings>_sourceAligned_<imagenet|conr>/`, with the resolved
inputs saved in `source_data_config.json`. SLURM logs go to `postadapt/log/{output,error}`.
