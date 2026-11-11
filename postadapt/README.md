# ConR postadapt submission

Run from the ConR checkout on `dir-pipeline-aligned` after pulling the source-contract fixes.
The scripts use your currently active Python environment. Set `PYTHON_BIN` to an absolute
Python executable if needed. `CONR_ROOT` can identify the checkout when submitting elsewhere.

## Submit the default grid

```bash
cd /path/to/ConR/postadapt
bash submit_postadapt.sh --dry-run
bash submit_postadapt.sh
```

Default selection: finished **AgeDB DIR-aligned** source runs, whose folder name matches
`*diraligned*`. A source needs `ckpt.best.pth.tar` and a completed final test block in
`training.log`; merely starting test evaluation is insufficient. Checkpoint roots are scanned
recursively, including `checkpoint/agedb/diraligned`, and overlapping roots are deduplicated.

The default grid mirrors the current RankSim submission settings:

| Setting | Default |
| --- | --- |
| Bandwidth | 0.1 |
| Conditional loss | w2 |
| L2-SP weight | 10.0 |
| Adaptation batch size | 64 |
| Learning rate | 1e-5 |
| Optimizer / parameters | SGD / all |
| Steps | 2000 AgeDB; 10000 IMDB-WIKI |

Image normalization, image root, split CSV, image size and backbone remain governed by the
corrected source-contract configuration. The launcher does not substitute a bundled CSV or
force a backbone. Keep each source run's `training.log` beside its checkpoint.

Change the grid, datasets, or run filter:

```bash
BANDWIDTHS="0.1 0.5 1.0 2.0 3.0 5.0" L2SP_WEIGHTS="1.0 10.0" \
    bash submit_postadapt.sh --dry-run

# Include original ConR runs as well as DIR-aligned runs:
DATASETS="agedb imdb_wiki" RUNS='*' bash submit_postadapt.sh --dry-run

# Select a method/seed by its source run folder name:
RUNS='*conr_lds_fds*seed0*' bash submit_postadapt.sh

# Custom checkpoint roots, recursive; quote paths containing spaces:
bash submit_postadapt.sh --roots /path/to/checkpoints --runs '*'

# Optional overrides apply last; repeat --override as needed:
bash submit_postadapt.sh --override trainer.max_steps=500 --limit 2 --dry-run

# Execute sequentially without SLURM:
bash submit_postadapt.sh local
```

Other grid environment variables are `COND_LOSSES`, `BATCH_SIZES`, and `LRS`.
`OPTIMIZER=adamw` or `OPTIM_PARAM=backbone` changes the corresponding defaults.
Default checkpoint roots are `checkpoint/<dataset>` and `<dataset>-dir/checkpoint`.

Real submission creates `postadapt/log/output` and `postadapt/log/error`, then uses absolute
script/checkpoint/log paths and an explicit SLURM working directory. Dry runs only print the
planned commands; they do not submit jobs, run model inference, or create log folders.
Re-running submission resubmits matching runs; it does not query the queue or skip previous
adaptation outputs. Use the filter or `--limit` to control repeated submissions.

## One source checkpoint

From the repository root:

```bash
bash postadapt/run_postadapt.sh --dry-run agedb /absolute/path/to/ckpt.best.pth.tar

# Run immediately in the current environment:
bash postadapt/run_postadapt.sh agedb /absolute/path/to/ckpt.best.pth.tar 0.1 w2 10.0 64 1e-5

# Submit one job directly (create SLURM log directories before submitting):
mkdir -p postadapt/log/output postadapt/log/error
sbatch postadapt/run_postadapt.sh agedb /absolute/path/to/ckpt.best.pth.tar 0.1 w2 10.0 64 1e-5
```

The positional interface is `<dataset> <checkpoint> [bandwidth loss l2sp batch lr] [key=value ...]`.
Defaults can be omitted; optional `key=value` overrides begin after the positional settings.
For IMDB-WIKI, train prediction export is disabled by default because of the large train split.
Explicit `logger.save_train_predictions=true` overrides this.

Corrected model outputs remain beside the source checkpoint, with the `_sourceAligned_imagenet`
or `_sourceAligned_conr` suffix. Actual resolved input settings are saved in
`source_data_config.json`.

## Checks

```bash
python -m unittest discover -s tests -p test_postadapt_submission.py -v
```

Tests use temporary fake source folders and mock job submission; no SLURM jobs are created.
The shell smoke test runs syntax checks and dry runs when Bash is available and runnable.
