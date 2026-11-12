# DIR-aligned ConR training — AgeDB

This branch derives from `codex/conr-agedb-imdb-reproduction`. It adds a **new** AgeDB entrypoint, `job_dir_aligned.sh`, to make ConR comparable with `Kishore4399/ImbalanceRegression@dir-pipeline` on AgeDB. The original `submission.sh` (paper-style ResNet50 reproduction) is preserved.

## Run on HPC

From the ConR repository root:

```bash
git fetch origin
git checkout dir-pipeline-aligned
sbatch --export=ALL job_dir_aligned.sh conr 0
sbatch --export=ALL job_dir_aligned.sh conr_fds 0
sbatch --export=ALL job_dir_aligned.sh conr_lds_fds 0
```

For five seeds:

```bash
for s in 0 1 2 3 4; do
  for method in conr conr_fds conr_lds_fds; do
    sbatch --export=ALL job_dir_aligned.sh "$method" "$s"
  done
done
```

Activate the desired Python environment before submission or set `PYTHON_BIN=/absolute/path/to/env/bin/python`. There is no `conda.sh` initialization and no use of `dirname "$0"` within Slurm. The script uses `SLURM_SUBMIT_DIR` to resolve the checkout and validates repository and data paths. To submit elsewhere, set `CONR_ROOT=/absolute/path/to/ConR`.

The AgeDB image root defaults to `/isilon/datalake/dma_research/scratch/damalab/Data/Shihua/82bd6-main`. The CSV is the **same** file referenced by `ImbalanceRegression@dir-pipeline/job_sub.sh`: `/isilon/datalake/dma_research/scratch/damalab/Tools/Rank-N-Contrast/data/agedb.csv`. Override with `AGEDB_DATA_DIR` and `AGEDB_CSV`.

## Matched settings

| Parameter | DIR-aligned ConR AgeDB |
| --- | --- |
| Backbone | ImageNet-pretrained ResNet18 (torchvision V1 weights) |
| Input | 224×224; resize/crop with padding 16/horizontal flip; ImageNet mean/std |
| Epochs | 300 |
| Batch size | 64 |
| Optimizer | Adam, learning rate 1e-3, weight decay 1e-4 |
| Scheduler | x0.5 every 50 epochs |
| Checkpoint | lowest validation L1 from epoch 80 onward |
| Dataset | shared DIR AgeDB split CSV |
| ConR-specific | original 3-view setup, window=1, temperature=0.2, beta=4, eta=0.01 |
| Output | `checkpoint/agedb/diraligned` |

## Experiment modes

| Mode | ConR | FDS | LDS (sqrt_inv) |
| --- | --- | --- | --- |
| `conr` | Yes | No | No |
| `conr_fds` | Yes | Yes | No |
| `conr_lds_fds` | Yes | Yes | Yes |

FDS and LDS share Gaussian kernel size 5, sigma 2; FDS uses start_update 0, start_smooth 1, momentum 0.9. When FDS is used, its feature statistics are collected in `eval()` mode from *raw* backbone features (no FDS smoothing), avoiding unwanted BatchNorm running-stat updates during the extra collection pass. ConR's contrastive branch retains `reg=False` so it also uses raw features. Output names include the mode and seed.

The legacy invocation `sbatch job_dir_aligned.sh 0` still means `conr`, seed 0.

The ConR regularizer has **not** been changed; only the backbone selection, ImageNet loading, normalization switch, scheduler and checkpoint eligibility were added. The custom ResNet18 uses the same standard BasicBlock topology and pooling as torchvision. The regression head is randomly initialized; the pretrained feature trunk weights are transferred by name/shape.

`postadapt.py` / `conr_backbone.py` support both DIR-aligned ResNet18 and original ConR ResNet50 checkpoints. The backbone is inferred from checkpoint weights, and the default data settings (including ImageNet versus 0.5 normalization) follow the source checkpoint's sibling `training.log`. Keep that log with the checkpoint, or provide the data settings explicitly. IMDB-WIKI backbone training was not modified by this AgeDB-only entrypoint.

**Verification:** Config options and code paths were inspected; no full GPU training has been run here. Torchvision weights must be cached or downloadable on the cluster.


# DIR-aligned ConR training — IMDB-WIKI

`job_dir_aligned_imdb.sh` is the IMDB-WIKI counterpart of `job_dir_aligned.sh`. It matches `Kishore4399/ImbalanceRegression@dir-pipeline/job_sub.sh imdb_wiki`. `imdb-wiki-dir/train.py` got the same changes as the aligned `agedb-dir/train.py`: `--imagenet_pretrained` with the ImageNet normalization switch, a ResNet18 option, `--lr_step`, `--lr_gamma`, `--save_start`, and FDS statistics collected in `eval()` mode from raw features. It also gained `--amp`.

```bash
for s in 0 1 2 3 4; do
  for method in conr conr_fds conr_lds_fds; do
    sbatch --export=ALL job_dir_aligned_imdb.sh "$method" "$s"
  done
done
```

Override the paths with `IMDB_DATA_DIR` and `IMDB_CSV`. Output goes to `checkpoint/imdb_wiki/diraligned`.

| Parameter | DIR-aligned ConR IMDB-WIKI |
| --- | --- |
| Backbone | ImageNet-pretrained ResNet50 (torchvision V2 weights) |
| Input | 224×224; resize/crop with padding 16/horizontal flip; ImageNet mean/std |
| Epochs | 90 |
| Batch size | 256 (evaluation 256) |
| Optimizer | Adam, learning rate 1e-3, weight decay 1e-4 |
| Scheduler | ×0.5 every 30 epochs |
| Checkpoint | lowest validation L1 from epoch 20 onward |
| Precision | mixed (`--amp`); features, losses and ConR in float32 |
| Dataset | shared DIR IMDB-WIKI split CSV |
| ConR-specific | original 3-view setup, window=1, temperature=0.2, beta=4, eta=0.01 (as `submission.sh`) |

**Temperature note.** `ConR()` in `imdb-wiki-dir/loss.py` uses the `-t` it is given, so IMDB-WIKI really runs at 0.2. `ConR()` in `agedb-dir/loss.py` overwrites `t` with 0.07 internally, so the AgeDB runs use 0.07 whatever `-t` says. The loss code was not changed here.

The partition list leaves out `gpu-mig`: each step forwards 3 × 256 images through ResNet50.
