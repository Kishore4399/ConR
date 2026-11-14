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


# ConR training — IMDB-WIKI (shared DIR recipe, from scratch)

`job_dir_aligned_imdb.sh` trains IMDB-WIKI with the recipe shared by the vanilla, SRL, RankSim and ConR runs: the DIR recipe of Yang et al. (ICML 2021). It matches the IMDB-WIKI jobs on `Kishore4399/SLR@dir-pipeline-aligned` and `Kishore4399/RankSim@dir-pipeline-aligned`, so `vanilla` here is the same baseline as theirs and only the ConR regularizer differs. No Python code changed.

```bash
for s in 0 1 2 3 4; do
  for method in vanilla conr; do
    sbatch --export=ALL job_dir_aligned_imdb.sh "$method" "$s"
  done
done
```

`conr_fds` and `conr_lds_fds` still work as optional extra rows with the same recipe. Override the paths with `IMDB_DATA_DIR` and `IMDB_CSV`. Output goes to `checkpoint/imdb_wiki/dirrecipe`.

| Parameter | ConR IMDB-WIKI (shared recipe) |
| --- | --- |
| Backbone | ResNet50 from scratch (no `--imagenet_pretrained`) |
| Input | 224×224; resize/crop with padding 16/horizontal flip; DIR mean/std 0.5 |
| Epochs | 90 |
| Batch size | 256 (evaluation 256) |
| Optimizer | Adam, learning rate 1e-3, **no weight decay** (the public ConR, RankSim and DIR code pass weight decay only to SGD) |
| Scheduler | ×0.1 at epochs 60 and 80 |
| Loss | L1 regression; plain shuffled batches, no resampling or reweighting for `vanilla` / `conr` |
| Checkpoint | lowest validation L1 over all epochs (`--save_start 1`) |
| Precision | mixed (`--amp`); features, losses and ConR in float32 |
| Dataset | shared DIR IMDB-WIKI split CSV |
| ConR-specific | 3-view setup (regression on the original image, ConR on two augmented views), window 1, temperature 0.2, beta 4, eta 0.01 |

**Differences from the ConR paper and public code.** The paper trains IMDB-WIKI with batch 64 and LR 2.5e-4; the shared recipe uses DIR's batch 256 and LR 1e-3 so all methods share one backbone recipe. The public README command leaves `-t` at its default 0.07, while the paper (App. A.4) says 0.2; this job uses 0.2. The public `train.py` adds `beta * L_ConR` to the loss twice (effective weight 8); this fork adds it once (weight 4, as in the paper). The public pushing weight is `w_j * exp(eta * |y_j - y_q|)`, not the paper's linear `eta * MAE`; `loss.py` is unchanged from the public code.

**Temperature note.** `ConR()` in `imdb-wiki-dir/loss.py` uses the `-t` it is given, so IMDB-WIKI really runs at 0.2. `ConR()` in `agedb-dir/loss.py` overwrites `t` with 0.07 internally, so the AgeDB runs use 0.07 whatever `-t` says. The loss code was not changed here.

**Memory.** Each ConR step forwards 3 × 256 images through ResNet50, so it needs a full GPU; it does not fit a `gpu-mig` slice.
