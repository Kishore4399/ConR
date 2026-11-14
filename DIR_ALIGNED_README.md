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


# ConR training — IMDB-WIKI (paper reproduction, from scratch)

`job_dir_aligned_imdb.sh` reproduces the ConR paper's IMDB-WIKI-DIR setup (Keramati et al., ICLR 2024, App. A.4). No Python code changed.

```bash
for s in 0 1 2 3 4; do
  sbatch --export=ALL job_dir_aligned_imdb.sh conr "$s"
done
```

`vanilla` (same flags without `--conr`), `conr_fds` and `conr_lds_fds` also work. Override the paths with `IMDB_DATA_DIR` and `IMDB_CSV`. Output goes to `checkpoint/imdb_wiki/paper`.

| Parameter | ConR paper, App. A.4 |
| --- | --- |
| Backbone | ResNet50 from scratch (no `--imagenet_pretrained`) + one linear regression layer |
| Input | 224×224; resize/crop with padding 16/horizontal flip; DIR mean/std 0.5 |
| Epochs | 90 |
| Batch size | 64 |
| Optimizer | Adam, learning rate 2.5e-4, weight decay 1e-4 |
| Scheduler | ×0.1 at epochs 60 and 80 |
| Loss | L1 regression (alpha 1) + beta × ConR |
| ConR | two augmented views per image, window ω 1, temperature τ 0.2, beta 4, eta 0.01 |
| Checkpoint | lowest validation L1 over all epochs (`--save_start 1`), then tested |
| Precision | full (fp32), as in the paper |
| Dataset | shared DIR IMDB-WIKI split CSV |

Target (paper Table 2, ConR-only): MAE 7.33 all / 6.75 many / 11.99 medium / 22.22 few; GM 4.02 / 3.79 / 6.98 / 12.95. The paper's own ablations (Tables 12 and 16) report about 7.84 for the same setting, so expect a result somewhere in that range.

**Where the public code differs from the paper text.** This job follows the text. The public `imdb-wiki-dir/train.py` (BorealisAI/ConR) differs in three ways:
- Its Adam call takes no weight decay (the flag only reaches SGD). Run with `CONR_WD=0` to match it.
- Its README command leaves `-t` at the default 0.07 instead of 0.2. Run with `CONR_TEMP=0.07` to match it.
- It adds `beta * L_ConR` to the loss twice (effective weight 8). This fork adds it once, as the paper describes; reproducing the double-add needs `--beta 8`.

The pushing weight in `loss.py` (unchanged from the public code) is `w_j * exp(eta * |y_j - y_q|)` rather than the paper's linear `eta * MAE`. Without reweighting, `w_j` is 1.

**Temperature note.** `ConR()` in `imdb-wiki-dir/loss.py` uses the `-t` it is given. `ConR()` in `agedb-dir/loss.py` overwrites `t` with 0.07 internally, so the AgeDB runs use 0.07 whatever `-t` says. The loss code was not changed here.

**Memory.** Each ConR step forwards 3 × 64 images through ResNet50 in fp32.
