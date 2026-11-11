# ConR reproduction for AgeDB-DIR and IMDB-WIKI-DIR

This repository adapts the official implementation of **ConR, Contrastive Regularizer for Deep Imbalanced Regression**, Mahsa Keramati, Lili Meng, and R. David Evans, ICLR 2024. It adds a cluster launcher and ports post-training adaptation from `Kishore4399/PostTrainingAdpt`.

[Paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/704357127afabbd5a6eb979a57810767-Paper-Conference.pdf) · [Official source revision](https://github.com/BorealisAI/ConR/tree/ad6a8c90426da9243f9e1de7bb37c7429bb99256) · [Adaptation source](https://github.com/Kishore4399/PostTrainingAdpt/tree/condition-variance-cosine)

ConR penalizes pairs whose labels differ while their predictions are similar. Two augmented views provide positive pairs. The contrastive term regularizes the ResNet-50 representation alongside supervised regression.

## Setup and data

Use Linux, Python 3.10 or newer, and a compatible PyTorch/torchvision pair. Install PyTorch for the cluster's CUDA version, then install the remaining dependencies.

```bash
pip install -r requirements.txt
```

Activate the environment before running or submitting a job. Set `PYTHON=/path/to/python` to choose a particular interpreter.

| Dataset | Default image root | Default CSV |
| --- | --- | --- |
| AgeDB-DIR | `/isilon/datalake/dma_research/scratch/damalab/Data/Shihua/82bd6-main` | `agedb-dir/data/agedb.csv` |
| IMDB-WIKI-DIR | `/isilon/datalake/dma_research/scratch/damalab/Tools/imbalanced-regression/imdb-wiki-dir/data` | `imdb-wiki-dir/data/imdb_wiki.csv` |

These roots come from your adaptation branch's `job_sub.sh`, `run_cv.sh`, and YAML configuration. Override them with `AGEDB_DATA_DIR` and `IMDB_WIKI_DATA_DIR`. Override manifests with `AGEDB_CSV` and `IMDB_WIKI_CSV`. CSV paths are relative to the image roots. AgeDB filenames begin with `AgeDB/`, and IMDB-WIKI filenames begin with `imdb_crop/` or `wiki_crop/`.

The official CSVs contain 12,208/2,140/2,140 AgeDB train/val/test rows and 191,509/11,022/11,022 IMDB-WIKI rows. Your source repo references ignored local CSVs, so those split lists could not be compared. Check the paths and counts on the cluster.

The large IMDB-WIKI CSV is fetched automatically from the pinned official revision when first needed. Its Git blob hash is verified before it is saved. Prepare it on an internet-connected login node before submitting offline GPU jobs.

```bash
python dir_data.py imdb_wiki --prepare
```

If the cluster already has the CSV, set `IMDB_WIKI_CSV` to that file.

## Train

Run from the checkout root.

```bash
bash submission.sh both --check-data
bash submission.sh both --dry-run

bash submission.sh agedb
bash submission.sh imdb_wiki
bash submission.sh both

sbatch submission.sh agedb
sbatch submission.sh imdb_wiki
```

`both` trains sequentially. Independent Slurm submissions allocate one GPU each. Slurm defaults follow your existing scripts with `gpu-mig,defq` partitions, eight CPU cores, and 50 GB host memory. Set your partition's wall-time limit when submitting, for example `sbatch --time=24:00:00 submission.sh agedb`. Set `CONR_ROOT` to the checkout's absolute path when submitting from another directory.

Both datasets explicitly use the paper's Appendix A.4 age-estimation settings.

| Setting | Value |
| --- | --- |
| Backbone / regression loss | ResNet-50 / L1 |
| Epochs / batch size | 90 / 64 |
| Adam learning rate / weight decay | 0.00025 / 0.0001 |
| Learning-rate drops | 10× at epochs 60 and 80 |
| ConR window / temperature | 1 / 0.2 |
| ConR coefficient / pushing scale | 4 / 0.01 |
| Image size / augmentation | 224 / padded random crop and horizontal flip |

Additional training arguments override earlier values. Set `CONR_OUTPUT_DIR` to change the output root.

```bash
CONR_SEED=1 bash submission.sh agedb --batch_size 32
CONR_OUTPUT_DIR=/scratch/conr bash submission.sh both

for seed in 0 1 2 3 4; do
    CONR_SEED=$seed bash submission.sh both
done
```

Five seeds follow the paper's reporting protocol. Changing batch size changes ConR's pair set. Hardware, GPU count, and nondeterministic operations can affect results.

Each timestamped folder under `checkpoint/agedb/` or `checkpoint/imdb_wiki/` contains `ckpt.pth.tar`, the validation-selected `ckpt.best.pth.tar`, and `outputs.csv` with MAE and geometric mean for all/many/median/few shots.

```bash
bash submission.sh agedb --resume /path/to/ckpt.pth.tar
bash submission.sh imdb_wiki --evaluate --resume /path/to/ckpt.best.pth.tar
```

Use the same model and training options when resuming.

## Post-training adaptation

RankSim-style single-run and batch submission scripts are available in
[`postadapt/`](postadapt/README.md). From that directory, run
`bash submit_postadapt.sh --dry-run` to inspect the default AgeDB DIR-aligned jobs,
then `bash submit_postadapt.sh` to submit them. Grid settings and run filters are documented there.

`postadapt.py` ports the original `postadpt.py`. It strictly loads ConR's encoder and `linear.*` regression head, including DataParallel `module.*` prefixes. FDS training statistics are omitted because evaluation uses raw encoder features.

```bash
python postadapt.py -c PostTrainAdpt.yaml \
    model.pretrained=/path/to/agedb/ckpt.best.pth.tar

python postadapt.py -c PostTrainAdpt.yaml \
    dataset.name=imdb_wiki \
    model.pretrained=/path/to/imdb_wiki/ckpt.best.pth.tar \
    trainer.max_steps=10000
```

The default uses labeled validation images, a frozen head, W2 conditional error matching, L2-SP weight 1, and a source error trend fitted on the adaptation split. Validation also selects the adapted checkpoint, following your source workflow. A custom CSV can supply a separate `dataset.adaptation_split`. Test metrics are reported independently of checkpoint selection.

W2/L1/KL/symmetric-KL losses, label-balanced sampling, cosine learning-rate scheduling, source bias correction, and validation DC/MAE selection are retained. `model.optim.param=all` also trains the head. A separate frozen source head keeps reference predictions stable in that mode.

Evaluation resizes to 224×224 and inherits the source normalization: ImageNet statistics for `--imagenet_pretrained` runs, mean/std 0.5 for original ConR. Adaptation uses the same normalization and defaults to crop and flip. Full precision is the default. Override `trainer.precision=bf16-mixed` on supported hardware.

Post-adaptation supports both ResNet18 (512 features) and ResNet50 (2048). The default `model.backbone.params.name=auto` infers the architecture from checkpoint weights; an explicit name must match. `dataset.name`, `data_folder`, `data_csv`, `img_size`, and `normalization` default to `auto` and follow the source checkpoint's sibling `training.log`. Explicit dataset values override these defaults. For old ConR logs without an ImageNet flag, normalization remains 0.5. Missing logs require explicit settings; the code does not guess normalization. Source relative paths are resolved against the dataset training directory used by the supplied launch scripts; custom training working directories need explicit absolute overrides.

Resolved inputs are printed and saved to `source_data_config.json`. Default corrected output directories have a `_sourceAligned_imagenet` or `_sourceAligned_conr` suffix, keeping older exports separate. `logger.save_dir` remains an explicit output-directory override. Existing before/after predictions must be regenerated with the corrected inputs, and an adapted model trained with incorrect normalization needs adaptation rerun. The source backbone does not need retraining solely for this issue.

Run `python -m unittest discover -s tests -v` for source-contract regression checks and CPU checkpoint prediction parity for ResNet18/50. Parity tests skip if the training libraries are unavailable.

Outputs beside the source checkpoint include source bias coefficients, before/after predictions, candidate checkpoints, the selected adapted checkpoint, and `metrics_best.csv`. Override `logger.save_dir` to choose another location. W&B logging is offline. Adaptation currently uses one device.

## Corrections and reproduction limits

- IMDB-WIKI discovers allocated GPUs, replacing four hardcoded GPU IDs.
- IMDB-WIKI adds the ConR term once. Upstream added it twice.
- Adam receives the configured weight decay, matching the paper.
- Both scripts use consistent validation return values and successfully export test metrics.
- CSV paths are separate from image roots. Output folders are created recursively and named portably. Empty shot groups report NaN during small smoke tests.
- CPU execution supports local validation.

Pair selection, exponential pushing weights, the contrastive denominator, and the released three-view training layout follow the official executable implementation. Appendix A.4 describes a different pushing-weight expression. The port retains the released loss.

Full dataset training and paper metrics require the cluster's images and GPU. Local synthetic-image tests verify execution and checkpoint compatibility.

## Attribution

Age-estimation code comes from BorealisAI/ConR at `ad6a8c90426da9243f9e1de7bb37c7429bb99256`. The adaptation source blobs are `238df619d94ba5c9968e0d4b2af280f774071000` for `postadpt.py` and `eab4a396399abadb244671f9ea380b6fa606fdb0` for `dataset.py` on `condition-variance-cosine`.

The original source notices and [LICENSE](LICENSE), CC BY-NC-SA 4.0, are preserved. ConR builds on Yang et al., Delving into Deep Imbalanced Regression, and Gong et al., RankSim.

