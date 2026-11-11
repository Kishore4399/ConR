# Adapted from Kishore4399/PostTrainingAdpt, condition-variance-cosine.
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, Sampler
import pytorch_lightning as pl
from torchvision import transforms


def get_transforms(split, aug):
    operations = [transforms.Resize((224, 224))]
    if split == 'train':
        if 'crop' in aug.split(','):
            operations.append(transforms.RandomCrop(224, padding=16))
        if 'flip' in aug.split(','):
            operations.append(transforms.RandomHorizontalFlip())
        if 'color' in aug.split(','):
            operations.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
        if 'grayscale' in aug.split(','):
            operations.append(transforms.RandomGrayscale(p=0.2))
    operations.extend([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    return transforms.Compose(operations)


class AgeDB_(data.Dataset):
    def __init__(self, data_folder, data_csv='./data/agedb.csv', label_name='age', transform=None, split='train'):
        df = pd.read_csv(data_csv)
        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder
        self.transform = transform
        self.label_name = label_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = np.asarray([row[self.label_name]]).astype(np.float32)
        img = Image.open(os.path.join(self.data_folder, row['path'])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "label": label,
        }


class AgeDB_PTA(data.Dataset):
    """Adaptation set for postadpt.py: the ``adpt_split`` rows, with training augmentation."""

    def __init__(self, data_folder, data_csv='./data/agedb.csv', label_name='age', transform=None, adpt_split='val'):
        df = pd.read_csv(data_csv)
        self.adpt_df = df[df['split'] == adpt_split]
        self.adpt_split = adpt_split
        self.data_folder = data_folder
        self.transform = transform
        self.label_name = label_name

    def __len__(self):
        return len(self.adpt_df)

    def __getitem__(self, index):
        row = self.adpt_df.iloc[index]
        img = Image.open(os.path.join(self.data_folder, row["path"])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": np.float32(row[self.label_name])}


class LabelBalancedBatchSampler(Sampler):
    """Stratified batches that even out the label distribution (training only).

    The label range is split into ``num_bins`` equal-width bins (empty bins are dropped). Bin b
    gets the share pi_b ~ n_b ** (1 - alpha) of every batch, where n_b is its sample count:
    alpha = 1 gives every bin the same share (flat labels), alpha = 0 keeps the natural
    distribution, values in between (e.g. 0.5) are a compromise that repeats rare labels less.

    Every batch is built from per-bin quotas (expected count B * pi_b, remainders rounded at
    random), so each batch covers the whole label range rather than being flat only on average.
    Within a bin, samples are drawn without replacement from a shuffled order and the bin is
    reshuffled when exhausted, so rare samples are repeated as evenly as possible.

    ``num_batches`` defaults to len(labels) // batch_size, the same number of steps per epoch as
    shuffle with drop_last=False. Single process only (no DDP sharding). Call ``set_epoch`` (done by
    Lightning) for a different, reproducible order every epoch.
    """

    def __init__(self, labels, batch_size, num_bins=10, alpha=1.0, num_batches=None, seed=0):
        labels = np.asarray(labels, dtype=np.float64).ravel()
        if labels.size == 0:
            raise ValueError("LabelBalancedBatchSampler needs at least one label")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if int(num_bins) < 1 or int(batch_size) < 1:
            raise ValueError("num_bins and batch_size must be >= 1")
        self.batch_size = int(batch_size)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.num_batches = int(num_batches) if num_batches is not None else max(labels.size // self.batch_size, 1)
        self.epoch = 0
        self._iter_count = 0

        lo, hi = labels.min(), labels.max()
        edges = np.linspace(lo, hi, int(num_bins) + 1)
        bin_of = np.clip(np.searchsorted(edges, labels, side="right") - 1, 0, int(num_bins) - 1)
        self.bin_indices = [np.flatnonzero(bin_of == b) for b in range(int(num_bins))]
        self.bin_edges = [(edges[b], edges[b + 1]) for b in range(int(num_bins))]
        keep = [i for i, idx in enumerate(self.bin_indices) if idx.size > 0]
        self.bin_indices = [self.bin_indices[i] for i in keep]
        self.bin_edges = [self.bin_edges[i] for i in keep]

        counts = np.array([idx.size for idx in self.bin_indices], dtype=np.float64)
        share = counts ** (1.0 - self.alpha)
        self.bin_share = share / share.sum()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._iter_count = 0

    def __len__(self):
        return self.num_batches

    def expected_per_batch(self):
        return self.batch_size * self.bin_share

    def summary(self):
        """Per-bin label range, count, expected samples per batch and repeats per epoch."""
        expected = self.expected_per_batch()
        rows = []
        for (a, b), idx, e in zip(self.bin_edges, self.bin_indices, expected):
            draws = e * self.num_batches
            rows.append({"bin": f"[{a:.1f}, {b:.1f}]", "n": int(idx.size),
                         "per_batch": round(float(e), 2), "repeats_per_epoch": round(float(draws / idx.size), 2)})
        return rows

    def _quotas(self, rng):
        expected = self.expected_per_batch()
        base = np.floor(expected).astype(int)
        remainder = self.batch_size - base.sum()
        if remainder > 0:
            frac = expected - base
            extra = rng.choice(len(base), size=remainder, replace=False, p=frac / frac.sum())
            base[extra] += 1
        return base

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch, self._iter_count])
        self._iter_count += 1
        orders = [rng.permutation(idx) for idx in self.bin_indices]
        pos = [0] * len(orders)
        for _ in range(self.num_batches):
            batch = []
            for b, q in enumerate(self._quotas(rng)):
                for _ in range(q):
                    if pos[b] == len(orders[b]):            # bin exhausted: reshuffle it
                        orders[b] = rng.permutation(self.bin_indices[b])
                        pos[b] = 0
                    batch.append(int(orders[b][pos[b]]))
                    pos[b] += 1
            rng.shuffle(batch)
            yield batch


class AgeDBPTADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_folder,
        data_csv='./data/agedb.csv', 
        label_name='age',
        aug="crop,flip",
        batch_size=256,
        val_batch_size=256,
        num_workers=8,
        shuffle=True,
        balance=None,
        adaptation_split='val',
    ):
        super().__init__()
        self.adaptation_split = adaptation_split
        self.data_folder = data_folder
        self.aug = aug
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_csv = data_csv
        self.label_name = label_name
        # Label-balanced sampling of the adaptation (training) batches; validation, test and the
        # source-trend fit (adaptation_dataloader) always use the natural distribution.
        balance = dict(balance or {})
        self.balance = bool(balance.get("enabled", False))
        self.balance_alpha = float(balance.get("alpha", 1.0))
        self.balance_num_bins = int(balance.get("num_bins", 10))
        self.balance_seed = int(balance.get("seed", 0))

    def setup(self, stage=None):
        self.train_dataset = AgeDB_PTA(
            self.data_folder,
            data_csv=self.data_csv,
            label_name=self.label_name,
            transform=get_transforms(split="train", aug=self.aug),
            adpt_split=self.adaptation_split)
        
        self.val_dataset = AgeDB_(
            self.data_folder,
            data_csv=self.data_csv,
            label_name=self.label_name,
            transform=get_transforms(split="val", aug=""),
            split='val')

        self.test_dataset = AgeDB_(
            self.data_folder,
            data_csv=self.data_csv,
            label_name=self.label_name,
            transform=get_transforms(split="val", aug=""),
            split='test')
        
    def adaptation_dataloader(self):
        """Every adaptation sample exactly once, without augmentation.

        Used to fit the source model's label-dependent error trend before adaptation.
        The split is read from the training dataset (``AgeDB_PTA.adpt_split``), so this
        always covers the same rows the adaptation loss trains on, and never the test split.
        """
        adpt_split = self.train_dataset.adpt_split
        if adpt_split == 'test':
            raise ValueError("The adaptation split must not be the test split.")
        adpt_dataset = AgeDB_(
            self.data_folder,
            data_csv=self.data_csv,
            label_name=self.label_name,
            transform=get_transforms(split="val", aug=""),
            split=adpt_split)

        return DataLoader(
            adpt_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=None
        )

    def train_eval_dataloader(self):
        """The source training split ('train'), each sample once, without augmentation.

        Only used to save predictions on the data the pretrained model was trained on
        (train_before.csv / train_after.csv); adaptation never trains on it.
        """
        return DataLoader(
            AgeDB_(
                self.data_folder,
                data_csv=self.data_csv,
                label_name=self.label_name,
                transform=get_transforms(split="val", aug=""),
                split='train'),
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        if self.balance:
            if self.trainer is not None and self.trainer.world_size > 1:
                raise ValueError("dataset.balance supports a single device only (no DDP sharding)")
            sampler = LabelBalancedBatchSampler(
                self.train_dataset.adpt_df[self.label_name].to_numpy(),
                batch_size=self.batch_size, num_bins=self.balance_num_bins,
                alpha=self.balance_alpha, seed=self.balance_seed)
            print(f"Label-balanced adaptation batches (alpha={self.balance_alpha}, "
                  f"{len(sampler.bin_indices)} non-empty bins, {len(sampler)} batches/epoch):")
            for row in sampler.summary():
                print(f"  {row['bin']:>16}  n={row['n']:>5}  per batch {row['per_batch']:>6}  "
                      f"repeats/epoch {row['repeats_per_epoch']}")
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
            return DataLoader(
                self.test_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
