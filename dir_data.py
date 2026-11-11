"""Locate and validate the DIR split manifests and image directories."""
import argparse
import csv
import hashlib
import math
import os
import sys
import tempfile
from urllib.request import urlopen
from pathlib import Path


REPO = Path(__file__).resolve().parent
CLUSTER_ROOT = Path("/isilon/datalake/dma_research/scratch/damalab")
DATASETS = {
    "agedb": ("agedb-dir", CLUSTER_ROOT / "Data/Shihua/82bd6-main", "AGEDB"),
    "imdb_wiki": (
        "imdb-wiki-dir",
        CLUSTER_ROOT / "Tools/imbalanced-regression/imdb-wiki-dir/data",
        "IMDB_WIKI",
    ),
}
SPLITS = ("train", "val", "test")
UPSTREAM = "ad6a8c90426da9243f9e1de7bb37c7429bb99256"
MANIFEST_BLOBS = {
    "agedb": "78abfdb939ff5cf68672cf11ee9b84a0eb9e990a",
    "imdb_wiki": "c1512e3da0de9076a0793b982c9a74cd872c0d8b",
}


def ensure_manifest(manifest):
    """Fetch a missing official CSV from a pinned revision and verify its Git hash."""
    manifest = Path(manifest).resolve()
    if manifest.is_file():
        return
    candidates = {REPO / folder / "data" / f"{name}.csv": name
                  for name, (folder, _, _) in DATASETS.items()}
    if manifest not in candidates:
        raise FileNotFoundError(f"CSV does not exist: {manifest}")
    dataset = candidates[manifest]
    relative = manifest.relative_to(REPO).as_posix()
    url = f"https://raw.githubusercontent.com/BorealisAI/ConR/{UPSTREAM}/{relative}"
    print(f"Downloading official split manifest from {url}", file=sys.stderr)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urlopen(url, timeout=120) as response:
            content = response.read()
        header = f"blob {len(content)}\0".encode()
        digest = hashlib.sha1(header + content).hexdigest()
        if digest != MANIFEST_BLOBS[dataset]:
            raise ValueError(f"Official split manifest checksum mismatch: {manifest}")
        with tempfile.NamedTemporaryFile(dir=manifest.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
        os.replace(temporary, manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def dataset_paths(dataset, data_dir=None, data_csv=None):
    folder, default_images, prefix = DATASETS[dataset]
    images = Path(data_dir or os.getenv(f"{prefix}_DATA_DIR") or default_images)
    manifest = Path(data_csv or os.getenv(f"{prefix}_CSV") or REPO / folder / "data" / f"{dataset}.csv")
    return images.expanduser().resolve(), manifest.expanduser().resolve()


def check_data(images, manifest, label="age", all_images=False):
    if not images.is_dir():
        raise ValueError(f"Image directory does not exist: {images}")
    ensure_manifest(manifest)
    counts = dict.fromkeys(SPLITS, 0)
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        if not {"path", "split", label}.issubset(rows.fieldnames or []):
            raise ValueError(f"{manifest} must contain path, split, and {label} columns")
        for row in rows:
            split = row["split"]
            if split not in counts:
                raise ValueError(f"{manifest}:{rows.line_num}: unexpected split {split!r}")
            age = float(row[label])
            if not math.isfinite(age) or age < 0:
                raise ValueError(f"{manifest}:{rows.line_num}: age must be finite and nonnegative")
            if all_images or counts[split] == 0:
                image = images / row["path"]
                if not image.is_file():
                    raise ValueError(f"{manifest}:{rows.line_num}: image is missing: {image}")
            counts[split] += 1
    if not all(counts.values()):
        raise ValueError(f"{manifest} needs nonempty train, val, and test splits: {counts}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Fetch a missing official CSV")
    args = parser.parse_args()
    images, manifest = dataset_paths(args.dataset)
    if args.prepare:
        ensure_manifest(manifest)
    if args.check:
        try:
            counts = check_data(images, manifest, all_images=True)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(f"{args.dataset}: {counts} in {manifest}\nImages: {images}")
    else:
        print(images)
        print(manifest)
