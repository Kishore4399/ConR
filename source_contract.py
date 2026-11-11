"""Source run input/architecture contract; no training dependencies required."""
import ast
from pathlib import Path


def infer_backbone(state):
    if "conv1.weight" not in state:
        raise ValueError("Not a ConR ResNet checkpoint: missing conv1.weight")
    blocks = [{key.split(".")[1] for key in state if key.startswith(f"layer{i}.")}
              for i in range(1, 5)]
    layout = tuple(len(block) for block in blocks)
    bottleneck = any(key.startswith("layer1.0.conv3.") for key in state)
    if bottleneck and layout == (3, 4, 6, 3):
        return "resnet50"
    if not bottleneck and layout == (2, 2, 2, 2):
        return "resnet18"
    raise ValueError(f"Unsupported ConR ResNet layout: {layout}, bottleneck={bottleneck}")


def resolve_backbone(configured, checkpoint_name):
    if configured in (None, "auto"):
        return checkpoint_name
    if configured != checkpoint_name:
        raise ValueError(f"Configured backbone {configured} does not match checkpoint {checkpoint_name}")
    return configured


def read_source_args(checkpoint):
    log = Path(checkpoint).resolve().parent / "training.log"
    if not log.exists():
        return {}
    result = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Args: Namespace(" not in line:
            continue
        try:
            node = ast.parse(line.split("Args: ", 1)[1], mode="eval").body
        except (SyntaxError, ValueError):
            continue
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Namespace":
            continue
        parsed = {}
        for item in node.keywords:
            try:
                parsed[item.arg] = ast.literal_eval(item.value)
            except (ValueError, TypeError):
                pass
        result = parsed
    return result


def resolve_source_data(dataset, checkpoint):
    dataset = dict(dataset)
    checkpoint = Path(checkpoint).resolve()
    source = read_source_args(checkpoint)
    # The supplied source-training scripts cd into <dataset>-dir before train.py.
    training_dir = checkpoint.parent.parent.parent

    def source_path(value):
        path = Path(value)
        return str((path if path.is_absolute() else training_dir / path).resolve())

    if dataset.get("name") == "auto":
        if source.get("dataset") not in ("agedb", "imdb_wiki"):
            raise ValueError("Restore source training.log or set dataset.name explicitly")
        dataset["name"] = source["dataset"]
    if dataset.get("normalization", "auto") == "auto":
        if not source:
            raise ValueError("Cannot infer source normalization: restore training.log or set "
                             "dataset.normalization=imagenet or conr explicitly")
        imagenet = source.get("imagenet_pretrained", False)  # original ConR logs predate this flag
        if not isinstance(imagenet, bool):
            raise ValueError("Source imagenet_pretrained must be a boolean")
        dataset["normalization"] = "imagenet" if imagenet else "conr"
    if dataset["normalization"] not in ("imagenet", "conr"):
        raise ValueError("dataset.normalization must be auto, imagenet, or conr")
    if dataset.get("data_folder") == "auto":
        if not source.get("data_dir"):
            raise ValueError("Restore source training.log or set dataset.data_folder explicitly")
        dataset["data_folder"] = source_path(source["data_dir"])
    if dataset.get("data_csv") == "auto":
        if source.get("data_csv"):
            dataset["data_csv"] = source_path(source["data_csv"])
        elif source.get("data_dir") and source.get("dataset") in ("agedb", "imdb_wiki"):
            dataset["data_csv"] = source_path(str(Path(source["data_dir"]) / f"{source['dataset']}.csv"))
        else:
            raise ValueError("Restore source training.log or set dataset.data_csv explicitly")
    if dataset.get("img_size") == "auto":
        if "img_size" not in source:
            raise ValueError("Restore source training.log or set dataset.img_size explicitly")
        dataset["img_size"] = int(source["img_size"])
    # ConR's AvgPool2d(7) and fixed-size regressor require a 7x7 final feature map.
    if int(dataset.get("img_size", 224)) != 224:
        raise ValueError("ConR postadapt currently requires source img_size=224")
    return dataset
