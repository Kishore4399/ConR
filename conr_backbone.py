"""Feature-only ConR ResNet-18/50 wrappers and strict checkpoint loading."""
import sys
from pathlib import Path

import torch
from torch import nn

# Both official age encoders have the same backbone and parameter names.
sys.path.insert(0, str(Path(__file__).resolve().parent / "agedb-dir"))
from resnet import resnet18, resnet50
from source_contract import infer_backbone, resolve_backbone


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    stripped = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[len("module."):]
        stripped[key] = value
    name = infer_backbone(stripped)
    declared = checkpoint.get("model")
    if declared is not None and declared != name:
        raise ValueError(f"Checkpoint model={declared} disagrees with its {name} weights")
    return stripped, name


class Encoder(nn.Module):
    dim = 2048

    def __init__(self, name="resnet50"):
        super().__init__()
        builders = {"resnet18": resnet18, "resnet50": resnet50}
        if name not in builders:
            raise ValueError(f"Unsupported ConR backbone: {name}")
        self.name = name
        self.dim = 512 if name == "resnet18" else 2048
        self.encoder = builders[name](
            fds=False, bucket_num=100, bucket_start=3, start_update=0,
            start_smooth=1, kernel="gaussian", ks=5, sigma=2, momentum=0.9,
        )
        self.encoder.linear = nn.Identity()

    def forward(self, images):
        return self.encoder(images)[1]


def load_checkpoint(encoder, head, path):
    """Load official wrapped or unwrapped state_dict checkpoints strictly."""
    state, name = read_checkpoint(path)
    resolve_backbone(encoder.name, name)
    # FDS statistics affect training-time smoothing only. Evaluation uses raw features.
    backbone = {key: value for key, value in state.items()
                if not key.startswith(("linear.", "FDS."))}
    encoder.encoder.load_state_dict(backbone, strict=True)
    head.load_state_dict(
        {key.removeprefix("linear."): value for key, value in state.items()
         if key.startswith("linear.")},
        strict=True,
    )

