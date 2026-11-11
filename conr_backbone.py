"""Feature-only wrapper around the official ConR ResNet-50."""
import sys
from pathlib import Path

import torch
from torch import nn

# Both official age encoders have the same backbone and parameter names.
sys.path.insert(0, str(Path(__file__).resolve().parent / "agedb-dir"))
from resnet import resnet50


class Encoder(nn.Module):
    dim = 2048

    def __init__(self, name="resnet50"):
        super().__init__()
        if name != "resnet50":
            raise ValueError("ConR age checkpoints use resnet50")
        self.encoder = resnet50(
            fds=False, bucket_num=100, bucket_start=3, start_update=0,
            start_smooth=1, kernel="gaussian", ks=5, sigma=2, momentum=0.9,
        )
        self.encoder.linear = nn.Identity()

    def forward(self, images):
        return self.encoder(images)[1]


def load_checkpoint(encoder, head, path):
    """Load official wrapped or unwrapped state_dict checkpoints strictly."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    # FDS statistics affect training-time smoothing only. Evaluation uses raw features.
    backbone = {key: value for key, value in state.items()
                if not key.startswith(("linear.", "FDS."))}
    encoder.encoder.load_state_dict(backbone, strict=True)
    head.load_state_dict(
        {key.removeprefix("linear."): value for key, value in state.items()
         if key.startswith("linear.")},
        strict=True,
    )
