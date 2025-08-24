from __future__ import annotations
import torch
import torch.nn as nn
from monai.losses import TverskyLoss

def make_loss(name: str, out_channels: int=2):
    name = name.lower()
    if name in ("ce","crossentropy","cross_entropy"):
        return nn.CrossEntropyLoss()
    if name in ("tversky","tversky_loss"):
        # foreground emphasis (alpha<beta)
        return TverskyLoss(include_background=True, alpha=0.3, beta=0.7, to_onehot_y=True, softmax=True)
    raise ValueError(f"Unknown loss: {name}")