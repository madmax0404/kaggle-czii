from __future__ import annotations
import os, numpy as np
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Spacing, Orientation,
    RandSpatialCrop, RandFlip, RandRotate90, ScaleIntensityRange, EnsureType
)

# Minimal dataset: expects each sample to have volume.npy and mask.npy
# For zarr, you can change LoadImage to zarr loader or np.load inside __getitem__.
class CryoETSegDataset(Dataset):
    def __init__(self, root: str, patch_size=(64,64,64), spacing=None, augment=False):
        self.root = root
        self.ids = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
        self.patch = patch_size
        self.spacing = spacing
        self.augment = augment

        transforms = [
            EnsureChannelFirst(),
            ScaleIntensityRange(a_min=0, a_max=1, b_min=0.0, b_max=1.0, clip=True),
            EnsureType(),
        ]
        if spacing:
            transforms.insert(0, Spacing(pixdim=spacing, mode=("bilinear")))
        if augment:
            transforms += [
                RandSpatialCrop(roi_size=self.patch, random_size=False, random_center=True),
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandRotate90(prob=0.2, max_k=3),
            ]
        else:
            transforms += [RandSpatialCrop(roi_size=self.patch, random_size=False, random_center=False)]
        self.tx = Compose(transforms)

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        v = np.load(os.path.join(self.root, sid, "volume.npy"))  # (D,H,W) or (H,W)
        m = np.load(os.path.join(self.root, sid, "mask.npy"))    # integers {0..C-1}
        v = np.expand_dims(v, 0) if v.ndim==3 else v  # MONAI EnsureChannelFirst will handle
        return {"image": torch.from_numpy(v).float(),
                "label": torch.from_numpy(m).long()}