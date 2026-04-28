"""Data utilities for CPI images."""
import os
import glob
from PIL import Image
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CPIDataset(Dataset):
    """Load PNG images from one or more directories.

    Returns (image_tensor, filepath)
    """

    def __init__(self, dirs: List[str], image_size: int = 64, augment: bool = False):
        paths = []
        for d in dirs:
            paths += glob.glob(os.path.join(d, "*.png"))
        self.paths = sorted(paths)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])
        self.augment = augment
        if augment:
            self.aug = T.RandomApply([T.ColorJitter(0.15, 0.15, 0.15)], p=0.5)
        else:
            self.aug = lambda x: x

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        im = self.transform(im)
        im = self.aug(im)
        return im, p
