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

    def __init__(self, dirs: List[str], image_size: int = 224, augment: bool = False):
        paths = []
        for d in dirs:
            paths += glob.glob(os.path.join(d, "*.png"))
        self.paths = sorted(paths)
        
        # Base transforms (resizing + normalization)
        self.base_transform = T.Compose([
            T.ToTensor(),
        ])
        
        # Augmentation transforms (applied before base transforms if augment=True)
        # Based on: Govindarajan et al. modified for CPI images
        if augment:
            self.transform = T.Compose([
                # Random horizontal and vertical flips
                # (crystals can be freely rotated in space)
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # Random resized crop with constrained aspect ratio [0.9, 1.1]
                # (preserves spike thickness which is important for crystal classification)
                T.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                ),
                # Brightness and contrast jitter only (no saturation/hue)
                # (CPI images are monochromatic, saturation/hue jitter not applicable)
                T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
                self.base_transform
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.CenterCrop(image_size),
                self.base_transform,
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        p = self.paths[idx]
        # Load as single-channel (grayscale) image
        im = Image.open(p).convert("L")
        im = self.transform(im)
        return im, p
