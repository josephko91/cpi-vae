"""Utility helpers for training and evaluation."""
import os
import torch
import torchvision.utils as vutils


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_reconstructions(real: torch.Tensor, recon: torch.Tensor, out_path: str, n: int = 8):
    """Save a vertical stack image: top = real, bottom = recon."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    real_grid = vutils.make_grid(real[:n], nrow=n, normalize=True, scale_each=True)
    recon_grid = vutils.make_grid(recon[:n], nrow=n, normalize=True, scale_each=True)
    combined = torch.cat([real_grid, recon_grid], dim=1)
    vutils.save_image(combined, out_path)
