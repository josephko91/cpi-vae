"""Validation utilities for trained VAE models."""
import os
import torch
from .data import CPIDataset
from .model import ConvVAE
from .utils import save_reconstructions
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt = torch.load(checkpoint_path, map_location=device)
    z_dim = ckpt.get("z_dim") or ckpt.get("model", {}).get("z_dim", 128)
    model = ConvVAE(in_channels=1, z_dim=z_dim).to(device)
    model.load_state_dict(ckpt.get("model_state") or ckpt.get("model") or ckpt.get("model_state_dict", {}))
    model.eval()
    return model


def evaluate(checkpoint_path: str, data_dirs, out_dir: str, batch_size: int = 64):
    model = load_model(checkpoint_path)
    dataset = CPIDataset(data_dirs, image_size=224, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (xb, paths) in enumerate(loader):
            xb = xb.to(device)
            recon, _, _ = model(xb)
            save_reconstructions(xb.cpu(), recon.cpu(), os.path.join(out_dir, f"recon_batch{i:03d}.png"))
            if i >= 4:
                break
