"""Training wrapper that uses cpi_vae.data and cpi_vae.model."""
from typing import Sequence
import os

import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from .data import CPIDataset
from .model import ConvVAE
from .utils import save_reconstructions, set_seed


def vae_loss(recon, x, mu, logvar, recon_type="mse"):
    import torch.nn.functional as F
    if recon_type == "mse":
        recon_loss = F.mse_loss(recon, x, reduction="sum")
    else:
        recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld


def train(config):
    """Train the model. `config` may be an argparse.Namespace or mapping."""
    set_seed(int(getattr(config, "seed", 42)))
    dirs = list(getattr(config, "data_dirs", []))
    max_samples = getattr(config, "max_samples", None)
    if max_samples is not None:
        max_samples = int(max_samples)
    print(f"[DEBUG train] Initializing CPIDataset with {len(dirs)} directories, max_samples={max_samples}")
    print(f"[DEBUG train] Dirs: {dirs}")
    dataset = CPIDataset(dirs, image_size=getattr(config, "image_size", 224), augment=True, max_samples=max_samples)
    if len(dataset) == 0:
        raise RuntimeError("No images found in data directories")
    val_frac = float(getattr(config, "val_frac", 0.05))
    n_val = max(int(len(dataset) * val_frac), 1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=getattr(config, "batch_size", 128), shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=getattr(config, "batch_size", 128), shuffle=False, num_workers=4)

    device = torch.device(getattr(config, "device", "cuda") if torch.cuda.is_available() else "cpu")
    model = ConvVAE(in_channels=1, z_dim=getattr(config, "z_dim", 128), image_size=getattr(config, "image_size", 224)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=getattr(config, "lr", 1e-3))

    best_val = float("inf")
    epochs = int(getattr(config, "epochs", 10))
    out_dir = getattr(config, "out_dir", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, _ in tqdm(train_loader, desc=f"train {epoch}"):
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, _, _ = vae_loss(recon, xb, mu, logvar)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, _ in tqdm(val_loader, desc=f"val  {epoch}"):
                xb = xb.to(device)
                recon, mu, logvar = model(xb)
                loss, _, _ = vae_loss(recon, xb, mu, logvar)
                val_loss += loss.item()

        n = len(dataset)
        print(f"Epoch {epoch} TrainLoss {train_loss/n:.6f} ValLoss {val_loss/n:.6f}")

        # save checkpoint and reconstructions
        ckpt = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()}
        torch.save(ckpt, os.path.join(out_dir, f"vae_epoch{epoch:03d}.pt"))
        xb, _ = next(iter(val_loader))
        xb = xb.to(device)
        with torch.no_grad():
            recon, _, _ = model(xb)
        save_reconstructions(xb.cpu(), recon.cpu(), os.path.join(out_dir, f"recon_epoch{epoch:03d}.png"))
