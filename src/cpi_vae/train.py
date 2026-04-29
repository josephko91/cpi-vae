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
    elif recon_type == "l1":
        recon_loss = F.l1_loss(recon, x, reduction="sum")
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
    # how often to save checkpoints/reconstructions (every N epochs)
    save_every = int(getattr(config, "save_every", 1))
    # reconstruction loss type and KLD weighting (beta-VAE)
    recon_type = getattr(config, "recon_type", "mse")
    beta = float(getattr(config, "beta", 1.0))
    os.makedirs(out_dir, exist_ok=True)
    # create a unique run subdirectory so checkpoints/reconstructions
    # from different runs don't overwrite each other
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{run_id}")
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(out_dir, f"run_{run_id}_{suffix}")
        suffix += 1
    os.makedirs(run_dir, exist_ok=False)

    # save the run/config metadata so the exact training settings are preserved
    try:
        # try to coerce config into a plain dict
        from collections.abc import Mapping
        if isinstance(config, Mapping):
            cfg = dict(config)
        else:
            try:
                cfg = vars(config)
            except Exception:
                cfg = {k: getattr(config, k) for k in dir(config) if not k.startswith('_')}
    except Exception:
        cfg = {}

    # write YAML if available, otherwise JSON; always write a simple text summary
    try:
        import yaml
        with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print('Saved config:', os.path.join(run_dir, 'config.yaml'))
    except Exception:
        import json
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)
        print('Saved config:', os.path.join(run_dir, 'config.json'))

    try:
        with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
            for k, v in sorted(cfg.items()):
                f.write(f"{k}: {v}\n")
        print('Saved config summary:', os.path.join(run_dir, 'config.txt'))
    except Exception:
        pass

    # warn if save_every is larger than total epochs
    if save_every > epochs:
        print(f"Warning: save_every={save_every} > epochs={epochs}. Only saving final epoch checkpoints.")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, _ in tqdm(train_loader, desc=f"train {epoch}"):
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss_recon, recon_loss, kld = vae_loss(recon, xb, mu, logvar)
            loss = recon_loss + beta * kld
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, _ in tqdm(val_loader, desc=f"val  {epoch}"):
                xb = xb.to(device)
                recon, mu, logvar = model(xb)
                _, recon_loss, kld = vae_loss(recon, xb, mu, logvar, recon_type=recon_type)
                loss = recon_loss + beta * kld
                val_loss += loss.item()

        n = len(dataset)
        print(f"Epoch {epoch} TrainLoss {train_loss/n:.6f} ValLoss {val_loss/n:.6f}")

        # save checkpoint and reconstructions according to save_every
        do_save = (epoch == epochs) or (save_every > 0 and epoch % save_every == 0)
        if save_every > epochs:
            # if user requested saving less often than total epochs, only save final
            do_save = (epoch == epochs)

        if do_save:
            # include the run config dict in the checkpoint for reproducibility
            ckpt = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "config": cfg}
            torch.save(ckpt, os.path.join(run_dir, f"vae_epoch{epoch:03d}.pt"))
            xb, _ = next(iter(val_loader))
            xb = xb.to(device)
            with torch.no_grad():
                recon, _, _ = model(xb)
            save_reconstructions(xb.cpu(), recon.cpu(), os.path.join(run_dir, f"recon_epoch{epoch:03d}.png"))
