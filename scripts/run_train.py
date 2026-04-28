#!/usr/bin/env python3
"""Small CLI wrapper that calls cpi_vae.train.train with argparse args."""
import sys
import os
from pathlib import Path

# Add src to path so we can import cpi_vae
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

import argparse
from cpi_vae import train as train_mod


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    train_mod.train(args)


if __name__ == '__main__':
    main()
