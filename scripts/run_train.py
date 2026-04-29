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


def load_data_dirs_from_config(config_path):
    """Load data directories from a config file (YAML or plain text).
    
    YAML format (preferred):
        data_dirs:
          - /path/to/dir1
          - /path/to/dir2
    
    Plain text format (one directory per line):
        /path/to/dir1
        /path/to/dir2
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    content = config_path.read_text().strip()
    
    # Try YAML format first
    try:
        import yaml
        data = yaml.safe_load(content)
        if isinstance(data, dict) and 'data_dirs' in data:
            dirs = data['data_dirs']
            if isinstance(dirs, list):
                return dirs
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARNING] YAML parsing failed: {e}, falling back to plain text")
    
    # Fall back to plain text format (one directory per line, skip comments/blank lines)
    dirs = []
    for line in content.split('\n'):
        line = line.strip()
        # Skip empty lines, comments, and YAML syntax lines
        if not line or line.startswith('#') or line in ('data_dirs:', 'data_dirs') or line == '-':
            continue
        # Remove leading dash from YAML list items
        if line.startswith('- '):
            line = line[2:].strip()
        # Only add if it looks like a path
        if line and not line.startswith('data_'):
            dirs.append(line)
    return dirs


def build_parser():
    p = argparse.ArgumentParser(
        description="Train a ConvVAE on CPI imagery data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using CLI arguments:
  python scripts/run_train.py --data_dirs /path/to/dir1 /path/to/dir2
  
  # Using a config file:
  python scripts/run_train.py --config configs/data_dirs.yaml
  
  # Config file can be YAML or plain text (one dir per line)
        """
    )
    
    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data_dirs", nargs="+",
                            help="Directories containing CPI image data")
    data_group.add_argument("--config", type=str,
                            help="Path to config file listing data directories (YAML or plain text)")
    
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=1,
                    help="Save checkpoints and reconstructions every N epochs (default: 1)")
    p.add_argument("--recon_type", type=str, default="mse",
                    choices=["mse", "bce", "l1"],
                    help="Reconstruction loss type: mse, bce, or l1 (default: mse)")
    p.add_argument("--beta", type=float, default=1.0,
                    help="Beta weight for KLD term (beta-VAE). Default 1.0")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--max_samples", type=int, default=None, help="Limit total dataset to this many random samples (optional)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", action="store_true", help="Enable torch.profiler and write TensorBoard traces to run_dir/profile")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    
    # Resolve data directories from config file or CLI args
    if args.config:
        args.data_dirs = load_data_dirs_from_config(args.config)
        print(f"Loaded {len(args.data_dirs)} directories from {args.config}")
    
    train_mod.train(args)


if __name__ == '__main__':
    main()
