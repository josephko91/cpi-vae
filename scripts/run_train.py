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
    except Exception:
        pass
    
    # Fall back to plain text format (one directory per line)
    dirs = [line.strip() for line in content.split('\n') if line.strip()]
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
    
    # Resolve data directories from config file or CLI args
    if args.config:
        args.data_dirs = load_data_dirs_from_config(args.config)
        print(f"Loaded {len(args.data_dirs)} directories from {args.config}")
    
    train_mod.train(args)


if __name__ == '__main__':
    main()
