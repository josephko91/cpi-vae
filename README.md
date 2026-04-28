# CPI ConvVAE

PyTorch training framework for a convolutional VAE on CPI cloud particle imagery.

## Project Structure

```
cpi-vae/
├── src/cpi_vae/           # Main package (src layout)
│   ├── __init__.py
│   ├── data.py            # CPIDataset: loads PNG images
│   ├── model.py           # ConvEncoder, ConvDecoder, ConvVAE
│   ├── train.py           # Training loop and loss functions
│   ├── validate.py        # Validation utilities and model loading
│   └── utils.py           # Helpers (seeding, reconstruction saving)
├── scripts/
│   └── run_train.py       # CLI entry point
├── configs/
│   ├── data_dirs.yaml     # YAML config: list of data directories
│   └── data_dirs.txt      # Plain text config: one directory per line
├── requirements.txt       # Dependencies
└── pyproject.toml        # Package metadata
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training with CLI arguments:

```bash
python scripts/run_train.py \
  --data_dirs /home/vanessa/hulk/cocpit/cpi_data/campaigns/MC3E/single_imgs_v1.4.0 \
             /home/vanessa/hulk/cocpit/cpi_data/campaigns/ARM/single_imgs_v1.4.0 \
             /home/vanessa/hulk/cocpit/cpi_data/campaigns/MPACE/single_imgs_v1.4.0 \
  --image_size 64 --z_dim 128 --batch_size 256 --epochs 100 \
  --out_dir ./checkpoints
```

3. Or run training with a config file (recommended for many directories):

```bash
# Edit configs/data_dirs.yaml or configs/data_dirs.txt to specify directories
python scripts/run_train.py \
  --config configs/data_dirs.yaml \
  --image_size 64 --z_dim 128 --batch_size 256 --epochs 100 \
  --out_dir ./checkpoints
```

### Config File Format

Two formats are supported:

**YAML format** (`configs/data_dirs.yaml`):
```yaml
data_dirs:
  - /path/to/campaign1/images
  - /path/to/campaign2/images
  - /path/to/campaign3/images
```

**Plain text format** (`configs/data_dirs.txt`, one directory per line):
```
/path/to/campaign1/images
/path/to/campaign2/images
/path/to/campaign3/images
```

4. Customize hyperparameters as needed (see `python scripts/run_train.py --help`)
# cpi-vae
Unsupervised clustering of ice crystal images
