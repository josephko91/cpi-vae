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
├── requirements.txt       # Dependencies
└── pyproject.toml        # Package metadata
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training:

```bash
python scripts/run_train.py \
  --data_dirs /home/vanessa/hulk/cocpit/cpi_data/campaigns/MC3E/single_imgs_v1.4.0 \
             /home/vanessa/hulk/cocpit/cpi_data/campaigns/ARM/single_imgs_v1.4.0 \
             /home/vanessa/hulk/cocpit/cpi_data/campaigns/MPACE/single_imgs_v1.4.0 \
  --image_size 64 --z_dim 128 --batch_size 256 --epochs 100 \
  --out_dir ./checkpoints
```

3. Customize hyperparameters as needed (see `scripts/run_train.py --help`)
# cpi-vae
Unsupervised clustering of ice crystal images
