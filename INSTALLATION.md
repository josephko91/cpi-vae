# Installation Guide

Choose one method based on your setup:

## Option 1: Conda/Mamba (Recommended)

Create a lightweight conda environment with core dependencies, then install PyTorch from the official PyTorch pip wheels that match your CUDA driver. This avoids conda solver issues and ensures a compatible PyTorch build for your GPU.

```bash
# Create the environment (environment.yml contains core deps only)
mamba env create -f environment.yml
mamba activate cpi-vae

# Install PyTorch via official PyTorch pip wheels for your CUDA version.
# Example: CUDA 12.2
pip install --index-url https://download.pytorch.org/whl/cu122 \
  torch torchvision torchaudio

# Or for CUDA 13.0
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch torchvision torchaudio

# For CPU-only
pip install torch torchvision torchaudio

# Verify CUDA availability
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
PY
```

---

## Option 2: Pip with CUDA Support

If you prefer a pip-only workflow, install core Python deps first and then install the PyTorch wheels that match your CUDA driver.

1. Check your driver/CUDA version:

```bash
nvidia-smi
```

2. Install core deps (inside a venv or conda env):

```bash
pip install -r requirements.txt
```

3. Install PyTorch wheels using the appropriate index URL:

```bash
# CUDA 12.2 (example)
pip install --index-url https://download.pytorch.org/whl/cu122 \
  torch torchvision torchaudio

# CUDA 13.0
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch torchvision torchaudio

# CPU-only
pip install torch torchvision torchaudio
```

Verify as above with the small Python snippet.

---

## Option 3: pip + pyproject.toml (Modern)

Install with optional CUDA dependencies:

```bash
# For CUDA 12.2
pip install ".[cu122]" --index-url https://download.pytorch.org/whl/cu122

# For CUDA 13.0
pip install ".[cu130]" --index-url https://download.pytorch.org/whl/cu130

# For CPU-only
pip install ".[cpu]"

# For development
pip install ".[dev,cu122]" --index-url https://download.pytorch.org/whl/cu122
```

---

## Troubleshooting

### CUDA not available after install?

**Problem:** `torch.cuda.is_available()` returns False  
**Solution:** Ensure PyTorch CUDA version matches your driver:

```bash
# Check your driver
nvidia-smi | grep "CUDA Version"

# Check PyTorch
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"
```

If they don't match (e.g., driver has CUDA 12.2 but PyTorch has cu130), reinstall PyTorch with the correct index URL.

### "Old driver" warning?

Update your driver or reinstall PyTorch for an older CUDA version that matches your driver.

```bash
# Example: If driver is CUDA 12.2
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
```

---

## Running Training

Once installed, run training:

```bash
# CPU (slow, for testing)
python scripts/run_train.py --config configs/data_dirs.yaml \
  --max_samples 1000 --image_size 224 --epochs 1 --batch_size 8 --device cpu

# GPU (fast)
python scripts/run_train.py --config configs/data_dirs.yaml \
  --max_samples 5000 --image_size 224 --epochs 10 --batch_size 32 --device cuda
```
