# Installation Guide

> **TL;DR:** Install core deps, add PyTorch for your GPU, run training script directly from the repo.

## Quick Start (Recommended)

### 1. Create environment

**With Mamba (fastest):**
```bash
mamba env create -f environment.yml
mamba activate cpi-vae
```

**With Conda:**
```bash
conda env create -f environment.yml
conda activate cpi-vae
```

**With venv (Python 3.8+):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install PyTorch

**Check your GPU setup:**
```bash
nvidia-smi  # Shows GPU and CUDA version
```

**Install PyTorch matching your CUDA version:**

```bash
# CUDA 12.2
pip install --index-url https://download.pytorch.org/whl/cu122 \
  torch torchvision torchaudio

# CUDA 12.4
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio

# CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio

# CPU-only (no GPU)
pip install torch torchvision torchaudio
```

**Verify installation:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 3. Run training

```bash
# Quick test (CPU)
python scripts/run_train.py --config configs/data_dirs.yaml \
  --max_samples 100 --image_size 224 --epochs 1 --batch_size 8 --device cpu

# Full training (GPU)
python scripts/run_train.py --config configs/data_dirs.yaml \
  --max_samples 5000 --image_size 224 --epochs 10 --batch_size 32 --device cuda
```

---

## Alternative: Install from GitHub (for development)

---

If you want to contribute to the project, you can install in editable mode:

```bash
# This will allow you to edit the code and see changes immediately
pip install -e .
```

---

## Troubleshooting

### Build failed: "email must be idn-email"

**Problem:**
```
error: subprocess-exited-with-error
configuration error: `project.authors[0].email` must be idn-email
```

**Solution:** This was a bug in older versions. Update to the latest code:
```bash
git pull
```

The `pyproject.toml` has been fixed to remove the invalid empty email field. If you still have the old code, simply delete it and clone fresh:
```bash
cd /path/to/parent && rm -rf cpi-vae
git clone <repo-url>
```

### CUDA not available after PyTorch install?

**Problem:** `torch.cuda.is_available()` returns `False`

**Solution:** Ensure PyTorch CUDA version matches your GPU driver version.

1. Check your driver:
```bash
nvidia-smi | grep "CUDA Version"
```

2. Check PyTorch:
```bash
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"
```

3. If versions don't match, reinstall PyTorch with the correct index URL:
```bash
pip uninstall torch torchvision torchaudio -y
# For CUDA 12.2 (most common)
pip install --index-url https://download.pytorch.org/whl/cu122 \
  torch torchvision torchaudio
# Or use cu121, cu124, etc. to match your driver version
```

### Import errors when running scripts?

**Problem:** `ModuleNotFoundError: No module named 'cpi_vae'`

**Solution:** Run scripts from the repo root:
```bash
cd /path/to/cpi-vae
python scripts/run_train.py --config configs/data_dirs.yaml --device cpu
```

Or add repo to PYTHONPATH:
```bash
export PYTHONPATH="/path/to/cpi-vae:$PYTHONPATH"
python scripts/run_train.py --config configs/data_dirs.yaml --device cpu
```

### Dependency conflicts?

**Problem:** Conda solver is stuck or returns errors

**Solution:** Use mamba instead (faster solver):
```bash
pip install mamba  # or: conda install -c conda-forge mamba
mamba env create -f environment.yml
mamba activate cpi-vae
```

### Still having issues?

1. **Verify environment:** 
   ```bash
   pip list | grep -E "torch|numpy|Pillow"
   ```

2. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Clean reinstall:**
   ```bash
   pip uninstall cpi-vae torch torchvision torchaudio numpy Pillow -y
   pip install -r requirements.txt
   # For CUDA 12.2
   pip install --index-url https://download.pytorch.org/whl/cu122 torch torchvision torchaudio
   # Or cu121, cu124, etc. to match your driver
   ```
