#!/usr/bin/env bash
set -euo pipefail

# Recommended target:
# - Ubuntu 22.04 / 24.04
# - NVIDIA GPU with >= 16 GB VRAM preferred
# - CUDA driver available (nvidia-smi should work)

ENV_NAME="${ENV_NAME:-tabular-cloud}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "[1/6] Checking GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. TabPFN will likely be impractical on CPU for this dataset."
fi

echo "[2/6] Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y

echo "[3/6] Upgrading pip toolchain"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
python -m pip install -U pip setuptools wheel uv

echo "[4/6] Installing core science stack"
python -m pip install -U numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel

echo "[5/6] Installing baseline libraries"
python -m pip install xgboost lightgbm catboost tabpfn

# Official AutoGluon GPU install for Linux via pip / uv:
# python -m uv pip install autogluon
python -m uv pip install autogluon

echo "[6/6] Sanity check"
python - <<'PY'
import importlib

mods = [
    "numpy",
    "pandas",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "tabpfn",
    "autogluon",
]

for m in mods:
    mod = importlib.import_module(m)
    print(m, getattr(mod, "__version__", "unknown"))

try:
    import torch
    print("torch", torch.__version__)
    print("torch.cuda.is_available", torch.cuda.is_available())
    print("torch.cuda.device_count", torch.cuda.device_count())
except Exception as e:
    print("torch check failed:", e)
PY

echo
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
