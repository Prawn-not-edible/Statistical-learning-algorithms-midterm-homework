$ErrorActionPreference = "Stop"

# Recommended: create a fresh conda env for broadest compatibility on Windows.
conda create -n tabular-baselines python=3.11 -y
conda activate tabular-baselines

python -m pip install -U pip setuptools wheel
python -m pip install -U numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel

# Tree baselines
python -m pip install xgboost lightgbm catboost

# TabPFN
python -m pip install tabpfn

# AutoGluon CPU install (official docs recommend the PyTorch CPU index for CPU-only installs)
python -m pip install -U uv
python -m uv pip install autogluon --extra-index-url https://download.pytorch.org/whl/cpu

# Optional: register Jupyter kernel
python -m ipykernel install --user --name tabular-baselines --display-name "Python (tabular-baselines)"

# Quick sanity check
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
PY
