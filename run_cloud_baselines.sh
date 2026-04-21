#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-tabular-cloud}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
WEIGHT_MODE="${WEIGHT_MODE:-raw}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${PROJECT_DIR}"

echo "[1/3] Running tree baselines"
python run_baselines.py --models xgboost lightgbm catboost --weight-mode "${WEIGHT_MODE}"

echo "[2/3] Running TabPFN"
python run_baselines.py --models tabpfn --weight-mode none --output-dir baseline_results_tabpfn

echo "[3/3] Running AutoGluon"
python run_baselines.py --models autogluon --weight-mode raw --output-dir baseline_results_autogluon

echo "All done."
