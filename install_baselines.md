# 基线环境安装

你当前本机环境是：

- Python `3.13.5`
- `pip 25.1`
- `conda 25.5.1`

虽然当前 Python 3.13 对其中一些库已经可用，但为了减少 Windows 下的轮子兼容问题，我更推荐单独建一个 `Python 3.11` 的 conda 环境来跑这批 baseline。

## 推荐安装方案（Windows + CPU）

在 Anaconda Prompt 或 PowerShell 里运行：

```powershell
conda create -n tabular-baselines python=3.11 -y
conda activate tabular-baselines

python -m pip install -U pip setuptools wheel
python -m pip install -U numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel
python -m pip install xgboost lightgbm catboost
python -m pip install tabpfn

python -m pip install -U uv
python -m uv pip install autogluon --extra-index-url https://download.pytorch.org/whl/cpu
```

我也把这份命令保存成了脚本：[install_baselines.ps1](D:/文件下载/统计学习算法导论/期中大作业/install_baselines.ps1)

## 如果你就想直接装在当前环境

不新建环境也可以，直接：

```powershell
python -m pip install -U pip setuptools wheel
python -m pip install xgboost lightgbm catboost tabpfn
python -m pip install -U uv
python -m uv pip install autogluon --extra-index-url https://download.pytorch.org/whl/cpu
```

但这条路的风险更高一点，因为你现在的 base 环境是 Python 3.13，而且已经带着 Anaconda 里的一堆依赖了，后面如果版本打架，排查会烦。

## 装完怎么验证

```powershell
python - <<'PY'
import importlib
mods = ["xgboost", "lightgbm", "catboost", "tabpfn", "autogluon"]
for m in mods:
    mod = importlib.import_module(m)
    print(m, getattr(mod, "__version__", "unknown"))
PY
```

## 跑 baseline

我已经把训练脚本也搭好了：[run_baselines.py](D:/文件下载/统计学习算法导论/期中大作业/run_baselines.py)

最小运行示例：

```powershell
python .\run_baselines.py --models xgboost lightgbm catboost --weight-mode raw
```

如果你想加上 TabPFN 和 AutoGluon：

```powershell
python .\run_baselines.py --models xgboost lightgbm catboost tabpfn autogluon --weight-mode raw
```

如果类别权重太激进，可以裁剪：

```powershell
python .\run_baselines.py --models xgboost lightgbm catboost --weight-mode clipped --class-weight-clip 100
```

结果会保存到：

- [baseline_results/baseline_metrics.csv](D:/文件下载/统计学习算法导论/期中大作业/baseline_results/baseline_metrics.csv)
- [baseline_results/reports](D:/文件下载/统计学习算法导论/期中大作业/baseline_results/reports)

## 和你这份数据最相关的提醒

- TabPFN 官方 README 说更推荐 GPU；在 CPU 上通常只适合很小的数据集。你这份训练集有 `18789` 行，所以 TabPFN 这项在纯 CPU 机器上可能会非常慢，甚至不太现实。
- AutoGluon 官方文档在 Windows 上明确推荐优先使用 Anaconda 环境。

## 官方安装文档

- XGBoost: https://xgboost.readthedocs.io/en/stable/install.html
- LightGBM: https://lightgbm.readthedocs.io/en/latest/
- CatBoost: https://catboost.ai/docs/en/installation/python-installation-method-pip-install
- TabPFN: https://github.com/PriorLabs/TabPFN
- AutoGluon: https://auto.gluon.ai/dev/install.html
