# 云上运行指南

适合你的方案：

- 系统：`Ubuntu 22.04` 或 `Ubuntu 24.04`
- Python：`3.11`
- GPU：优先 NVIDIA，`>= 16GB VRAM` 更稳
- 运行顺序：先树模型，再 `TabPFN`，最后 `AutoGluon`

## 为什么建议上 GPU

- `TabPFN` 官方 README 明确写了：`GPU Recommended`
- README 还写了：CPU 只适合大约 `<= 1000` 样本的小数据
- 你的训练集是 `18789 x 137`，所以 `TabPFN` 在 CPU 上大概率不现实

## 先上传项目

把整个项目目录传到云机器，比如：

```bash
scp -r ./期中大作业 user@your-server:/home/user/
```

或者直接在云端 `git clone` 你的仓库。

## 安装环境

进入项目目录后运行：

```bash
chmod +x cloud_setup_gpu.sh
./cloud_setup_gpu.sh
```

这会：

- 创建 `conda` 环境 `tabular-cloud`
- 安装 `xgboost / lightgbm / catboost / tabpfn / autogluon`
- 做一次版本和 GPU 检查

## 开始跑

先跑树模型：

```bash
conda activate tabular-cloud
python run_baselines.py --models xgboost lightgbm catboost --weight-mode raw
```

再跑 `TabPFN`：

```bash
conda activate tabular-cloud
python run_baselines.py --models tabpfn --weight-mode none --output-dir baseline_results_tabpfn
```

再跑 `AutoGluon`：

```bash
conda activate tabular-cloud
python run_baselines.py --models autogluon --weight-mode raw --output-dir baseline_results_autogluon
```

如果你想一口气跑完，可以直接：

```bash
chmod +x run_cloud_baselines.sh
./run_cloud_baselines.sh
```

## 建议的云主机配置

最低可用：

- 4 vCPU
- 16 GB RAM
- 1 块 NVIDIA T4 / L4 / 3090 / A10 / A100 这类 GPU 之一

更稳一点：

- 8 vCPU
- 32 GB RAM
- 16 GB 以上显存

## 结果文件

树模型结果：

- `baseline_results/baseline_metrics.csv`
- `baseline_results/reports/`

TabPFN 结果：

- `baseline_results_tabpfn/baseline_metrics.csv`

AutoGluon 结果：

- `baseline_results_autogluon/baseline_metrics.csv`

## 官方文档

- XGBoost install: https://xgboost.readthedocs.io/en/stable/install.html
- LightGBM install: https://lightgbm.org/
- CatBoost pip install: https://catboost.ai/docs/en/installation/python-installation-method-pip-install.html
- TabPFN README: https://github.com/PriorLabs/TabPFN
- AutoGluon install: https://auto.gluon.ai/dev/install.html
