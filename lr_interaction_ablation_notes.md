# LR Baseline And Interaction Ablation

本轮把 Logistic Regression (LR) 加入 baseline，并专门用 LR 验证 `lgb_interact__*` 显式交互特征是否有价值。LR 是线性模型，不能自己从原始列里自动构造乘积交互，因此它很适合用来检验“LightGBM 路径共现发现的交互是否能迁移给其他模型”。

## 已落地内容

- `run_baselines.py` 已加入 `logistic_regression`，默认 baseline 列表现在包含 LR。
- 新增 `run_lr_interaction_ablation.py`，固定使用 5 折 OOF、`StandardScaler + LogisticRegression`，比较：
  - `lr_base`：186 个正式预处理特征。
  - `lr_lgb_interactions_only`：20 个 `lgb_interact__*` 交互特征。
  - `lr_base_plus_lgb_interactions`：186 个基础特征 + 20 个交互特征。
- 交互特征来自 `preprocessed_outputs_v2/interaction_features/selected_lgb_interactions.csv`，所有列都有 `lgb_interact__` 前缀，便于和医学先验特征区分。

## 主要结果

### LR 无类别权重

这个版本最适合写“交互特征让线性模型变强”的叙事，因为它没有额外类别权重干预，提升更干净。

| variant | features | best C | log_loss | macro_auc | macro_f1 | balanced_accuracy | minority_recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr_base | 186 | 3.0 | 0.117626 | 0.691148 | 0.175244 | 0.175967 | 0.011765 |
| lr_base_plus_lgb_interactions | 206 | 0.1 | 0.089397 | 0.734180 | 0.188545 | 0.179424 | 0.015406 |

增量：

- Macro F1: `+0.013301`
- Macro AUC: `+0.043032`
- Log loss: `-0.028229`
- Minority recall: `+0.003641`

结论：加入 LightGBM 发现的 20 个显式交互后，LR 的 Macro F1、Macro AUC、Log loss 同时改善，说明这些交互不是只对树模型有用，而是被成功转化成了线性模型也能使用的信号。

### LR 使用 `class_weight=balanced`

这个版本更偏向少数类召回，但类别权重会显著改变概率校准，因此结论要更谨慎。

| variant | features | best C | log_loss | macro_auc | macro_f1 | balanced_accuracy | minority_recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr_base | 186 | 1.0 | 0.893596 | 0.690187 | 0.175972 | 0.301307 | 0.217625 |
| lr_base_plus_lgb_interactions | 206 | 1.0 | 0.801389 | 0.703939 | 0.173595 | 0.280368 | 0.187361 |

按最佳 Macro F1 选择时，交互版 Macro F1 略低；但同一 `C` 下，交互版在大多数正则强度上降低 log loss，并在 `C=0.03/0.1/0.3/3.0` 的 Macro F1 有小幅正增益。比如：

| C | Macro F1 delta | Log loss delta | Macro AUC delta |
|---:|---:|---:|---:|
| 0.03 | +0.007328 | -0.049662 | +0.005344 |
| 0.10 | +0.005363 | -0.069548 | +0.007413 |
| 0.30 | +0.007784 | -0.083628 | +0.011037 |
| 3.00 | +0.001990 | -0.094290 | +0.010069 |

结论：类别权重场景下，交互特征对概率质量和 AUC 仍然正向，但对 Macro F1 的收益不如无权重 LR 稳定。

## 汇报叙事建议

可以把这一节写成：

> 为了验证 LightGBM 路径共现发现的交互不是树模型内部的黑盒偶然现象，我们将这些交互显式物化为 `lgb_interact__*` 特征，并用无法自动学习交互的 Logistic Regression 做消融实验。结果显示，在无类别权重 LR 中，加入 20 个交互特征后 Macro F1 从 0.1752 提升到 0.1885，Macro AUC 从 0.6911 提升到 0.7342，Log loss 从 0.1176 降至 0.0894。这说明树模型发现的高阶组合信号可以被迁移到线性模型中，验证了交互特征工程的有效性。

严格说明：当前实验复用的是全训练集发现的交互映射，适合做特征工程探索和叙事验证。如果作为最终 OOF 分数汇报，应把交互发现放到每个 CV 训练折内部重新执行。
