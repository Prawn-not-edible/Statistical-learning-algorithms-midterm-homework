# Cardiovascular Temporal Feature Ablation

本轮构造了一组心血管时序特征，用来把“慢性累积暴露 + 当前状态 + 急性触发”的医学机制显式编码进模型。

## 已落地内容

- `temporal_features.py`
  - 构造时序特征块。
  - 保留树模型可用的 NaN。
  - 为 LR 提供 `-1` unknown 填充值，避免和 0/正常值混淆。
- `run_temporal_feature_ablation.py`
  - 5 折 OOF 消融。
  - 比较 `base`、`temporal_only`、`base_plus_temporal`。
  - 模型包括 LR 无权重、LR balanced、LightGBM balanced。

## 新增特征

共 14 个：

- `CAD_发病年龄`
- `高血压_发病年龄`
- `糖尿病_发病年龄`
- `is_早发冠心病`
- `高血压_血管损伤负荷`
- `糖尿病_糖毒性负荷`
- `冠心病_脂质暴露负荷`
- `冠心病_急性应激叠加`
- `肾脏_急慢性打击指数`
- `吸烟_残余风险指数`
- `饮酒_残余风险指数`
- `冠心病年限_cleaned`
- `高血压年限_cleaned`
- `糖尿病年限_cleaned`

输出文件：

- `preprocessed_outputs_v2/temporal_features/X_train_temporal_features.csv`
- `preprocessed_outputs_v2/temporal_features/X_test_temporal_features.csv`
- `preprocessed_outputs_v2/temporal_features/temporal_feature_metadata.csv`

## 消融结果

完整结果在：

- `temporal_feature_ablation_results/temporal_feature_ablation_metrics.csv`
- `temporal_feature_ablation_results/temporal_feature_ablation_report.md`

### LightGBM balanced

| variant | features | log_loss | macro_auc | macro_f1 | balanced_accuracy | minority_recall |
|---|---:|---:|---:|---:|---:|---:|
| base | 186 | 0.097137 | 0.729918 | 0.199197 | 0.187343 | 0.024930 |
| base_plus_temporal | 200 | 0.096766 | 0.732052 | 0.207333 | 0.192254 | 0.030812 |

增量：

- Macro F1: `+0.008136`
- Macro AUC: `+0.002134`
- Log loss: `-0.000371`
- Minority recall: `+0.005882`

这是本轮最明确的正向结果。说明显式的发病年龄、累积负荷、急慢性打击等特征对树模型是有帮助的。

### LR 无类别权重

按最佳 Macro F1 选择：

| variant | features | best C | log_loss | macro_auc | macro_f1 | minority_recall |
|---|---:|---:|---:|---:|---:|---:|
| base | 186 | 3.0 | 0.117588 | 0.691294 | 0.175244 | 0.011765 |
| base_plus_temporal | 200 | 0.3 | 0.097470 | 0.701572 | 0.173181 | 0.005882 |

这个结果不能证明时序特征显著增强无权重 LR。它改善了部分概率/AUC表现，但 Macro F1 轻微下降。

### LR balanced

按最佳 Macro F1 选择：

| variant | features | best C | log_loss | macro_auc | macro_f1 | minority_recall |
|---|---:|---:|---:|---:|---:|---:|
| base | 186 | 1.0 | 0.893196 | 0.690289 | 0.176179 | 0.217625 |
| base_plus_temporal | 200 | 1.0 | 0.857662 | 0.686929 | 0.165829 | 0.178129 |

matched C 下，balanced LR 的 log loss 基本都有改善，例如：

| C | Macro F1 delta | Log loss delta |
|---:|---:|---:|
| 0.03 | +0.000396 | -0.023825 |
| 0.10 | +0.001825 | -0.029478 |
| 0.30 | -0.005139 | -0.034346 |
| 1.00 | -0.010350 | -0.035534 |
| 3.00 | -0.006766 | -0.035760 |

结论是：balanced LR 从时序特征中获得了更好的概率质量，但分类阈值/少数类召回没有稳定改善。

## 结论

这组特征建议作为树模型方向的候选增强模块，而不是作为“LR 证明非线性医学逻辑”的主证据。

报告叙事可以这样写：

> 我们进一步构造了心血管时序特征，将发病年龄、慢性病程累积暴露、急性触发和戒断残余风险显式编码。消融显示，该模块对 LightGBM 有稳定正向收益：Macro F1 从 0.1992 提升到 0.2073，minority recall 从 0.0249 提升到 0.0308，log loss 也小幅下降。这说明时序病程信息确实补充了静态化验指标。与此同时，LR 的 Macro F1 未稳定提升，说明这些特征并不像 LightGBM 路径交互那样能直接线性化类别边界，因此我们将其定位为树模型友好的候选增强模块。

注意：`1月内手术/创伤史` 中取值 `1` 占绝大多数，`2` 极少，因此本实现把 `2/3` 视作近期急性应激阳性，而不是照搬“1=阳性”的假设。
