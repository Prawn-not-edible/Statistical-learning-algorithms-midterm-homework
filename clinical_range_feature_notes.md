# Clinical Reference Range Features

本轮新增一组“临床参考范围”特征：对生命体征、糖脂代谢、肝肾功能、电解质、炎症/心衰指标等 24 个核心指标，生成 `is_normal_*` 二值指示特征。

## 模块实现

- `clinical_range_features.py`
  - `CLINICAL_RANGES`：临床正常范围字典。
  - `add_clinical_normal_indicators`：新增 `is_normal_*` 特征。
  - `build_clinical_indicator_block`：只返回临床范围特征块，方便和主特征矩阵拼接。
  - `impute_clinical_indicator_block`：给 LR 这类不能处理 NaN 的模型使用，默认把 unknown 填成 `-1`，不和 0/1 混淆。
- `run_clinical_range_ablation.py`
  - 用 5 折 OOF 对 `base`、`clinical_ranges_only`、`base_plus_clinical_ranges` 做消融。
  - LR 使用 `-1` 表示临床指示特征 unknown。
  - LightGBM 保留临床指示特征中的 NaN，让树模型自己学习缺失流向。

## 特征数量

原计划 23 个核心指标；实际落地为 24 个，因为额外加入了 `*肌酐(酶法)`，它和既有 `肌酐` 都是肾功能关键指标，且前面特征工程多次用到 `*肌酐(酶法)`。

输出文件：

- `preprocessed_outputs_v2/clinical_range_features/X_train_clinical_range_features.csv`
- `preprocessed_outputs_v2/clinical_range_features/X_test_clinical_range_features.csv`
- `preprocessed_outputs_v2/clinical_range_features/clinical_range_feature_metadata.csv`

## 消融结果

完整结果在：

- `clinical_range_ablation_results_nan_lgbm/clinical_range_ablation_metrics.csv`
- `clinical_range_ablation_results_nan_lgbm/clinical_range_ablation_report.md`

### LR 无类别权重

| variant | features | best C | log_loss | macro_auc | macro_f1 | balanced_accuracy | minority_recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 186 | 3.0 | 0.117588 | 0.691294 | 0.175244 | 0.175967 | 0.011765 |
| base_plus_clinical_ranges | 210 | 1.0 | 0.109583 | 0.692230 | 0.176815 | 0.176012 | 0.011765 |

增量：

- Macro F1: `+0.001571`
- Macro AUC: `+0.000936`
- Log loss: `-0.008005`

### LR 使用 `class_weight=balanced`

| variant | features | best C | log_loss | macro_auc | macro_f1 | balanced_accuracy | minority_recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 186 | 1.0 | 0.893196 | 0.690289 | 0.176179 | 0.301478 | 0.217625 |
| base_plus_clinical_ranges | 210 | 0.3 | 0.804654 | 0.693916 | 0.171603 | 0.284159 | 0.193378 |

增量：

- Macro F1: `-0.004576`
- Macro AUC: `+0.003627`
- Log loss: `-0.088542`

### LightGBM balanced

| variant | features | log_loss | macro_auc | macro_f1 | balanced_accuracy | minority_recall |
|---|---:|---:|---:|---:|---:|---:|
| base | 186 | 0.097137 | 0.729918 | 0.199197 | 0.187343 | 0.024930 |
| base_plus_clinical_ranges | 210 | 0.096983 | 0.732938 | 0.198830 | 0.187343 | 0.024930 |

增量：

- Macro F1: `-0.000367`
- Macro AUC: `+0.003020`
- Log loss: `-0.000154`

## 结论

这组特征有解释价值，但性能收益不够强，不建议现在默认并入正式主 pipeline。

更稳妥的叙事是：

> 我们系统化构造了 24 个临床正常范围指示特征，尝试把连续化验值转换为可解释的“是否异常”医学先验信号。消融显示，该特征块对概率质量有轻微正向作用：无权重 LR 的 log loss 从 0.1176 降到 0.1096，LightGBM 的 macro AUC 从 0.7299 提升到 0.7329。但 Macro F1 提升不稳定，说明简单正常/异常二值化会损失连续值细节，因此当前将其保留为可插拔模块，而不默认写入最终 pipeline。

这个结论本身有报告价值：不是所有医学先验都无脑加入，经过消融后只保留稳定增益更强的模块。
