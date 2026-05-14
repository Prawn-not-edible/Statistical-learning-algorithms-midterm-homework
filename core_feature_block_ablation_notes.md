# Core Feature Block Ablation Notes

这次把你关心的三个前置模块拆开做了统一消融：

1. 三分类缺失处理：`simple_impute` vs `missing_abc`
2. 长尾三版本：`missing_abc` vs `missing_abc_plus_long_tail`
3. 医学先验特征：`missing_abc` vs `missing_abc_plus_medical_prior`

实验脚本：

- `run_core_feature_block_ablation.py`

新增可插拔模块：

- `long_tail_features.py`
- `medical_prior_features.py`

输出目录：

- `core_feature_block_ablation_results/`
- `preprocessed_outputs_v2/core_feature_blocks/`

## 模块状态

### 三分类缺失处理

这部分之前已经在主流程 `baseline_preprocess.py` 里默认启用，但没有单独做过 LR + LightGBM 的完整模块消融。这次补上了。

比较方式：

- baseline：`simple_impute`，只做普通 median/mode 填充，无缺失指示。
- module：`missing_abc`，按 A/B/C 缺失医学含义处理，并为 A/C 类缺失加 indicator。

生成结果：

- 原始简单填充特征数：128
- 三分类缺失处理后特征数：175
- 新增缺失指示特征：47

### 长尾三版本

这部分之前做过 LightGBM 方向的探索，结论是 expanded log / clinical grade 不应默认并入主 pipeline。但没有和 LR 放在同一套 OOF 框架里单独比较。这次补上了。

当前实现采用偏度驱动：

- 保留原始值：来自 `missing_abc` 矩阵。
- 增加 winsor q99 版本：`longtail__winsor_q99__*`
- 增加 log1p 版本：`longtail__log1p__*`

生成结果：

- 偏度 `skew > 2` 的源特征：24 个
- 新增长尾特征：48 个
- 总特征数：175 -> 223

### 医学先验特征

这部分之前在主流程里有，但没有打标签，也不是独立可插拔模块。现在已经单独抽成 `medical_prior_features.py`。

新模块特征统一使用：

`prior__*`

生成结果：

- 医学先验特征：11 个
- 总特征数：175 -> 186

这满足后续换模型时的需求：医学先验特征可以单独拼接、单独消融，而不是只能跟主 preprocessing 绑定。

## 消融结果摘要

### LightGBM balanced

| comparison | log_loss delta | macro_auc delta | macro_f1 delta | minority_recall delta |
|---|---:|---:|---:|---:|
| missing_abc - simple_impute | +0.000223 | -0.005549 | +0.008655 | +0.005882 |
| long_tail block | +0.000602 | +0.003600 | +0.010163 | +0.009524 |
| medical_prior block | +0.000527 | -0.010215 | +0.001308 | +0.000000 |

解释：

- 三分类缺失处理对 LightGBM 的 Macro F1 和 minority recall 有正向贡献，但概率质量略差。
- 长尾三版本对 LightGBM 的 Macro F1 提升最明显，是这三块里对树模型最强的模块。
- 医学先验特征对 LightGBM 只有很小 Macro F1 提升，AUC/log loss 反而变差，因此不应无脑默认加入所有模型。

### LR 无类别权重

| comparison | log_loss delta | macro_auc delta | macro_f1 delta | minority_recall delta |
|---|---:|---:|---:|---:|
| missing_abc - simple_impute | +0.001314 | -0.014997 | -0.011549 | -0.009524 |
| long_tail block | +0.005066 | -0.027791 | +0.004022 | +0.003641 |
| medical_prior block | -0.007920 | -0.002637 | +0.007519 | +0.003641 |

解释：

- 无权重 LR 不喜欢三分类缺失处理，可能是大量 missing indicator 增加了稀疏噪声。
- 长尾特征对 Macro F1 有小幅帮助，但会伤害 log loss / AUC。
- 医学先验特征对无权重 LR 是正向的：Macro F1 和 log loss 都改善，这是该模块最适合写进报告的证据。

### LR balanced

| comparison | log_loss delta | macro_auc delta | macro_f1 delta | minority_recall delta |
|---|---:|---:|---:|---:|
| missing_abc - simple_impute | -0.045240 | -0.022157 | +0.015688 | -0.005176 |
| long_tail block | -0.126152 | -0.022169 | +0.003966 | -0.033782 |
| medical_prior block | -0.047215 | +0.000744 | -0.001415 | +0.000000 |

解释：

- balanced LR 下，三分类缺失处理和长尾特征都能改善 Macro F1 或 log loss，但少数类召回不稳定。
- 医学先验特征主要改善概率质量，Macro F1 没有稳定收益。

## 最终建议

当前不建议把三个模块都无条件写死进主流程。

建议定位如下：

- `missing_abc`：主流程可以继续保留，因为它是缺失机制建模的基础，且对 LightGBM 和 balanced LR 的 Macro F1 有帮助。
- `long_tail`：保留为可插拔增强模块；如果最终模型偏向 LightGBM，可以重点考虑。
- `medical_prior`：必须保持可插拔；它对 LR 无权重有较好收益，但对 LightGBM 不稳定。现在已用 `prior__` 前缀完成标签化。

报告叙事可以写成：

> 我们没有把所有医学特征工程无差别并入最终模型，而是将其拆成可插拔模块并分别做 OOF 消融。结果显示，三分类缺失处理主要改善树模型和类别加权 LR 的 Macro F1；偏度驱动的长尾三版本对 LightGBM 的 Macro F1 提升最明显；医学先验特征对无权重 LR 有较好增益，但对 LightGBM 不稳定。因此最终模型选择阶段应按模型类型决定是否启用对应模块。
