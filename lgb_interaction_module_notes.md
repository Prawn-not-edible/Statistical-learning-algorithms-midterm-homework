# LightGBM 监督式交互特征模块

这个模块的定位是：用 LightGBM 发现高频路径共现的特征对，再把它们显式物化为一个可插拔特征块。LightGBM 在这里是“交互发现器”，不是最终模型选择结论。

## 1. 模块入口

- `lgb_interaction_features.py`
  - `LGBTreePathInteractionTransformer`：sklearn 风格 transformer。
  - `analyze_tree_paths`：遍历 LightGBM root-to-leaf 路径，统计特征对共现。
  - `materialize_interaction_outputs`：统一输出交互矩阵、拼接矩阵、映射表和报告。
- `discover_lgb_interactions.py`
  - 命令行入口，用当前 `preprocessed_outputs_v2` 产物发现并落盘交互特征。

## 2. 命令行用法

首次发现交互并生成可插拔特征块：

```bash
python discover_lgb_interactions.py --top-k 20 --n-estimators 160
```

复用已经选好的交互映射，只做 transform：

```bash
python discover_lgb_interactions.py ^
  --selection-file preprocessed_outputs_v2/interaction_features/selected_lgb_interactions.csv ^
  --output-dir preprocessed_outputs_v2/interaction_features_reuse_test ^
  --no-combined
```

## 3. Python 接入方式

```python
import pandas as pd
from pathlib import Path
from lgb_interaction_features import LGBTreePathInteractionTransformer

finder = LGBTreePathInteractionTransformer(top_k=20, n_estimators=160)
X_lgb_inter = finder.fit_transform(X_train, y_train)
X_train_aug = pd.concat([X_train, X_lgb_inter], axis=1)

finder.save_selection(Path("preprocessed_outputs_v2/interaction_features"))

# 后续任意模型都可以复用同一份映射，不需要重新训练 LightGBM。
applier = LGBTreePathInteractionTransformer.from_selection_file(
    Path("preprocessed_outputs_v2/interaction_features/selected_lgb_interactions.csv")
)
X_test_lgb_inter = applier.transform(X_test)
X_test_aug = pd.concat([X_test, X_test_lgb_inter], axis=1)
```

## 4. 输出文件

默认输出目录：

`preprocessed_outputs_v2/interaction_features/`

核心文件：

- `lgb_tree_path_interactions.csv`：全部候选特征对及路径共现统计。
- `selected_lgb_interactions.csv`：最终选入的显式交互特征映射表。
- `X_train_lgb_interaction_features.csv`：只包含 `lgb_interact__` 前缀列的训练集交互块。
- `X_test_lgb_interaction_features.csv`：只包含 `lgb_interact__` 前缀列的测试集交互块。
- `X_train_with_lgb_interactions.csv`：基础特征 + 交互块的拼接训练矩阵。
- `X_test_with_lgb_interactions.csv`：基础特征 + 交互块的拼接测试矩阵。
- `lgb_interaction_report.md`：方法说明、Top 共现对、选中特征表。
- `lgb_interaction_metadata.json`：模块配置和元数据。

## 5. 特征标注规则

所有由这个模块生成的显式交互特征都使用：

`lgb_interact__001`, `lgb_interact__002`, ...

`selected_lgb_interactions.csv` 中保留：

- `feature_a`
- `feature_b`
- `operation=product`
- `formula`
- `interaction_family=lgb_tree_path`
- `source=LightGBM tree-path co-occurrence`

这样可以和原始特征、缺失指示特征、医学先验交互特征清楚区分。

## 6. 当前实验注意点

当前交互发现使用完整训练集完成，适合做解释、特征工程探索和下一轮对比实验。如果要汇报严格 OOF 性能，应在每个交叉验证训练折内部重新发现交互，再 transform 对应验证折，避免监督式特征选择带来的轻微乐观偏差。
