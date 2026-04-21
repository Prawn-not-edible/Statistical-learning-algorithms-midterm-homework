# Baseline 模型输入要求速记

这份速记只保留后面跑基线时最需要的输入约束。当前这份数据是纯数值表格，`序号` 是 ID，`label` 是多分类目标。

## 统一方案

- 基线统一使用 `train` 的中位数填补 `train/test` 缺失值，避免数据泄露。
- 不做标准化。
- 保留原始列名，方便树模型和 AutoGluon 直接读取。
- 统一产出：
  - `X_train_processed.csv`
  - `X_test_processed.csv`
  - `y_train.csv`
  - `train_sample_weights.csv`
  - `class_weights.json`

## XGBoost

- 可直接吃 `pandas.DataFrame` / `numpy.ndarray`。
- `fit(..., sample_weight=...)` 支持逐样本权重。
- 官方文档说明 `missing=None` 时默认按 `np.nan` 处理缺失值。
- 现在这份数据全部是数值列，因此不需要额外编码。

## LightGBM

- 可直接吃 `pandas.DataFrame` / `numpy.ndarray`。
- `fit(..., sample_weight=...)` 支持逐样本权重。
- 也支持构造器里的 `class_weight=`。
- 官方文档说明默认支持缺失值，`NaN` 会被当作缺失。
- 当前数据没有显式类别列声明需求，先按全数值输入即可。

## CatBoost

- 可直接吃 `pandas.DataFrame` / `numpy.ndarray`，也可以用 `Pool`。
- 如果有类别特征，可以通过 `cat_features` 传列名或列索引；但当前这份数据先按全数值处理。
- 支持 `sample_weight`，也支持 `class_weights`。
- 官方文档说明数值缺失值可以直接保留给模型处理。

## TabPFN

- 基本接口是 `clf.fit(X_train, y_train)`。
- 官方 README 建议不要做标准化和 one-hot。
- 官方 README 说明当前版本可以处理 missing values，但为了和其他基线统一、减少版本差异，仍建议先喂我们统一填补后的数值矩阵。
- 官方 README 建议数据规模最好在 `<100000` 行、`<2000` 特征；当前数据 `18789 x 137`，在建议范围内。
- GPU 更合适，CPU 可以跑但会慢很多。

## AutoGluon

- 训练接口是 `TabularPredictor(label='label').fit(train_data)`，输入通常是带标签列的 `pandas.DataFrame`。
- 官方 FAQ 说明它不会做“统一的通用缺失值插补”，而是把缺失交给底层模型各自处理。
- `sample_weight` 是在 `TabularPredictor(...)` 初始化时指定“哪一列是权重列”，所以如果后面要给 AutoGluon 加权，需要把权重列拼回训练表。
- 为了和其他 baseline 对齐，我们可以优先使用已经填补好的训练表，再按需额外拼接 `sample_weight` 列。

## 这份数据的直接结论

- 对 XGBoost / LightGBM / CatBoost：当前预处理已经够跑第一版 baseline。
- 对 TabPFN：当前预处理也能直接用，不需要再缩放。
- 对 AutoGluon：直接把 `train_processed.csv` 读成 DataFrame 就能开跑；如果要加权，再 merge `train_sample_weights.csv`。

## 官方文档

- XGBoost: https://xgboost.readthedocs.io/en/release_2.1.0/python/python_api.html
- LightGBM: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
- LightGBM Missing Values: https://lightgbm.readthedocs.io/en/v4.5.0/Advanced-Topics.html
- CatBoost fit: https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit
- CatBoost missing values: https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing.html
- TabPFN README: https://github.com/PriorLabs/TabPFN
- AutoGluon fit: https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html
- AutoGluon FAQ: https://auto.gluon.ai/stable/tutorials/tabular/tabular-faq.html
