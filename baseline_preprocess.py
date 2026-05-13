import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "preprocessed_outputs_v2"
TRAIN_FILE = "training_dataset.csv"
TEST_FILE = "test_dataset.csv"
ENCODING = "gbk"
ID_COL = "序号"
TARGET_COL = "label"

TYPE_A_MISSING_COL_CANDIDATES = [
    # 缺失=未做该检查/不适用，有诊断或行为含义。
    "TnI峰值",
    "CK-MB峰值",
    "MYO峰值",
    "TIMI危险评分(STEMI)",
    "TIMI危险评分（STEMI）",
    "戒烟时间",
    "戒酒年限",
]

MISSING_RATE_TYPE_B_THRESHOLD = 0.30
MISSING_DIFF_TYPE_B_THRESHOLD = 0.10
BINARY_SIGNIFICANCE_ALPHA = 0.05
WINSOR_OUTLIER_PCT_THRESHOLD = 0.05
WINSOR_UPPER_QUANTILE = 0.99
CONTINUOUS_N_UNIQUE_MIN = 10

ENDPOINT_LIKE_FEATURE_CANDIDATES = [
    # 语义接近结局变量；消融显示删除后指标几乎不变，默认剔除以降低泄露质疑风险。
    "心源性死亡",
    "卒中",
    "靶血管重建",
    "MACE事件",
    "再发心梗",
    "脑梗死",
    "脑出血",
    "急性支架内血栓",
]

DEFAULT_DROP_FEATURE_CANDIDATES = [
    *ENDPOINT_LIKE_FEATURE_CANDIDATES,
    # 与 生化C11 完全重复：corr=1 且逐行相同。
    "肝功全套",
]

CLINICAL_THRESHOLDS = {
    "心肌肌钙蛋白I": {"normal": 0.04, "high": 1.0, "critical": 10.0},
    "入院BNP": {"normal": 100.0, "high": 400.0, "critical": 1000.0},
    "超敏感C-反应蛋白": {"normal": 3.0, "high": 10.0, "critical": 50.0},
    "*肌酐(酶法)": {"normal": 115.0, "high": 200.0, "critical": 400.0},
    "EGFR": {"normal": 90.0, "low": 60.0, "critical": 30.0},
}
CLINICAL_REVERSE_COLS = {"EGFR"}

CARDIAC_PANEL_COLS = ["TnI峰值", "CK-MB峰值", "MYO峰值", "CK同工酶(质量)", "心肌肌钙蛋白I"]
RENAL_PANEL_COLS = ["*肌酐(酶法)", "肌酐", "EGFR", "*尿素", "*尿酸"]
BIOCHEM_PANEL_COLS = [
    "*谷草转氨酶",
    "*谷丙转氨酶",
    "谷草/谷丙",
    "*碱性磷酸酶",
    "*谷氨酰转酞酶",
    "*总蛋白",
    "*白蛋白（溴甲酚绿法）",
    "球蛋白",
    "白球比值",
    "*总胆红素",
    "直接胆红素",
    "间接胆红素",
    "胆碱脂酶",
    "*尿素",
    "*肌酐(酶法)",
    "*尿酸",
    "*钙",
    "*磷",
    "*总胆固醇",
    "*甘油三酯",
    "高密度脂蛋白胆固醇",
    "低密度脂蛋白胆固醇",
    "*葡萄糖",
    "*钠",
    "*钾",
    "*氯",
    "二氧化碳",
    "阴离子间隙",
]


@dataclass
class PreprocessBundle:
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_processed: pd.DataFrame
    test_processed: pd.DataFrame
    base_feature_cols: list[str]
    feature_cols: list[str]
    dropped_default_cols: list[str]
    missing_cols: list[str]
    added_missing_indicator_cols: list[str]
    added_log_cols: list[str]
    added_clinical_grade_cols: list[str]
    added_prior_cols: list[str]
    median_values: pd.Series
    fill_values: pd.Series
    imputation_plan: pd.DataFrame
    binary_chi2_results: pd.DataFrame
    significant_binary_cols: list[str]
    winsorization_caps: pd.DataFrame
    winsorized_cols: list[str]
    class_weight_dict: dict[int, float]
    class_weight_dict_clipped: dict[int, float] | None
    sample_weights: np.ndarray
    sample_weights_clipped: np.ndarray | None

    @property
    def X_train(self) -> pd.DataFrame:
        return self.train_processed[self.feature_cols]

    @property
    def y_train(self) -> pd.Series:
        return self.train_processed[TARGET_COL]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.test_processed[self.feature_cols]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding=ENCODING)


def compute_balanced_class_weights(y: pd.Series) -> dict[int, float]:
    counts = y.value_counts().sort_index()
    n_samples = len(y)
    n_classes = counts.shape[0]
    weights = n_samples / (n_classes * counts.astype(float))
    return {int(label): float(weight) for label, weight in weights.items()}


def clip_weights(weight_dict: dict[int, float], max_weight: float | None) -> dict[int, float] | None:
    if max_weight is None:
        return None
    return {label: float(min(weight, max_weight)) for label, weight in weight_dict.items()}


def resolve_existing_columns(columns: list[str], candidates: list[str]) -> list[str]:
    existing = []
    seen = set()
    for col in candidates:
        if col in columns and col not in seen:
            existing.append(col)
            seen.add(col)
    return existing


def add_missing_indicators(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    added_cols = []
    for col in cols:
        new_col = f"{col}_missing"
        df[new_col] = df[col].isna().astype(int)
        added_cols.append(new_col)
    return df, added_cols


def add_log_transforms(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    added_cols = []
    for col in cols:
        new_col = f"{col}_log"
        df[new_col] = np.log1p(df[col].clip(lower=0).fillna(0))
        added_cols.append(new_col)
    return df, added_cols


def safe_median(series: pd.Series, fallback: float = 0.0) -> float:
    value = series.median()
    return float(value) if not pd.isna(value) else fallback


def safe_mode(series: pd.Series, fallback: float = 0.0) -> float:
    mode = series.dropna().mode()
    if mode.empty or pd.isna(mode.iloc[0]):
        return fallback
    return float(mode.iloc[0])


def is_continuous_like(series: pd.Series) -> bool:
    return series.dropna().nunique() > CONTINUOUS_N_UNIQUE_MIN


def compute_iqr_outlier_pct(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return 0.0
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr <= 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(((values < lower) | (values > upper)).mean())


def resolve_default_drop_cols(columns: list[str]) -> list[str]:
    return resolve_existing_columns(columns, DEFAULT_DROP_FEATURE_CANDIDATES)


def build_winsorization_caps(train: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if not is_continuous_like(train[col]):
            continue
        outlier_pct = compute_iqr_outlier_pct(train[col])
        if outlier_pct <= WINSOR_OUTLIER_PCT_THRESHOLD:
            continue
        upper_cap = train[col].dropna().quantile(WINSOR_UPPER_QUANTILE)
        if pd.isna(upper_cap):
            continue
        rows.append(
            {
                "feature": col,
                "upper_quantile": WINSOR_UPPER_QUANTILE,
                "upper_cap": float(upper_cap),
                "iqr_outlier_pct": outlier_pct,
            }
        )
    return pd.DataFrame(rows).sort_values("iqr_outlier_pct", ascending=False).reset_index(drop=True)


def apply_winsorization_caps(df: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for row in caps.itertuples(index=False):
        if row.feature in df.columns:
            df[row.feature] = df[row.feature].clip(upper=row.upper_cap)
    return df


def compute_missing_by_label(train: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    missing_by_label = train.groupby(TARGET_COL)[feature_cols].agg(lambda x: x.isna().mean()).T
    missing_diff = missing_by_label.max(axis=1) - missing_by_label.min(axis=1)
    return missing_by_label, missing_diff


def build_binary_chi2_results(train: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        non_missing_unique = train[col].dropna().nunique()
        if non_missing_unique > 2:
            continue

        contingency = pd.crosstab(train[TARGET_COL], train[col].fillna(-999999))
        if contingency.shape[1] < 2:
            chi2, p_value = 0.0, 1.0
        else:
            chi2, p_value, _, _ = chi2_contingency(contingency)

        rows.append(
            {
                "feature": col,
                "n_unique_non_missing": int(non_missing_unique),
                "missing_rate": float(train[col].isna().mean()),
                "chi2": float(chi2),
                "p_value": float(p_value),
                "significant_0_05": bool(p_value < BINARY_SIGNIFICANCE_ALPHA),
            }
        )

    return pd.DataFrame(rows).sort_values("chi2", ascending=False).reset_index(drop=True)


def build_imputation_plan(
    train: pd.DataFrame,
    feature_cols: list[str],
    missing_diff: pd.Series,
) -> pd.DataFrame:
    type_a_cols = set(resolve_existing_columns(feature_cols, TYPE_A_MISSING_COL_CANDIDATES))
    missing_rate = train[feature_cols].isna().mean()
    median_values = train[feature_cols].median()

    rows = []
    for col in feature_cols:
        is_binary = train[col].dropna().nunique() <= 2
        col_missing_rate = float(missing_rate[col])
        col_missing_diff = float(missing_diff.get(col, 0.0))

        if col in type_a_cols:
            missing_type = "A_missing_is_signal_fill_zero"
            fill_strategy = "zero"
            fill_value = 0.0
            add_missing_indicator = True
        elif (
            col_missing_rate < MISSING_RATE_TYPE_B_THRESHOLD
            and col_missing_diff < MISSING_DIFF_TYPE_B_THRESHOLD
        ):
            missing_type = "B_low_missing_low_label_diff"
            if is_binary:
                fill_strategy = "mode"
                fill_value = safe_mode(train[col])
            else:
                fill_strategy = "median"
                fill_value = safe_median(train[col])
            add_missing_indicator = False
        else:
            missing_type = "C_high_missing_or_label_diff"
            fill_strategy = "median"
            fill_value = safe_median(train[col])
            add_missing_indicator = True

        rows.append(
            {
                "feature": col,
                "missing_type": missing_type,
                "missing_rate": col_missing_rate,
                "missing_diff_by_label": col_missing_diff,
                "is_binary": bool(is_binary),
                "fill_strategy": fill_strategy,
                "fill_value": fill_value,
                "add_missing_indicator": bool(add_missing_indicator),
                "missing_indicator_col": f"{col}_missing" if add_missing_indicator else "",
            }
        )

    return pd.DataFrame(rows)


def add_missing_indicators_from_plan(df: pd.DataFrame, imputation_plan: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    added_cols = []
    for row in imputation_plan.itertuples(index=False):
        col = row.feature
        if not row.add_missing_indicator or col not in df.columns:
            continue
        new_col = row.missing_indicator_col
        df[new_col] = df[col].isna().astype(int)
        added_cols.append(new_col)
    return df, added_cols


def apply_imputation_plan(df: pd.DataFrame, imputation_plan: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for row in imputation_plan.itertuples(index=False):
        col = row.feature
        if col in df.columns:
            df[col] = df[col].fillna(row.fill_value)
    return df


def clinical_bin(series: pd.Series, thresholds: dict[str, float], reverse: bool = False) -> pd.Series:
    if not reverse:
        bins = [-np.inf, thresholds["normal"], thresholds["high"], thresholds["critical"], np.inf]
        labels = [0, 1, 2, 3]
    else:
        bins = [-np.inf, thresholds["critical"], thresholds["low"], thresholds["normal"], np.inf]
        labels = [3, 2, 1, 0]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(float)


def add_clinical_grades(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    added_cols = []
    for col, thresholds in CLINICAL_THRESHOLDS.items():
        if col not in df.columns:
            continue
        new_col = f"{col}_grade"
        df[new_col] = clinical_bin(df[col], thresholds, reverse=col in CLINICAL_REVERSE_COLS)
        added_cols.append(new_col)
    return df, added_cols


def fill_with_train_median(df: pd.DataFrame, col: str, median_values: pd.Series, fallback: float = 0.0) -> pd.Series:
    fill_value = median_values[col] if col in median_values.index else fallback
    if pd.isna(fill_value):
        fill_value = fallback
    if col not in df.columns:
        return pd.Series(fill_value, index=df.index, dtype=float)
    return df[col].fillna(fill_value)


def fill_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return df[col].fillna(0)


def add_prior_interaction_features(df: pd.DataFrame, median_values: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    added_cols = []

    cardiac_panel = resolve_existing_columns(list(df.columns), CARDIAC_PANEL_COLS)
    renal_panel = resolve_existing_columns(list(df.columns), RENAL_PANEL_COLS)
    biochem_panel = resolve_existing_columns(list(df.columns), BIOCHEM_PANEL_COLS)

    df["cardiac_panel_count"] = df[cardiac_panel].notna().sum(axis=1) if cardiac_panel else 0
    df["renal_panel_count"] = df[renal_panel].notna().sum(axis=1) if renal_panel else 0
    df["has_biochem"] = df[biochem_panel].notna().any(axis=1).astype(int) if biochem_panel else 0
    df["biochem_panel_count"] = df[biochem_panel].notna().sum(axis=1) if biochem_panel else 0
    added_cols.extend(["cardiac_panel_count", "renal_panel_count", "has_biochem", "biochem_panel_count"])

    df["crp_albumin_ratio"] = np.log1p(fill_zero(df, "超敏感C-反应蛋白")) / (
        fill_with_train_median(df, "*白蛋白（溴甲酚绿法）", median_values) + 1
    )
    df["cardiorenal_index"] = np.log1p(fill_zero(df, "入院BNP")) / (
        fill_with_train_median(df, "EGFR", median_values) + 1
    )
    df["bun_creatinine_ratio"] = fill_with_train_median(df, "*尿素", median_values) / (
        fill_with_train_median(df, "*肌酐(酶法)", median_values) + 0.01
    )
    df["cardiac_triple_hit"] = (
        np.log1p(fill_zero(df, "心肌肌钙蛋白I"))
        + np.log1p(fill_zero(df, "CK同工酶(质量)"))
        + np.log1p(fill_zero(df, "MYO峰值"))
    )
    df["metabolic_burden"] = fill_zero(df, "糖尿病年限") * np.log1p(
        fill_with_train_median(df, "*葡萄糖", median_values)
    )
    df["troponin_hr"] = fill_zero(df, "心肌肌钙蛋白I") * fill_with_train_median(df, "心率", median_values)
    df["hospitalization_severity"] = fill_with_train_median(df, "住院日", median_values).clip(lower=0) * fill_zero(
        df, "压疮评分"
    ).clip(lower=0)
    added_cols.extend(
        [
            "crp_albumin_ratio",
            "cardiorenal_index",
            "bun_creatinine_ratio",
            "cardiac_triple_hit",
            "metabolic_burden",
            "troponin_hr",
            "hospitalization_severity",
        ]
    )

    return df, added_cols


def preprocess_baseline_data(
    data_dir: Path = DATA_DIR,
    train_file: str = TRAIN_FILE,
    test_file: str = TEST_FILE,
    class_weight_clip: float | None = None,
) -> PreprocessBundle:
    train_raw = read_csv(data_dir / train_file)
    test_raw = read_csv(data_dir / test_file)
    dropped_default_cols = resolve_default_drop_cols([c for c in train_raw.columns if c != TARGET_COL])
    if dropped_default_cols:
        train_raw = train_raw.drop(columns=dropped_default_cols)
        test_raw = test_raw.drop(columns=[c for c in dropped_default_cols if c in test_raw.columns])

    base_feature_cols = [c for c in train_raw.columns if c not in [ID_COL, TARGET_COL]]
    missing_cols = train_raw[base_feature_cols].columns[train_raw[base_feature_cols].isna().any()].tolist()
    median_values = train_raw[base_feature_cols].median()
    missing_by_label, missing_diff = compute_missing_by_label(train_raw, base_feature_cols)
    imputation_plan = build_imputation_plan(train_raw, base_feature_cols, missing_diff)
    fill_values = imputation_plan.set_index("feature")["fill_value"]
    binary_chi2_results = build_binary_chi2_results(train_raw, base_feature_cols)
    significant_binary_cols = binary_chi2_results.loc[
        binary_chi2_results["significant_0_05"], "feature"
    ].tolist()
    winsorization_caps = build_winsorization_caps(train_raw, base_feature_cols)
    winsorized_cols = winsorization_caps["feature"].tolist() if not winsorization_caps.empty else []

    train_processed = train_raw.copy()
    test_processed = test_raw.copy()

    # 先根据训练集缺失模式创建缺失指示，再填充原始列，避免测试集统计量泄露。
    train_processed, added_missing_indicator_cols = add_missing_indicators_from_plan(train_processed, imputation_plan)
    test_processed, _ = add_missing_indicators_from_plan(test_processed, imputation_plan)

    # 异常值截断参数只从训练集估计，再用于训练集/测试集。
    train_processed = apply_winsorization_caps(train_processed, winsorization_caps)
    test_processed = apply_winsorization_caps(test_processed, winsorization_caps)

    # 医学先验交互特征需要使用原始缺失模式，因此放在填充前创建。
    train_processed, added_prior_cols = add_prior_interaction_features(train_processed, median_values)
    test_processed, _ = add_prior_interaction_features(test_processed, median_values)

    train_processed = apply_imputation_plan(train_processed, imputation_plan)
    test_processed = apply_imputation_plan(test_processed, imputation_plan)

    # LightGBM 消融显示 expanded_log 和 clinical_grade 未带来稳定收益，正式预处理默认不生成。
    added_log_cols: list[str] = []
    added_clinical_grade_cols: list[str] = []

    feature_cols = [c for c in train_processed.columns if c not in [ID_COL, TARGET_COL]]

    class_weight_dict = compute_balanced_class_weights(train_processed[TARGET_COL])
    class_weight_dict_clipped = clip_weights(class_weight_dict, class_weight_clip)
    active_weight_dict = class_weight_dict_clipped or class_weight_dict
    sample_weights = train_processed[TARGET_COL].map(class_weight_dict).to_numpy(dtype=float)
    sample_weights_clipped = (
        train_processed[TARGET_COL].map(active_weight_dict).to_numpy(dtype=float)
        if class_weight_dict_clipped is not None
        else None
    )

    return PreprocessBundle(
        train_raw=train_raw,
        test_raw=test_raw,
        train_processed=train_processed,
        test_processed=test_processed,
        base_feature_cols=base_feature_cols,
        feature_cols=feature_cols,
        dropped_default_cols=dropped_default_cols,
        missing_cols=missing_cols,
        added_missing_indicator_cols=added_missing_indicator_cols,
        added_log_cols=added_log_cols,
        added_clinical_grade_cols=added_clinical_grade_cols,
        added_prior_cols=added_prior_cols,
        median_values=median_values,
        fill_values=fill_values,
        imputation_plan=imputation_plan,
        binary_chi2_results=binary_chi2_results,
        significant_binary_cols=significant_binary_cols,
        winsorization_caps=winsorization_caps,
        winsorized_cols=winsorized_cols,
        class_weight_dict=class_weight_dict,
        class_weight_dict_clipped=class_weight_dict_clipped,
        sample_weights=sample_weights,
        sample_weights_clipped=sample_weights_clipped,
    )


def build_summary(bundle: PreprocessBundle, class_weight_clip: float | None) -> dict[str, Any]:
    train_missing_after = int(bundle.X_train.isna().sum().sum())
    test_missing_after = int(bundle.X_test.isna().sum().sum())
    label_counts = bundle.y_train.value_counts().sort_index().to_dict()
    imputation_type_counts = bundle.imputation_plan["missing_type"].value_counts().to_dict()

    summary = {
        "encoding": ENCODING,
        "train_shape": list(bundle.train_processed.shape),
        "test_shape": list(bundle.test_processed.shape),
        "base_feature_count": len(bundle.base_feature_cols),
        "feature_count_after_feature_engineering": len(bundle.feature_cols),
        "dropped_default_feature_count": len(bundle.dropped_default_cols),
        "dropped_default_cols": bundle.dropped_default_cols,
        "missing_feature_count_before_fill": len(bundle.missing_cols),
        "added_missing_indicator_count": len(bundle.added_missing_indicator_cols),
        "added_missing_indicator_cols": bundle.added_missing_indicator_cols,
        "winsorized_feature_count": len(bundle.winsorized_cols),
        "winsorized_cols": bundle.winsorized_cols,
        "added_log_feature_count": len(bundle.added_log_cols),
        "added_log_cols": bundle.added_log_cols,
        "added_clinical_grade_feature_count": len(bundle.added_clinical_grade_cols),
        "added_clinical_grade_cols": bundle.added_clinical_grade_cols,
        "added_prior_feature_count": len(bundle.added_prior_cols),
        "added_prior_cols": bundle.added_prior_cols,
        "missing_imputation_type_counts": {str(k): int(v) for k, v in imputation_type_counts.items()},
        "binary_feature_count": int(len(bundle.binary_chi2_results)),
        "significant_binary_feature_count_p_lt_0_05": int(len(bundle.significant_binary_cols)),
        "significant_binary_cols_p_lt_0_05": bundle.significant_binary_cols,
        "missing_values_after_fill_train": train_missing_after,
        "missing_values_after_fill_test": test_missing_after,
        "class_weight_clip": class_weight_clip,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "class_weights_raw": {str(k): round(v, 6) for k, v in bundle.class_weight_dict.items()},
        "class_weights_clipped": (
            {str(k): round(v, 6) for k, v in bundle.class_weight_dict_clipped.items()}
            if bundle.class_weight_dict_clipped is not None
            else None
        ),
        "sample_weight_summary_raw": {
            "min": float(np.min(bundle.sample_weights)),
            "max": float(np.max(bundle.sample_weights)),
            "mean": float(np.mean(bundle.sample_weights)),
        },
        "sample_weight_summary_clipped": (
            {
                "min": float(np.min(bundle.sample_weights_clipped)),
                "max": float(np.max(bundle.sample_weights_clipped)),
                "mean": float(np.mean(bundle.sample_weights_clipped)),
            }
            if bundle.sample_weights_clipped is not None
            else None
        ),
        "notes": [
            "Missing values are imputed by a train-derived A/B/C clinical strategy, not by one global median rule.",
            "Type A features receive missing indicators and zero fill; Type C features receive missing indicators and train medians.",
            "Low-missingness Type B binary features use the train mode; Type B continuous features use the train median.",
            "All imputation statistics are computed only from training data and then reused on test data.",
            "Endpoint-like features are removed by default after leakage-sensitivity ablation showed negligible benefit.",
            "Only the exactly duplicated liver-function flag is removed; pulse is retained despite high correlation with heart rate.",
            "Outlier caps are estimated from the training set 99th percentile for high-IQR-outlier continuous features.",
            "Expanded log and clinical-grade derived features are disabled after LightGBM ablation showed no stable gain.",
            "Literature-motivated prior interaction features are created before refined imputation.",
            "No standardization is applied.",
        ],
    }
    return summary


def save_outputs(bundle: PreprocessBundle, output_dir: Path, class_weight_clip: float | None):
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle.train_processed.to_csv(output_dir / "train_processed.csv", index=False, encoding="utf-8-sig")
    bundle.test_processed.to_csv(output_dir / "test_processed.csv", index=False, encoding="utf-8-sig")
    bundle.X_train.to_csv(output_dir / "X_train_processed.csv", index=False, encoding="utf-8-sig")
    bundle.X_test.to_csv(output_dir / "X_test_processed.csv", index=False, encoding="utf-8-sig")
    bundle.y_train.to_csv(output_dir / "y_train.csv", index=False, encoding="utf-8-sig")
    pd.Series(bundle.missing_cols, name="missing_col").to_csv(
        output_dir / "missing_cols.csv", index=False, encoding="utf-8-sig"
    )
    pd.Series(bundle.dropped_default_cols, name="feature").to_csv(
        output_dir / "dropped_default_cols.csv", index=False, encoding="utf-8-sig"
    )
    pd.Series(bundle.added_missing_indicator_cols, name="feature").to_csv(
        output_dir / "added_missing_indicator_cols.csv", index=False, encoding="utf-8-sig"
    )
    pd.Series(bundle.added_log_cols, name="feature").to_csv(
        output_dir / "added_log_cols.csv", index=False, encoding="utf-8-sig"
    )
    pd.Series(bundle.added_clinical_grade_cols, name="feature").to_csv(
        output_dir / "added_clinical_grade_cols.csv", index=False, encoding="utf-8-sig"
    )
    pd.Series(bundle.added_prior_cols, name="feature").to_csv(
        output_dir / "added_prior_feature_cols.csv", index=False, encoding="utf-8-sig"
    )
    bundle.median_values.rename("median").to_csv(output_dir / "train_feature_medians.csv", encoding="utf-8-sig")
    bundle.fill_values.rename("fill_value").to_csv(output_dir / "train_feature_fill_values.csv", encoding="utf-8-sig")
    bundle.imputation_plan.to_csv(output_dir / "missing_imputation_plan.csv", index=False, encoding="utf-8-sig")
    bundle.winsorization_caps.to_csv(output_dir / "winsorization_caps.csv", index=False, encoding="utf-8-sig")
    bundle.binary_chi2_results.to_csv(output_dir / "binary_chi2_results.csv", index=False, encoding="utf-8-sig")
    pd.Series(bundle.significant_binary_cols, name="feature").to_csv(
        output_dir / "significant_binary_cols.csv", index=False, encoding="utf-8-sig"
    )

    raw_weight_df = pd.DataFrame(
        {
            ID_COL: bundle.train_processed[ID_COL],
            TARGET_COL: bundle.y_train,
            "sample_weight": bundle.sample_weights,
        }
    )
    raw_weight_df.to_csv(output_dir / "train_sample_weights.csv", index=False, encoding="utf-8-sig")
    np.save(output_dir / "train_sample_weights.npy", bundle.sample_weights)

    class_weights_export = {
        "raw": bundle.class_weight_dict,
        "clipped": bundle.class_weight_dict_clipped,
    }
    with open(output_dir / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump(class_weights_export, f, ensure_ascii=False, indent=2)

    if bundle.sample_weights_clipped is not None:
        clipped_weight_df = pd.DataFrame(
            {
                ID_COL: bundle.train_processed[ID_COL],
                TARGET_COL: bundle.y_train,
                "sample_weight_clipped": bundle.sample_weights_clipped,
            }
        )
        clipped_weight_df.to_csv(output_dir / "train_sample_weights_clipped.csv", index=False, encoding="utf-8-sig")
        np.save(output_dir / "train_sample_weights_clipped.npy", bundle.sample_weights_clipped)

    summary = build_summary(bundle, class_weight_clip)
    with open(output_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_console_summary(bundle: PreprocessBundle, class_weight_clip: float | None):
    print(f"Train shape: {bundle.train_processed.shape}")
    print(f"Test shape: {bundle.test_processed.shape}")
    print(f"Base feature count: {len(bundle.base_feature_cols)}")
    print(f"Feature count after feature engineering: {len(bundle.feature_cols)}")
    print(f"Dropped default features: {len(bundle.dropped_default_cols)}")
    print(f"Missing columns before fill: {len(bundle.missing_cols)}")
    print(f"Added missing indicators: {len(bundle.added_missing_indicator_cols)}")
    print(f"Winsorized features: {len(bundle.winsorized_cols)}")
    print(f"Added log features: {len(bundle.added_log_cols)}")
    print(f"Added clinical grade features: {len(bundle.added_clinical_grade_cols)}")
    print(f"Added prior interaction features: {len(bundle.added_prior_cols)}")
    print("Missing imputation type counts:")
    for missing_type, count in bundle.imputation_plan["missing_type"].value_counts().items():
        print(f"  {missing_type}: {count}")
    print(f"Binary features tested by chi-square: {len(bundle.binary_chi2_results)}")
    print(f"Significant binary features (p<0.05): {len(bundle.significant_binary_cols)}")
    print(f"Missing values after fill (train): {int(bundle.X_train.isna().sum().sum())}")
    print(f"Missing values after fill (test): {int(bundle.X_test.isna().sum().sum())}")
    print("Raw class weights:")
    for label, weight in bundle.class_weight_dict.items():
        print(f"  label={label}: {weight:.6f}")
    if class_weight_clip is not None and bundle.class_weight_dict_clipped is not None:
        print(f"Clipped class weights (max={class_weight_clip}):")
        for label, weight in bundle.class_weight_dict_clipped.items():
            print(f"  label={label}: {weight:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced baseline preprocessing for tabular models.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--class-weight-clip",
        type=float,
        default=None,
        help="Optional max cap for balanced class weights. Raw weights are always saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = preprocess_baseline_data(
        data_dir=args.data_dir,
        class_weight_clip=args.class_weight_clip,
    )
    save_outputs(bundle, args.output_dir, args.class_weight_clip)
    print_console_summary(bundle, args.class_weight_clip)


if __name__ == "__main__":
    main()
