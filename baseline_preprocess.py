import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "preprocessed_outputs_v2"
TRAIN_FILE = "training_dataset.csv"
TEST_FILE = "test_dataset.csv"
ENCODING = "gbk"
ID_COL = "序号"
TARGET_COL = "label"

CLINICAL_MISSING_COL_CANDIDATES = [
    "TnI峰值",
    "CK-MB峰值",
    "MYO峰值",
    "TIMI危险评分(STEMI)",
    "TIMI危险评分（STEMI）",
    "戒酒年限",
    "戒烟时间",
    "糖尿病年限",
]

SKEWED_LOG_COLS = [
    "心肌肌钙蛋白I",
    "CK同工酶(质量)",
    "入院BNP",
    "TnI峰值",
    "CK-MB峰值",
    "MYO峰值",
    "空腹血糖",
]


@dataclass
class PreprocessBundle:
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_processed: pd.DataFrame
    test_processed: pd.DataFrame
    base_feature_cols: list[str]
    feature_cols: list[str]
    missing_cols: list[str]
    added_missing_indicator_cols: list[str]
    added_log_cols: list[str]
    median_values: pd.Series
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
        new_col = f"{col}_缺失"
        df[new_col] = df[col].isna().astype(int)
        added_cols.append(new_col)
    return df, added_cols


def add_log_transforms(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    added_cols = []
    for col in cols:
        new_col = f"{col}_log"
        df[new_col] = np.log1p(df[col].clip(lower=0))
        added_cols.append(new_col)
    return df, added_cols


def preprocess_baseline_data(
    data_dir: Path = DATA_DIR,
    train_file: str = TRAIN_FILE,
    test_file: str = TEST_FILE,
    class_weight_clip: float | None = None,
) -> PreprocessBundle:
    train_raw = read_csv(data_dir / train_file)
    test_raw = read_csv(data_dir / test_file)

    base_feature_cols = [c for c in train_raw.columns if c not in [ID_COL, TARGET_COL]]
    missing_cols = train_raw[base_feature_cols].columns[train_raw[base_feature_cols].isna().any()].tolist()
    median_values = train_raw[base_feature_cols].median()

    train_processed = train_raw.copy()
    test_processed = test_raw.copy()

    # 先根据原始缺失创建“缺失即信号”的标志特征，再做中位数填充。
    clinical_missing_cols = resolve_existing_columns(base_feature_cols, CLINICAL_MISSING_COL_CANDIDATES)
    train_processed, added_missing_indicator_cols = add_missing_indicators(train_processed, clinical_missing_cols)
    test_processed, _ = add_missing_indicators(test_processed, clinical_missing_cols)

    train_processed[base_feature_cols] = train_processed[base_feature_cols].fillna(median_values)
    test_processed[base_feature_cols] = test_processed[base_feature_cols].fillna(median_values)

    # 对右偏医学指标追加 log1p 特征，不覆盖原值。
    skewed_cols = resolve_existing_columns(base_feature_cols, SKEWED_LOG_COLS)
    train_processed, added_log_cols = add_log_transforms(train_processed, skewed_cols)
    test_processed, _ = add_log_transforms(test_processed, skewed_cols)

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
        missing_cols=missing_cols,
        added_missing_indicator_cols=added_missing_indicator_cols,
        added_log_cols=added_log_cols,
        median_values=median_values,
        class_weight_dict=class_weight_dict,
        class_weight_dict_clipped=class_weight_dict_clipped,
        sample_weights=sample_weights,
        sample_weights_clipped=sample_weights_clipped,
    )


def build_summary(bundle: PreprocessBundle, class_weight_clip: float | None) -> dict[str, Any]:
    train_missing_after = int(bundle.X_train.isna().sum().sum())
    test_missing_after = int(bundle.X_test.isna().sum().sum())
    label_counts = bundle.y_train.value_counts().sort_index().to_dict()

    summary = {
        "encoding": ENCODING,
        "train_shape": list(bundle.train_processed.shape),
        "test_shape": list(bundle.test_processed.shape),
        "base_feature_count": len(bundle.base_feature_cols),
        "feature_count_after_feature_engineering": len(bundle.feature_cols),
        "missing_feature_count_before_fill": len(bundle.missing_cols),
        "added_missing_indicator_count": len(bundle.added_missing_indicator_cols),
        "added_missing_indicator_cols": bundle.added_missing_indicator_cols,
        "added_log_feature_count": len(bundle.added_log_cols),
        "added_log_cols": bundle.added_log_cols,
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
            "Clinical missing indicators are created before median imputation.",
            "Median statistics are computed only from the original training features.",
            "Right-skewed clinical markers receive extra log1p features while raw values are preserved.",
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
    pd.Series(bundle.added_missing_indicator_cols, name="feature").to_csv(
        output_dir / "added_missing_indicator_cols.csv", index=False, encoding="utf-8-sig"
    )
    pd.Series(bundle.added_log_cols, name="feature").to_csv(
        output_dir / "added_log_cols.csv", index=False, encoding="utf-8-sig"
    )
    bundle.median_values.rename("median").to_csv(output_dir / "train_feature_medians.csv", encoding="utf-8-sig")

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
    print(f"Missing columns before fill: {len(bundle.missing_cols)}")
    print(f"Added missing indicators: {len(bundle.added_missing_indicator_cols)}")
    print(f"Added log features: {len(bundle.added_log_cols)}")
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
