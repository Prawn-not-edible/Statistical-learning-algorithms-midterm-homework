from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CLINICAL_RANGE_PREFIX = "is_normal_"
CLINICAL_RANGE_FAMILY = "clinical_reference_range"


# Adult clinical reference ranges. Several indicators have sex-specific cutoffs
# in clinical practice; here we use broad combined intervals for robust tabular modeling.
CLINICAL_RANGES: dict[str, tuple[float, float]] = {
    # Vital signs and body size
    "入院收缩压": (90.0, 140.0),
    "入院舒张压": (60.0, 90.0),
    "心率": (60.0, 100.0),
    "BMI计算": (18.5, 23.9),
    # Glucose and lipid metabolism
    "空腹血糖": (3.9, 6.1),
    "*糖化血红蛋白": (4.0, 6.5),
    "*总胆固醇": (0.0, 5.2),
    "*甘油三酯": (0.0, 1.7),
    "低密度脂蛋白胆固醇": (0.0, 3.4),
    # Liver, biliary, and kidney function
    "*谷丙转氨酶": (0.0, 40.0),
    "*谷草转氨酶": (0.0, 40.0),
    "*白蛋白（溴甲酚绿法）": (40.0, 55.0),
    "*总胆红素": (0.0, 21.0),
    "肌酐": (44.0, 106.0),
    "*肌酐(酶法)": (44.0, 106.0),
    "*尿酸": (150.0, 420.0),
    "*尿素": (2.9, 8.2),
    # Electrolytes and acid-base balance
    "*钠": (135.0, 145.0),
    "*钾": (3.5, 5.5),
    "*氯": (96.0, 106.0),
    "*钙": (2.11, 2.52),
    "阴离子间隙": (8.0, 16.0),
    # Cardiovascular and inflammatory markers
    "超敏感C-反应蛋白": (0.0, 3.0),
    "入院BNP": (0.0, 100.0),
}


def add_clinical_normal_indicators(
    df: pd.DataFrame,
    ranges_dict: dict[str, tuple[float, float]] | None = None,
    prefix: str = CLINICAL_RANGE_PREFIX,
    keep_missing: bool = True,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Add binary indicators for whether numeric clinical values are in reference range.

    Missing source values stay missing by default. This preserves the distinction
    between "abnormal result" and "test not observed"; downstream non-missing-tolerant
    models can impute these new indicators explicitly.
    """
    ranges = ranges_dict or CLINICAL_RANGES
    df_out = df.copy()
    added_features: list[str] = []
    metadata_rows = []

    for col, (lower, upper) in ranges.items():
        exists = col in df_out.columns
        new_col = f"{prefix}{col}"
        if not exists:
            metadata_rows.append(
                {
                    "source_feature": col,
                    "clinical_lower": float(lower),
                    "clinical_upper": float(upper),
                    "indicator_feature": new_col,
                    "exists_in_input": False,
                    "missing_rate": np.nan,
                    "normal_rate_non_missing": np.nan,
                    "abnormal_rate_non_missing": np.nan,
                }
            )
            continue

        source = df_out[col]
        indicator = pd.Series(0.0, index=df_out.index, dtype=float)
        indicator[(source >= lower) & (source <= upper)] = 1.0
        if keep_missing:
            indicator[source.isna()] = np.nan
        df_out[new_col] = indicator
        added_features.append(new_col)

        non_missing = indicator.dropna()
        normal_rate = float((non_missing == 1.0).mean()) if len(non_missing) else np.nan
        abnormal_rate = float((non_missing == 0.0).mean()) if len(non_missing) else np.nan
        metadata_rows.append(
            {
                "source_feature": col,
                "clinical_lower": float(lower),
                "clinical_upper": float(upper),
                "indicator_feature": new_col,
                "exists_in_input": True,
                "missing_rate": float(source.isna().mean()),
                "normal_rate_non_missing": normal_rate,
                "abnormal_rate_non_missing": abnormal_rate,
            }
        )

    return df_out, added_features, pd.DataFrame(metadata_rows)


def build_clinical_indicator_block(
    df: pd.DataFrame,
    ranges_dict: dict[str, tuple[float, float]] | None = None,
    prefix: str = CLINICAL_RANGE_PREFIX,
    keep_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enhanced, added_cols, metadata = add_clinical_normal_indicators(
        df=df,
        ranges_dict=ranges_dict,
        prefix=prefix,
        keep_missing=keep_missing,
    )
    return enhanced[added_cols].copy(), metadata


def impute_clinical_indicator_block(block: pd.DataFrame, unknown_value: float = -1.0) -> pd.DataFrame:
    """Impute missing clinical indicators for models such as LR that cannot consume NaN."""
    return block.fillna(unknown_value)


def save_clinical_range_outputs(
    output_dir: Path,
    train_block: pd.DataFrame,
    test_block: pd.DataFrame | None,
    metadata: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_block.to_csv(output_dir / "X_train_clinical_range_features.csv", index=False, encoding="utf-8-sig")
    if test_block is not None:
        test_block.to_csv(output_dir / "X_test_clinical_range_features.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(output_dir / "clinical_range_feature_metadata.csv", index=False, encoding="utf-8-sig")
