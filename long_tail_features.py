from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LONG_TAIL_PREFIX = "longtail__"
LONG_TAIL_FAMILY = "skew_driven_long_tail_versions"
DEFAULT_SKEW_THRESHOLD = 2.0
DEFAULT_UPPER_QUANTILE = 0.99
CONTINUOUS_N_UNIQUE_MIN = 10


def discover_long_tail_columns(
    train: pd.DataFrame,
    feature_cols: list[str],
    skew_threshold: float = DEFAULT_SKEW_THRESHOLD,
) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        values = train[col].dropna()
        if values.nunique() <= CONTINUOUS_N_UNIQUE_MIN:
            continue
        skew = values.skew()
        if pd.isna(skew) or skew <= skew_threshold:
            continue
        rows.append(
            {
                "source_feature": col,
                "skew": float(skew),
                "missing_rate": float(train[col].isna().mean()),
                "median": float(values.median()) if len(values) else np.nan,
                "upper_q99": float(values.quantile(DEFAULT_UPPER_QUANTILE)) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("skew", ascending=False).reset_index(drop=True)


def build_long_tail_feature_block(
    df: pd.DataFrame,
    long_tail_plan: pd.DataFrame,
    prefix: str = LONG_TAIL_PREFIX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features: dict[str, pd.Series] = {}
    metadata_rows = []
    for row in long_tail_plan.itertuples(index=False):
        col = row.source_feature
        if col not in df.columns:
            continue
        source = df[col]
        fill_value = row.median if not pd.isna(row.median) else 0.0
        clipped = source.clip(upper=row.upper_q99).fillna(fill_value)
        log_values = np.log1p(source.clip(lower=0).fillna(0))

        winsor_col = f"{prefix}winsor_q99__{col}"
        log_col = f"{prefix}log1p__{col}"
        features[winsor_col] = clipped
        features[log_col] = log_values
        metadata_rows.extend(
            [
                {
                    "feature": winsor_col,
                    "source_feature": col,
                    "feature_family": LONG_TAIL_FAMILY,
                    "operation": "winsor_q99",
                    "formula": f"clip({col}, upper=train_q99).fillna(train_median)",
                    "skew": row.skew,
                    "source_missing_rate": row.missing_rate,
                    "train_median": row.median,
                    "train_upper_q99": row.upper_q99,
                },
                {
                    "feature": log_col,
                    "source_feature": col,
                    "feature_family": LONG_TAIL_FAMILY,
                    "operation": "log1p",
                    "formula": f"log1p(max({col}, 0).fillna(0))",
                    "skew": row.skew,
                    "source_missing_rate": row.missing_rate,
                    "train_median": row.median,
                    "train_upper_q99": row.upper_q99,
                },
            ]
        )

    return pd.DataFrame(features, index=df.index), pd.DataFrame(metadata_rows)


def save_long_tail_outputs(
    output_dir: Path,
    train_block: pd.DataFrame,
    test_block: pd.DataFrame | None,
    plan: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_block.to_csv(output_dir / "X_train_long_tail_features.csv", index=False, encoding="utf-8-sig")
    if test_block is not None:
        test_block.to_csv(output_dir / "X_test_long_tail_features.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(output_dir / "long_tail_feature_plan.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(output_dir / "long_tail_feature_metadata.csv", index=False, encoding="utf-8-sig")
