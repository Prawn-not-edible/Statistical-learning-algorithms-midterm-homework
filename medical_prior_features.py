from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline_preprocess import (
    BIOCHEM_PANEL_COLS,
    CARDIAC_PANEL_COLS,
    RENAL_PANEL_COLS,
    fill_with_train_median,
    fill_zero,
    resolve_existing_columns,
)


MEDICAL_PRIOR_PREFIX = "prior__"
MEDICAL_PRIOR_FAMILY = "literature_guided_medical_prior"


def _add_metadata(
    rows: list[dict],
    df: pd.DataFrame,
    feature: str,
    formula: str,
    source_features: list[str],
    rationale: str,
) -> None:
    values = df[feature]
    rows.append(
        {
            "feature": feature,
            "formula": formula,
            "source_features": ", ".join(source_features),
            "feature_family": MEDICAL_PRIOR_FAMILY,
            "rationale": rationale,
            "missing_rate": float(values.isna().mean()),
            "non_missing_count": int(values.notna().sum()),
            "median": float(values.median(skipna=True)) if values.notna().any() else np.nan,
            "max": float(values.max(skipna=True)) if values.notna().any() else np.nan,
        }
    )


def add_medical_prior_features(
    df: pd.DataFrame,
    median_values: pd.Series,
    prefix: str = MEDICAL_PRIOR_PREFIX,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Build a tagged, pluggable medical-prior feature block.

    These formulas mirror the current literature/medical-prior features in
    baseline_preprocess.py, but every generated column receives a `prior__`
    prefix so downstream ablations can identify and switch the block cleanly.
    """
    df_out = df.copy()
    added_cols: list[str] = []
    metadata_rows: list[dict] = []

    cardiac_panel = resolve_existing_columns(list(df_out.columns), CARDIAC_PANEL_COLS)
    renal_panel = resolve_existing_columns(list(df_out.columns), RENAL_PANEL_COLS)
    biochem_panel = resolve_existing_columns(list(df_out.columns), BIOCHEM_PANEL_COLS)

    panel_specs = [
        (
            f"{prefix}cardiac_panel_count",
            df_out[cardiac_panel].notna().sum(axis=1) if cardiac_panel else 0,
            "count(non-missing cardiac panel)",
            cardiac_panel,
            "Testing behavior itself can carry diagnostic information.",
        ),
        (
            f"{prefix}renal_panel_count",
            df_out[renal_panel].notna().sum(axis=1) if renal_panel else 0,
            "count(non-missing renal panel)",
            renal_panel,
            "Renal testing density captures cardiorenal concern.",
        ),
        (
            f"{prefix}has_biochem",
            df_out[biochem_panel].notna().any(axis=1).astype(float) if biochem_panel else 0,
            "I(any biochemistry panel observed)",
            biochem_panel,
            "Shared biochemistry missingness is a structural testing signal.",
        ),
        (
            f"{prefix}biochem_panel_count",
            df_out[biochem_panel].notna().sum(axis=1) if biochem_panel else 0,
            "count(non-missing biochemistry panel)",
            biochem_panel,
            "Amount of biochemistry testing reflects clinical workup intensity.",
        ),
    ]
    for feature, values, formula, sources, rationale in panel_specs:
        df_out[feature] = values
        added_cols.append(feature)
        _add_metadata(metadata_rows, df_out, feature, formula, sources, rationale)

    formulas = {
        f"{prefix}crp_albumin_ratio": (
            np.log1p(fill_zero(df_out, "超敏感C-反应蛋白"))
            / (fill_with_train_median(df_out, "*白蛋白（溴甲酚绿法）", median_values) + 1),
            "log1p(超敏感C-反应蛋白) / (*白蛋白（溴甲酚绿法） + 1)",
            ["超敏感C-反应蛋白", "*白蛋白（溴甲酚绿法）"],
            "CRP/albumin captures inflammation plus nutritional reserve.",
        ),
        f"{prefix}cardiorenal_index": (
            np.log1p(fill_zero(df_out, "入院BNP")) / (fill_with_train_median(df_out, "EGFR", median_values) + 1),
            "log1p(入院BNP) / (EGFR + 1)",
            ["入院BNP", "EGFR"],
            "BNP pressure signal combined with renal function.",
        ),
        f"{prefix}bun_creatinine_ratio": (
            fill_with_train_median(df_out, "*尿素", median_values)
            / (fill_with_train_median(df_out, "*肌酐(酶法)", median_values) + 0.01),
            "*尿素 / (*肌酐(酶法) + 0.01)",
            ["*尿素", "*肌酐(酶法)"],
            "BUN/creatinine-like cardiorenal and mortality risk signal.",
        ),
        f"{prefix}cardiac_triple_hit": (
            np.log1p(fill_zero(df_out, "心肌肌钙蛋白I"))
            + np.log1p(fill_zero(df_out, "CK同工酶(质量)"))
            + np.log1p(fill_zero(df_out, "MYO峰值")),
            "log1p(心肌肌钙蛋白I) + log1p(CK同工酶(质量)) + log1p(MYO峰值)",
            ["心肌肌钙蛋白I", "CK同工酶(质量)", "MYO峰值"],
            "Multi-marker myocardial injury score.",
        ),
        f"{prefix}metabolic_burden": (
            fill_zero(df_out, "糖尿病年限") * np.log1p(fill_with_train_median(df_out, "*葡萄糖", median_values)),
            "糖尿病年限 * log1p(*葡萄糖)",
            ["糖尿病年限", "*葡萄糖"],
            "Duration-weighted acute glucose burden.",
        ),
        f"{prefix}troponin_hr": (
            fill_zero(df_out, "心肌肌钙蛋白I") * fill_with_train_median(df_out, "心率", median_values),
            "心肌肌钙蛋白I * 心率",
            ["心肌肌钙蛋白I", "心率"],
            "Troponin injury signal combined with hemodynamic stress.",
        ),
        f"{prefix}hospitalization_severity": (
            fill_with_train_median(df_out, "住院日", median_values).clip(lower=0)
            * fill_zero(df_out, "压疮评分").clip(lower=0),
            "住院日 * 压疮评分",
            ["住院日", "压疮评分"],
            "Hospitalization complexity and frailty proxy.",
        ),
    }

    for feature, (values, formula, sources, rationale) in formulas.items():
        df_out[feature] = values
        added_cols.append(feature)
        _add_metadata(metadata_rows, df_out, feature, formula, sources, rationale)

    return df_out, added_cols, pd.DataFrame(metadata_rows)


def build_medical_prior_block(
    df: pd.DataFrame,
    median_values: pd.Series,
    prefix: str = MEDICAL_PRIOR_PREFIX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enhanced, added_cols, metadata = add_medical_prior_features(df, median_values=median_values, prefix=prefix)
    return enhanced[added_cols].copy(), metadata


def save_medical_prior_outputs(
    output_dir: Path,
    train_block: pd.DataFrame,
    test_block: pd.DataFrame | None,
    metadata: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_block.to_csv(output_dir / "X_train_medical_prior_features.csv", index=False, encoding="utf-8-sig")
    if test_block is not None:
        test_block.to_csv(output_dir / "X_test_medical_prior_features.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(output_dir / "medical_prior_feature_metadata.csv", index=False, encoding="utf-8-sig")
