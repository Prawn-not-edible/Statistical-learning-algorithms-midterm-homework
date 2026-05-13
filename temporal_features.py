from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TEMPORAL_FEATURE_FAMILY = "cardiovascular_temporal_features"
TEMPORAL_FEATURE_UNKNOWN_VALUE = -1.0


def _series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index, dtype=float)


def _series_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _positive_recent_stress(series: pd.Series) -> pd.Series:
    # In this dataset, value 1 appears in almost all rows for 1月内手术/创伤史,
    # while value 2 is rare. Treat 2/3 as positive acute stress and 1/0 as non-positive/unknown.
    return series.isin([2, 3]).astype(float)


def _residual_risk_from_quit_time(
    history: pd.Series,
    quit_years: pd.Series,
    active_codes: set[int],
    former_codes: set[int],
) -> pd.Series:
    risk = pd.Series(0.0, index=history.index, dtype=float)
    quit_observed = quit_years.notna()
    positive_quit_time = quit_observed & (quit_years > 0)
    zero_quit_time = quit_observed & (quit_years <= 0)

    risk[positive_quit_time] = 1.0 / (1.0 + quit_years[positive_quit_time])
    risk[zero_quit_time] = 1.0

    active_without_quit_time = history.isin(active_codes) & ~quit_observed
    risk[active_without_quit_time] = 1.0

    former_without_quit_time = history.isin(former_codes) & ~quit_observed
    risk[former_without_quit_time] = np.nan
    return risk


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
            "rationale": rationale,
            "missing_rate": float(values.isna().mean()),
            "non_missing_count": int(values.notna().sum()),
            "min": float(values.min(skipna=True)) if values.notna().any() else np.nan,
            "median": float(values.median(skipna=True)) if values.notna().any() else np.nan,
            "max": float(values.max(skipna=True)) if values.notna().any() else np.nan,
        }
    )


def add_temporal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Build cardiovascular temporal features from chronic duration and acute state."""
    df_out = df.copy()
    added_cols: list[str] = []
    metadata_rows: list[dict] = []

    age = _series_or_nan(df_out, "年龄")
    sex = _series_or_nan(df_out, "性别")
    is_male = sex == 1
    is_female = sex == 2

    onset_specs = [
        ("冠心病年限", "CAD_发病年龄", "年龄 - 冠心病年限"),
        ("高血压年限", "高血压_发病年龄", "年龄 - 高血压年限"),
        ("糖尿病年限", "糖尿病_发病年龄", "年龄 - 糖尿病年限"),
    ]
    for duration_col, new_col, formula in onset_specs:
        if "年龄" not in df_out.columns or duration_col not in df_out.columns:
            continue
        duration = df_out[duration_col]
        onset_age = age - duration
        invalid = duration.isna() | onset_age.lt(0) | duration.gt(age)
        df_out[new_col] = onset_age.mask(invalid)
        added_cols.append(new_col)
        _add_metadata(
            metadata_rows,
            df_out,
            new_col,
            formula,
            ["年龄", duration_col],
            "Earlier onset can indicate stronger inherited or metabolic vulnerability.",
        )

    if {"性别", "年龄", "冠心病年限"}.issubset(df_out.columns):
        cad_onset = df_out["CAD_发病年龄"] if "CAD_发病年龄" in df_out.columns else age - df_out["冠心病年限"]
        premature = pd.Series(0.0, index=df_out.index, dtype=float)
        premature[is_male & cad_onset.lt(55)] = 1.0
        premature[is_female & cad_onset.lt(65)] = 1.0
        df_out["is_早发冠心病"] = premature
        added_cols.append("is_早发冠心病")
        _add_metadata(
            metadata_rows,
            df_out,
            "is_早发冠心病",
            "male CAD onset <55 or female CAD onset <65",
            ["性别", "年龄", "冠心病年限"],
            "Premature CAD is a clinically recognized high-risk marker.",
        )

    if {"高血压年限", "入院收缩压"}.issubset(df_out.columns):
        htn_years = df_out["高血压年限"].fillna(0)
        sbp = df_out["入院收缩压"]
        sbp_excess = (sbp - 140.0).clip(lower=0)
        burden = htn_years * sbp_excess
        burden[df_out["高血压年限"].notna() & sbp.isna()] = np.nan
        df_out["高血压_血管损伤负荷"] = burden
        added_cols.append("高血压_血管损伤负荷")
        _add_metadata(
            metadata_rows,
            df_out,
            "高血压_血管损伤负荷",
            "高血压年限 * max(0, 入院收缩压 - 140)",
            ["高血压年限", "入院收缩压"],
            "Chronic pressure exposure plus current excess systolic pressure.",
        )

    if {"糖尿病年限", "*糖化血红蛋白"}.issubset(df_out.columns):
        dm_years = df_out["糖尿病年限"].fillna(0)
        hba1c = df_out["*糖化血红蛋白"]
        hba1c_excess = (hba1c - 6.5).clip(lower=0)
        burden = dm_years * hba1c_excess
        burden[df_out["糖尿病年限"].notna() & hba1c.isna()] = np.nan
        df_out["糖尿病_糖毒性负荷"] = burden
        added_cols.append("糖尿病_糖毒性负荷")
        _add_metadata(
            metadata_rows,
            df_out,
            "糖尿病_糖毒性负荷",
            "糖尿病年限 * max(0, *糖化血红蛋白 - 6.5)",
            ["糖尿病年限", "*糖化血红蛋白"],
            "Duration-weighted hyperglycemic exposure.",
        )

    if {"冠心病年限", "低密度脂蛋白胆固醇"}.issubset(df_out.columns):
        cad_years = df_out["冠心病年限"].fillna(0)
        ldl = df_out["低密度脂蛋白胆固醇"]
        burden = cad_years * ldl
        burden[df_out["冠心病年限"].notna() & ldl.isna()] = np.nan
        df_out["冠心病_脂质暴露负荷"] = burden
        added_cols.append("冠心病_脂质暴露负荷")
        _add_metadata(
            metadata_rows,
            df_out,
            "冠心病_脂质暴露负荷",
            "冠心病年限 * 低密度脂蛋白胆固醇",
            ["冠心病年限", "低密度脂蛋白胆固醇"],
            "Duration-weighted atherosclerotic lipid burden.",
        )

    if {"冠心病年限", "1月内手术/创伤史"}.issubset(df_out.columns):
        acute_stress = _positive_recent_stress(df_out["1月内手术/创伤史"])
        df_out["冠心病_急性应激叠加"] = df_out["冠心病年限"].fillna(0) * acute_stress
        added_cols.append("冠心病_急性应激叠加")
        _add_metadata(
            metadata_rows,
            df_out,
            "冠心病_急性应激叠加",
            "冠心病年限 * I(1月内手术/创伤史 in {2,3})",
            ["冠心病年限", "1月内手术/创伤史"],
            "Acute stress on top of chronic coronary disease.",
        )

    creatinine_col = "肌酐" if "肌酐" in df_out.columns else "*肌酐(酶法)"
    if {"高血压年限", "糖尿病年限", creatinine_col}.issubset(df_out.columns):
        chronic_years = df_out["高血压年限"].fillna(0) + df_out["糖尿病年限"].fillna(0)
        creatinine_ratio = df_out[creatinine_col] / 106.0
        kidney_hit = chronic_years * creatinine_ratio
        kidney_hit[chronic_years.gt(0) & df_out[creatinine_col].isna()] = np.nan
        df_out["肾脏_急慢性打击指数"] = kidney_hit
        added_cols.append("肾脏_急慢性打击指数")
        _add_metadata(
            metadata_rows,
            df_out,
            "肾脏_急慢性打击指数",
            "(高血压年限 + 糖尿病年限) * (肌酐 / 106)",
            ["高血压年限", "糖尿病年限", creatinine_col],
            "Chronic hypertensive/diabetic kidney exposure plus current renal dysfunction.",
        )

    if {"吸烟史_x", "戒烟时间"}.issubset(df_out.columns):
        df_out["吸烟_残余风险指数"] = _residual_risk_from_quit_time(
            history=df_out["吸烟史_x"],
            quit_years=df_out["戒烟时间"],
            active_codes={1},
            former_codes={2},
        )
        added_cols.append("吸烟_残余风险指数")
        _add_metadata(
            metadata_rows,
            df_out,
            "吸烟_残余风险指数",
            "active smoker=1; former smoker=1/(1+戒烟时间); never smoker=0",
            ["吸烟史_x", "戒烟时间"],
            "Residual smoking risk decays non-linearly after quitting.",
        )

    if {"饮酒史_x", "戒酒年限"}.issubset(df_out.columns):
        df_out["饮酒_残余风险指数"] = _residual_risk_from_quit_time(
            history=df_out["饮酒史_x"],
            quit_years=df_out["戒酒年限"],
            active_codes={2, 3},
            former_codes={4},
        )
        added_cols.append("饮酒_残余风险指数")
        _add_metadata(
            metadata_rows,
            df_out,
            "饮酒_残余风险指数",
            "active drinking=1; former drinking=1/(1+戒酒年限); no drinking=0",
            ["饮酒史_x", "戒酒年限"],
            "Residual alcohol-related risk decays non-linearly after quitting.",
        )

    for col in ["冠心病年限", "高血压年限", "糖尿病年限"]:
        if col not in df_out.columns:
            continue
        new_col = f"{col}_cleaned"
        df_out[new_col] = df_out[col].fillna(0)
        added_cols.append(new_col)
        _add_metadata(
            metadata_rows,
            df_out,
            new_col,
            f"{col}.fillna(0)",
            [col],
            "Duration value for linear models where missing disease duration means no recorded disease history.",
        )

    metadata = pd.DataFrame(metadata_rows)
    return df_out, added_cols, metadata


def build_temporal_feature_block(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enhanced, added_cols, metadata = add_temporal_features(df)
    return enhanced[added_cols].copy(), metadata


def impute_temporal_feature_block(
    block: pd.DataFrame,
    unknown_value: float = TEMPORAL_FEATURE_UNKNOWN_VALUE,
) -> pd.DataFrame:
    return block.fillna(unknown_value)


def save_temporal_feature_outputs(
    output_dir: Path,
    train_block: pd.DataFrame,
    test_block: pd.DataFrame | None,
    metadata: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_block.to_csv(output_dir / "X_train_temporal_features.csv", index=False, encoding="utf-8-sig")
    if test_block is not None:
        test_block.to_csv(output_dir / "X_test_temporal_features.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(output_dir / "temporal_feature_metadata.csv", index=False, encoding="utf-8-sig")
