from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_preprocess import (
    DATA_DIR,
    ID_COL,
    TARGET_COL,
    TEST_FILE,
    TRAIN_FILE,
    apply_imputation_plan,
    build_imputation_plan,
    compute_missing_by_label,
    read_csv,
    resolve_default_drop_cols,
    safe_median,
    safe_mode,
)
from long_tail_features import (
    DEFAULT_SKEW_THRESHOLD,
    LONG_TAIL_FAMILY,
    build_long_tail_feature_block,
    discover_long_tail_columns,
    save_long_tail_outputs,
)
from medical_prior_features import (
    MEDICAL_PRIOR_FAMILY,
    build_medical_prior_block,
    save_medical_prior_outputs,
)
from run_clinical_range_ablation import run_oof_lgbm, run_oof_lr


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "core_feature_block_ablation_results"
DEFAULT_FEATURE_OUTPUT_DIR = ROOT / "preprocessed_outputs_v2" / "core_feature_blocks"


@dataclass
class FeatureVariant:
    name: str
    x: pd.DataFrame
    feature_family: str
    feature_count: int
    comparison_base: str


def read_and_drop_defaults(data_dir: Path, train_file: str, test_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = read_csv(data_dir / train_file)
    test_raw = read_csv(data_dir / test_file)
    dropped = resolve_default_drop_cols([c for c in train_raw.columns if c != TARGET_COL])
    if dropped:
        train_raw = train_raw.drop(columns=dropped)
        test_raw = test_raw.drop(columns=[c for c in dropped if c in test_raw.columns])
    return train_raw, test_raw


def build_simple_impute_matrix(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fill_rows = []
    train_out = train_raw[feature_cols].copy()
    test_out = test_raw[feature_cols].copy()
    for col in feature_cols:
        is_binary = train_raw[col].dropna().nunique() <= 2
        if is_binary:
            fill_value = safe_mode(train_raw[col])
            strategy = "mode"
        else:
            fill_value = safe_median(train_raw[col])
            strategy = "median"
        train_out[col] = train_out[col].fillna(fill_value)
        test_out[col] = test_out[col].fillna(fill_value)
        fill_rows.append(
            {
                "feature": col,
                "strategy": strategy,
                "fill_value": fill_value,
                "is_binary": bool(is_binary),
                "missing_rate": float(train_raw[col].isna().mean()),
            }
        )
    return train_out, test_out, pd.DataFrame(fill_rows)


def build_missing_abc_matrix(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    _, missing_diff = compute_missing_by_label(train_raw, feature_cols)
    imputation_plan = build_imputation_plan(train_raw, feature_cols, missing_diff)

    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        out = df[feature_cols].copy()
        for row in imputation_plan.itertuples(index=False):
            if row.add_missing_indicator and row.feature in out.columns:
                out[row.missing_indicator_col] = out[row.feature].isna().astype(float)
        return out

    train_out = apply_imputation_plan(add_indicators(train_raw), imputation_plan)
    test_out = apply_imputation_plan(add_indicators(test_raw), imputation_plan)
    added = [row.missing_indicator_col for row in imputation_plan.itertuples(index=False) if row.add_missing_indicator]
    return train_out, test_out, imputation_plan, added


def build_variants(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_output_dir: Path,
    skew_threshold: float,
) -> tuple[list[FeatureVariant], pd.Series, dict[str, pd.DataFrame]]:
    feature_cols = [c for c in train_raw.columns if c not in [ID_COL, TARGET_COL]]
    y = train_raw[TARGET_COL].astype(int)
    train_medians = train_raw[feature_cols].median()

    simple_train, simple_test, simple_plan = build_simple_impute_matrix(train_raw, test_raw, feature_cols)
    abc_train, abc_test, abc_plan, abc_indicators = build_missing_abc_matrix(train_raw, test_raw, feature_cols)

    long_tail_plan = discover_long_tail_columns(train_raw, feature_cols, skew_threshold=skew_threshold)
    long_tail_train, long_tail_meta = build_long_tail_feature_block(train_raw, long_tail_plan)
    long_tail_test, _ = build_long_tail_feature_block(test_raw, long_tail_plan)

    prior_train, prior_meta = build_medical_prior_block(train_raw, median_values=train_medians)
    prior_test, _ = build_medical_prior_block(test_raw, median_values=train_medians)

    feature_output_dir.mkdir(parents=True, exist_ok=True)
    simple_plan.to_csv(feature_output_dir / "simple_imputation_plan.csv", index=False, encoding="utf-8-sig")
    abc_plan.to_csv(feature_output_dir / "missing_abc_imputation_plan.csv", index=False, encoding="utf-8-sig")
    pd.Series(abc_indicators, name="feature").to_csv(
        feature_output_dir / "missing_abc_indicator_features.csv", index=False, encoding="utf-8-sig"
    )
    save_long_tail_outputs(feature_output_dir / "long_tail", long_tail_train, long_tail_test, long_tail_plan, long_tail_meta)
    save_medical_prior_outputs(feature_output_dir / "medical_prior", prior_train, prior_test, prior_meta)

    variants = [
        FeatureVariant(
            name="simple_impute",
            x=simple_train,
            feature_family="simple_median_mode_imputation",
            feature_count=simple_train.shape[1],
            comparison_base="none",
        ),
        FeatureVariant(
            name="missing_abc",
            x=abc_train,
            feature_family="missing_abc_imputation_and_indicators",
            feature_count=abc_train.shape[1],
            comparison_base="simple_impute",
        ),
        FeatureVariant(
            name="missing_abc_plus_long_tail",
            x=pd.concat([abc_train.reset_index(drop=True), long_tail_train.reset_index(drop=True)], axis=1),
            feature_family=f"missing_abc_plus_{LONG_TAIL_FAMILY}",
            feature_count=abc_train.shape[1] + long_tail_train.shape[1],
            comparison_base="missing_abc",
        ),
        FeatureVariant(
            name="missing_abc_plus_medical_prior",
            x=pd.concat([abc_train.reset_index(drop=True), prior_train.reset_index(drop=True)], axis=1),
            feature_family=f"missing_abc_plus_{MEDICAL_PRIOR_FAMILY}",
            feature_count=abc_train.shape[1] + prior_train.shape[1],
            comparison_base="missing_abc",
        ),
    ]

    artifacts = {
        "simple_plan": simple_plan,
        "abc_plan": abc_plan,
        "long_tail_plan": long_tail_plan,
        "long_tail_metadata": long_tail_meta,
        "medical_prior_metadata": prior_meta,
    }
    return variants, y, artifacts


def select_best_lr(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    lr_rows = summary[summary["model"].str.startswith("lr_")].copy()
    ascending = metric == "log_loss"
    return (
        lr_rows.sort_values(["model", "variant", metric], ascending=[True, True, ascending])
        .groupby(["model", "variant"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    best_lr: pd.DataFrame,
    artifacts: dict[str, pd.DataFrame],
    n_folds: int,
    c_grid: list[float],
) -> None:
    def delta_table(model: str) -> pd.DataFrame:
        if model == "lgbm_balanced":
            rows = summary[summary["model"] == model].copy()
            key_cols = ["variant", "log_loss", "macro_auc", "macro_f1", "balanced_accuracy", "minority_recall"]
        else:
            rows = best_lr[best_lr["model"] == model].copy()
            key_cols = ["variant", "C", "log_loss", "macro_auc", "macro_f1", "balanced_accuracy", "minority_recall"]
        indexed = rows.set_index("variant")
        pairs = [
            ("missing_abc", "simple_impute", "missing_abc - simple_impute"),
            ("missing_abc_plus_long_tail", "missing_abc", "long_tail block"),
            ("missing_abc_plus_medical_prior", "missing_abc", "medical_prior block"),
        ]
        delta_rows = []
        for plus, base, label in pairs:
            if plus not in indexed.index or base not in indexed.index:
                continue
            delta_rows.append(
                {
                    "comparison": label,
                    "log_loss_delta": float(indexed.loc[plus, "log_loss"] - indexed.loc[base, "log_loss"]),
                    "macro_auc_delta": float(indexed.loc[plus, "macro_auc"] - indexed.loc[base, "macro_auc"]),
                    "macro_f1_delta": float(indexed.loc[plus, "macro_f1"] - indexed.loc[base, "macro_f1"]),
                    "minority_recall_delta": float(
                        indexed.loc[plus, "minority_recall"] - indexed.loc[base, "minority_recall"]
                    ),
                }
            )
        return pd.DataFrame(delta_rows), rows[key_cols]

    lr_none_delta, lr_none_best = delta_table("lr_none")
    lr_bal_delta, lr_bal_best = delta_table("lr_balanced")
    lgb_delta, lgb_rows = delta_table("lgbm_balanced")

    lines = [
        "# Core Feature Block Ablation",
        "",
        "本实验把三块前置特征处理拆开验证：三分类缺失处理、偏度驱动长尾三版本、医学先验特征块。",
        "",
        "## Setup",
        "",
        f"- OOF folds: {n_folds}",
        f"- LR C grid: `{c_grid}`",
        "- Models: LR without class weight, LR balanced, LightGBM balanced",
        "- Base comparison for missing block: `simple_impute`",
        "- Base comparison for long-tail and medical-prior blocks: `missing_abc`",
        "",
        "## Feature Blocks",
        "",
        f"- Missing A/B/C indicators: `{int(artifacts['abc_plan']['add_missing_indicator'].sum())}` generated indicators",
        f"- Long-tail skewed source features: `{len(artifacts['long_tail_plan'])}`; generated features: `{len(artifacts['long_tail_metadata'])}`",
        f"- Medical-prior generated features: `{len(artifacts['medical_prior_metadata'])}`, all tagged with `prior__`",
        "",
        "## LR None Best Results",
        "",
        lr_none_best.to_markdown(index=False),
        "",
        "### LR None Deltas",
        "",
        lr_none_delta.to_markdown(index=False),
        "",
        "## LR Balanced Best Results",
        "",
        lr_bal_best.to_markdown(index=False),
        "",
        "### LR Balanced Deltas",
        "",
        lr_bal_delta.to_markdown(index=False),
        "",
        "## LightGBM Balanced Results",
        "",
        lgb_rows.to_markdown(index=False),
        "",
        "### LightGBM Deltas",
        "",
        lgb_delta.to_markdown(index=False),
        "",
        "## Medical Prior Feature Tags",
        "",
        artifacts["medical_prior_metadata"][
            ["feature", "formula", "feature_family", "rationale"]
        ].to_markdown(index=False),
        "",
        "## Long-Tail Plan",
        "",
        artifacts["long_tail_plan"].head(30).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "这些结果用于判断每个模块是否应默认进入最终 pipeline。医学先验现在以 `prior__` 前缀作为独立可插拔特征块输出，不再只能依赖主预处理里的未标记列。",
        "",
        "## Full Grid",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (output_dir / "core_feature_block_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate missing A/B/C, long-tail, and medical-prior feature blocks.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--train-file", default=TRAIN_FILE)
    parser.add_argument("--test-file", default=TEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-output-dir", type=Path, default=DEFAULT_FEATURE_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--skew-threshold", type=float, default=DEFAULT_SKEW_THRESHOLD)
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "balanced_accuracy", "minority_recall", "log_loss"],
        default="macro_f1",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = args.output_dir / "oof_predictions"
    oof_dir.mkdir(exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building feature variants...", flush=True)
    train_raw, test_raw = read_and_drop_defaults(args.data_dir, args.train_file, args.test_file)
    variants, y, artifacts = build_variants(
        train_raw=train_raw,
        test_raw=test_raw,
        feature_output_dir=args.feature_output_dir,
        skew_threshold=args.skew_threshold,
    )

    rows = []
    for variant in variants:
        print(f"\n[{time.strftime('%H:%M:%S')}] Variant {variant.name}: {variant.x.shape[1]} features", flush=True)
        for model_name, class_weight in [("lr_none", None), ("lr_balanced", "balanced")]:
            for c_value in args.c_grid:
                print(f"  {model_name}, C={c_value}", flush=True)
                metrics, oof = run_oof_lr(
                    x=variant.x,
                    y=y,
                    c_value=c_value,
                    class_weight=class_weight,
                    n_folds=args.n_folds,
                    seed=args.seed,
                )
                rows.append(
                    {
                        "model": model_name,
                        "variant": variant.name,
                        "feature_family": variant.feature_family,
                        "comparison_base": variant.comparison_base,
                        "feature_count": int(variant.feature_count),
                        "C": float(c_value),
                        **metrics,
                    }
                )
                np.save(oof_dir / f"{model_name}_{variant.name}_C{str(c_value).replace('.', 'p')}_oof.npy", oof)
                print(
                    f"    macro_f1={metrics['macro_f1']:.6f}, "
                    f"minority_recall={metrics['minority_recall']:.6f}, "
                    f"log_loss={metrics['log_loss']:.6f}",
                    flush=True,
                )

        print("  lgbm_balanced", flush=True)
        metrics, oof = run_oof_lgbm(variant.x, y, n_folds=args.n_folds, seed=args.seed)
        rows.append(
            {
                "model": "lgbm_balanced",
                "variant": variant.name,
                "feature_family": variant.feature_family,
                "comparison_base": variant.comparison_base,
                "feature_count": int(variant.feature_count),
                "C": np.nan,
                **metrics,
            }
        )
        np.save(oof_dir / f"lgbm_balanced_{variant.name}_oof.npy", oof)
        print(
            f"    macro_f1={metrics['macro_f1']:.6f}, "
            f"minority_recall={metrics['minority_recall']:.6f}, "
            f"log_loss={metrics['log_loss']:.6f}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "core_feature_block_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    best_lr = select_best_lr(summary, args.selection_metric)
    best_lr.to_csv(args.output_dir / "core_feature_block_ablation_best_lr.csv", index=False, encoding="utf-8-sig")
    config = {
        "data_dir": str(args.data_dir),
        "train_file": args.train_file,
        "test_file": args.test_file,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "c_grid": args.c_grid,
        "skew_threshold": args.skew_threshold,
        "selection_metric": args.selection_metric,
    }
    (args.output_dir / "core_feature_block_ablation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir, summary, best_lr, artifacts, args.n_folds, args.c_grid)

    print("\nBest LR by variant:")
    print(best_lr.to_string(index=False))
    print("\nLightGBM rows:")
    print(summary[summary["model"] == "lgbm_balanced"].to_string(index=False))
    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
