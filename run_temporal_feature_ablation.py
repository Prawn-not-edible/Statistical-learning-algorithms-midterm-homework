from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_preprocess import DATA_DIR, TEST_FILE, TRAIN_FILE, preprocess_baseline_data, read_csv
from run_clinical_range_ablation import run_oof_lgbm, run_oof_lr
from temporal_features import (
    TEMPORAL_FEATURE_FAMILY,
    TEMPORAL_FEATURE_UNKNOWN_VALUE,
    build_temporal_feature_block,
    impute_temporal_feature_block,
    save_temporal_feature_outputs,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "temporal_feature_ablation_results"
DEFAULT_FEATURE_OUTPUT_DIR = ROOT / "preprocessed_outputs_v2" / "temporal_features"


@dataclass
class FeatureVariant:
    name: str
    x: pd.DataFrame
    feature_family: str
    feature_count: int


def build_feature_variants(
    base_x: pd.DataFrame,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    feature_output_dir: Path,
    lr_unknown_value: float,
) -> tuple[list[FeatureVariant], list[FeatureVariant], pd.DataFrame]:
    temporal_train, metadata = build_temporal_feature_block(raw_train)
    temporal_test, _ = build_temporal_feature_block(raw_test)
    save_temporal_feature_outputs(feature_output_dir, temporal_train, temporal_test, metadata)

    temporal_train_lr = impute_temporal_feature_block(temporal_train, unknown_value=lr_unknown_value)
    base_plus_lr = pd.concat([base_x.reset_index(drop=True), temporal_train_lr.reset_index(drop=True)], axis=1)
    base_plus_lgbm = pd.concat([base_x.reset_index(drop=True), temporal_train.reset_index(drop=True)], axis=1)

    lr_variants = [
        FeatureVariant("base", base_x, "base_preprocessed_features", base_x.shape[1]),
        FeatureVariant("temporal_only", temporal_train_lr, TEMPORAL_FEATURE_FAMILY, temporal_train_lr.shape[1]),
        FeatureVariant(
            "base_plus_temporal",
            base_plus_lr,
            "base_plus_cardiovascular_temporal_features",
            base_plus_lr.shape[1],
        ),
    ]
    lgbm_variants = [
        FeatureVariant("base", base_x, "base_preprocessed_features", base_x.shape[1]),
        FeatureVariant("temporal_only", temporal_train, TEMPORAL_FEATURE_FAMILY, temporal_train.shape[1]),
        FeatureVariant(
            "base_plus_temporal",
            base_plus_lgbm,
            "base_plus_cardiovascular_temporal_features",
            base_plus_lgbm.shape[1],
        ),
    ]
    return lr_variants, lgbm_variants, metadata


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
    metadata: pd.DataFrame,
    n_folds: int,
    c_grid: list[float],
    lr_unknown_value: float,
) -> None:
    def best_delta(model: str, metric: str) -> float:
        rows = best_lr[best_lr["model"] == model].set_index("variant")
        if "base" not in rows.index or "base_plus_temporal" not in rows.index:
            return float("nan")
        return float(rows.loc["base_plus_temporal", metric] - rows.loc["base", metric])

    lgb_rows = summary[summary["model"] == "lgbm_balanced"].set_index("variant")
    lgb_delta_macro = float(lgb_rows.loc["base_plus_temporal", "macro_f1"] - lgb_rows.loc["base", "macro_f1"])
    lgb_delta_loss = float(lgb_rows.loc["base_plus_temporal", "log_loss"] - lgb_rows.loc["base", "log_loss"])
    lgb_delta_auc = float(lgb_rows.loc["base_plus_temporal", "macro_auc"] - lgb_rows.loc["base", "macro_auc"])

    lines = [
        "# Temporal Cardiovascular Feature Ablation",
        "",
        "本实验把“慢性累积 + 当前状态 + 急性触发”的医学时序逻辑显式编码为可插拔特征块。",
        "",
        "## Setup",
        "",
        f"- OOF folds: {n_folds}",
        f"- LR C grid: `{c_grid}`",
        f"- LR temporal NaN imputation: `{lr_unknown_value}`",
        f"- Feature family: `{TEMPORAL_FEATURE_FAMILY}`",
        "",
        "## Temporal Feature Metadata",
        "",
        metadata[["feature", "formula", "missing_rate", "non_missing_count", "median", "max"]].to_markdown(index=False),
        "",
        "## Best LR Results",
        "",
        best_lr.to_markdown(index=False),
        "",
        "## LightGBM Results",
        "",
        summary[summary["model"] == "lgbm_balanced"].to_markdown(index=False),
        "",
        "## Main Deltas",
        "",
        f"- LR none Macro F1 delta: {best_delta('lr_none', 'macro_f1'):+.6f}",
        f"- LR none Log loss delta: {best_delta('lr_none', 'log_loss'):+.6f}",
        f"- LR balanced Macro F1 delta: {best_delta('lr_balanced', 'macro_f1'):+.6f}",
        f"- LR balanced Log loss delta: {best_delta('lr_balanced', 'log_loss'):+.6f}",
        f"- LightGBM balanced Macro F1 delta: {lgb_delta_macro:+.6f}",
        f"- LightGBM balanced Macro AUC delta: {lgb_delta_auc:+.6f}",
        f"- LightGBM balanced Log loss delta: {lgb_delta_loss:+.6f}",
        "",
        "## Interpretation",
        "",
        "这些特征包含发病年龄、早发冠心病、慢性暴露负荷、急慢性肾脏打击以及戒断残余风险。它们主要测试线性模型是否能从医学非线性映射中受益，同时观察树模型是否也能从显式代数关系中获得额外增益。",
        "",
        "## Full Grid",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (output_dir / "temporal_feature_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate cardiovascular temporal features.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--train-file", default=TRAIN_FILE)
    parser.add_argument("--test-file", default=TEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-output-dir", type=Path, default=DEFAULT_FEATURE_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--lr-unknown-value", type=float, default=TEMPORAL_FEATURE_UNKNOWN_VALUE)
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

    print(f"[{time.strftime('%H:%M:%S')}] Loading baseline preprocessing...", flush=True)
    bundle = preprocess_baseline_data(data_dir=args.data_dir, train_file=args.train_file, test_file=args.test_file)
    raw_test = read_csv(args.data_dir / args.test_file)
    if bundle.dropped_default_cols:
        raw_test = raw_test.drop(columns=[c for c in bundle.dropped_default_cols if c in raw_test.columns])

    lr_variants, lgbm_variants, metadata = build_feature_variants(
        base_x=bundle.X_train.copy(),
        raw_train=bundle.train_raw.copy(),
        raw_test=raw_test,
        feature_output_dir=args.feature_output_dir,
        lr_unknown_value=args.lr_unknown_value,
    )
    y = bundle.y_train.copy()

    rows = []
    for variant in lr_variants:
        print(f"\n[{time.strftime('%H:%M:%S')}] LR variant {variant.name}: {variant.x.shape[1]} features", flush=True)
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

    for variant in lgbm_variants:
        print(f"\n[{time.strftime('%H:%M:%S')}] LightGBM variant {variant.name}: {variant.x.shape[1]} features", flush=True)
        metrics, oof = run_oof_lgbm(variant.x, y, n_folds=args.n_folds, seed=args.seed)
        rows.append(
            {
                "model": "lgbm_balanced",
                "variant": variant.name,
                "feature_family": variant.feature_family,
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
    summary.to_csv(args.output_dir / "temporal_feature_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(args.output_dir / "temporal_feature_metadata.csv", index=False, encoding="utf-8-sig")
    best_lr = select_best_lr(summary, args.selection_metric)
    best_lr.to_csv(args.output_dir / "temporal_feature_ablation_best_lr.csv", index=False, encoding="utf-8-sig")

    config = {
        "data_dir": str(args.data_dir),
        "train_file": args.train_file,
        "test_file": args.test_file,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "c_grid": args.c_grid,
        "lr_unknown_value": args.lr_unknown_value,
        "selection_metric": args.selection_metric,
        "feature_family": TEMPORAL_FEATURE_FAMILY,
    }
    (args.output_dir / "temporal_feature_ablation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir, summary, best_lr, metadata, args.n_folds, args.c_grid, args.lr_unknown_value)

    print("\nBest LR by variant:")
    print(best_lr.to_string(index=False))
    print("\nLightGBM rows:")
    print(summary[summary["model"] == "lgbm_balanced"].to_string(index=False))
    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
