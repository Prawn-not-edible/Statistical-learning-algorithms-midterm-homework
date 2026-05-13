from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline_preprocess import DATA_DIR, ID_COL, TARGET_COL, TEST_FILE, TRAIN_FILE, preprocess_baseline_data, read_csv
from clinical_range_features import (
    CLINICAL_RANGE_FAMILY,
    CLINICAL_RANGE_PREFIX,
    build_clinical_indicator_block,
    impute_clinical_indicator_block,
    save_clinical_range_outputs,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "clinical_range_ablation_results"
DEFAULT_FEATURE_OUTPUT_DIR = ROOT / "preprocessed_outputs_v2" / "clinical_range_features"
ALL_LABELS = list(range(6))
EPS = 1e-12


@dataclass
class FeatureVariant:
    name: str
    x: pd.DataFrame
    feature_family: str
    feature_count: int


def align_proba(proba: Any, classes: Any, n_rows: int) -> np.ndarray:
    full = np.zeros((n_rows, len(ALL_LABELS)), dtype=float)
    arr = np.asarray(proba, dtype=float)
    for idx, label in enumerate(classes):
        full[:, int(label)] = arr[:, idx]
    full = np.clip(full, EPS, 1.0)
    full /= full.sum(axis=1, keepdims=True)
    return full


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(axis=1)
    try:
        macro_auc = roc_auc_score(
            y_true,
            proba,
            labels=ALL_LABELS,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        macro_auc = float("nan")
    metrics = {
        "log_loss": float(log_loss(y_true, proba, labels=ALL_LABELS)),
        "macro_auc": float(macro_auc),
        "macro_f1": float(f1_score(y_true, pred, labels=ALL_LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "minority_recall": float(
            recall_score(y_true, pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)
        ),
    }
    for label in ALL_LABELS:
        metrics[f"class_{label}_recall"] = float(
            recall_score(y_true, pred, labels=[label], average="macro", zero_division=0)
        )
    return metrics


def make_lr(c_value: float, class_weight: str | None):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=5000,
            class_weight=class_weight,
            random_state=42,
        ),
    )


def make_lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=len(ALL_LABELS),
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def run_oof_lr(
    x: pd.DataFrame,
    y: pd.Series,
    c_value: float,
    class_weight: str | None,
    n_folds: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    y_arr = y.to_numpy(dtype=int)
    oof = np.zeros((len(x), len(ALL_LABELS)), dtype=float)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x, y_arr), start=1):
        model = make_lr(c_value, class_weight)
        model.fit(x.iloc[train_idx], y_arr[train_idx])
        lr_step = model.named_steps["logisticregression"]
        oof[val_idx] = align_proba(model.predict_proba(x.iloc[val_idx]), lr_step.classes_, len(val_idx))
        print(f"    LR fold {fold_idx}/{n_folds} done", flush=True)
    return compute_metrics(y_arr, oof), oof


def run_oof_lgbm(
    x: pd.DataFrame,
    y: pd.Series,
    n_folds: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    y_arr = y.to_numpy(dtype=int)
    oof = np.zeros((len(x), len(ALL_LABELS)), dtype=float)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x, y_arr), start=1):
        model = make_lgbm(seed + fold_idx * 1009)
        model.fit(x.iloc[train_idx], y_arr[train_idx])
        oof[val_idx] = align_proba(model.predict_proba(x.iloc[val_idx]), model.classes_, len(val_idx))
        print(f"    LGBM fold {fold_idx}/{n_folds} done", flush=True)
    return compute_metrics(y_arr, oof), oof


def build_feature_variants(
    base_x: pd.DataFrame,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    feature_output_dir: Path,
    lr_unknown_value: float,
) -> tuple[list[FeatureVariant], list[FeatureVariant], pd.DataFrame]:
    clinical_train, metadata = build_clinical_indicator_block(raw_train, keep_missing=True)
    clinical_test, _ = build_clinical_indicator_block(raw_test, keep_missing=True)
    save_clinical_range_outputs(feature_output_dir, clinical_train, clinical_test, metadata)

    clinical_train_lr = impute_clinical_indicator_block(clinical_train, unknown_value=lr_unknown_value)
    base_plus_lr = pd.concat([base_x.reset_index(drop=True), clinical_train_lr.reset_index(drop=True)], axis=1)
    base_plus_lgbm = pd.concat([base_x.reset_index(drop=True), clinical_train.reset_index(drop=True)], axis=1)

    # LightGBM can consume NaN in the clinical block, but LR cannot.
    lr_variants = [
        FeatureVariant("base", base_x, "base_preprocessed_features", base_x.shape[1]),
        FeatureVariant(
            "clinical_ranges_only",
            clinical_train_lr,
            CLINICAL_RANGE_FAMILY,
            clinical_train_lr.shape[1],
        ),
        FeatureVariant(
            "base_plus_clinical_ranges",
            base_plus_lr,
            "base_plus_clinical_reference_ranges",
            base_plus_lr.shape[1],
        ),
    ]
    lgbm_variants = [
        FeatureVariant("base", base_x, "base_preprocessed_features", base_x.shape[1]),
        FeatureVariant(
            "clinical_ranges_only",
            clinical_train,
            CLINICAL_RANGE_FAMILY,
            clinical_train.shape[1],
        ),
        FeatureVariant(
            "base_plus_clinical_ranges",
            base_plus_lgbm,
            "base_plus_clinical_reference_ranges",
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
    def metric_delta(model: str, metric: str) -> float:
        rows = best_lr[best_lr["model"] == model].set_index("variant")
        if "base" not in rows.index or "base_plus_clinical_ranges" not in rows.index:
            return float("nan")
        return float(rows.loc["base_plus_clinical_ranges", metric] - rows.loc["base", metric])

    lgb_rows = summary[summary["model"] == "lgbm_balanced"].set_index("variant")
    lgb_delta_macro = (
        float(lgb_rows.loc["base_plus_clinical_ranges", "macro_f1"] - lgb_rows.loc["base", "macro_f1"])
        if {"base", "base_plus_clinical_ranges"}.issubset(lgb_rows.index)
        else float("nan")
    )
    lgb_delta_loss = (
        float(lgb_rows.loc["base_plus_clinical_ranges", "log_loss"] - lgb_rows.loc["base", "log_loss"])
        if {"base", "base_plus_clinical_ranges"}.issubset(lgb_rows.index)
        else float("nan")
    )

    existing_metadata = metadata[metadata["exists_in_input"]].copy()
    lines = [
        "# Clinical Reference Range Feature Ablation",
        "",
        "本实验把临床正常参考范围显式编码为 `is_normal_*` 二值特征，用来验证医学先验切点是否能补充连续化验值。",
        "",
        "## Setup",
        "",
        f"- OOF folds: {n_folds}",
        f"- LR C grid: `{c_grid}`",
        f"- LR missing indicator imputation: clinical-range NaN -> `{lr_unknown_value}` for LR compatibility",
        f"- Feature prefix: `{CLINICAL_RANGE_PREFIX}`",
        f"- Feature family: `{CLINICAL_RANGE_FAMILY}`",
        "",
        "## Clinical Range Metadata",
        "",
        existing_metadata[
            [
                "source_feature",
                "clinical_lower",
                "clinical_upper",
                "indicator_feature",
                "missing_rate",
                "normal_rate_non_missing",
                "abnormal_rate_non_missing",
            ]
        ].to_markdown(index=False),
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
        f"- LR none Macro F1 delta: {metric_delta('lr_none', 'macro_f1'):+.6f}",
        f"- LR none Log loss delta: {metric_delta('lr_none', 'log_loss'):+.6f}",
        f"- LR balanced Macro F1 delta: {metric_delta('lr_balanced', 'macro_f1'):+.6f}",
        f"- LR balanced Log loss delta: {metric_delta('lr_balanced', 'log_loss'):+.6f}",
        f"- LightGBM balanced Macro F1 delta: {lgb_delta_macro:+.6f}",
        f"- LightGBM balanced Log loss delta: {lgb_delta_loss:+.6f}",
        "",
        "## Interpretation",
        "",
        "这些特征不是替代原始连续值，而是额外告诉模型该指标是否落在临床正常区间。若消融结果为正，可以把它写成“医学先验阈值把长尾连续指标压缩为可解释的异常信号”；若结果不稳定，则保留为可插拔模块，而不默认并入正式 pipeline。",
        "",
        "## Full Grid",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (output_dir / "clinical_range_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate clinical reference-range indicator features.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--train-file", default=TRAIN_FILE)
    parser.add_argument("--test-file", default=TEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-output-dir", type=Path, default=DEFAULT_FEATURE_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--lr-unknown-value", type=float, default=-1.0)
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
        print("  lgbm_balanced", flush=True)
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
    summary.to_csv(args.output_dir / "clinical_range_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(args.output_dir / "clinical_range_feature_metadata.csv", index=False, encoding="utf-8-sig")
    best_lr = select_best_lr(summary, args.selection_metric)
    best_lr.to_csv(args.output_dir / "clinical_range_ablation_best_lr.csv", index=False, encoding="utf-8-sig")

    config = {
        "data_dir": str(args.data_dir),
        "train_file": args.train_file,
        "test_file": args.test_file,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "c_grid": args.c_grid,
        "lr_unknown_value": args.lr_unknown_value,
        "selection_metric": args.selection_metric,
        "clinical_indicator_prefix": CLINICAL_RANGE_PREFIX,
    }
    (args.output_dir / "clinical_range_ablation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir=args.output_dir,
        summary=summary,
        best_lr=best_lr,
        metadata=metadata,
        n_folds=args.n_folds,
        c_grid=args.c_grid,
        lr_unknown_value=args.lr_unknown_value,
    )

    print("\nBest LR by variant:")
    print(best_lr.to_string(index=False))
    print("\nLightGBM rows:")
    print(summary[summary["model"] == "lgbm_balanced"].to_string(index=False))
    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
