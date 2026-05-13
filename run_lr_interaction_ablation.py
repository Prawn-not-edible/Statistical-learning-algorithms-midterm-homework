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

from lgb_interaction_features import LGBTreePathInteractionTransformer


ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = ROOT / "preprocessed_outputs_v2"
DEFAULT_INTERACTION_SELECTION = DEFAULT_PREPROCESSED_DIR / "interaction_features" / "selected_lgb_interactions.csv"
DEFAULT_OUTPUT_DIR = ROOT / "lr_interaction_ablation_results"
ALL_LABELS = list(range(6))
EPS = 1e-12


@dataclass
class Variant:
    name: str
    x: pd.DataFrame
    feature_family: str


def read_processed(preprocessed_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_csv(preprocessed_dir / "X_train_processed.csv", encoding="utf-8-sig")
    y = pd.read_csv(preprocessed_dir / "y_train.csv", encoding="utf-8-sig").iloc[:, 0].astype(int)
    return x, y


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


def make_lr(c_value: float, class_weight: str | None) -> Any:
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
        model = make_lr(c_value=c_value, class_weight=class_weight)
        model.fit(x.iloc[train_idx], y_arr[train_idx])
        lr_step = model.named_steps["logisticregression"]
        oof[val_idx] = align_proba(model.predict_proba(x.iloc[val_idx]), lr_step.classes_, len(val_idx))
        print(f"    fold {fold_idx}/{n_folds} done", flush=True)

    return compute_metrics(y_arr, oof), oof


def build_selected_interactions(x: pd.DataFrame, selection_file: Path) -> pd.DataFrame:
    transformer = LGBTreePathInteractionTransformer.from_selection_file(selection_file)
    return transformer.transform(x)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    best_by_variant: pd.DataFrame,
    selected_interactions: pd.DataFrame,
    n_folds: int,
    class_weight: str | None,
    c_grid: list[float],
) -> None:
    best_lookup = best_by_variant.set_index("variant")
    base = best_lookup.loc["lr_base"]
    plus = best_lookup.loc["lr_base_plus_lgb_interactions"]
    delta_macro_f1 = plus["macro_f1"] - base["macro_f1"]
    delta_bal_acc = plus["balanced_accuracy"] - base["balanced_accuracy"]
    delta_minority_recall = plus["minority_recall"] - base["minority_recall"]
    delta_log_loss = plus["log_loss"] - base["log_loss"]

    lines = [
        "# LR Interaction Ablation",
        "",
        "本实验把 Logistic Regression 作为线性基线模型，用来验证 LightGBM 路径共现发现的显式交互特征是否能被迁移到非树模型中。",
        "",
        "## Setup",
        "",
        f"- OOF folds: {n_folds}",
        f"- LR class_weight: `{class_weight}`",
        f"- LR C grid: `{c_grid}`",
        "- Scaling: `StandardScaler` before LR",
        "- Interaction source: `selected_lgb_interactions.csv`",
        "- Interaction naming: `lgb_interact__*`",
        "",
        "## Best Result By Variant",
        "",
        best_by_variant.to_markdown(index=False),
        "",
        "## Main Delta",
        "",
        f"- Macro F1: {base['macro_f1']:.6f} -> {plus['macro_f1']:.6f} (delta {delta_macro_f1:+.6f})",
        f"- Balanced accuracy: {base['balanced_accuracy']:.6f} -> {plus['balanced_accuracy']:.6f} (delta {delta_bal_acc:+.6f})",
        f"- Minority recall: {base['minority_recall']:.6f} -> {plus['minority_recall']:.6f} (delta {delta_minority_recall:+.6f})",
        f"- Log loss: {base['log_loss']:.6f} -> {plus['log_loss']:.6f} (delta {delta_log_loss:+.6f}; lower is better)",
        "",
        "## Interpretation",
        "",
        "LR 本身只能学习线性加和关系，无法自动生成 `feature_a * feature_b` 这样的交互项。因此，如果 `lr_base_plus_lgb_interactions` 相比 `lr_base` 在 Macro F1、Balanced Accuracy 或 Minority Recall 上提升，就说明树模型路径中发现的组合信号被成功显式化，并且能被线性模型利用。",
        "",
        "需要注意：当前使用的是全训练集发现出的 20 个交互特征，适合做特征工程探索和叙事验证。若要把该结果作为严格 OOF 分数汇报，交互发现应放进每个训练折内部重新执行。",
        "",
        "## Selected Interaction Features",
        "",
        selected_interactions[
            ["interaction_id", "feature_a", "feature_b", "interaction_score", "path_count", "tree_count"]
        ].to_markdown(index=False),
        "",
        "## Full Grid",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (output_dir / "lr_interaction_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Logistic Regression ablation for LGB-discovered interactions.")
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_INTERACTION_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "balanced_accuracy", "minority_recall", "log_loss"],
        default="macro_f1",
        help="Metric used to pick the best C for each variant. log_loss is minimized; other metrics are maximized.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_weight = None if args.class_weight == "none" else args.class_weight

    print(f"[{time.strftime('%H:%M:%S')}] Loading processed data...", flush=True)
    x_base, y = read_processed(args.preprocessed_dir)
    selected_interactions = pd.read_csv(args.selection_file, encoding="utf-8-sig")
    x_lgb_interactions = build_selected_interactions(x_base, args.selection_file)
    x_plus = pd.concat([x_base.reset_index(drop=True), x_lgb_interactions.reset_index(drop=True)], axis=1)

    variants = [
        Variant("lr_base", x_base, "base_preprocessed_features"),
        Variant("lr_lgb_interactions_only", x_lgb_interactions, "lgb_tree_path_interactions_only"),
        Variant("lr_base_plus_lgb_interactions", x_plus, "base_plus_lgb_tree_path_interactions"),
    ]

    rows = []
    oof_dir = args.output_dir / "oof_predictions"
    oof_dir.mkdir(exist_ok=True)
    for variant in variants:
        print(f"\n[{time.strftime('%H:%M:%S')}] Variant: {variant.name} ({variant.x.shape[1]} features)", flush=True)
        for c_value in args.c_grid:
            print(f"  C={c_value}", flush=True)
            metrics, oof = run_oof_lr(
                x=variant.x,
                y=y,
                c_value=c_value,
                class_weight=class_weight,
                n_folds=args.n_folds,
                seed=args.seed,
            )
            row = {
                "variant": variant.name,
                "feature_family": variant.feature_family,
                "feature_count": int(variant.x.shape[1]),
                "C": float(c_value),
                **metrics,
            }
            rows.append(row)
            np.save(oof_dir / f"{variant.name}_C{str(c_value).replace('.', 'p')}_oof.npy", oof)
            print(
                f"    macro_f1={metrics['macro_f1']:.6f}, "
                f"balanced_acc={metrics['balanced_accuracy']:.6f}, "
                f"minority_recall={metrics['minority_recall']:.6f}, "
                f"log_loss={metrics['log_loss']:.6f}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "lr_interaction_ablation_metrics.csv", index=False, encoding="utf-8-sig")

    ascending = args.selection_metric == "log_loss"
    best_by_variant = (
        summary.sort_values(["variant", args.selection_metric], ascending=[True, ascending])
        .groupby("variant", as_index=False)
        .head(1)
        .sort_values("variant")
        .reset_index(drop=True)
    )
    best_by_variant.to_csv(args.output_dir / "lr_interaction_ablation_best_by_variant.csv", index=False, encoding="utf-8-sig")

    selected_interactions.to_csv(args.output_dir / "selected_lgb_interactions_used.csv", index=False, encoding="utf-8-sig")
    config = {
        "preprocessed_dir": str(args.preprocessed_dir),
        "selection_file": str(args.selection_file),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "c_grid": args.c_grid,
        "class_weight": class_weight,
        "selection_metric": args.selection_metric,
        "note": "This ablation uses the selected interaction map as a pluggable feature block for LR.",
    }
    (args.output_dir / "lr_interaction_ablation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir=args.output_dir,
        summary=summary,
        best_by_variant=best_by_variant,
        selected_interactions=selected_interactions,
        n_folds=args.n_folds,
        class_weight=class_weight,
        c_grid=args.c_grid,
    )

    print("\nBest by variant:")
    print(best_by_variant.to_string(index=False))
    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
