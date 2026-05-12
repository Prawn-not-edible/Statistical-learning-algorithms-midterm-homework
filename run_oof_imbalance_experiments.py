import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold

from baseline_preprocess import preprocess_baseline_data
from run_imbalance_experiments import (
    align_proba,
    fit_predict_tabpfn,
    generate_synthetic_class1,
    make_tabpfn,
    score,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "oof_imbalance_experiment_results"
DEFAULT_TABPFN_MODEL_PATH = ROOT / "tabpfn-v2.6-classifier-v2.6_default.ckpt"
RANDOM_STATE = 42


def normalize_rows(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    row_sum = proba.sum(axis=1, keepdims=True)
    np.divide(proba, row_sum, out=proba, where=row_sum > 0)
    return proba


def build_target_priors(n_classes: int) -> dict[str, np.ndarray]:
    if n_classes != 6:
        raise ValueError("This experiment expects 6 classes.")

    target_priors = {
        "prior_uniform": np.ones(n_classes, dtype=float) / n_classes,
        "prior_manual_class1_030": np.array([0.10, 0.30, 0.15, 0.15, 0.15, 0.15], dtype=float),
    }
    for w1 in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        target_priors[f"prior_grid_class1_{w1:.2f}"] = np.array(
            [0.60 - w1, w1, 0.10, 0.10, 0.10, 0.10],
            dtype=float,
        )
    return target_priors


def correct_proba_with_prior(proba, train_y, labels, target_prior):
    counts = np.array([(train_y == label).sum() for label in labels], dtype=float)
    train_prior = counts / counts.sum()
    corrected = proba / (train_prior + 1e-12) * target_prior
    return normalize_rows(corrected)


def predict_from_proba(proba, labels):
    return np.asarray(labels)[np.argmax(proba, axis=1)]


def evaluate_oof_experiments(X, y, labels, args):
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
    n_rows = len(y)
    n_classes = len(labels)
    y_array = y.to_numpy()

    target_priors = build_target_priors(n_classes)
    synthetic_counts = args.synthetic_counts
    stage1_weights = args.stage1_weights
    stage1_thresholds = args.stage1_thresholds

    oof_base_proba = np.zeros((n_rows, n_classes), dtype=float)
    oof_prior_proba = {name: np.zeros((n_rows, n_classes), dtype=float) for name in target_priors}
    oof_synthetic_proba = {n: np.zeros((n_rows, n_classes), dtype=float) for n in synthetic_counts}
    oof_stage_proba = {w: np.zeros((n_rows, n_classes), dtype=float) for w in stage1_weights}
    oof_stage_pred = {
        (w, t): np.zeros(n_rows, dtype=int)
        for w in stage1_weights
        for t in stage1_thresholds
    }
    fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_array), start=1):
        print(f"\n===== Fold {fold_idx}/{args.n_splits} =====")
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_train = pd.Series(y_array[train_idx]).reset_index(drop=True)
        y_val = pd.Series(y_array[val_idx]).reset_index(drop=True)

        fold_counts = y_val.value_counts().sort_index().to_dict()
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                **{f"val_label_{label}_count": int(fold_counts.get(label, 0)) for label in labels},
            }
        )

        print("Running base TabPFN ...")
        _, base_proba, _ = fit_predict_tabpfn(
            X_train,
            y_train,
            X_val,
            labels,
            args.tabpfn_model_path,
            args.tabpfn_device,
            args.tabpfn_estimators,
        )
        oof_base_proba[val_idx] = base_proba

        print("Applying fold-wise prior correction ...")
        for name, target_prior in target_priors.items():
            oof_prior_proba[name][val_idx] = correct_proba_with_prior(
                base_proba,
                y_train.to_numpy(),
                labels,
                target_prior,
            )

        print("Running synthetic Class 1 augmentation ...")
        for n_synth in synthetic_counts:
            rng = np.random.default_rng(RANDOM_STATE + fold_idx * 1000 + n_synth)
            X_synth = generate_synthetic_class1(X_train, y_train, n_synth, rng)
            y_synth = pd.Series(np.ones(n_synth, dtype=int))
            X_aug = pd.concat([X_train, X_synth], ignore_index=True)
            y_aug = pd.concat([y_train, y_synth], ignore_index=True)
            _, synth_proba, _ = fit_predict_tabpfn(
                X_aug,
                y_aug,
                X_val,
                labels,
                args.tabpfn_model_path,
                args.tabpfn_device,
                args.tabpfn_estimators,
            )
            oof_synthetic_proba[n_synth][val_idx] = synth_proba

        print("Running two-stage LightGBM + TabPFN ...")
        stage2_mask = y_train.to_numpy() > 0
        stage2_model = make_tabpfn(args.tabpfn_model_path, args.tabpfn_device, args.tabpfn_estimators)
        stage2_model.fit(X_train[stage2_mask].reset_index(drop=True), y_train[stage2_mask].reset_index(drop=True))
        stage2_proba = stage2_model.predict_proba(X_val)
        stage2_full_proba = align_proba(stage2_proba, stage2_model.classes_, labels)
        stage2_pred = predict_from_proba(stage2_full_proba, labels)

        y_train_binary = (y_train.to_numpy() > 0).astype(int)
        for weight in stage1_weights:
            clf = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                class_weight={0: 1, 1: weight},
                random_state=RANDOM_STATE,
                verbosity=-1,
            )
            clf.fit(X_train, y_train_binary)
            p_abnormal = clf.predict_proba(X_val)[:, 1]

            mixture_proba = np.zeros((len(val_idx), n_classes), dtype=float)
            mixture_proba[:, 0] = 1.0 - p_abnormal
            mixture_proba[:, 1:] = stage2_full_proba[:, 1:] * p_abnormal.reshape(-1, 1)
            mixture_proba = normalize_rows(mixture_proba)
            oof_stage_proba[weight][val_idx] = mixture_proba

            for thresh in stage1_thresholds:
                pred = np.zeros(len(val_idx), dtype=int)
                abnormal_mask = p_abnormal >= thresh
                pred[abnormal_mask] = stage2_pred[abnormal_mask]
                oof_stage_pred[(weight, thresh)][val_idx] = pred

    return {
        "fold_summary": pd.DataFrame(fold_rows),
        "base_proba": oof_base_proba,
        "prior_proba": oof_prior_proba,
        "synthetic_proba": oof_synthetic_proba,
        "stage_proba": oof_stage_proba,
        "stage_pred": oof_stage_pred,
    }


def summarize_results(oof, y, labels, args):
    y_true = y.to_numpy()
    rows = []
    reports = {}

    base_pred = predict_from_proba(oof["base_proba"], labels)
    base_metrics = score(y_true, base_pred, oof["base_proba"], labels)
    rows.append({"experiment": "tabpfn_oof_base", **{k: v for k, v in base_metrics.items() if k != "classification_report"}})
    reports["tabpfn_oof_base"] = base_metrics["classification_report"]

    prior_rows = []
    for name, proba in oof["prior_proba"].items():
        pred = predict_from_proba(proba, labels)
        metrics = score(y_true, pred, proba, labels)
        prior_rows.append({"experiment": name, **{k: v for k, v in metrics.items() if k != "classification_report"}})
    prior_df = pd.DataFrame(prior_rows).sort_values("macro_f1", ascending=False)
    best_prior = prior_df.iloc[0].to_dict()
    rows.append({"experiment": f"best_prior:{best_prior['experiment']}", **{k: best_prior[k] for k in ["log_loss", "macro_f1", "accuracy", "class1_recall", "minority_recall"]}})
    best_prior_pred = predict_from_proba(oof["prior_proba"][best_prior["experiment"]], labels)
    reports[f"best_prior:{best_prior['experiment']}"] = score(
        y_true,
        best_prior_pred,
        oof["prior_proba"][best_prior["experiment"]],
        labels,
    )["classification_report"]

    synth_rows = []
    for n_synth, proba in oof["synthetic_proba"].items():
        pred = predict_from_proba(proba, labels)
        metrics = score(y_true, pred, proba, labels)
        synth_rows.append(
            {
                "experiment": f"synthetic_class1_{n_synth}",
                "synthetic_class1_count": n_synth,
                **{k: v for k, v in metrics.items() if k != "classification_report"},
            }
        )
    synth_df = pd.DataFrame(synth_rows).sort_values("macro_f1", ascending=False)
    best_synth = synth_df.iloc[0].to_dict()
    rows.append({"experiment": f"best_synthetic:{best_synth['experiment']}", **{k: best_synth[k] for k in ["log_loss", "macro_f1", "accuracy", "class1_recall", "minority_recall"]}})
    best_synth_key = int(best_synth["synthetic_class1_count"])
    best_synth_pred = predict_from_proba(oof["synthetic_proba"][best_synth_key], labels)
    reports[f"best_synthetic:{best_synth['experiment']}"] = score(
        y_true,
        best_synth_pred,
        oof["synthetic_proba"][best_synth_key],
        labels,
    )["classification_report"]

    stage_rows = []
    y_binary = (y_true > 0).astype(int)
    for weight in args.stage1_weights:
        proba = oof["stage_proba"][weight]
        for thresh in args.stage1_thresholds:
            pred = oof["stage_pred"][(weight, thresh)]
            metrics = score(y_true, pred, proba, labels)
            pred_binary = (pred > 0).astype(int)
            stage_rows.append(
                {
                    "experiment": f"two_stage_w{weight}_t{thresh}",
                    "stage1_pos_weight": weight,
                    "stage1_threshold": thresh,
                    "stage1_minority_recall": float(recall_score(y_binary, pred_binary, zero_division=0)),
                    "stage1_false_positives": int(((pred_binary == 1) & (y_binary == 0)).sum()),
                    "stage1_flagged": int(pred_binary.sum()),
                    **{k: v for k, v in metrics.items() if k != "classification_report"},
                }
            )
    stage_df = pd.DataFrame(stage_rows).sort_values("macro_f1", ascending=False)
    best_stage = stage_df.iloc[0].to_dict()
    rows.append({"experiment": f"best_two_stage:{best_stage['experiment']}", **{k: best_stage[k] for k in ["log_loss", "macro_f1", "accuracy", "class1_recall", "minority_recall"]}})
    best_stage_weight = int(best_stage["stage1_pos_weight"])
    best_stage_thresh = float(best_stage["stage1_threshold"])
    reports[f"best_two_stage:{best_stage['experiment']}"] = score(
        y_true,
        oof["stage_pred"][(best_stage_weight, best_stage_thresh)],
        oof["stage_proba"][best_stage_weight],
        labels,
    )["classification_report"]

    return pd.DataFrame(rows), prior_df, synth_df, stage_df, reports


def write_oof_report(output_dir, final_df, fold_df, prior_df, synth_df, stage_df, reports, args):
    report_path = output_dir / "OOF交叉验证不平衡实验结果与分析.md"

    lines = [
        "# OOF Cross-Validation Imbalance Experiment Results and Analysis",
        "",
        "## 1. Why OOF Cross-Validation Was Added",
        "",
        "The previous hold-out validation split contained only one Class 1 sample. Under that split, Class 1 Recall can only be either 0% or 100%, which makes threshold tuning and model selection statistically unstable. This run replaces the single hold-out split with Stratified K-Fold out-of-fold predictions. Metrics are computed after concatenating all OOF predictions, so all five Class 1 samples contribute to the final evaluation.",
        "",
        f"Number of folds: {args.n_splits}",
        "",
        "## 2. Fold Label Distribution",
        "",
        "```text",
        fold_df.to_string(index=False),
        "```",
        "",
        "## 3. Final OOF Comparison",
        "",
        "```text",
        final_df.to_string(index=False),
        "```",
        "",
        "## 4. Selected Classification Reports",
        "",
    ]
    for name, report in reports.items():
        lines.extend([f"### {name}", "", "```text", report.strip(), "```", ""])

    lines.extend(
        [
            "## 5. Prior Correction OOF Scan",
            "",
            "```text",
            prior_df.to_string(index=False),
            "```",
            "",
            "## 6. Synthetic Class 1 Augmentation OOF Scan",
            "",
            "```text",
            synth_df.to_string(index=False),
            "```",
            "",
            "## 7. Two-stage OOF Scan",
            "",
            "```text",
            stage_df.to_string(index=False),
            "```",
            "",
            "## 8. Interpretation",
            "",
            "These OOF results should replace the earlier single hold-out results for model selection. The key comparison is Macro F1, but Class 1 Recall and overall minority Recall must be reported as well. If the best model still has Class 1 Recall equal to 0, then the report should explicitly state that the model improves aggregate Macro F1 but still fails to solve the rarest-class recognition problem.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_outputs(output_dir, final_df, fold_df, prior_df, synth_df, stage_df, reports, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_dir / "oof_final_model_comparison.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(output_dir / "oof_fold_summary.csv", index=False, encoding="utf-8-sig")
    prior_df.to_csv(output_dir / "oof_experiment1_prior_correction.csv", index=False, encoding="utf-8-sig")
    synth_df.to_csv(output_dir / "oof_experiment2_synthetic_class1.csv", index=False, encoding="utf-8-sig")
    stage_df.to_csv(output_dir / "oof_experiment3_two_stage.csv", index=False, encoding="utf-8-sig")

    report_dir = output_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    for name, report in reports.items():
        safe_name = name.replace(":", "_").replace("/", "_")
        (report_dir / f"{safe_name}.txt").write_text(report, encoding="utf-8")

    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stratified K-Fold OOF imbalance experiments.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tabpfn-model-path", type=Path, default=DEFAULT_TABPFN_MODEL_PATH)
    parser.add_argument("--tabpfn-device", default="auto")
    parser.add_argument("--tabpfn-estimators", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--synthetic-counts", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--stage1-weights", nargs="+", type=int, default=[50, 100, 200, 500])
    parser.add_argument(
        "--stage1-thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    bundle = preprocess_baseline_data()
    X = bundle.X_train.reset_index(drop=True)
    y = bundle.y_train.reset_index(drop=True)
    labels = sorted(y.unique().tolist())

    min_class_count = int(y.value_counts().min())
    if args.n_splits > min_class_count:
        raise ValueError(
            f"n_splits={args.n_splits} is larger than the smallest class count={min_class_count}. "
            "Reduce --n-splits."
        )

    print(f"Running {args.n_splits}-fold Stratified OOF experiments ...")
    oof = evaluate_oof_experiments(X, y, labels, args)
    final_df, prior_df, synth_df, stage_df, reports = summarize_results(oof, y, labels, args)

    config = {
        "validation": "StratifiedKFold OOF",
        "n_splits": args.n_splits,
        "tabpfn_model_path": str(args.tabpfn_model_path),
        "tabpfn_device": args.tabpfn_device,
        "tabpfn_estimators": args.tabpfn_estimators,
        "synthetic_counts": args.synthetic_counts,
        "stage1_weights": args.stage1_weights,
        "stage1_thresholds": args.stage1_thresholds,
        "runtime_seconds": round(time.time() - started, 3),
    }

    save_outputs(args.output_dir, final_df, oof["fold_summary"], prior_df, synth_df, stage_df, reports, config)
    report_path = write_oof_report(args.output_dir, final_df, oof["fold_summary"], prior_df, synth_df, stage_df, reports, args)

    print("\nFinal OOF comparison:")
    print(final_df.to_string(index=False))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
