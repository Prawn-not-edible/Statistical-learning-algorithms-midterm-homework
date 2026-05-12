import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss, recall_score
from sklearn.model_selection import train_test_split

from baseline_preprocess import preprocess_baseline_data


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "imbalance_experiment_results"
DEFAULT_TABPFN_MODEL_PATH = ROOT / "tabpfn-v2.6-classifier-v2.6_default.ckpt"
RANDOM_STATE = 42


def split_data(bundle, test_size: float):
    X = bundle.X_train.copy()
    y = bundle.y_train.copy()
    idx = np.arange(len(X))
    idx_train, idx_val, X_train, X_val, y_train, y_val = train_test_split(
        idx,
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return {
        "idx_train": idx_train,
        "idx_val": idx_val,
        "X_train": X_train.reset_index(drop=True),
        "X_val": X_val.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
    }


def align_proba(proba, model_classes, labels):
    proba = np.asarray(proba, dtype=float)
    out = np.zeros((proba.shape[0], len(labels)), dtype=float)
    label_to_pos = {label: i for i, label in enumerate(labels)}
    for src_idx, cls in enumerate(model_classes):
        if int(cls) in label_to_pos:
            out[:, label_to_pos[int(cls)]] = proba[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    np.divide(out, row_sum, out=out, where=row_sum > 0)
    return out


def score(y_true, pred, proba, labels):
    return {
        "log_loss": float(log_loss(y_true, proba, labels=labels)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "class1_recall": float(recall_score(y_true == 1, pred == 1, zero_division=0)),
        "minority_recall": float(recall_score(y_true > 0, pred > 0, zero_division=0)),
        "classification_report": classification_report(y_true, pred, digits=4, zero_division=0),
    }


def make_tabpfn(model_path: Path, device: str, n_estimators: int):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        model_path=str(model_path),
        device=device,
        n_estimators=n_estimators,
        ignore_pretraining_limits=True,
        random_state=RANDOM_STATE,
    )


def fit_predict_tabpfn(X_train, y_train, X_val, labels, model_path, device, n_estimators):
    model = make_tabpfn(model_path, device, n_estimators)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)
    proba = align_proba(proba, model.classes_, labels)
    pred = np.asarray(labels)[np.argmax(proba, axis=1)]
    return model, proba, pred


def get_or_run_base_tabpfn(split, labels, args):
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    proba_path = cache_dir / "tabpfn_base_val_proba.npy"
    pred_path = cache_dir / "tabpfn_base_val_pred.npy"

    if args.use_cache and proba_path.exists() and pred_path.exists():
        proba = np.load(proba_path)
        pred = np.load(pred_path)
        return proba, pred

    _, proba, pred = fit_predict_tabpfn(
        split["X_train"],
        split["y_train"],
        split["X_val"],
        labels,
        args.tabpfn_model_path,
        args.tabpfn_device,
        args.tabpfn_estimators,
    )
    np.save(proba_path, proba)
    np.save(pred_path, pred)
    return proba, pred


def run_prior_correction(split, labels, base_proba):
    y_train = split["y_train"].to_numpy()
    y_val = split["y_val"].to_numpy()
    counts = np.array([(y_train == label).sum() for label in labels], dtype=float)
    train_prior = counts / counts.sum()

    target_priors = {
        "tabpfn_raw": train_prior,
        "prior_uniform": np.ones(len(labels), dtype=float) / len(labels),
        "prior_manual_class1_030": np.array([0.10, 0.30, 0.15, 0.15, 0.15, 0.15], dtype=float),
    }
    for w1 in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        target_priors[f"prior_grid_class1_{w1:.2f}"] = np.array(
            [0.60 - w1, w1, 0.10, 0.10, 0.10, 0.10],
            dtype=float,
        )

    rows = []
    best = None
    eps = 1e-12
    for name, target_prior in target_priors.items():
        corrected = base_proba / (train_prior + eps) * target_prior
        corrected /= corrected.sum(axis=1, keepdims=True)
        pred = np.asarray(labels)[np.argmax(corrected, axis=1)]
        metrics = score(y_val, pred, corrected, labels)
        row = {"experiment": name, **{k: v for k, v in metrics.items() if k != "classification_report"}}
        row["target_prior"] = json.dumps(target_prior.round(6).tolist())
        rows.append(row)
        if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
            best = {"name": name, "target_prior": target_prior, "proba": corrected, "pred": pred, "metrics": metrics}

    return pd.DataFrame(rows), best, train_prior


def generate_synthetic_class1(X_train, y_train, n_samples: int, rng: np.random.Generator):
    X = X_train.reset_index(drop=True)
    y = y_train.reset_index(drop=True)
    class1 = X[y == 1].reset_index(drop=True)
    if class1.empty:
        raise ValueError("No class 1 samples in the training split.")

    global_min = X.min(axis=0)
    global_max = X.max(axis=0)
    global_std = X.std(axis=0).replace(0, 1.0).fillna(1.0)
    class1_std = class1.std(axis=0).replace(0, np.nan).fillna(global_std * 0.05)

    categorical_cols = []
    continuous_cols = []
    for col in X.columns:
        unique_count = X[col].nunique(dropna=True)
        if unique_count <= 20:
            categorical_cols.append(col)
        else:
            continuous_cols.append(col)

    rows = []
    for _ in range(n_samples):
        base = class1.iloc[int(rng.integers(0, len(class1)))].copy()
        row = base.copy()

        for col in categorical_cols:
            values = class1[col].dropna().to_numpy()
            if len(values) > 0 and rng.random() < 0.25:
                row[col] = values[int(rng.integers(0, len(values)))]

        for col in continuous_cols:
            scale = max(float(class1_std[col]) * 0.25, float(global_std[col]) * 0.02, 1e-8)
            value = float(base[col]) + float(rng.normal(0, scale))
            row[col] = np.clip(value, float(global_min[col]), float(global_max[col]))

        rows.append(row)

    return pd.DataFrame(rows, columns=X.columns)


def run_synthetic_augmentation(split, labels, args):
    rng = np.random.default_rng(RANDOM_STATE)
    y_val = split["y_val"].to_numpy()
    rows = []
    best = None

    for n_synth in args.synthetic_counts:
        X_synth = generate_synthetic_class1(split["X_train"], split["y_train"], n_synth, rng)
        y_synth = pd.Series(np.ones(n_synth, dtype=int), name=split["y_train"].name)
        X_aug = pd.concat([split["X_train"], X_synth], ignore_index=True)
        y_aug = pd.concat([split["y_train"], y_synth], ignore_index=True)

        _, proba, pred = fit_predict_tabpfn(
            X_aug,
            y_aug,
            split["X_val"],
            labels,
            args.tabpfn_model_path,
            args.tabpfn_device,
            args.tabpfn_estimators,
        )
        metrics = score(y_val, pred, proba, labels)
        row = {
            "experiment": f"synthetic_class1_{n_synth}",
            "synthetic_class1_count": n_synth,
            **{k: v for k, v in metrics.items() if k != "classification_report"},
        }
        rows.append(row)
        if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
            best = {"name": row["experiment"], "n_synth": n_synth, "proba": proba, "pred": pred, "metrics": metrics}

    return pd.DataFrame(rows), best


def run_two_stage(split, labels, args):
    y_train = split["y_train"].to_numpy()
    y_val = split["y_val"].to_numpy()
    y_train_binary = (y_train > 0).astype(int)
    y_val_binary = (y_val > 0).astype(int)

    stage2_mask = y_train > 0
    stage2_model = make_tabpfn(args.tabpfn_model_path, args.tabpfn_device, args.tabpfn_estimators)
    stage2_model.fit(split["X_train"][stage2_mask].reset_index(drop=True), pd.Series(y_train[stage2_mask]))
    stage2_proba = stage2_model.predict_proba(split["X_val"])
    stage2_full_proba = align_proba(stage2_proba, stage2_model.classes_, labels)
    stage2_pred_all = np.asarray(labels)[np.argmax(stage2_full_proba, axis=1)]

    rows = []
    best = None
    for class_weight_pos in args.stage1_weights:
        clf = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            class_weight={0: 1, 1: class_weight_pos},
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        clf.fit(split["X_train"], y_train_binary)
        p_abnormal = clf.predict_proba(split["X_val"])[:, 1]

        mixture_proba = np.zeros((len(y_val), len(labels)), dtype=float)
        mixture_proba[:, 0] = 1.0 - p_abnormal
        mixture_proba[:, 1:] = stage2_full_proba[:, 1:] * p_abnormal.reshape(-1, 1)
        mixture_proba /= mixture_proba.sum(axis=1, keepdims=True)

        for thresh in args.stage1_thresholds:
            pred = np.zeros(len(y_val), dtype=int)
            abnormal_mask = p_abnormal >= thresh
            pred[abnormal_mask] = stage2_pred_all[abnormal_mask]
            metrics = score(y_val, pred, mixture_proba, labels)
            binary_pred = (p_abnormal >= thresh).astype(int)
            row = {
                "experiment": f"two_stage_w{class_weight_pos}_t{thresh}",
                "stage1_pos_weight": class_weight_pos,
                "stage1_threshold": thresh,
                "stage1_minority_recall": float(recall_score(y_val_binary, binary_pred, zero_division=0)),
                "stage1_false_positives": int(((binary_pred == 1) & (y_val_binary == 0)).sum()),
                "stage1_flagged": int(binary_pred.sum()),
                **{k: v for k, v in metrics.items() if k != "classification_report"},
            }
            rows.append(row)
            if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
                best = {
                    "name": row["experiment"],
                    "stage1_pos_weight": class_weight_pos,
                    "stage1_threshold": thresh,
                    "proba": mixture_proba,
                    "pred": pred,
                    "metrics": metrics,
                    "row": row,
                }

    return pd.DataFrame(rows), best


def write_report(output_dir, final_rows, prior_df, synth_df, stage_df, reports):
    def table_text(df: pd.DataFrame) -> str:
        return df.to_string(index=False)

    report_path = output_dir / "不平衡处理实验结果与分析.md"
    lines = [
        "# Imbalance Experiment Results and Analysis",
        "",
        "## 1. Experiment Goal",
        "",
        "The previous five baseline models achieved high Accuracy and relatively low Log Loss, but Macro F1 remained very low. In practice, the models still failed to identify minority classes `label=1~5`. This experiment compares three imbalance-oriented strategies: TabPFN prior correction, Class 1 synthetic augmentation, and a two-stage LightGBM + TabPFN pipeline.",
        "",
        "## 2. Final Comparison",
        "",
        "```text",
        table_text(final_rows),
        "```",
        "",
        "## 3. Classification Reports of Selected Runs",
        "",
    ]
    for name, text in reports.items():
        lines.extend([f"### {name}", "", "```text", text.strip(), "```", ""])

    lines.extend(
        [
            "## 4. Experiment 1: TabPFN Prior Correction Scan",
            "",
            "This experiment replaces the original training prior with a target prior. The goal is to make TabPFN less dominated by the majority class. This can increase minority recall, but it may also introduce more false positives and worsen Log Loss.",
            "",
            "```text",
            table_text(prior_df.drop(columns=["target_prior"], errors="ignore")),
            "```",
            "",
            "## 5. Experiment 2: TabPFN with Synthetic Class 1 Augmentation",
            "",
            "This experiment augments the very rare Class 1 samples with reproducible synthetic samples, then refits TabPFN. The goal is to test whether increasing the number of Class 1 examples helps the model detect that class.",
            "",
            "```text",
            table_text(synth_df) if not synth_df.empty else "Synthetic augmentation was skipped.",
            "```",
            "",
            "## 6. Experiment 3: Two-stage LightGBM + TabPFN Scan",
            "",
            "This experiment first uses LightGBM to detect whether a sample is abnormal, i.e. `label>0`. Samples predicted as abnormal are then passed to TabPFN for minority-class refinement. The motivation is to decouple abnormality detection from fine-grained minority classification.",
            "",
            "```text",
            table_text(stage_df),
            "```",
            "",
            "## 7. Interpretation Notes",
            "",
            "For this highly imbalanced dataset, Accuracy is dominated by the majority class `label=0`. The most informative metrics are Macro F1, Class 1 Recall, and overall minority Recall. If a method improves Macro F1 but Class 1 Recall remains 0, it should be described as improving minority-class behavior only partially, not as fully solving the rare-class recognition problem.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run imbalance-focused TabPFN experiments.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tabpfn-model-path", type=Path, default=DEFAULT_TABPFN_MODEL_PATH)
    parser.add_argument("--tabpfn-device", default="auto")
    parser.add_argument("--tabpfn-estimators", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--use-cache", action="store_true")
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
    split = split_data(bundle, args.test_size)
    labels = sorted(bundle.y_train.unique().tolist())
    y_val = split["y_val"].to_numpy()

    print("Running TabPFN base probabilities ...")
    base_proba, base_pred = get_or_run_base_tabpfn(split, labels, args)
    base_metrics = score(y_val, base_pred, base_proba, labels)

    print("Running experiment 1: prior correction ...")
    prior_df, best_prior, train_prior = run_prior_correction(split, labels, base_proba)

    print("Running experiment 2: synthetic class 1 augmentation ...")
    synth_df, best_synth = run_synthetic_augmentation(split, labels, args)

    print("Running experiment 3: two-stage LightGBM + TabPFN ...")
    stage_df, best_stage = run_two_stage(split, labels, args)

    final_rows = pd.DataFrame(
        [
            {
                "experiment": "tabpfn_base",
                **{k: v for k, v in base_metrics.items() if k != "classification_report"},
            },
            {
                "experiment": f"best_prior:{best_prior['name']}",
                **{k: v for k, v in best_prior["metrics"].items() if k != "classification_report"},
            },
            {
                "experiment": f"best_synthetic:{best_synth['name']}",
                **{k: v for k, v in best_synth["metrics"].items() if k != "classification_report"},
            },
            {
                "experiment": f"best_two_stage:{best_stage['name']}",
                **{k: v for k, v in best_stage["metrics"].items() if k != "classification_report"},
            },
        ]
    )

    prior_df.to_csv(args.output_dir / "experiment1_prior_correction.csv", index=False, encoding="utf-8-sig")
    synth_df.to_csv(args.output_dir / "experiment2_synthetic_class1.csv", index=False, encoding="utf-8-sig")
    stage_df.to_csv(args.output_dir / "experiment3_two_stage.csv", index=False, encoding="utf-8-sig")
    final_rows.to_csv(args.output_dir / "final_model_comparison.csv", index=False, encoding="utf-8-sig")

    reports = {
        "tabpfn_base": base_metrics["classification_report"],
        best_prior["name"]: best_prior["metrics"]["classification_report"],
        best_synth["name"]: best_synth["metrics"]["classification_report"],
        best_stage["name"]: best_stage["metrics"]["classification_report"],
    }
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    for name, text in reports.items():
        safe_name = name.replace(":", "_").replace("/", "_")
        (report_dir / f"{safe_name}.txt").write_text(text, encoding="utf-8")

    config = {
        "tabpfn_model_path": str(args.tabpfn_model_path),
        "tabpfn_device": args.tabpfn_device,
        "tabpfn_estimators": args.tabpfn_estimators,
        "test_size": args.test_size,
        "synthetic_counts": args.synthetic_counts,
        "stage1_weights": args.stage1_weights,
        "stage1_thresholds": args.stage1_thresholds,
        "train_prior": {str(label): float(value) for label, value in zip(labels, train_prior)},
        "runtime_seconds": round(time.time() - started, 3),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = write_report(args.output_dir, final_rows, prior_df, synth_df, stage_df, reports)
    print("\nFinal comparison:")
    print(final_rows.to_string(index=False))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
