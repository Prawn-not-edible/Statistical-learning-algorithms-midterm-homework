import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import train_test_split

from baseline_preprocess import preprocess_baseline_data


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "baseline_results"
RANDOM_STATE = 42


def get_weight_array(bundle, mode: str):
    if mode == "none":
        return None
    if mode == "raw":
        return bundle.sample_weights
    if mode == "clipped":
        if bundle.sample_weights_clipped is None:
            raise ValueError("You selected clipped weights but did not generate clipped weights.")
        return bundle.sample_weights_clipped
    raise ValueError(f"Unsupported weight mode: {mode}")


def split_data(bundle, weight_mode: str, test_size: float):
    X = bundle.X_train.copy()
    y = bundle.y_train.copy()
    weights = get_weight_array(bundle, weight_mode)

    indices = np.arange(len(X))
    split_items = [indices, X, y]
    if weights is not None:
        split_items.append(weights)

    parts = train_test_split(
        *split_items,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if weights is None:
        idx_train, idx_val, X_train, X_val, y_train, y_val = parts
        w_train = None
        w_val = None
    else:
        idx_train, idx_val, X_train, X_val, y_train, y_val, w_train, w_val = parts

    return {
        "idx_train": idx_train,
        "idx_val": idx_val,
        "X_train": X_train.reset_index(drop=True),
        "X_val": X_val.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "w_train": w_train,
        "w_val": w_val,
    }


def score_predictions(y_true, y_pred, y_pred_proba, labels):
    return {
        "log_loss": float(log_loss(y_true, y_pred_proba, labels=labels)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def ensure_proba_array(model_proba, classes):
    if isinstance(model_proba, pd.DataFrame):
        available = [c for c in classes if c in model_proba.columns]
        arr = model_proba.reindex(columns=classes, fill_value=0.0).to_numpy()
        if len(available) != len(classes):
            warnings.warn("Some class probability columns were missing and filled with 0.")
        return arr
    return np.asarray(model_proba)


def run_xgboost(split, labels):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(labels),
        n_estimators=500,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
    fit_kwargs = {}
    if split["w_train"] is not None:
        fit_kwargs["sample_weight"] = split["w_train"]
    model.fit(split["X_train"], split["y_train"], **fit_kwargs)
    proba = ensure_proba_array(model.predict_proba(split["X_val"]), labels)
    pred = model.predict(split["X_val"])
    return score_predictions(split["y_val"], pred, proba, labels)


def run_lightgbm(split, labels):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(labels),
        n_estimators=500,
        random_state=RANDOM_STATE,
    )
    fit_kwargs = {}
    if split["w_train"] is not None:
        fit_kwargs["sample_weight"] = split["w_train"]
    model.fit(split["X_train"], split["y_train"], **fit_kwargs)
    proba = ensure_proba_array(model.predict_proba(split["X_val"]), labels)
    pred = model.predict(split["X_val"])
    return score_predictions(split["y_val"], pred, proba, labels)


def run_catboost(split, labels):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=500,
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )
    fit_kwargs = {}
    if split["w_train"] is not None:
        fit_kwargs["sample_weight"] = split["w_train"]
    model.fit(split["X_train"], split["y_train"], **fit_kwargs)
    proba = ensure_proba_array(model.predict_proba(split["X_val"]), labels)
    pred = model.predict(split["X_val"]).reshape(-1)
    pred = np.asarray(pred, dtype=int)
    return score_predictions(split["y_val"], pred, proba, labels)


def run_tabpfn(split, labels, model_path: Path | None, device: str):
    from tabpfn import TabPFNClassifier

    model_kwargs = {}
    if model_path is not None:
        model_kwargs["model_path"] = str(model_path)
    if device != "auto":
        model_kwargs["device"] = device

    model = TabPFNClassifier(**model_kwargs)
    model.fit(split["X_train"], split["y_train"])
    proba = ensure_proba_array(model.predict_proba(split["X_val"]), labels)
    pred = model.predict(split["X_val"])
    return score_predictions(split["y_val"], pred, proba, labels)


def run_autogluon(split, labels, output_dir: Path, use_weights: bool):
    from autogluon.tabular import TabularPredictor

    train_df = split["X_train"].copy()
    val_df = split["X_val"].copy()
    train_df["label"] = split["y_train"].to_numpy()
    val_df["label"] = split["y_val"].to_numpy()

    predictor_kwargs = {
        "label": "label",
        "problem_type": "multiclass",
        "eval_metric": "log_loss",
        "path": str(output_dir / "autogluon_artifacts"),
    }
    if use_weights and split["w_train"] is not None:
        train_df["sample_weight"] = split["w_train"]
        val_df["sample_weight"] = split["w_val"]
        predictor_kwargs["sample_weight"] = "sample_weight"

    predictor = TabularPredictor(**predictor_kwargs)
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        presets="medium_quality",
    )

    feature_cols = [c for c in split["X_train"].columns]
    proba_df = predictor.predict_proba(split["X_val"][feature_cols])
    proba = ensure_proba_array(proba_df, labels)
    pred = predictor.predict(split["X_val"][feature_cols]).to_numpy()
    return score_predictions(split["y_val"], pred, proba, labels)


MODEL_RUNNERS = {
    "xgboost": run_xgboost,
    "lightgbm": run_lightgbm,
    "catboost": run_catboost,
    "tabpfn": run_tabpfn,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run local baseline models on the processed dataset.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "lightgbm", "catboost", "tabpfn", "autogluon"],
        help="Models to run.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--weight-mode", choices=["none", "raw", "clipped"], default="raw")
    parser.add_argument(
        "--class-weight-clip",
        type=float,
        default=None,
        help="If provided, preprocessing will also generate clipped sample weights.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--tabpfn-model-path",
        type=Path,
        default=None,
        help="Optional local TabPFN classifier checkpoint path, e.g. tabpfn_weights/tabpfn-v2.6-classifier-v2.6_default.ckpt.",
    )
    parser.add_argument(
        "--tabpfn-device",
        default="auto",
        help="Device passed to TabPFNClassifier. Use auto, cuda, cuda:0, cuda:1, or cpu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = preprocess_baseline_data(class_weight_clip=args.class_weight_clip)
    split = split_data(bundle, weight_mode=args.weight_mode, test_size=args.test_size)
    labels = sorted(bundle.y_train.unique().tolist())

    rows = []
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(exist_ok=True)

    for model_name in args.models:
        print(f"Running {model_name} ...")
        try:
            if model_name == "autogluon":
                result = run_autogluon(
                    split,
                    labels=labels,
                    output_dir=args.output_dir,
                    use_weights=args.weight_mode != "none",
                )
            elif model_name == "tabpfn":
                result = run_tabpfn(
                    split,
                    labels=labels,
                    model_path=args.tabpfn_model_path,
                    device=args.tabpfn_device,
                )
            else:
                result = MODEL_RUNNERS[model_name](split, labels)

            rows.append(
                {
                    "model": model_name,
                    "log_loss": result["log_loss"],
                    "macro_f1": result["macro_f1"],
                    "accuracy": result["accuracy"],
                }
            )
            (report_dir / f"{model_name}_classification_report.txt").write_text(
                result["classification_report"],
                encoding="utf-8",
            )
        except Exception as e:
            rows.append(
                {
                    "model": model_name,
                    "log_loss": np.nan,
                    "macro_f1": np.nan,
                    "accuracy": np.nan,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            print(f"{model_name} failed: {type(e).__name__}: {e}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": args.models,
                "test_size": args.test_size,
                "weight_mode": args.weight_mode,
                "class_weight_clip": args.class_weight_clip,
                "tabpfn_model_path": str(args.tabpfn_model_path) if args.tabpfn_model_path else None,
                "tabpfn_device": args.tabpfn_device,
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nBaseline results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
