from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lgb_interaction_features import (
    INTERACTION_PREFIX,
    LGBTreePathInteractionTransformer,
    materialize_interaction_outputs,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = ROOT / "preprocessed_outputs_v2"
DEFAULT_OUTPUT_DIR = DEFAULT_PREPROCESSED_DIR / "interaction_features"


def read_processed(preprocessed_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    x_train = pd.read_csv(preprocessed_dir / "X_train_processed.csv", encoding="utf-8-sig")
    x_test = pd.read_csv(preprocessed_dir / "X_test_processed.csv", encoding="utf-8-sig")
    y_train = pd.read_csv(preprocessed_dir / "y_train.csv", encoding="utf-8-sig").iloc[:, 0].astype(int)
    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("X_train_processed.csv and X_test_processed.csv do not have aligned columns.")
    return x_train, y_train, x_test


def assert_clean_matrix(name: str, frame: pd.DataFrame) -> None:
    values = frame.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError(f"{name} contains NaN values.")
    if np.isinf(values).any():
        raise ValueError(f"{name} contains infinite values.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover LightGBM tree-path interactions and materialize them as a pluggable feature block."
    )
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-direct-missing-pairs",
        action="store_true",
        help="Keep pairs like feature and feature_missing. By default these direct companion pairs are filtered.",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=None,
        help="Reuse an existing selected_lgb_interactions.csv and only transform matrices, without fitting LightGBM.",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Only write the standalone interaction block, not X_*_with_lgb_interactions.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test = read_processed(args.preprocessed_dir)

    if args.selection_file is not None:
        transformer = LGBTreePathInteractionTransformer.from_selection_file(args.selection_file)
        train_interactions = transformer.transform(x_train)
        test_interactions = transformer.transform(x_test)
    else:
        transformer = LGBTreePathInteractionTransformer(
            top_k=args.top_k,
            n_estimators=args.n_estimators,
            random_state=args.seed,
            exclude_direct_missing_pairs=not args.include_direct_missing_pairs,
            interaction_prefix=INTERACTION_PREFIX,
        )
        train_interactions = transformer.fit_transform(x_train, y_train)
        test_interactions = transformer.transform(x_test)

    assert_clean_matrix("train interaction matrix", train_interactions)
    assert_clean_matrix("test interaction matrix", test_interactions)

    materialize_interaction_outputs(
        transformer=transformer,
        x_train=x_train,
        train_interactions=train_interactions,
        output_dir=args.output_dir,
        x_test=x_test,
        test_interactions=test_interactions,
        write_combined=not args.no_combined,
    )

    print(f"Input features: {x_train.shape[1]}")
    if hasattr(transformer, "interactions_"):
        print(f"Discovered interaction pairs: {len(transformer.interactions_)}")
    print(f"Selected explicit interactions: {len(transformer.selected_interactions_)}")
    print(f"Train interaction matrix: {train_interactions.shape}")
    print(f"Test interaction matrix: {test_interactions.shape}")
    print(f"Output directory: {args.output_dir}")
    print("Top selected interactions:")
    preview_cols = [
        col
        for col in ["interaction_id", "feature_a", "feature_b", "interaction_score", "path_count", "tree_count"]
        if col in transformer.selected_interactions_.columns
    ]
    print(transformer.selected_interactions_[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
