from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin


INTERACTION_PREFIX = "lgb_interact__"
INTERACTION_FAMILY = "lgb_tree_path"
INTERACTION_SOURCE = "LightGBM tree-path co-occurrence"


@dataclass
class LGBInteractionConfig:
    top_k: int = 20
    n_estimators: int = 160
    random_state: int = 42
    learning_rate: float = 0.04
    num_leaves: int = 15
    max_depth: int = 5
    min_child_samples: int = 80
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.1
    reg_lambda: float = 2.0
    n_jobs: int = -1
    class_weight: str | dict[int, float] | None = "balanced"
    exclude_direct_missing_pairs: bool = True
    interaction_prefix: str = INTERACTION_PREFIX


def _collect_leaf_paths(
    node: dict[str, Any],
    feature_names: list[str],
    path: list[str],
    paths: list[tuple[list[str], int]],
) -> None:
    if "leaf_index" in node:
        paths.append((path.copy(), int(node.get("leaf_count", 1))))
        return

    split_feature = node.get("split_feature")
    if split_feature is not None:
        path.append(feature_names[int(split_feature)])

    left_child = node.get("left_child")
    right_child = node.get("right_child")
    if left_child is not None:
        _collect_leaf_paths(left_child, feature_names, path, paths)
    if right_child is not None:
        _collect_leaf_paths(right_child, feature_names, path, paths)

    if split_feature is not None:
        path.pop()


def _is_direct_missing_pair(a: str, b: str) -> bool:
    return a == f"{b}_missing" or b == f"{a}_missing"


def analyze_tree_paths(model: LGBMClassifier, exclude_direct_missing_pairs: bool = True) -> pd.DataFrame:
    """Count feature-pair co-occurrence along LightGBM root-to-leaf paths."""
    booster = model.booster_
    dumped = booster.dump_model()
    feature_names = booster.feature_name()

    path_count: Counter[tuple[str, str]] = Counter()
    weighted_leaf_count: Counter[tuple[str, str]] = Counter()
    depth_sum: Counter[tuple[str, str]] = Counter()
    tree_count: Counter[tuple[str, str]] = Counter()
    first_seen_tree: dict[tuple[str, str], int] = {}

    for tree_idx, tree_info in enumerate(dumped["tree_info"]):
        paths: list[tuple[list[str], int]] = []
        _collect_leaf_paths(tree_info["tree_structure"], feature_names, [], paths)

        pairs_in_tree = set()
        for path_features, leaf_count in paths:
            unique_path_features = list(dict.fromkeys(path_features))
            if len(unique_path_features) < 2:
                continue

            for a, b in combinations(sorted(unique_path_features), 2):
                if exclude_direct_missing_pairs and _is_direct_missing_pair(a, b):
                    continue
                pair = (a, b)
                path_count[pair] += 1
                weighted_leaf_count[pair] += leaf_count
                depth_sum[pair] += len(unique_path_features)
                pairs_in_tree.add(pair)
                first_seen_tree.setdefault(pair, tree_idx)

        for pair in pairs_in_tree:
            tree_count[pair] += 1

    rows = []
    for pair, count in path_count.items():
        weighted = weighted_leaf_count[pair]
        trees = tree_count[pair]
        score = count * np.log1p(trees) * np.log1p(weighted)
        rows.append(
            {
                "feature_a": pair[0],
                "feature_b": pair[1],
                "path_count": int(count),
                "tree_count": int(trees),
                "weighted_leaf_count": int(weighted),
                "avg_unique_features_in_path": float(depth_sum[pair] / count),
                "first_seen_tree": int(first_seen_tree[pair]),
                "interaction_score": float(score),
                "feature_a_missing_indicator": bool(pair[0].endswith("_missing")),
                "feature_b_missing_indicator": bool(pair[1].endswith("_missing")),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "path_count",
                "tree_count",
                "weighted_leaf_count",
                "avg_unique_features_in_path",
                "first_seen_tree",
                "interaction_score",
                "feature_a_missing_indicator",
                "feature_b_missing_indicator",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["interaction_score", "tree_count", "path_count", "weighted_leaf_count"],
        ascending=False,
    )


def make_selected_interactions(
    interactions: pd.DataFrame,
    top_k: int,
    interaction_prefix: str = INTERACTION_PREFIX,
) -> pd.DataFrame:
    selected = interactions.head(top_k).copy().reset_index(drop=True)
    selected.insert(0, "interaction_id", [f"{interaction_prefix}{i + 1:03d}" for i in range(len(selected))])
    selected["generated_feature"] = selected["interaction_id"]
    selected["interaction_family"] = INTERACTION_FAMILY
    selected["operation"] = "product"
    selected["formula"] = selected["interaction_id"] + " = " + selected["feature_a"] + " * " + selected["feature_b"]
    selected["source"] = INTERACTION_SOURCE
    return selected


def build_interaction_matrix(x: pd.DataFrame, selected_interactions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(
        {
            col
            for col in selected_interactions[["feature_a", "feature_b"]].to_numpy().ravel().tolist()
            if col not in x.columns
        }
    )
    if missing:
        raise ValueError(f"Input data is missing required interaction source columns: {missing}")

    features: dict[str, np.ndarray] = {}
    for row in selected_interactions.itertuples(index=False):
        features[row.interaction_id] = x[row.feature_a].to_numpy(dtype=float) * x[row.feature_b].to_numpy(dtype=float)
    return pd.DataFrame(features, index=x.index)


class LGBTreePathInteractionTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-style transformer that discovers and materializes LightGBM path interactions."""

    def __init__(
        self,
        top_k: int = 20,
        n_estimators: int = 160,
        random_state: int = 42,
        learning_rate: float = 0.04,
        num_leaves: int = 15,
        max_depth: int = 5,
        min_child_samples: int = 80,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_alpha: float = 0.1,
        reg_lambda: float = 2.0,
        n_jobs: int = -1,
        class_weight: str | dict[int, float] | None = "balanced",
        exclude_direct_missing_pairs: bool = True,
        interaction_prefix: str = INTERACTION_PREFIX,
        include_original: bool = False,
        model_params: dict[str, Any] | None = None,
    ):
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.exclude_direct_missing_pairs = exclude_direct_missing_pairs
        self.interaction_prefix = interaction_prefix
        self.include_original = include_original
        self.model_params = model_params

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray):
        if not isinstance(x, pd.DataFrame):
            raise TypeError("LGBTreePathInteractionTransformer requires a pandas DataFrame with feature names.")

        self.feature_names_in_ = x.columns.tolist()
        self.num_class_ = int(np.max(y)) + 1
        params = self._build_lgbm_params()
        self.model_ = LGBMClassifier(**params)
        self.model_.fit(x, y)

        self.interactions_ = analyze_tree_paths(
            self.model_,
            exclude_direct_missing_pairs=self.exclude_direct_missing_pairs,
        )
        self.selected_interactions_ = make_selected_interactions(
            self.interactions_,
            top_k=self.top_k,
            interaction_prefix=self.interaction_prefix,
        )
        self.interaction_feature_names_ = self.selected_interactions_["interaction_id"].tolist()
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        self._check_ready_for_transform()
        if not isinstance(x, pd.DataFrame):
            raise TypeError("LGBTreePathInteractionTransformer requires a pandas DataFrame with feature names.")

        interaction_matrix = build_interaction_matrix(x, self.selected_interactions_)
        if self.include_original:
            return pd.concat([x.reset_index(drop=True), interaction_matrix.reset_index(drop=True)], axis=1)
        return interaction_matrix

    def fit_transform(self, x: pd.DataFrame, y: pd.Series | np.ndarray, **fit_params: Any) -> pd.DataFrame:
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        self._check_ready_for_transform()
        if self.include_original:
            base = input_features if input_features is not None else getattr(self, "feature_names_in_", [])
            return np.asarray(list(base) + self.interaction_feature_names_, dtype=object)
        return np.asarray(self.interaction_feature_names_, dtype=object)

    def _build_lgbm_params(self) -> dict[str, Any]:
        params = {
            "objective": "multiclass",
            "num_class": self.num_class_,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": -1,
        }
        if self.model_params:
            params.update(self.model_params)
        return params

    def _check_ready_for_transform(self) -> None:
        if not hasattr(self, "selected_interactions_"):
            raise ValueError("The transformer is not fitted or no selected interaction table has been loaded.")

    def config(self) -> LGBInteractionConfig:
        return LGBInteractionConfig(
            top_k=self.top_k,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
            exclude_direct_missing_pairs=self.exclude_direct_missing_pairs,
            interaction_prefix=self.interaction_prefix,
        )

    def save_selection(self, output_dir: Path) -> None:
        self._check_ready_for_transform()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.selected_interactions_.to_csv(output_dir / "selected_lgb_interactions.csv", index=False, encoding="utf-8-sig")
        if hasattr(self, "interactions_"):
            self.interactions_.to_csv(output_dir / "lgb_tree_path_interactions.csv", index=False, encoding="utf-8-sig")
        num_trees_dumped = int(self.model_.booster_.num_trees()) if hasattr(self, "model_") else None
        metadata = {
            "config": asdict(self.config()),
            "input_feature_count": len(getattr(self, "feature_names_in_", [])),
            "input_feature_names": getattr(self, "feature_names_in_", []),
            "selected_interaction_count": int(len(self.selected_interactions_)),
            "interaction_prefix": self.interaction_prefix,
            "interaction_family": INTERACTION_FAMILY,
            "source": INTERACTION_SOURCE,
            "num_trees_dumped": num_trees_dumped,
        }
        (output_dir / "lgb_interaction_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model_info = {
            "n_features": metadata["input_feature_count"],
            "n_estimators": self.n_estimators,
            "num_trees_dumped": num_trees_dumped,
            "interaction_prefix": self.interaction_prefix,
        }
        (output_dir / "lgb_interaction_model_info.json").write_text(
            json.dumps(model_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_selection_file(cls, selection_file: Path, include_original: bool = False):
        selected = pd.read_csv(selection_file, encoding="utf-8-sig")
        required = {"interaction_id", "feature_a", "feature_b"}
        missing = required - set(selected.columns)
        if missing:
            raise ValueError(f"Selection file is missing required columns: {sorted(missing)}")
        obj = cls(top_k=len(selected), include_original=include_original)
        obj.selected_interactions_ = selected
        obj.interaction_feature_names_ = selected["interaction_id"].tolist()
        return obj


def write_interaction_report(
    output_dir: Path,
    transformer: LGBTreePathInteractionTransformer,
    x_train: pd.DataFrame,
) -> None:
    def as_markdown(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return "```\n" + df.to_string(index=False) + "\n```"

    top_pairs = transformer.interactions_.head(30)[
        ["feature_a", "feature_b", "interaction_score", "path_count", "tree_count", "weighted_leaf_count"]
    ]
    selected = transformer.selected_interactions_[
        ["interaction_id", "feature_a", "feature_b", "interaction_score", "path_count", "tree_count", "source"]
    ]

    lines = [
        "# LightGBM Tree-Path Interaction Module",
        "",
        "本模块把 LightGBM 作为监督式交互发现器，而不是最终模型选择结论。方法是遍历每棵树的 root-to-leaf 路径，统计哪些特征经常在同一路径中共同出现；高频共现对表示模型倾向于联合使用这些特征做分裂。",
        "",
        "## Data And Model",
        "",
        f"- Input feature count: {x_train.shape[1]}",
        f"- Selected explicit interactions: {len(transformer.selected_interactions_)}",
        f"- Explicit interaction prefix: `{transformer.interaction_prefix}`",
        f"- Interaction family tag: `{INTERACTION_FAMILY}`",
        f"- LightGBM estimators: {transformer.n_estimators}",
        f"- Dumped trees: {transformer.model_.booster_.num_trees()}",
        "",
        "## Top Tree-Path Co-Occurrences",
        "",
        as_markdown(top_pairs),
        "",
        "## Materialized Interaction Features",
        "",
        as_markdown(selected),
        "",
        "## Usage Notes",
        "",
        "- 生成列统一使用 `lgb_interact__` 前缀，并在 `selected_lgb_interactions.csv` 中保留来源、公式和 LightGBM 路径统计。",
        "- 当前显式交互采用乘积形式：`feature_a * feature_b`。线性模型、距离模型或神经网络可在建模阶段再做标准化。",
        "- LightGBM 本身已经能隐式利用这些路径交互；显式构造主要用于模型无关复用、解释展示和对比实验。",
        "- 若要做严格 OOF 性能估计，应把交互发现放进每个训练折内部，避免全训练集监督式特征选择带来的轻微乐观偏差。",
        "",
    ]
    (output_dir / "lgb_interaction_report.md").write_text("\n".join(lines), encoding="utf-8")


def materialize_interaction_outputs(
    transformer: LGBTreePathInteractionTransformer,
    x_train: pd.DataFrame,
    train_interactions: pd.DataFrame,
    output_dir: Path,
    x_test: pd.DataFrame | None = None,
    test_interactions: pd.DataFrame | None = None,
    write_combined: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transformer.save_selection(output_dir)
    train_interactions.to_csv(output_dir / "X_train_lgb_interaction_features.csv", index=False, encoding="utf-8-sig")

    if x_test is not None and test_interactions is not None:
        test_interactions.to_csv(output_dir / "X_test_lgb_interaction_features.csv", index=False, encoding="utf-8-sig")

    if write_combined:
        pd.concat([x_train.reset_index(drop=True), train_interactions.reset_index(drop=True)], axis=1).to_csv(
            output_dir / "X_train_with_lgb_interactions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if x_test is not None and test_interactions is not None:
            pd.concat([x_test.reset_index(drop=True), test_interactions.reset_index(drop=True)], axis=1).to_csv(
                output_dir / "X_test_with_lgb_interactions.csv",
                index=False,
                encoding="utf-8-sig",
            )

    if hasattr(transformer, "model_") and hasattr(transformer, "interactions_"):
        write_interaction_report(output_dir, transformer, x_train)
