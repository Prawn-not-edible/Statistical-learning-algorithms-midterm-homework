import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.feature_selection import mutual_info_classif


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "eda_outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "training_dataset.csv"
TEST_PATH = DATA_DIR / "test_dataset.csv"
ENCODING = "gbk"
ID_COL = "序号"
TARGET_COL = "label"
EVENT_COL = "event_flag"


def configure_plotting():
    font_candidates = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    chosen = None
    for name in font_candidates:
        try:
            font_path = font_manager.findfont(name, fallback_to_default=False)
            font_manager.fontManager.addfont(font_path)
            chosen = name
            break
        except Exception:
            continue

    if chosen is None:
        chosen = "DejaVu Sans"

    matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["font.sans-serif"] = [chosen]
    matplotlib.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", rc={"font.family": chosen, "font.sans-serif": [chosen]})


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def load_data():
    train = pd.read_csv(TRAIN_PATH, encoding=ENCODING)
    test = pd.read_csv(TEST_PATH, encoding=ENCODING)
    return train, test


def classify_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]
    unique_counts = df[feature_cols].nunique(dropna=True)

    binary_cols = unique_counts[unique_counts <= 2].index.tolist()
    low_card_cols = unique_counts[(unique_counts >= 3) & (unique_counts <= 10)].index.tolist()
    continuous_cols = unique_counts[unique_counts > 10].index.tolist()

    return feature_cols, binary_cols, low_card_cols, continuous_cols, unique_counts


def make_dataset_overview(train: pd.DataFrame, test: pd.DataFrame, binary_cols, low_card_cols, continuous_cols):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    overview = pd.DataFrame(
        {
            "dataset": ["train", "test"],
            "rows": [len(train), len(test)],
            "columns": [train.shape[1], test.shape[1]],
        }
    )
    axes[0].bar(overview["dataset"], overview["rows"], color=["#4C78A8", "#F58518"])
    for idx, value in enumerate(overview["rows"]):
        axes[0].text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=10)
    axes[0].set_title("Train/Test 样本量")
    axes[0].set_ylabel("样本数")

    type_df = pd.DataFrame(
        {
            "类型": ["二值/近二值", "低基数离散", "连续/高基数"],
            "数量": [len(binary_cols), len(low_card_cols), len(continuous_cols)],
        }
    )
    sns.barplot(data=type_df, x="类型", y="数量", hue="类型", palette=["#54A24B", "#E45756", "#72B7B2"], legend=False, ax=axes[1])
    for idx, value in enumerate(type_df["数量"]):
        axes[1].text(idx, value, str(value), ha="center", va="bottom", fontsize=10)
    axes[1].set_title("训练集特征类型划分（按唯一值个数近似）")
    axes[1].set_ylabel("特征数")

    savefig(OUT_DIR / "01_dataset_overview.png")


def make_label_distribution(train: pd.DataFrame):
    label_counts = train[TARGET_COL].value_counts().sort_index()
    label_pct = label_counts / label_counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(x=label_counts.index.astype(str), y=label_counts.values, color="#4C78A8", ax=axes[0])
    axes[0].set_title("标签分布（线性尺度）")
    axes[0].set_xlabel("label")
    axes[0].set_ylabel("样本数")
    for idx, (count, pct) in enumerate(zip(label_counts.values, label_pct.values)):
        axes[0].text(idx, count, f"{count}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=9)

    sns.barplot(x=label_counts.index.astype(str), y=label_counts.values, color="#E45756", ax=axes[1])
    axes[1].set_yscale("log")
    axes[1].set_title("标签分布（对数尺度）")
    axes[1].set_xlabel("label")
    axes[1].set_ylabel("样本数（log）")

    savefig(OUT_DIR / "02_label_distribution.png")


def make_missingness(train: pd.DataFrame, test: pd.DataFrame):
    train_missing = train.isna().mean().sort_values(ascending=False).head(20)
    test_missing = test.isna().mean().sort_values(ascending=False).head(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    sns.barplot(x=train_missing.values * 100, y=train_missing.index, color="#4C78A8", ax=axes[0])
    axes[0].set_title("训练集缺失率 Top 20")
    axes[0].set_xlabel("缺失率（%）")
    axes[0].set_ylabel("")

    sns.barplot(x=test_missing.values * 100, y=test_missing.index, color="#F58518", ax=axes[1])
    axes[1].set_title("测试集缺失率 Top 20")
    axes[1].set_xlabel("缺失率（%）")
    axes[1].set_ylabel("")

    savefig(OUT_DIR / "03_missing_top20.png")

    feature_missing = pd.DataFrame(
        {
            "train_missing_pct": train.drop(columns=[TARGET_COL], errors="ignore").isna().mean().values * 100,
            "test_missing_pct": test.isna().mean().values * 100,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(feature_missing["train_missing_pct"], bins=20, color="#4C78A8", alpha=0.6, label="train", ax=ax)
    sns.histplot(feature_missing["test_missing_pct"], bins=20, color="#F58518", alpha=0.6, label="test", ax=ax)
    ax.set_title("特征缺失率分布")
    ax.set_xlabel("每列缺失率（%）")
    ax.set_ylabel("特征数")
    ax.legend()

    savefig(OUT_DIR / "04_missing_distribution.png")


def get_top_continuous_features(train: pd.DataFrame, continuous_cols):
    usable = []
    y = train[TARGET_COL]
    for col in continuous_cols:
        non_null_ratio = train[col].notna().mean()
        if non_null_ratio >= 0.55:
            usable.append(col)

    if not usable:
        return []

    X = train[usable].copy()
    X = X.fillna(X.median())
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi, index=usable).sort_values(ascending=False)
    return mi_series


def make_feature_signal_plots(train: pd.DataFrame, mi_series: pd.Series):
    top20 = mi_series.head(20).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20.index, top20.values, color="#72B7B2")
    ax.set_title("连续特征与标签的互信息 Top 20")
    ax.set_xlabel("Mutual Information")
    ax.set_ylabel("")
    savefig(OUT_DIR / "05_continuous_feature_signal.png")

    heatmap_cols = mi_series.head(12).index.tolist()
    corr_df = train[heatmap_cols + [TARGET_COL]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_df, cmap="RdBu_r", center=0, annot=False, square=True, ax=ax)
    ax.set_title("高信号连续特征 Spearman 相关热图")
    savefig(OUT_DIR / "06_correlation_heatmap.png")

    plot_cols = mi_series.head(8).index.tolist()
    rows = math.ceil(len(plot_cols) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4.2 * rows))
    axes = np.array(axes).reshape(-1)

    plot_df = train[plot_cols + [TARGET_COL]].copy()
    plot_df[EVENT_COL] = np.where(plot_df[TARGET_COL] == 0, "label=0", "label>0")
    for ax, col in zip(axes, plot_cols):
        sns.kdeplot(
            data=plot_df,
            x=col,
            hue=EVENT_COL,
            common_norm=False,
            fill=True,
            alpha=0.25,
            linewidth=1.1,
            ax=ax,
        )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("density")
    for ax in axes[len(plot_cols):]:
        ax.axis("off")

    savefig(OUT_DIR / "07_classwise_numeric_distributions.png")


def make_outlier_plot(train: pd.DataFrame, continuous_cols):
    outlier_rates = {}
    for col in continuous_cols:
        series = train[col].dropna()
        if series.nunique() <= 10 or len(series) < 100:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_rates[col] = ((series < lower) | (series > upper)).mean()

    top_cols = pd.Series(outlier_rates).sort_values(ascending=False).head(12).index.tolist()
    scaled = train[top_cols].copy()
    scaled = (scaled - scaled.median()) / scaled.std(ddof=0)

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(data=scaled, orient="h", color="#54A24B", ax=ax, showfliers=True)
    ax.set_title("异常值比例较高的连续特征（标准化后箱线图）")
    ax.set_xlabel("标准化取值")
    ax.set_ylabel("")

    savefig(OUT_DIR / "08_outlier_boxplots.png")
    return pd.Series(outlier_rates).sort_values(ascending=False)


def make_binary_feature_plot(train: pd.DataFrame, binary_cols):
    event = (train[TARGET_COL] > 0).astype(int)
    positive_rate_gap = {}
    for col in binary_cols:
        col_series = train[col]
        non_null = col_series.notna()
        if non_null.sum() == 0:
            continue
        values = col_series[non_null]
        if values.nunique() != 2:
            continue
        tmp = pd.DataFrame({"x": values, "event": event[non_null]})
        rate_by_event = tmp.groupby("event")["x"].mean()
        if len(rate_by_event) == 2:
            positive_rate_gap[col] = rate_by_event.loc[1] - rate_by_event.loc[0]

    gap = pd.Series(positive_rate_gap).sort_values(key=lambda s: s.abs(), ascending=False).head(15)
    plot_df = gap.sort_values().rename("差值").reset_index().rename(columns={"index": "特征"})

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#E45756" if v > 0 else "#4C78A8" for v in plot_df["差值"]]
    ax.barh(plot_df["特征"], plot_df["差值"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("二值特征在 label>0 与 label=0 中的阳性率差异 Top 15")
    ax.set_xlabel("阳性率差值（event - non-event）")
    ax.set_ylabel("")

    savefig(OUT_DIR / "09_binary_feature_gap.png")


def make_train_test_shift(train: pd.DataFrame, test: pd.DataFrame, continuous_cols):
    stats = []
    for col in continuous_cols:
        train_non_null = train[col].dropna()
        test_non_null = test[col].dropna()
        if len(train_non_null) < 50 or len(test_non_null) < 50:
            continue
        train_mean = train_non_null.mean()
        test_mean = test_non_null.mean()
        pooled_std = np.nanmean([train_non_null.std(ddof=0), test_non_null.std(ddof=0)])
        if pooled_std and not np.isnan(pooled_std):
            shift = abs(train_mean - test_mean) / pooled_std
            stats.append((col, shift, train_mean, test_mean))

    shift_df = pd.DataFrame(stats, columns=["特征", "标准化均值差", "train_mean", "test_mean"])
    shift_df = shift_df.sort_values("标准化均值差", ascending=False).head(15).sort_values("标准化均值差")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(shift_df["特征"], shift_df["标准化均值差"], color="#B279A2")
    ax.set_title("训练集 vs 测试集 分布漂移较大的连续特征 Top 15")
    ax.set_xlabel("标准化均值差")
    ax.set_ylabel("")

    savefig(OUT_DIR / "10_train_test_shift.png")
    return shift_df


def build_summary(train: pd.DataFrame, test: pd.DataFrame, feature_cols, binary_cols, low_card_cols, continuous_cols, unique_counts, mi_series, outlier_series, shift_df):
    label_counts = train[TARGET_COL].value_counts().sort_index()
    label_pct = (label_counts / label_counts.sum() * 100).round(4)

    missing_train = train[feature_cols].isna().mean().sort_values(ascending=False)
    missing_test = test[feature_cols].isna().mean().sort_values(ascending=False)

    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "feature_count": len(feature_cols),
        "binary_feature_count": len(binary_cols),
        "low_cardinality_feature_count": len(low_card_cols),
        "continuous_feature_count": len(continuous_cols),
        "label_counts": label_counts.to_dict(),
        "label_pct": label_pct.to_dict(),
        "top_missing_train_pct": (missing_train.head(15) * 100).round(2).to_dict(),
        "top_missing_test_pct": (missing_test.head(15) * 100).round(2).to_dict(),
        "top_signal_continuous_features": mi_series.head(15).round(4).to_dict(),
        "top_outlier_rate_features": (outlier_series.head(15) * 100).round(2).to_dict(),
        "top_train_test_shift_features": shift_df.sort_values("标准化均值差", ascending=False).head(15).round(4).to_dict(orient="records"),
        "columns_with_no_missing_train": int((missing_train == 0).sum()),
        "columns_with_missing_train": int((missing_train > 0).sum()),
        "columns_with_missing_over_30pct_train": int((missing_train > 0.3).sum()),
        "columns_with_missing_over_50pct_train": int((missing_train > 0.5).sum()),
    }

    with open(OUT_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# EDA 文字摘要")
    lines.append("")
    lines.append(f"- 训练集规模：{train.shape[0]} 行，{train.shape[1]} 列；测试集规模：{test.shape[0]} 行，{test.shape[1]} 列。")
    lines.append(f"- 可用特征共 {len(feature_cols)} 个（不含 `{ID_COL}` 与 `{TARGET_COL}`）。")
    lines.append(f"- 按唯一值个数近似划分：二值/近二值特征 {len(binary_cols)} 个，低基数离散特征 {len(low_card_cols)} 个，连续/高基数特征 {len(continuous_cols)} 个。")
    lines.append("- 标签分布：")
    for cls, cnt in label_counts.items():
        lines.append(f"  - label={cls}: {cnt} 条，占 {label_pct[cls]:.2f}%")
    lines.append(f"- 训练集共有 {(missing_train > 0).sum()} 个特征存在缺失，其中 {(missing_train > 0.3).sum()} 个特征缺失率超过 30%，{(missing_train > 0.5).sum()} 个特征缺失率超过 50%。")
    lines.append("- 训练集缺失最重的前 10 列：")
    for col, pct in (missing_train.head(10) * 100).round(2).items():
        lines.append(f"  - {col}: {pct:.2f}%")
    lines.append("- 连续特征与标签互信息较高的前 10 列：")
    for col, score in mi_series.head(10).round(4).items():
        lines.append(f"  - {col}: {score:.4f}")
    lines.append("- 异常值比例较高的前 10 个连续特征：")
    for col, pct in (outlier_series.head(10) * 100).round(2).items():
        lines.append(f"  - {col}: {pct:.2f}%")
    lines.append("- 训练集与测试集均值漂移较大的前 10 个连续特征：")
    for item in shift_df.sort_values("标准化均值差", ascending=False).head(10).to_dict(orient="records"):
        lines.append(f"  - {item['特征']}: 标准化均值差 {item['标准化均值差']:.3f}")

    (OUT_DIR / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    configure_plotting()
    train, test = load_data()
    feature_cols, binary_cols, low_card_cols, continuous_cols, unique_counts = classify_features(train)

    make_dataset_overview(train, test, binary_cols, low_card_cols, continuous_cols)
    make_label_distribution(train)
    make_missingness(train, test)

    mi_series = get_top_continuous_features(train, continuous_cols)
    make_feature_signal_plots(train, mi_series)

    outlier_series = make_outlier_plot(train, continuous_cols)
    make_binary_feature_plot(train, binary_cols)
    shift_df = make_train_test_shift(train, test, continuous_cols)

    build_summary(
        train,
        test,
        feature_cols,
        binary_cols,
        low_card_cols,
        continuous_cols,
        unique_counts,
        mi_series,
        outlier_series,
        shift_df,
    )
    print(f"EDA outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
