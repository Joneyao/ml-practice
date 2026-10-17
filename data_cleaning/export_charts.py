"""运行 A1-A3 模块，导出真实数据配图（PNG + CSV）到 charts_out/。

图与文章的映射：
- A1：missing_heatmap.png（每列缺失值分布热力图）、
       naive_vs_honest.png（dropna 建模用到的样本 vs 被丢弃样本的存活率对比，
       揭示"假高分"背后丢的是系统性偏差而非随机噪声）
- A2：imputation_comparison.png（四种填补策略的交叉验证分数对比）、
       missing_by_group.png（按舱位分组的缺失率条形图，直观展示 MAR）
- A3：outlier_scatter.png（面积-房价散点图，标出 IQR 判定的异常值）、
       strategy_comparison.png（保留/删除/截尾三种策略的 R² 对比）
"""
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common.data_loader import load_house_prices, load_titanic
from common.plotting import save_fig, setup_cjk_font
from data_cleaning import explore, imputation, outliers

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_a1_missing_heatmap():
    """每列缺失值分布热力图：真实反映哪些列缺、缺多少、缺在哪些行。"""
    df = load_titanic()
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = mcolors.ListedColormap(["#dfe6e9", "#d63031"])
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap=cmap, ax=ax)
    ax.set_xlabel("列")
    ax.set_title("Titanic 各列缺失值分布（红色=缺失）")
    return save_fig(fig, OUT, "missing_heatmap.png")


def export_a1_naive_vs_honest():
    """对比 dropna 建模用到的样本 vs 被丢弃样本的存活率，揭示"假高分"背后的偏差。"""
    df = load_titanic()
    naive_score = explore.naive_cv_score(df, target_col="survived")
    age_missing = df["age"].isna()
    rates = pd.Series({
        "全量乘客\n(诚实基线)": df["survived"].mean(),
        "年龄非缺失\n(naive 建模用到的样本)": df.loc[~age_missing, "survived"].mean(),
        "年龄缺失\n(被 dropna 丢弃的样本)": df.loc[age_missing, "survived"].mean(),
    })
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(rates.index, rates.values, color=["#636e72", "#00b894", "#d63031"])
    ax.set_ylabel("存活率")
    ax.set_title(f"被丢弃样本的存活率明显偏低（naive CV 分数={naive_score:.3f}）")
    return save_fig(fig, OUT, "naive_vs_honest.png")


def export_a2_imputation_comparison():
    df = load_titanic()
    result = imputation.compare_strategies(df, target_col="survived")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(result["strategy"], result["cv_score"])
    ax.set_xlabel("填补策略")
    ax.set_ylabel("5 折交叉验证平均分")
    ax.set_title("缺失值填补策略对模型效果的影响")
    result.to_csv(os.path.join(OUT, "imputation_comparison.csv"), index=False)
    return save_fig(fig, OUT, "imputation_comparison.png")


def export_a2_missing_by_group():
    """按舱位分组的 age 缺失率条形图，直观展示 MAR（缺失和舱位相关）。"""
    df = load_titanic()
    label = imputation.classify_missingness(df, col="age", group_col="pclass")
    rates = df.groupby("pclass")["age"].apply(lambda s: s.isna().mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rates.index.astype(str), rates.values, color="#0984e3")
    ax.set_xlabel("舱位等级 (pclass)")
    ax.set_ylabel("age 缺失率")
    ax.set_title(f"分组缺失率差异明显 → 判定为 {label}")
    return save_fig(fig, OUT, "missing_by_group.png")


def export_a3_outlier_scatter():
    """面积-房价散点图，标出被 IQR 判为异常值的点。"""
    df = load_house_prices()
    mask = outliers.detect_iqr(df["GrLivArea"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.loc[~mask, "GrLivArea"], df.loc[~mask, "SalePrice"],
               s=12, alpha=0.5, color="#0984e3", label="正常值")
    ax.scatter(df.loc[mask, "GrLivArea"], df.loc[mask, "SalePrice"],
               s=30, alpha=0.9, color="#d63031", label="IQR 判定异常值")
    ax.set_xlabel("居住面积 GrLivArea")
    ax.set_ylabel("房价 SalePrice")
    ax.set_title(f"面积-房价散点图（{int(mask.sum())} 个点被 IQR 判为异常值）")
    ax.legend()
    return save_fig(fig, OUT, "outlier_scatter.png")


def export_a3_strategy_comparison():
    df = load_house_prices()
    result = outliers.compare_outlier_strategies(
        df, col="GrLivArea", target_col="SalePrice"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(result["strategy"], result["r2"])
    ax.set_xlabel("异常值处理策略")
    ax.set_ylabel("R²")
    ax.set_title("异常值处理策略对回归效果的影响")
    result.to_csv(os.path.join(OUT, "strategy_comparison.csv"), index=False)
    return save_fig(fig, OUT, "strategy_comparison.png")


def main():
    setup_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    paths = [
        export_a1_missing_heatmap(),
        export_a1_naive_vs_honest(),
        export_a2_imputation_comparison(),
        export_a2_missing_by_group(),
        export_a3_outlier_scatter(),
        export_a3_strategy_comparison(),
    ]
    # 注：不在 stdout 中回显文件名——部分产物文件名本身包含 "missing"
    # （如 missing_heatmap.png），会被字体缺字检查的 grep 误判为警告。
    print(f"导出完成：共 {len(paths)} 张图表，输出目录 {OUT}")


if __name__ == "__main__":
    main()
