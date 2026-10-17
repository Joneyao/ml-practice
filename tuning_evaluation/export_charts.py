"""运行 D1/D2 模块，导出真实数据配图（PNG + CSV）到 charts_out/。"""
import os
import matplotlib.pyplot as plt
from common.plotting import setup_cjk_font, save_fig
from common.data_loader import load_house_prices, load_telco_churn
from tuning_evaluation import cross_validation, hyperparam_search

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_d1_single_vs_kfold():
    df = load_house_prices()
    num = df.select_dtypes(include="number").fillna(
        df.select_dtypes(include="number").median(numeric_only=True))
    y = num["SalePrice"]
    X = num.drop(columns=["SalePrice"])
    single = cross_validation.single_split_scores(X, y, seeds=range(10))
    kf = cross_validation.kfold_scores(X, y, n_splits=5)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(["单次切分"] * len(single), single["r2"], alpha=0.6,
               s=60, color="#4C72B0", label="10 次单切分")
    ax.errorbar(["K-Fold"], [kf["mean"]], yerr=[kf["std"]], fmt="o",
                capsize=8, markersize=10, color="#C44E52",
                label="K-Fold 均值±标准差")
    ax.set_ylabel("R²")
    ax.set_title("单次切分 vs K-Fold：分数稳定性")
    ax.legend()
    single.to_csv(os.path.join(OUT, "single_split.csv"), index=False)
    return save_fig(fig, OUT, "single_vs_kfold.png")


def export_d2_search_tradeoff():
    X, y = hyperparam_search.prepare_churn_simple(load_telco_churn())
    result = hyperparam_search.compare_search(X, y)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(result["method"], result["best_score"],
                  color=["#4C72B0", "#55A868"])
    for i, row in result.reset_index().iterrows():
        ax.text(i, row["best_score"] + 0.003,
                f"{row['n_candidates']:.0f} 个候选\n{row['elapsed_seconds']:.1f}s",
                ha="center", fontsize=9)
    ax.set_ylabel("最佳 ROC-AUC")
    ax.set_ylim(0, max(result["best_score"]) * 1.15)
    ax.set_title("GridSearch vs RandomSearch 性价比")
    result.to_csv(os.path.join(OUT, "search_tradeoff.csv"), index=False)
    return save_fig(fig, OUT, "search_tradeoff.png")


def main():
    setup_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    paths = [export_d1_single_vs_kfold(), export_d2_search_tradeoff()]
    print(f"导出完成：共 {len(paths)} 张图表，输出目录 {OUT}")


if __name__ == "__main__":
    main()
