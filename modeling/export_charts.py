"""运行 C1-C3 模块，导出真实数据配图（PNG + CSV）到 charts_out/。"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.plotting import setup_cjk_font, save_fig
from common.data_loader import load_house_prices, load_telco_churn, load_credit_fraud
from modeling import regression, classification, imbalance

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_c1_regularization():
    df = load_house_prices()
    X, y = regression.prepare_numeric(df)
    result = regression.compare_regularization(X, y, alphas=[0.1, 1.0, 10.0, 100.0])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, color in [("ridge", "#1976D2"), ("lasso", "#D32F2F")]:
        sub = result[result["model"] == model].sort_values("alpha")
        ax.plot(sub["alpha"], sub["rmse"], "o-", label=model, color=color)
    lin = result[result["model"] == "linear"]["rmse"].iloc[0]
    ax.axhline(lin, ls="--", color="#555", label="线性回归（无正则）")
    ax.set_xscale("log")
    ax.set_xlabel("正则强度 alpha（对数坐标）")
    ax.set_ylabel("5 折交叉验证 RMSE")
    ax.set_title("正则化对回归误差的影响")
    ax.legend()
    result.to_csv(os.path.join(OUT, "regularization_rmse.csv"), index=False)
    return save_fig(fig, OUT, "regularization_rmse.png")


def export_c1_coef_shrinkage():
    df = load_house_prices()
    X, y = regression.prepare_numeric(df)
    # 取与售价相关性最高的几列，避免图太乱
    top = X.corrwith(y).abs().sort_values(ascending=False).head(5).index.tolist()
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    traj = regression.coefficient_shrinkage(X[top], y, alphas)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for feat in top:
        sub = traj[traj["feature"] == feat].sort_values("alpha")
        ax.plot(sub["alpha"], sub["coef"], "o-", label=feat)
    ax.set_xscale("log")
    ax.set_xlabel("正则强度 alpha（对数坐标）")
    ax.set_ylabel("Ridge 系数")
    ax.set_title("alpha 越大，系数被压得越小（收缩）")
    ax.legend(fontsize=8)
    return save_fig(fig, OUT, "coef_shrinkage.png")


def export_c2_algo_comparison():
    df = load_telco_churn()
    X, y = classification.prepare_churn(df)
    result = classification.compare_algorithms(X, y)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(result))
    w = 0.35
    ax.bar(x - w / 2, result["accuracy"], w, label="准确率", color="#4C72B0")
    ax.bar(x + w / 2, result["roc_auc"], w, label="ROC-AUC", color="#55A868")
    for i, row in result.reset_index().iterrows():
        ax.text(i, max(row["accuracy"], row["roc_auc"]) + 0.01,
                f"{row['fit_seconds']:.1f}s", ha="center", fontsize=9, color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(result["algorithm"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("分数")
    ax.set_title("三种算法：精度接近，训练成本差很多（红字为耗时）")
    ax.legend()
    result.to_csv(os.path.join(OUT, "algo_comparison.csv"), index=False)
    return save_fig(fig, OUT, "algo_comparison.png")


def _fraud_subset(seed=0, neg_per_pos=50):
    """分层子集：保留全部欺诈 + 按比例抽正常样本，秒级可跑。"""
    df = load_credit_fraud()
    pos = df[df["Class"] == 1]
    n_neg = min(len(df[df["Class"] == 0]), len(pos) * neg_per_pos)
    neg = df[df["Class"] == 0].sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed)


def export_c3_accuracy_trap():
    df = load_credit_fraud()
    X, y = imbalance.prepare_fraud(df)
    trap = imbalance.naive_accuracy_trap(X, y)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(["准确率", "召回率"], [trap["accuracy"], trap["recall"]],
                  color=["#4C72B0", "#C44E52"])
    ax.set_ylim(0, 1.05)
    ax.set_title("永远预测'不是欺诈'：准确率的假象")
    for b, v in zip(bars, [trap["accuracy"], trap["recall"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    return save_fig(fig, OUT, "accuracy_trap.png")


def export_c3_strategy_comparison():
    sub = _fraud_subset()
    X, y = imbalance.prepare_fraud(sub)
    result = imbalance.compare_imbalance_strategies(X, y)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(result))
    metrics = [("recall", "#C44E52"), ("f1", "#DD8452"), ("roc_auc", "#55A868")]
    w = 0.25
    for i, (m, c) in enumerate(metrics):
        ax.bar(x + (i - 1) * w, result[m], w, label=m, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(result["strategy"])
    ax.set_ylabel("分数")
    ax.set_title("三种不均衡策略：看 recall/F1/AUC，不看 accuracy")
    ax.legend()
    result.to_csv(os.path.join(OUT, "imbalance_strategies.csv"), index=False)
    return save_fig(fig, OUT, "imbalance_strategies.png")


def main():
    setup_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    paths = [
        export_c1_regularization(),
        export_c1_coef_shrinkage(),
        export_c2_algo_comparison(),
        export_c3_accuracy_trap(),
        export_c3_strategy_comparison(),
    ]
    print(f"导出完成：共 {len(paths)} 张图表，输出目录 {OUT}")


if __name__ == "__main__":
    main()
