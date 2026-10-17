"""运行 B1-B3 模块，导出真实数据配图（PNG + CSV）到 charts_out/。"""
import os

import matplotlib.pyplot as plt

from common.data_loader import load_telco_churn, load_titanic_raw
from common.plotting import save_fig, setup_cjk_font
from feature_engineering import construction, encoding, pipeline_leakage

OUT = os.path.join(os.path.dirname(__file__), "charts_out")

_SERVICE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def export_b1_survival_spread():
    """title/deck/交叉特征 vs 原始 pclass/sex 的生还率组间差异对比。"""
    df = load_titanic_raw()
    df["title"] = df["name"].apply(construction.extract_title)
    df["deck"] = df["cabin"].apply(construction.extract_deck)
    df["pclass_sex"] = construction.make_cross_feature(df, "pclass", "sex")
    spreads = {
        "pclass\n(原始)": construction.survival_rate_spread(df, "pclass"),
        "sex\n(原始)": construction.survival_rate_spread(df, "sex"),
        "title\n(从 name 构造)": construction.survival_rate_spread(df, "title"),
        "deck\n(从 cabin 构造)": construction.survival_rate_spread(df, "deck"),
        "pclass_sex\n(交叉特征)": construction.survival_rate_spread(df, "pclass_sex"),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(spreads.keys(), spreads.values(), color="#0984e3")
    ax.set_ylabel("生还率组间差异（spread）")
    ax.set_title("构造出的特征比原始列区分度更高")
    return save_fig(fig, OUT, "b1_survival_spread.png")


def export_b1_feature_set_comparison():
    """baseline（含原始 age+sex）vs engineered（title 替代，不碰缺失的 age）。"""
    df = load_titanic_raw()
    result = construction.compare_feature_sets(df, target_col="survived")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(result["feature_set"], result["cv_score"], color=["#636e72", "#00b894"])
    ax.set_ylabel("5 折交叉验证平均分")
    ax.set_title("从永不缺失的 name 列构造特征，效果不输原始列+填补")
    result.to_csv(os.path.join(OUT, "b1_feature_set_comparison.csv"), index=False)
    return save_fig(fig, OUT, "b1_feature_set_comparison.png")


def export_b2_encoding_dimension():
    """低基数列 vs 拼出来的高基数组合列，One-Hot 维度对比（对数坐标）。"""
    df = load_telco_churn()
    df["customer_profile"] = df[_SERVICE_COLS].astype(str).agg("_".join, axis=1)
    result = encoding.compare_encoding_dimensions(
        df, ["Contract", "PaymentMethod", "customer_profile"]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(result["column"], result["onehot_dims"], color="#d63031")
    ax.set_yscale("log")
    ax.set_ylabel("One-Hot 展开维度（对数坐标）")
    ax.set_title("16 个类别列拼成一个组合列，维度从个位数炸到 4000+")
    result.to_csv(os.path.join(OUT, "b2_encoding_dimension.csv"), index=False)
    return save_fig(fig, OUT, "b2_encoding_dimension.png")


def export_b2_singleton_ratio():
    """高基数组合列中"只出现一次"的类别占比。"""
    df = load_telco_churn()
    df["customer_profile"] = df[_SERVICE_COLS].astype(str).agg("_".join, axis=1)
    report = encoding.cardinality_report(df, "customer_profile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["只出现 1 次的组合", "出现 ≥2 次的组合"],
        [report["singleton_ratio"], 1 - report["singleton_ratio"]],
        color=["#e17055", "#00b894"],
    )
    ax.set_ylabel("占比")
    ax.set_title(f"{report['n_unique']} 个类别中，{report['singleton_ratio']:.1%} 只见过一次")
    return save_fig(fig, OUT, "b2_singleton_ratio.png")


def export_b3_leakage_gap():
    """泄漏管道 vs 正确管道的交叉验证分数对比。"""
    df = load_telco_churn()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["customer_profile"] = df[_SERVICE_COLS].astype(str).agg("_".join, axis=1)
    result = pipeline_leakage.compare_leakage(
        df, "Churn", ["tenure", "MonthlyCharges", "TotalCharges"],
        "customer_profile", seed=0,
    )
    gap = result["cv_score"].iloc[0] - result["cv_score"].iloc[1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["泄漏管道\n(先 fit_transform 再切分)", "正确管道\n(Pipeline 按折 fit)"],
           result["cv_score"], color=["#d63031", "#00b894"])
    ax.set_ylabel("5 折交叉验证平均准确率")
    ax.set_title(f"离线评估虚高 {gap:.1%}——上线后这段差距就是事故")
    result.to_csv(os.path.join(OUT, "b3_leakage_gap.csv"), index=False)
    return save_fig(fig, OUT, "b3_leakage_gap.png")


def main():
    setup_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    paths = [
        export_b1_survival_spread(),
        export_b1_feature_set_comparison(),
        export_b2_encoding_dimension(),
        export_b2_singleton_ratio(),
        export_b3_leakage_gap(),
    ]
    print(f"导出完成：共 {len(paths)} 张图表，输出目录 {OUT}")


if __name__ == "__main__":
    main()
