"""D1：交叉验证的正确姿势——单次切分 vs K-Fold 的分数稳定性。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score


def single_split_scores(X, y, seeds=(0, 1, 2, 3, 4)):
    """对每个随机种子各做一次 train_test_split，返回每次的 R²。"""
    rows = []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        model = LinearRegression().fit(Xtr, ytr)
        rows.append({"seed": s, "r2": float(model.score(Xte, yte))})
    return pd.DataFrame(rows)


def kfold_scores(X, y, n_splits=5, seed=0):
    """K-Fold 交叉验证，返回每折 R² + 均值 + 标准差。"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
    return {
        "fold_scores": [float(s) for s in scores],
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def compare_cv_stability(X, y):
    """把单次切分的分数散布和 K-Fold 的均值±标准差放一起对比。"""
    single = single_split_scores(X, y, seeds=range(10))
    kf = kfold_scores(X, y, n_splits=5)
    return pd.DataFrame([
        {"method": "single_split", "center": float(single["r2"].mean()),
         "low": float(single["r2"].min()), "high": float(single["r2"].max())},
        {"method": "kfold", "center": kf["mean"],
         "low": kf["mean"] - kf["std"], "high": kf["mean"] + kf["std"]},
    ])
