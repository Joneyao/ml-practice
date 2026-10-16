"""A2：缺失值填补策略对比——删除 vs 均值 vs 分组中位数 vs KNN。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score


def classify_missingness(df: pd.DataFrame, col: str, group_col: str,
                          threshold: float = 0.1) -> str:
    """简化版缺失机制判定：按 group_col 分组比较缺失率，
    组间最大差异超过 threshold 判为 MAR，否则判为 MCAR。
    """
    rates = df.groupby(group_col)[col].apply(lambda s: s.isna().mean())
    if rates.max() - rates.min() > threshold:
        return "MAR"
    return "MCAR"


def impute_mean(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(df[col].mean())


def impute_grouped_median(df: pd.DataFrame, col: str, group_col: str) -> pd.Series:
    return df.groupby(group_col)[col].transform(lambda s: s.fillna(s.median()))


def impute_knn(df: pd.DataFrame, cols: list[str], n_neighbors: int = 5) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    filled = imputer.fit_transform(df[cols])
    return pd.DataFrame(filled, columns=cols, index=df.index)


def _cv_score(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> float:
    clf = RandomForestClassifier(n_estimators=50, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=5)
    return float(scores.mean())


def compare_strategies(df: pd.DataFrame, target_col: str,
                        age_col: str = "age", group_col: str = "pclass") -> pd.DataFrame:
    """对比四种策略对同一个随机森林交叉验证分数的影响。"""
    rows = []

    dropped = df.dropna(subset=[age_col])
    rows.append({
        "strategy": "dropna", "n_samples": len(dropped),
        "cv_score": _cv_score(dropped[[group_col, age_col]], dropped[target_col]),
    })

    mean_filled = df.copy()
    mean_filled[age_col] = impute_mean(df, age_col)
    rows.append({
        "strategy": "mean", "n_samples": len(mean_filled),
        "cv_score": _cv_score(mean_filled[[group_col, age_col]], mean_filled[target_col]),
    })

    grouped_filled = df.copy()
    grouped_filled[age_col] = impute_grouped_median(df, age_col, group_col)
    rows.append({
        "strategy": "grouped_median", "n_samples": len(grouped_filled),
        "cv_score": _cv_score(grouped_filled[[group_col, age_col]], grouped_filled[target_col]),
    })

    knn_filled = impute_knn(df, cols=[group_col, age_col])
    knn_filled[target_col] = df[target_col].values
    rows.append({
        "strategy": "knn", "n_samples": len(knn_filled),
        "cv_score": _cv_score(knn_filled[[group_col, age_col]], knn_filled[target_col]),
    })

    return pd.DataFrame(rows)
