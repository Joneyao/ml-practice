"""A1：数据探查——缺失值/类型统计，复现"未清洗先建模"的假高分。"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """每列缺失数和缺失率。"""
    counts = df.isna().sum()
    rates = counts / len(df)
    return pd.DataFrame({
        "column": counts.index,
        "missing_count": counts.values,
        "missing_rate": rates.values,
    })


def dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    """每列 dtype 和唯一值数量。"""
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "n_unique": [df[c].nunique() for c in df.columns],
    })


def naive_cv_score(df: pd.DataFrame, target_col: str, seed: int = 0) -> float:
    """不做任何清洗：直接 dropna，数值列喂给随机森林做 5 折交叉验证。
    用来复现"看起来能跑的模型"分数虚高的假象——dropna 丢掉的不是随机样本，
    而是系统性偏差（比如年龄缺失的乘客本身就和存活率相关）。
    """
    numeric = df.select_dtypes(include="number").dropna()
    y = numeric[target_col]
    X = numeric.drop(columns=[target_col])
    clf = RandomForestClassifier(n_estimators=50, random_state=seed)
    # dropna 后样本量可能很小，折数不能超过最小类别的样本数，
    # 否则 StratifiedKFold 无法分层——教学场景（如小样本单测）里真实存在。
    n_splits = min(5, y.value_counts().min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv)
    return float(scores.mean())
