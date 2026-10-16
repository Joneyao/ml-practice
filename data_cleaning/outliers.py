"""A3：异常值检测与处理——IQR/Z-score 检测，删除 vs 截尾的效果对比。"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error


def detect_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """IQR 法：超出 [Q1 - k*IQR, Q3 + k*IQR] 判为异常值。"""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def detect_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Z-score 法：标准化后绝对值超过 threshold 判为异常值。"""
    z = (series - series.mean()) / series.std()
    return z.abs() > threshold


def _winsorize(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """截尾：把超出分位数范围的值直接夹到边界，而不是删除整行。"""
    lower, upper = series.quantile(lower_q), series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def _fit_and_score(X: pd.DataFrame, y: pd.Series) -> dict:
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
    }


def compare_outlier_strategies(df: pd.DataFrame, col: str, target_col: str) -> pd.DataFrame:
    """对比"保留" vs "删除异常值" vs "截尾"三种策略对单变量线性回归的影响。"""
    rows = []

    keep_score = _fit_and_score(df[[col]], df[target_col])
    rows.append({"strategy": "keep", "n_samples": len(df), **keep_score})

    mask = detect_iqr(df[col])
    dropped = df[~mask]
    drop_score = _fit_and_score(dropped[[col]], dropped[target_col])
    rows.append({"strategy": "drop", "n_samples": len(dropped), **drop_score})

    winsorized = df.copy()
    winsorized[col] = _winsorize(df[col])
    win_score = _fit_and_score(winsorized[[col]], winsorized[target_col])
    rows.append({"strategy": "winsorize", "n_samples": len(winsorized), **win_score})

    return pd.DataFrame(rows)
