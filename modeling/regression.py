"""C1：回归与正则化——LinearRegression vs Ridge vs Lasso，系数收缩与多重共线性。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def prepare_numeric(df, target="SalePrice"):
    """取数值列，中位数填缺失，分离特征与目标。"""
    numeric = df.select_dtypes(include="number").copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    y = numeric[target]
    X = numeric.drop(columns=[target])
    return X, y


def _rmse_cv(model, X, y):
    scores = cross_val_score(model, X, y, cv=5,
                             scoring="neg_root_mean_squared_error")
    return float(-scores.mean())


def compare_regularization(X, y, alphas=(1.0, 10.0, 100.0)):
    """对比线性回归和不同 alpha 下的 Ridge/Lasso 的交叉验证 RMSE。
    所有线性模型都先标准化（正则化对特征尺度敏感）。
    """
    rows = [{
        "model": "linear", "alpha": None,
        "rmse": _rmse_cv(make_pipeline(StandardScaler(), LinearRegression()), X, y),
    }]
    for a in alphas:
        rows.append({
            "model": "ridge", "alpha": a,
            "rmse": _rmse_cv(make_pipeline(StandardScaler(), Ridge(alpha=a)), X, y),
        })
        rows.append({
            "model": "lasso", "alpha": a,
            "rmse": _rmse_cv(make_pipeline(StandardScaler(), Lasso(alpha=a, max_iter=10000)), X, y),
        })
    return pd.DataFrame(rows)


def coefficient_shrinkage(X, y, alphas):
    """记录 Ridge 系数随 alpha 变化的轨迹（标准化后），展示收缩效果。"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rows = []
    for a in alphas:
        model = Ridge(alpha=a).fit(Xs, y)
        for feat, coef in zip(X.columns, model.coef_):
            rows.append({"alpha": a, "feature": feat, "coef": float(coef)})
    return pd.DataFrame(rows)
