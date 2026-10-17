"""C2：分类算法对比——LogReg/RandomForest/GBDT 同协议对比。"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


def prepare_churn(df):
    """最简特征处理：数值列原样 + 类别列 One-Hot，Churn 转 0/1。
    聚焦算法对比，不做 B 批的精细特征工程。
    """
    data = df.copy()
    y = (data["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    data = data.drop(columns=["Churn"])
    num = data.select_dtypes(include="number")
    cat = data.select_dtypes(exclude="number")
    X = pd.concat([num, pd.get_dummies(cat, drop_first=True)], axis=1)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def _eval(model, X, y, cv):
    t0 = time.perf_counter()
    scores = cross_validate(model, X, y, cv=cv,
                            scoring=["accuracy", "roc_auc"])
    fit_seconds = time.perf_counter() - t0
    return {
        "accuracy": float(scores["test_accuracy"].mean()),
        "roc_auc": float(scores["test_roc_auc"].mean()),
        "fit_seconds": float(fit_seconds),
    }


def compare_algorithms(X, y, seed=0):
    """三种算法在同一 StratifiedKFold 下对比。"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "gbdt": GradientBoostingClassifier(random_state=seed),
    }
    rows = []
    for name, model in models.items():
        rows.append({"algorithm": name, **_eval(model, X, y, cv)})
    return pd.DataFrame(rows)
