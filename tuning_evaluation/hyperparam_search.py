"""D2：调参的性价比——GridSearchCV vs RandomizedSearchCV。"""
import time
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold


def prepare_churn_simple(df):
    """最简特征：数值列 + One-Hot 类别列，Churn 转 0/1。本模块自包含。"""
    data = df.copy()
    y = (data["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    data = data.drop(columns=["Churn"])
    num = data.select_dtypes(include="number")
    cat = data.select_dtypes(exclude="number")
    X = pd.concat([num, pd.get_dummies(cat, drop_first=True)], axis=1)
    return X.fillna(X.median(numeric_only=True)), y


def _cv(seed):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def run_grid_search(X, y, param_grid, seed=0):
    t0 = time.perf_counter()
    gs = GridSearchCV(RandomForestClassifier(random_state=seed), param_grid,
                      cv=_cv(seed), scoring="roc_auc")
    gs.fit(X, y)
    return {
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
        "n_candidates": len(gs.cv_results_["params"]),
        "elapsed_seconds": time.perf_counter() - t0,
    }


def run_random_search(X, y, param_distributions, n_iter=10, seed=0):
    t0 = time.perf_counter()
    rs = RandomizedSearchCV(RandomForestClassifier(random_state=seed),
                            param_distributions, n_iter=n_iter, cv=_cv(seed),
                            scoring="roc_auc", random_state=seed)
    rs.fit(X, y)
    return {
        "best_params": rs.best_params_,
        "best_score": float(rs.best_score_),
        "n_candidates": n_iter,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def compare_search(X, y, seed=0):
    """同一超参空间上对比两种搜索的 best_score / 候选数 / 耗时。"""
    grid = run_grid_search(X, y, param_grid={
        "n_estimators": [50, 100, 200], "max_depth": [3, 5, 10, None],
    }, seed=seed)
    rnd = run_random_search(X, y, param_distributions={
        "n_estimators": randint(50, 201), "max_depth": [3, 5, 10, None],
    }, n_iter=6, seed=seed)
    return pd.DataFrame([
        {"method": "grid", **{k: grid[k] for k in ["best_score", "n_candidates", "elapsed_seconds"]}},
        {"method": "random", **{k: rnd[k] for k in ["best_score", "n_candidates", "elapsed_seconds"]}},
    ])
