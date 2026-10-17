import numpy as np
import pandas as pd
from tuning_evaluation.cross_validation import (
    single_split_scores, kfold_scores, compare_cv_stability,
)


def _sample_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.normal(1500, 400, size=n)
    qual = rng.integers(3, 10, size=n)
    price = area * 100 + qual * 8000 + rng.normal(0, 20000, size=n)
    return pd.DataFrame({"GrLivArea": area, "OverallQual": qual, "SalePrice": price})


def _xy(df):
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    return X, y


def test_single_split_scores_varies():
    X, y = _xy(_sample_df())
    result = single_split_scores(X, y, seeds=[0, 1, 2, 3, 4])
    assert len(result) == 5
    # 不同种子的单次切分分数应有可观测的差异（方差 > 0）
    assert result["r2"].std() > 0


def test_kfold_scores_structure():
    X, y = _xy(_sample_df())
    res = kfold_scores(X, y, n_splits=5)
    assert len(res["fold_scores"]) == 5
    assert "mean" in res and "std" in res


def test_compare_cv_stability_has_both():
    X, y = _xy(_sample_df())
    result = compare_cv_stability(X, y)
    assert set(result["method"]) >= {"single_split", "kfold"}
