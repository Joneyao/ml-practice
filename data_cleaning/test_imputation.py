import numpy as np
import pandas as pd
from data_cleaning.imputation import (
    classify_missingness, impute_mean, impute_grouped_median,
    impute_knn, compare_strategies,
)


def _sample_df():
    rng = np.random.default_rng(0)
    n = 100
    pclass = rng.choice([1, 2, 3], size=n)
    age = rng.normal(35, 10, size=n)
    # 构造 MAR：三等舱乘客的年龄更容易缺失
    missing_mask = (rng.random(n) < np.where(pclass == 3, 0.5, 0.05))
    age_with_na = age.copy()
    age_with_na[missing_mask] = np.nan
    survived = (age > 30).astype(int)  # 和 age 相关，制造真实的系统性偏差
    return pd.DataFrame({"pclass": pclass, "age": age_with_na, "survived": survived})


def test_classify_missingness_detects_mar():
    df = _sample_df()
    label = classify_missingness(df, col="age", group_col="pclass")
    assert label == "MAR"


def test_impute_mean_fills_all_na():
    df = _sample_df()
    filled = impute_mean(df, col="age")
    assert filled.isna().sum() == 0


def test_impute_grouped_median_fills_all_na():
    df = _sample_df()
    filled = impute_grouped_median(df, col="age", group_col="pclass")
    assert filled.isna().sum() == 0


def test_impute_knn_fills_all_na():
    df = _sample_df()
    filled = impute_knn(df, cols=["pclass", "age"])
    assert filled["age"].isna().sum() == 0


def test_compare_strategies_returns_all_rows():
    df = _sample_df()
    result = compare_strategies(df, target_col="survived")
    assert set(result["strategy"]) == {
        "dropna", "mean", "grouped_median", "knn",
    }
    assert (result["n_samples"] > 0).all()
