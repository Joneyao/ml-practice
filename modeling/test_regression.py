import numpy as np
import pandas as pd
from modeling.regression import (
    prepare_numeric, compare_regularization, coefficient_shrinkage,
)


def _sample_df():
    rng = np.random.default_rng(0)
    n = 200
    area = rng.normal(1500, 400, size=n)
    # 构造一个与 area 高度共线的特征，制造多重共线性
    area_dup = area + rng.normal(0, 10, size=n)
    qual = rng.integers(3, 10, size=n)
    price = area * 100 + qual * 8000 + rng.normal(0, 15000, size=n)
    return pd.DataFrame({
        "GrLivArea": area, "GrLivArea2": area_dup,
        "OverallQual": qual, "SalePrice": price,
    })


def test_prepare_numeric_drops_target():
    X, y = prepare_numeric(_sample_df(), target="SalePrice")
    assert "SalePrice" not in X.columns
    assert len(X) == len(y) == 200
    assert X.isna().sum().sum() == 0


def test_compare_regularization_covers_three_models():
    df = _sample_df()
    X, y = prepare_numeric(df)
    result = compare_regularization(X, y, alphas=[1.0, 10.0])
    assert "linear" in set(result["model"])
    assert "ridge" in set(result["model"])
    assert "lasso" in set(result["model"])
    assert (result["rmse"] > 0).all()


def test_coefficient_shrinkage_decreases_with_alpha():
    df = _sample_df()
    X, y = prepare_numeric(df)
    traj = coefficient_shrinkage(X, y, alphas=[0.1, 1.0, 100.0, 1000.0])
    # alpha 越大，系数 L2 范数越小（收缩）
    norms = traj.groupby("alpha")["coef"].apply(lambda s: np.sqrt((s ** 2).sum()))
    assert norms.loc[1000.0] < norms.loc[0.1]
