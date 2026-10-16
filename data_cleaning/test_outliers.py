import numpy as np
import pandas as pd
from data_cleaning.outliers import (
    detect_iqr, detect_zscore, compare_outlier_strategies,
)


def _sample_df():
    rng = np.random.default_rng(1)
    n = 200
    area = rng.normal(1500, 300, size=n)
    # 插入几个真实存在的大户型（不是噪声）
    area[:5] = [4500, 4700, 5000, 4600, 4900]
    price = area * 120 + rng.normal(0, 20000, size=n)
    return pd.DataFrame({"GrLivArea": area, "SalePrice": price})


def test_detect_iqr_flags_large_areas():
    df = _sample_df()
    mask = detect_iqr(df["GrLivArea"])
    assert mask.iloc[:5].all()  # 插入的 5 个大户型应被标记
    assert mask.sum() < len(df) * 0.2  # 不应标记超过 20% 的数据


def test_detect_zscore_flags_large_areas():
    df = _sample_df()
    mask = detect_zscore(df["GrLivArea"])
    assert mask.iloc[:5].all()


def test_compare_outlier_strategies_returns_all_rows():
    df = _sample_df()
    result = compare_outlier_strategies(df, col="GrLivArea", target_col="SalePrice")
    assert set(result["strategy"]) == {"keep", "drop", "winsorize"}
    assert (result["r2"].notna()).all()
