import pandas as pd
from feature_engineering.encoding import (
    encode_onehot, encode_label, encode_ordinal, encode_target_mean,
    cardinality_report, compare_encoding_dimensions,
)


def _sample_df():
    return pd.DataFrame({
        "contract": ["Month-to-month", "One year", "Two year", "Month-to-month",
                     "One year", "Two year", "Month-to-month", "Month-to-month"],
        "customer_id_like": ["A", "B", "C", "D", "E", "F", "G", "H"],  # 高基数示例：每行唯一
        "churn": [1, 0, 0, 1, 0, 0, 1, 1],
    })


def test_encode_onehot_shape_matches_categories():
    df = _sample_df()
    result = encode_onehot(df, "contract")
    assert result.shape == (8, 3)
    assert result.sum(axis=1).eq(1).all()


def test_encode_label_returns_integers():
    df = _sample_df()
    result = encode_label(df, "contract")
    assert result.dtype.kind in "iu"
    assert result.nunique() == 3


def test_encode_ordinal_respects_given_order():
    df = _sample_df()
    order = ["Month-to-month", "One year", "Two year"]
    result = encode_ordinal(df, "contract", order)
    assert result.iloc[0] == 0  # Month-to-month
    assert result.iloc[1] == 1  # One year
    assert result.iloc[2] == 2  # Two year


def test_encode_target_mean_matches_group_mean():
    df = _sample_df()
    result = encode_target_mean(df, "contract", "churn")
    month_to_month_mean = df.loc[df["contract"] == "Month-to-month", "churn"].mean()
    assert abs(result.iloc[0] - month_to_month_mean) < 1e-9


def test_cardinality_report_flags_high_cardinality_column():
    df = _sample_df()
    report = cardinality_report(df, "customer_id_like")
    assert report["n_unique"] == 8
    assert report["singleton_ratio"] == 1.0  # 每个类别只出现一次


def test_compare_encoding_dimensions_returns_one_row_per_column():
    df = _sample_df()
    result = compare_encoding_dimensions(df, ["contract", "customer_id_like"])
    assert set(result["column"]) == {"contract", "customer_id_like"}
    high_card_row = result[result["column"] == "customer_id_like"].iloc[0]
    low_card_row = result[result["column"] == "contract"].iloc[0]
    assert high_card_row["onehot_dims"] > low_card_row["onehot_dims"]
