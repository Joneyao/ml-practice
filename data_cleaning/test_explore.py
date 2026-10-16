import pandas as pd
from data_cleaning.explore import missing_summary, dtype_summary, naive_cv_score


def _sample_df():
    return pd.DataFrame({
        "survived": [0, 1, 1, 0, 1, 0, 1, 0],
        "pclass": [3, 1, 2, 3, 1, 3, 2, 1],
        "age": [22.0, None, 26.0, None, 35.0, 28.0, None, 40.0],
        "fare": [7.25, 71.28, 7.92, 8.05, 53.1, 8.46, 13.0, 30.0],
    })


def test_missing_summary_counts_nulls():
    df = _sample_df()
    summary = missing_summary(df)
    age_row = summary[summary["column"] == "age"].iloc[0]
    assert age_row["missing_count"] == 3
    assert abs(age_row["missing_rate"] - 3 / 8) < 1e-9


def test_dtype_summary_has_all_columns():
    df = _sample_df()
    summary = dtype_summary(df)
    assert set(summary["column"]) == set(df.columns)


def test_naive_cv_score_returns_float_in_range():
    df = _sample_df()
    score = naive_cv_score(df, target_col="survived")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
