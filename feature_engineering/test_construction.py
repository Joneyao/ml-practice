import numpy as np
import pandas as pd
from feature_engineering.construction import (
    extract_title, extract_deck, bin_age, make_cross_feature,
    survival_rate_spread, compare_feature_sets,
)


def _sample_df():
    # 10 行、5 生还 5 未生还——保证 StratifiedKFold(n_splits=5) 每折每类至少 1 个样本。
    return pd.DataFrame({
        "name": [
            "Allen, Miss. Elisabeth Walton",
            "Allison, Master. Hudson Trevor",
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Moran, Mr. James",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
            "McCarthy, Mr. Timothy J",
            "Palsson, Master. Gosta Leonard",
            "Johnson, Mr. William Cahoone Jr",
        ],
        "cabin": ["B5", None, None, "C85", None, None, "C123", "E46", None, None],
        "pclass": [1, 1, 3, 1, 3, 3, 1, 1, 3, 3],
        "sex": ["female", "male", "male", "female", "male",
                "female", "female", "male", "male", "male"],
        "age": [29.0, 0.92, 22.0, None, 27.0, 26.0, 35.0, 54.0, 2.0, 19.0],
        "fare": [211.34, 151.55, 7.25, 53.1, 8.46, 7.92, 53.1, 51.86, 21.08, 8.05],
        "survived": [1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
    })


def test_extract_title_basic_and_rare():
    assert extract_title("Allen, Miss. Elisabeth Walton") == "Miss"
    assert extract_title("Allison, Master. Hudson Trevor") == "Master"
    assert extract_title("Smith, Dr. John") == "Rare"
    assert extract_title("Jones, Mlle. Anne") == "Miss"


def test_extract_deck_handles_missing():
    assert extract_deck("C85") == "C"
    assert extract_deck("B5") == "B"
    assert extract_deck(None) == "Unknown"
    assert extract_deck(float("nan")) == "Unknown"


def test_bin_age_boundaries():
    assert bin_age(5) == "child"
    assert bin_age(15) == "teen"
    assert bin_age(30) == "young_adult"
    assert bin_age(50) == "adult"
    assert bin_age(70) == "senior"
    assert bin_age(None) == "Unknown"


def test_make_cross_feature_concatenates():
    df = _sample_df()
    cross = make_cross_feature(df, "pclass", "sex")
    assert cross.iloc[0] == "1_female"
    assert cross.iloc[2] == "3_male"


def test_survival_rate_spread_is_nonnegative():
    df = _sample_df()
    df["title"] = df["name"].apply(extract_title)
    spread = survival_rate_spread(df, "title")
    assert spread >= 0.0


def test_compare_feature_sets_returns_two_rows():
    df = _sample_df()
    result = compare_feature_sets(df, target_col="survived")
    assert set(result["feature_set"]) == {"baseline", "engineered"}
    assert result["cv_score"].between(0.0, 1.0).all()
