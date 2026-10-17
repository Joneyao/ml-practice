import numpy as np
import pandas as pd
from feature_engineering.pipeline_leakage import (
    TargetMeanEncoder, build_preprocessor, build_pipeline,
    leaked_cv_score, correct_cv_score, compare_leakage,
)


def _sample_df(n=200, n_categories=150, seed=0):
    rng = np.random.default_rng(seed)
    categories = [f"combo_{i}" for i in range(n_categories)]
    cat_col = rng.choice(categories, size=n)
    tenure = rng.integers(1, 72, size=n)
    monthly = rng.uniform(20, 120, size=n)
    # churn 和 cat_col 弱相关，制造"高基数列对目标有轻微依赖"的场景
    base_rate = pd.Series(cat_col).map(
        {c: rng.uniform(0.1, 0.9) for c in categories}
    ).to_numpy()
    churn = (rng.random(n) < base_rate).astype(int)
    return pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "combo": cat_col,
        "Churn": churn,
    })


def test_target_mean_encoder_fit_transform_matches_group_mean():
    df = _sample_df(n=50, n_categories=5, seed=1)
    encoder = TargetMeanEncoder(col="combo")
    encoder.fit(df[["combo"]], df["Churn"])
    transformed = encoder.transform(df[["combo"]])
    expected = df.groupby("combo")["Churn"].transform("mean").to_numpy()
    assert np.allclose(transformed.ravel(), expected)


def test_target_mean_encoder_unseen_category_falls_back_to_global_mean():
    df = _sample_df(n=50, n_categories=5, seed=1)
    encoder = TargetMeanEncoder(col="combo")
    encoder.fit(df[["combo"]], df["Churn"])
    unseen = pd.DataFrame({"combo": ["never_seen_category"]})
    result = encoder.transform(unseen)
    assert abs(result[0, 0] - encoder.global_mean_) < 1e-9


def test_build_preprocessor_returns_column_transformer():
    pre = build_preprocessor(["tenure", "MonthlyCharges"], "combo")
    df = _sample_df(n=30, n_categories=5, seed=2)
    out = pre.fit_transform(df[["tenure", "MonthlyCharges", "combo"]], df["Churn"])
    assert out.shape == (30, 3)  # 2 数值列 + 1 target-encoded 列


def test_build_pipeline_fits_and_predicts():
    pipe = build_pipeline(["tenure", "MonthlyCharges"], "combo")
    df = _sample_df(n=60, n_categories=5, seed=3)
    pipe.fit(df[["tenure", "MonthlyCharges", "combo"]], df["Churn"])
    preds = pipe.predict(df[["tenure", "MonthlyCharges", "combo"]])
    assert len(preds) == 60


def test_leaked_score_exceeds_correct_score_on_high_cardinality_data():
    df = _sample_df(n=300, n_categories=250, seed=42)
    leaked = leaked_cv_score(df, "Churn", ["tenure", "MonthlyCharges"], "combo", seed=42)
    correct = correct_cv_score(df, "Churn", ["tenure", "MonthlyCharges"], "combo", seed=42)
    assert leaked > correct


def test_compare_leakage_returns_two_rows_with_gap():
    df = _sample_df(n=300, n_categories=250, seed=42)
    result = compare_leakage(df, "Churn", ["tenure", "MonthlyCharges"], "combo", seed=42)
    assert len(result) == 2
    assert set(result["pipeline"]) == {
        "leaked (fit_transform before split)",
        "correct (Pipeline, fit per fold)",
    }
