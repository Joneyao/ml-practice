import numpy as np
import pandas as pd
from tuning_evaluation.hyperparam_search import (
    prepare_churn_simple, run_grid_search, run_random_search, compare_search,
)


def _sample_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, size=n)
    charges = rng.normal(65, 30, size=n).clip(18, 120)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    logit = -0.04 * tenure + (contract == "Month-to-month") * 1.2
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "tenure": tenure, "MonthlyCharges": charges,
        "Contract": contract, "Churn": np.where(churn == 1, "Yes", "No"),
    })


def test_prepare_churn_simple_binarizes():
    X, y = prepare_churn_simple(_sample_df())
    assert set(y.unique()) <= {0, 1}
    assert "Churn" not in X.columns


def test_grid_search_returns_best():
    X, y = prepare_churn_simple(_sample_df())
    res = run_grid_search(X, y, param_grid={"n_estimators": [10, 20], "max_depth": [3, 5]})
    assert "best_params" in res and "best_score" in res
    assert res["n_candidates"] == 4           # 2 × 2
    assert res["elapsed_seconds"] >= 0


def test_compare_search_covers_both():
    X, y = prepare_churn_simple(_sample_df())
    result = compare_search(X, y)
    assert set(result["method"]) == {"grid", "random"}
    assert (result["best_score"] > 0.5).all()
