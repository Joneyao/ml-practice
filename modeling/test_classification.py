import numpy as np
import pandas as pd
from modeling.classification import prepare_churn, compare_algorithms


def _sample_df():
    rng = np.random.default_rng(0)
    n = 300
    tenure = rng.integers(0, 72, size=n)
    charges = rng.normal(65, 30, size=n).clip(18, 120)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    # churn 与 tenure 负相关、与 month-to-month 正相关
    logit = -0.04 * tenure + (contract == "Month-to-month") * 1.2
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")
    return pd.DataFrame({
        "tenure": tenure, "MonthlyCharges": charges,
        "Contract": contract, "Churn": churn_label,
    })


def test_prepare_churn_binarizes_target():
    X, y = prepare_churn(_sample_df())
    assert set(y.unique()) <= {0, 1}
    assert "Churn" not in X.columns
    assert X.isna().sum().sum() == 0


def test_compare_algorithms_covers_three():
    df = _sample_df()
    X, y = prepare_churn(df)
    result = compare_algorithms(X, y)
    assert set(result["algorithm"]) == {"logreg", "random_forest", "gbdt"}
    assert (result["roc_auc"] > 0.5).all()      # 都比瞎猜强
    assert (result["fit_seconds"] >= 0).all()
