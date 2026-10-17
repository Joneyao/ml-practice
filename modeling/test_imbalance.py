import numpy as np
import pandas as pd
from modeling.imbalance import (
    prepare_fraud, naive_accuracy_trap, compare_imbalance_strategies,
)


def _imbalanced_df(n=2000, pos=20, seed=0):
    """构造一个 1% 欺诈率的小数据集，特征对两类有可分性。"""
    rng = np.random.default_rng(seed)
    neg = pd.DataFrame(rng.normal(0, 1, size=(n - pos, 4)),
                       columns=["V1", "V2", "V3", "Amount"])
    neg["Class"] = 0
    posd = pd.DataFrame(rng.normal(2.5, 1, size=(pos, 4)),
                        columns=["V1", "V2", "V3", "Amount"])
    posd["Class"] = 1
    return pd.concat([neg, posd], ignore_index=True).sample(frac=1, random_state=seed)


def test_prepare_fraud_drops_time_if_present():
    df = _imbalanced_df()
    df["Time"] = 1.0
    X, y = prepare_fraud(df)
    assert "Time" not in X.columns
    assert "Class" not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_naive_accuracy_trap_high_acc_zero_recall():
    df = _imbalanced_df()
    X, y = prepare_fraud(df)
    res = naive_accuracy_trap(X, y)
    assert res["accuracy"] > 0.95       # 准确率虚高
    assert res["recall"] == 0.0         # 一个正例都没抓到


def test_compare_imbalance_strategies_recall_improves():
    df = _imbalanced_df()
    X, y = prepare_fraud(df)
    result = compare_imbalance_strategies(X, y)
    assert set(result["strategy"]) == {"baseline", "class_weight", "smote"}
    base_recall = result.loc[result["strategy"] == "baseline", "recall"].iloc[0]
    cw_recall = result.loc[result["strategy"] == "class_weight", "recall"].iloc[0]
    # 处理不均衡后召回应提升
    assert cw_recall >= base_recall
