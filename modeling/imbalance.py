"""C3：不均衡数据——accuracy 陷阱 + class_weight/SMOTE 修复。"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def prepare_fraud(df, target="Class"):
    """丢 Time 列，其余作特征，target 转整数 0/1。"""
    data = df.copy()
    if "Time" in data.columns:
        data = data.drop(columns=["Time"])
    y = data[target].astype(int)
    X = data.drop(columns=[target])
    return X, y


def naive_accuracy_trap(X, y, seed=0):
    """永远预测多数类：准确率虚高，recall=0。"""
    from sklearn.model_selection import cross_val_predict
    dummy = DummyClassifier(strategy="most_frequent")
    pred = cross_val_predict(dummy, X, y, cv=5)
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
    }


def _scored(estimator, X, y, cv):
    scores = cross_validate(
        estimator, X, y, cv=cv,
        scoring=["precision", "recall", "f1", "roc_auc", "average_precision"],
    )
    return {
        "precision": float(scores["test_precision"].mean()),
        "recall": float(scores["test_recall"].mean()),
        "f1": float(scores["test_f1"].mean()),
        "roc_auc": float(scores["test_roc_auc"].mean()),
        "average_precision": float(scores["test_average_precision"].mean()),
    }


def compare_imbalance_strategies(X, y, seed=0):
    """三种策略对比（不看 accuracy，看 PR/F1/AUC）。SMOTE 只在训练折过采样。"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    class_weight = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    smote = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    rows = [
        {"strategy": "baseline", **_scored(baseline, X, y, cv)},
        {"strategy": "class_weight", **_scored(class_weight, X, y, cv)},
        {"strategy": "smote", **_scored(smote, X, y, cv)},
    ]
    return pd.DataFrame(rows)
