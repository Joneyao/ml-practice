"""E1：训练 Telco Churn 预测 pipeline 并用 joblib 持久化。
注意：这是教学演示，特征只取 3 个数值列，便于接口演示。
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from common.data_loader import load_telco_churn

FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_pipeline():
    """标准化 + 逻辑回归的最简 pipeline。"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def _prepare(df):
    data = df.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    X = data[FEATURES].fillna(data[FEATURES].median(numeric_only=True))
    y = (data["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    return X, y


def train_and_save(model_path):
    """用 Telco 数据训练并 joblib.dump 到 model_path。"""
    X, y = _prepare(load_telco_churn())
    pipe = build_pipeline().fit(X, y)
    joblib.dump(pipe, model_path)
    return pipe
