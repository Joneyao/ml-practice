"""B3：ColumnTransformer + Pipeline，复现并修复"fit_transform 全量数据再切分"的数据泄漏。"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Target Mean 编码器，实现 sklearn Transformer 接口，可嵌进 Pipeline/
    ColumnTransformer。fit 时记录每个类别的目标均值，transform 时按映射
    替换；fit 阶段没见过的类别在 transform 时退回全局均值。
    """

    def __init__(self, col: str):
        self.col = col

    def fit(self, X: pd.DataFrame, y):
        y_series = pd.Series(np.asarray(y), index=X.index)
        merged = pd.DataFrame({self.col: X[self.col], "y": y_series})
        self.global_mean_ = float(merged["y"].mean())
        self.mapping_ = merged.groupby(self.col)["y"].mean().to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        return X[[self.col]].apply(
            lambda s: s.map(self.mapping_).fillna(self.global_mean_)
        ).to_numpy()


def build_preprocessor(numeric_cols: list, target_enc_col: str) -> ColumnTransformer:
    """构造数值列标准化 + 高基数列 Target Mean 编码的预处理器。"""
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), numeric_cols),
        ("target_enc", TargetMeanEncoder(target_enc_col), [target_enc_col]),
    ])


def build_pipeline(numeric_cols: list, target_enc_col: str) -> Pipeline:
    """构造完整 Pipeline：预处理器 + 逻辑回归分类器。"""
    return Pipeline([
        ("pre", build_preprocessor(numeric_cols, target_enc_col)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def leaked_cv_score(df: pd.DataFrame, target_col: str, numeric_cols: list,
                     target_enc_col: str, seed: int = 0) -> float:
    """事故复现：先在全量数据上 fit_transform 预处理器（Target Mean
    编码器看到了全部样本的目标值），再把转换后的矩阵喂给交叉验证——
    每一折的"训练"信息已经通过编码器泄漏进了"验证"折。
    """
    X = df[numeric_cols + [target_enc_col]]
    y = df[target_col]
    preprocessor = build_preprocessor(numeric_cols, target_enc_col)
    transformed = preprocessor.fit_transform(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(LogisticRegression(max_iter=1000), transformed, y, cv=cv)
    return float(scores.mean())


def correct_cv_score(df: pd.DataFrame, target_col: str, numeric_cols: list,
                      target_enc_col: str, seed: int = 0) -> float:
    """正确做法：把预处理器放进 Pipeline，交给 cross_val_score 整体处理——
    每一折里，预处理器只在当折的训练集上 fit，验证集只参与 transform。
    """
    X = df[numeric_cols + [target_enc_col]]
    y = df[target_col]
    pipe = build_pipeline(numeric_cols, target_enc_col)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv)
    return float(scores.mean())


def compare_leakage(df: pd.DataFrame, target_col: str, numeric_cols: list,
                     target_enc_col: str, seed: int = 0) -> pd.DataFrame:
    """对比泄漏管道 vs 正确管道的交叉验证分数，返回两行对比表。"""
    leaked = leaked_cv_score(df, target_col, numeric_cols, target_enc_col, seed)
    correct = correct_cv_score(df, target_col, numeric_cols, target_enc_col, seed)
    return pd.DataFrame([
        {"pipeline": "leaked (fit_transform before split)", "cv_score": leaked},
        {"pipeline": "correct (Pipeline, fit per fold)", "cv_score": correct},
    ])
