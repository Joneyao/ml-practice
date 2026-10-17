"""B1：从一列数据里构造特征——称谓/甲板/年龄分箱/交叉特征。"""
import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_RARE_TITLES = {
    "Dr", "Rev", "Col", "Major", "Capt", "Jonkheer", "Don",
    "the Countess", "Sir", "Lady", "Dona",
}
_TITLE_PATTERN = re.compile(r",\s*([^.]+)\.")


def extract_title(name: str) -> str:
    """从 "姓, 称谓. 名" 格式的姓名提取称谓，合并稀有/等价称谓。"""
    match = _TITLE_PATTERN.search(name)
    title = match.group(1).strip() if match else "Unknown"
    if title in _RARE_TITLES:
        return "Rare"
    if title in ("Mlle", "Ms"):
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return title


def extract_deck(cabin) -> str:
    """取船舱号首字母作为甲板标识，缺失返回 "Unknown"。"""
    if pd.isna(cabin):
        return "Unknown"
    return str(cabin)[0]


def bin_age(age) -> str:
    """年龄分箱：child/teen/young_adult/adult/senior，缺失返回 "Unknown"。"""
    if pd.isna(age):
        return "Unknown"
    bins = [0, 12, 18, 35, 60, 150]
    labels = ["child", "teen", "young_adult", "adult", "senior"]
    for i in range(len(bins) - 1):
        if bins[i] < age <= bins[i + 1]:
            return labels[i]
    return "Unknown"


def make_cross_feature(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    """两列值用下划线拼接成交叉特征，如 pclass=1 + sex=female -> "1_female"。"""
    return df[col_a].astype(str) + "_" + df[col_b].astype(str)


def survival_rate_spread(df: pd.DataFrame, col: str, target_col: str = "survived") -> float:
    """按 col 分组算 target_col 均值，组间最大值减最小值，衡量该特征的区分度。"""
    rates = df.groupby(col)[target_col].mean()
    return float(rates.max() - rates.min())


def _cv_score(X: pd.DataFrame, y: pd.Series, num_cols: list,
              cat_cols: list, seed: int = 42) -> float:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv)
    return float(scores.mean())


def compare_feature_sets(df: pd.DataFrame, target_col: str = "survived") -> pd.DataFrame:
    """对比 baseline（pclass+age+fare+sex，age 走中位数填补）vs engineered
    （pclass+fare+title，完全不碰缺失的 age 列）两套特征集的交叉验证分数。
    """
    working = df.copy()
    working["title"] = working["name"].apply(extract_title)
    y = working[target_col]

    baseline_cols = ["pclass", "age", "fare"]
    baseline_score = _cv_score(
        working[baseline_cols + ["sex"]], y, baseline_cols, ["sex"],
    )

    engineered_cols = ["pclass", "fare"]
    engineered_score = _cv_score(
        working[engineered_cols + ["title"]], y, engineered_cols, ["title"],
    )

    return pd.DataFrame([
        {"feature_set": "baseline", "cv_score": baseline_score},
        {"feature_set": "engineered", "cv_score": engineered_score},
    ])
