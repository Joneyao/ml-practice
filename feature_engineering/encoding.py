"""B2：类别变量编码——One-Hot/Label/Ordinal/Target，以及高基数编码的维度膨胀风险。"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_onehot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One-Hot 编码单列，返回每个类别一列的 0/1 DataFrame。"""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df[[col]])
    columns = encoder.get_feature_names_out([col])
    return pd.DataFrame(encoded, columns=columns, index=df.index)


def encode_label(df: pd.DataFrame, col: str) -> pd.Series:
    """Label 编码：类别 -> 任意整数（按字母序），不隐含真实大小关系。
    只适合树模型（分裂点不依赖数值大小含义），不建议喂给线性模型。
    """
    encoder = LabelEncoder()
    return pd.Series(encoder.fit_transform(df[col]), index=df.index, name=f"{col}_label")


def encode_ordinal(df: pd.DataFrame, col: str, order: list) -> pd.Series:
    """Ordinal 编码：按业务定义的顺序映射到整数，适用于真实有序的类别
    （如 Contract 的 Month-to-month < One year < Two year）。
    """
    mapping = {category: rank for rank, category in enumerate(order)}
    return df[col].map(mapping).rename(f"{col}_ordinal")


def encode_target_mean(df: pd.DataFrame, col: str, target_col: str) -> pd.Series:
    """Target Mean 编码：类别 -> 该类别下 target 的均值。

    注意：这是"整表一次性编码"的朴素实现，教学用于演示编码器怎么写。
    如果不加区分地在训练/测试切分之前对整表调用它，就是 B3 要专门
    复现的那种数据泄漏——B3 会把同样的思路包进符合 sklearn Transformer
    接口的类，正确地嵌进 Pipeline。
    """
    means = df.groupby(col)[target_col].mean()
    return df[col].map(means).rename(f"{col}_target_mean")


def cardinality_report(df: pd.DataFrame, col: str) -> dict:
    """报告某列的基数（唯一值数）、One-Hot 展开后的维度，
    以及"只出现一次的类别"占比——高基数编码过拟合风险的直接证据。
    """
    n_unique = df[col].nunique()
    n_rows = len(df)
    value_counts = df[col].value_counts()
    singleton_ratio = float((value_counts == 1).sum() / n_unique) if n_unique > 0 else 0.0
    return {
        "column": col,
        "n_rows": n_rows,
        "n_unique": n_unique,
        "onehot_dims": n_unique,
        "avg_rows_per_category": n_rows / n_unique if n_unique > 0 else 0.0,
        "singleton_ratio": singleton_ratio,
    }


def compare_encoding_dimensions(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """对比多列做 One-Hot 编码后的维度差异，用于展示"高基数列一编码，
    维度就爆炸"的坑。
    """
    return pd.DataFrame([cardinality_report(df, col) for col in cols])
