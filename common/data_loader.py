"""Titanic / House Prices (Ames) 数据集加载，统一走本地缓存。"""
import os
import pandas as pd


def load_titanic(cache_dir="data", force_download=True):
    """加载 Titanic 数据集。优先用缓存，没有缓存且允许下载时联网获取。"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "titanic.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    if not force_download:
        raise FileNotFoundError(f"缓存不存在且禁止下载: {cache_path}")
    import seaborn as sns
    df = sns.load_dataset("titanic")
    df.to_csv(cache_path, index=False)
    return df


def load_house_prices(cache_dir="data", force_download=True):
    """加载 House Prices (Ames) 数据集。优先用缓存，没有缓存且允许下载时联网获取。"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "house_prices.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    if not force_download:
        raise FileNotFoundError(f"缓存不存在且禁止下载: {cache_path}")
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(data_id=42165, as_frame=True)
    df = bunch.frame
    df.to_csv(cache_path, index=False)
    return df


def _strip_arff_quotes(value):
    """OpenML ARFF 来源的字符串列常带多余单引号（如 "'One year'"），统一去掉。"""
    if isinstance(value, str) and len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    return value


def load_telco_churn(cache_dir="data", force_download=True):
    """加载 Telco Customer Churn 数据集（OpenML data_id=42178，与 Kaggle
    blastchar/telco-customer-churn 同源）。优先用缓存，没有缓存且允许下载时
    联网获取。联网路径会清理 ARFF 遗留的引号，并把 TotalCharges 转成数值
    （11 行新客户的 TotalCharges 原始值是空格字符串，转换后变成 NaN，
    这是真实数据不是人为构造的缺陷）。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "telco_churn.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    if not force_download:
        raise FileNotFoundError(f"缓存不存在且禁止下载: {cache_path}")
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(data_id=42178, as_frame=True)
    df = bunch.frame.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(_strip_arff_quotes)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.to_csv(cache_path, index=False)
    return df


def load_titanic_raw(cache_dir="data", force_download=True):
    """加载 Titanic 原始版（OpenML data_id=40945），含 name/cabin 原始文本列，
    供特征工程篇从文本列构造 title/deck 特征。与 load_titanic() 不同：
    那个函数包装的是 seaborn 内置版本，deck 已被预处理过、没有 name 列，
    不适合本函数的用途。优先用缓存，没有缓存且允许下载时联网获取。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "titanic_raw.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    if not force_download:
        raise FileNotFoundError(f"缓存不存在且禁止下载: {cache_path}")
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(data_id=40945, as_frame=True)
    df = bunch.frame.copy()
    df["survived"] = df["survived"].astype(int)
    df.to_csv(cache_path, index=False)
    return df


def load_credit_fraud(cache_dir="data", force_download=True):
    """加载 Credit Card Fraud Detection 数据集（OpenML data_id=1597）。
    优先用缓存，没有缓存且允许下载时联网获取。Class 列统一转为整数 0/1。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "credit_fraud.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    if not force_download:
        raise FileNotFoundError(f"缓存不存在且禁止下载: {cache_path}")
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(data_id=1597, as_frame=True)
    df = bunch.frame
    # Class 列可能是带引号字符串或 category，统一转整数
    df["Class"] = pd.to_numeric(
        df["Class"].astype(str).str.strip("'\""), errors="coerce"
    ).astype(int)
    df.to_csv(cache_path, index=False)
    return df
