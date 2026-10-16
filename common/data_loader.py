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
