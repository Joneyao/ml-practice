import os
import pandas as pd
from common import plotting, data_loader


def test_setup_font_returns_name():
    name = plotting.setup_cjk_font()
    assert isinstance(name, str) and len(name) > 0


def test_load_titanic_from_cache(tmp_path):
    cache_dir = tmp_path
    fixture = pd.DataFrame({
        "survived": [0, 1, 1],
        "pclass": [3, 1, 2],
        "age": [22.0, None, 26.0],
        "sex": ["male", "female", "female"],
    })
    fixture.to_csv(cache_dir / "titanic.csv", index=False)
    df = data_loader.load_titanic(cache_dir=str(cache_dir), force_download=False)
    assert len(df) == 3
    assert "survived" in df.columns


def test_load_house_prices_from_cache(tmp_path):
    cache_dir = tmp_path
    fixture = pd.DataFrame({
        "SalePrice": [200000, 350000, 150000],
        "GrLivArea": [1500, 2200, 900],
        "OverallQual": [6, 8, 4],
    })
    fixture.to_csv(cache_dir / "house_prices.csv", index=False)
    df = data_loader.load_house_prices(cache_dir=str(cache_dir), force_download=False)
    assert len(df) == 3
    assert "SalePrice" in df.columns


def test_load_telco_churn_from_cache(tmp_path):
    cache_dir = tmp_path
    fixture = pd.DataFrame({
        "gender": ["Female", "Male", "Female"],
        "tenure": [1, 34, 2],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": [29.85, 1889.5, 108.15],
        "Churn": ["No", "No", "Yes"],
    })
    fixture.to_csv(cache_dir / "telco_churn.csv", index=False)
    df = data_loader.load_telco_churn(cache_dir=str(cache_dir), force_download=False)
    assert len(df) == 3
    assert "Churn" in df.columns


def test_load_titanic_raw_from_cache(tmp_path):
    cache_dir = tmp_path
    fixture = pd.DataFrame({
        "name": ["Allen, Miss. Elisabeth Walton", "Allison, Master. Hudson Trevor"],
        "cabin": ["B5", None],
        "pclass": [1, 1],
        "sex": ["female", "male"],
        "age": [29.0, 0.9167],
        "survived": [1, 1],
    })
    fixture.to_csv(cache_dir / "titanic_raw.csv", index=False)
    df = data_loader.load_titanic_raw(cache_dir=str(cache_dir), force_download=False)
    assert len(df) == 2
    assert "name" in df.columns
    assert "cabin" in df.columns


def test_load_credit_fraud_from_cache(tmp_path):
    cache_dir = tmp_path
    fixture = pd.DataFrame({
        "V1": [-1.2, 0.5, 2.1, -0.3],
        "V2": [0.3, -1.1, 0.8, 0.1],
        "Amount": [149.62, 2.69, 378.66, 12.99],
        "Class": [0, 0, 1, 0],
    })
    fixture.to_csv(cache_dir / "credit_fraud.csv", index=False)
    df = data_loader.load_credit_fraud(cache_dir=str(cache_dir), force_download=False)
    assert len(df) == 4
    assert "Class" in df.columns
    assert df["Class"].dtype.kind == "i"  # 整数类型
