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
