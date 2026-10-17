import os
from fastapi.testclient import TestClient
from deployment.train_and_save import build_pipeline, train_and_save
from deployment.app import create_app


def test_build_pipeline_predicts(tmp_path):
    import numpy as np, pandas as pd
    pipe = build_pipeline()
    X = pd.DataFrame({
        "tenure": [1, 60, 12], "MonthlyCharges": [70.0, 20.0, 99.0],
        "TotalCharges": [70.0, 1200.0, 1188.0],
    })
    y = np.array([1, 0, 1])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == 3


def test_train_and_save_creates_file(tmp_path):
    model_path = str(tmp_path / "model.pkl")
    train_and_save(model_path)
    assert os.path.exists(model_path)


def test_predict_endpoint_returns_probability(tmp_path):
    model_path = str(tmp_path / "model.pkl")
    train_and_save(model_path)
    app = create_app(model_path)
    client = TestClient(app)
    resp = client.post("/predict", json={
        "tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert "churn_prediction" in body
