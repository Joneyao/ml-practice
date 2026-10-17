"""E1：最小预测接口。
⚠️ 演示用：无鉴权/无限流/无 HTTPS。生产需另加认证、限流、网关、版本管理。
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class ChurnRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int


def create_app(model_path):
    """加载 joblib 模型，暴露 POST /predict。"""
    app = FastAPI(title="Telco Churn 预测（演示）")
    model = joblib.load(model_path)

    @app.post("/predict", response_model=ChurnResponse)
    def predict(req: ChurnRequest):
        X = pd.DataFrame([{
            "tenure": req.tenure,
            "MonthlyCharges": req.MonthlyCharges,
            "TotalCharges": req.TotalCharges,
        }])
        proba = float(model.predict_proba(X)[0, 1])
        return ChurnResponse(churn_probability=proba, churn_prediction=int(proba >= 0.5))

    return app
