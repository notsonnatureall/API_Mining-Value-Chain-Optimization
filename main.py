import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from report_anomaly import detect_anomaly, ProductionInput,AnomalyOutput
from actual_tons import tons_predict, generate_weather_system, ForecastOutput, ForecastRequest, feature_engineering_tons
from cycle_time import predict_util, generate_weather_system, UtilRequest, UtilOutput
from sales_risk import predict_sales_risk, ContractRequest

app = FastAPI(title="Production Anomaly Detection API", version="1.0")

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "Anomaly API is running ✅"}

# =========================
# ANOMALY DETECTION
# =========================
@app.post("/detect-anomaly", response_model=List[AnomalyOutput])
def predict(request: ProductionInput):
    result = detect_anomaly(request)
    return {
        "prediction_result" : result,
        "ai_recommendation" : "loremipsum del torot"
    }
    
# =========================
# ACTUAL TONS PREDICTION
# =========================
@app.post("/predict-tons")
def predict_tons_api(request: ForecastRequest):
    result = tons_predict(request)
    return {
        "prediction_result" : result,
        "ai_recommendation" : "loremipsum del torot"
    }

# =========================
#  FLEET UTILIZATION PREDICTION
# =========================
@app.post("/predict-utilization")
def predict_util_api(request: UtilRequest):
    result = predict_util(request)
    return {
        "prediction_result" : result,
        "ai_recommendation" : "loremipsum del torot"
    }

# =========================
#  SALES RISK PREDICTION
# =========================
@app.post("/predict-sales-risk")
def predict_sales_risk_api(request: ContractRequest):
    result = predict_sales_risk(request)
    return{
        "prediction_result" : result,
        "ai_recommendtaion" : "loremipsum del torot"
    }