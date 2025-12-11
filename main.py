import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from report_anomaly import detect_anomaly, ProductionInput,AnomalyOutput
from actual_tons import tons_predict, generate_weather_system, ForecastOutput, ForecastRequest, feature_engineering_tons
from cycle_time import predict_util, generate_weather_system, UtilRequest, UtilOutput
from sales_risk import predict_sales_risk, ContractRequest
from truck_inspection import predict_inspection, PredictionRequest
from failure_machine import predict_from_simulated_data, simulate_single_point_per_machine

app = FastAPI(title="Production Anomaly Detection API", version="1.0")

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "Anomaly API is running ✅"}

@app.get('/failure-machine')
def failure_machine():
    df_sim = simulate_single_point_per_machine()
    df_result = predict_from_simulated_data(df_sim)
    return df_result

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
    probability, days_left = predict_sales_risk(request)
    if probability > 0.7:
        status = "CRITICAL"
        rec = "Tolak kontrak atau negosiasi ulang deadline segera."
    elif probability > 0.4:
        status = "MODERATE"
        rec = "Pantau produksi ketat, siapkan armada cadangan."
    else:
        status = "SAFE"
        rec = "Kontrak aman untuk diproses."

    if days_left < 0:
        rec = "Tanggal deadline tidak valid atau sudah lewat."

    return {
        "prediction_result": {
            "delay_probability": round(probability, 4),
            "risk_status": status,
            "days_until_deadline": days_left,
            "description": rec
        },
        "ai_recommendation": "Loremipsum del torot"
    }

# =========================
#  SALES RISK PREDICTION
# =========================

@app.post("/predict-inspection")
def predict(request: PredictionRequest):
    result = predict_inspection(request)
    return {
        "prediction" : result,
        "ai_recommendation" : "loremipsum del torot"
    }