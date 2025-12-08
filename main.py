import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from report_anomaly import detect_anomaly, ProductionInput,AnomalyOutput
from actual_tons import tons_predict, generate_weather_system, ForecastOutput, ForecastRequest

app = FastAPI(title="Production Anomaly Detection API", version="1.0")


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "Anomaly API is running ✅"}


# =========================
# MAIN ANOMALY DETECTION
# =========================
@app.post("/detect-anomaly", response_model=List[AnomalyOutput])
def predict(request: ProductionInput):
    result = detect_anomaly(request)
    return {
        "prediction_result" : result,
        "ai_recommendation" : "loremipsum del torot"
    }
    
# =========================
# MAIN ACTUAL TONS PREDICTION
# =========================
@app.post("/predict-tons", response_model=List[ForecastOutput])
def predict_tons(request: ForecastRequest):
    result = tons_predict(request)
    return {
        "prediction_result" : result,
        "ai_recommendation" : "loremipsum del torot"
    }
