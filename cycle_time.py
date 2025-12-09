import fastapi 
import pandas as pd
import numpy as np
import joblib
import random
import sys
import __main__

from xgboost import XGBRegressor
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from datetime import date
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin

class feature_engineering_cycle(BaseEstimator, TransformerMixin):
  def __init__(self) -> None:
      super().__init__()

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X['broken_ratio'] = X['breakdown_hours'] / X['availability_hours']
    X['fuel_consumption_ratio'] = X['fuel_consumption_l'] / X['target_tons']
    X['rainy heavy flag'] = np.where(X['precip_mm'] > 30, 1, 0)
    X['wind_high_flag'] = np.where(X['wind_speed_kmh'] > 20, 1, 0)
    X['downtime_ratio'] = X['breakdown_hours'] / 24
    X['production_pressure'] = X['actual_tons'] / X['target_tons']
    X['wetness_factor'] = X['precip_mm'] * X['humidity']

    return X
  
if "feature_engineering_cycle" not in sys.modules['__main__'].__dict__:
    sys.modules['__main__'].feature_engineering_cycle = feature_engineering_cycle

model_actual_tons = joblib.load("model/cycle_prediction_pipeline.pkl")

# =========================
# INPUT SCHEMA
# =========================
class UtilRequest(BaseModel):
    date: date
    target_tons: float = Field(..., ge=0)
    actual_tons: float = Field(..., ge=0)
    hauling_distance_km: float = Field(..., ge=0)
    availability_hours: float = Field(..., ge=0, le=24)
    breakdown_hours: float = Field(..., ge=0)
    fuel_consumption_l: float = Field(..., ge=0)

# =========================
# OUTPUT SCHEMA
# =========================
class UtilOutput(BaseModel):
    utilization_percent: float

# =========================
# GENERATE WEATHER SYSTEM
# =========================
def generate_weather_system():
    
    weather_map = {
        0: "Clear",
        1: "Cloudy",
        2: "Light Rain",
        3: "Heavy Rain"
    }

    weather_condition = random.choice(list(weather_map.keys()))
    if weather_condition == 0:  # Clear
        precip_mm = round(np.random.uniform(0, 0.5), 2)
        humidity = round(np.random.uniform(55, 70), 1)
        weather_condition = "Clear"
    elif weather_condition == 1:  # Cloudy
        precip_mm = round(np.random.uniform(0, 1.5), 2)
        humidity = round(np.random.uniform(65, 80), 1)
        weather_condition = "Cloudy"
    elif weather_condition == 2:  # Light Rain
        precip_mm = round(np.random.uniform(2, 20), 2)
        humidity = round(np.random.uniform(75, 90), 1)
        weather_condition = "Light Rain"
    else:  # Heavy Rain
        precip_mm = round(np.random.uniform(30, 120), 2)
        humidity = round(np.random.uniform(85, 98), 1)
        weather_condition = "Heavy Rain"

    temperature_c = round(np.random.uniform(28, 33), 1)
    wind_speed_kmh = round(np.random.uniform(5, 35), 1)

    return {
        "temperature_c": temperature_c,
        "humidity": humidity,
        "precip_mm": precip_mm,
        "wind_speed_kmh": wind_speed_kmh,
        "weather_condition": weather_condition
    }

# =========================
# CYCLE PREDICTION
# =========================
def predict_util(data):
    data_weather = generate_weather_system()
    data = {
        "date": data.date,
        "target_tons": data.target_tons,
        "actual_tons": data.actual_tons,
        "hauling_distance_km": data.hauling_distance_km,
        "availability_hours": data.availability_hours,
        "breakdown_hours": data.breakdown_hours,
        "fuel_consumption_l": data.fuel_consumption_l
        }
    data.update(data_weather)
    data = pd.DataFrame([data])
    result = model_actual_tons.predict(data)
    return [{
                "utilization_percent": int(result[0])
    }]
