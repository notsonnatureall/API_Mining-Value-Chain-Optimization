from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Production Anomaly Detection API", version="1.0")

# === FIX CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProductionInput(BaseModel):
    date: str = Field(..., example="2025-12-06")
    unit_id: str = Field(..., example="EXC-001")

    target_tons: float = Field(..., ge=0)
    actual_tons: float = Field(..., ge=0)

    breakdown_hours: float = Field(..., ge=0)
    utilization_percent: float = Field(..., ge=0, le=100)

    precip_mm: float = Field(..., ge=0)
    fuel_consumption_l: float = Field(..., ge=0)


class AnomalyOutput(BaseModel):
    date: str
    type_anomaly: str
    trigger_value: float
    description: str


@app.get("/")
def home():
    return {"status": "Anomaly API is running ✅"}


@app.post("/detect-anomaly", response_model=List[AnomalyOutput])
def detect_anomaly(data: ProductionInput):
    try:
        feature = pd.DataFrame([data.model_dump()])

        feature["production_drop_pct"] = (
            (feature["actual_tons"] - feature["target_tons"]) / feature["target_tons"]
        )

        anomalies = []

        if feature.loc[0, "production_drop_pct"] < -0.3:
            anomalies.append({
                "date": data.date,
                "type_anomaly": "Production Anomaly",
                "trigger_value": round(feature.loc[0, "production_drop_pct"], 3),
                "description": "Produksi turun lebih dari 30% dari target"
            })

        if feature.loc[0, "breakdown_hours"] > 4 and feature.loc[0, "utilization_percent"] < 50:
            anomalies.append({
                "date": data.date,
                "type_anomaly": "Breakdown Anomaly",
                "trigger_value": feature.loc[0, "breakdown_hours"],
                "description": "Jam breakdown tinggi tapi utilisasi rendah"
            })

        if feature.loc[0, "precip_mm"] > 40 and feature.loc[0, "actual_tons"] > feature.loc[0, "target_tons"] * 1.1:
            anomalies.append({
                "date": data.date,
                "type_anomaly": "Rainfall Anomaly",
                "trigger_value": feature.loc[0, "precip_mm"],
                "description": "Curah hujan tinggi namun produksi sangat tinggi (tidak realistis)"
            })

        FUEL_THRESHOLD_DEFAULT = 300

        if feature.loc[0, "fuel_consumption_l"] > FUEL_THRESHOLD_DEFAULT:
            anomalies.append({
                "date": data.date,
                "type_anomaly": "Fuel Anomaly",
                "trigger_value": feature.loc[0, "fuel_consumption_l"],
                "description": "Konsumsi BBM melebihi batas wajar"
            })

        if len(anomalies) == 0:
            return [{
                "date": data.date,
                "type_anomaly": "Normal",
                "trigger_value": 0,
                "description": "Tidak terdeteksi anomali"
            }]

        return anomalies

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
