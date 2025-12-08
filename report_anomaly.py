import fastapi 
import pandas as pd

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

# =========================
# INPUT SCHEMA (FORM UI)
# =========================
class ProductionInput(BaseModel):
    date: str = Field(..., example="2025-12-06")
    unit_id: str = Field(..., example="EXC-001")

    target_tons: float = Field(..., ge=0)
    actual_tons: float = Field(..., ge=0)

    breakdown_hours: float = Field(..., ge=0)
    utilization_percent: float = Field(..., ge=0, le=100)

    precip_mm: float = Field(..., ge=0)
    fuel_consumption_l: float = Field(..., ge=0)


# =========================
# OUTPUT SCHEMA
# =========================
class AnomalyOutput(BaseModel):
    date: str
    type_anomaly: str
    trigger_value: float
    description: str

def detect_anomaly(data: ProductionInput):
    try:
        # =========================
        # CREATE DATAFRAME
        # =========================
        feature = pd.DataFrame([data.model_dump()])

        # =========================
        # FEATURE ENGINEERING
        # =========================
        feature["production_drop_pct"] = (
            (feature["actual_tons"] - feature["target_tons"]) / feature["target_tons"]
        )

        # =========================
        # INIT FLAGS
        # =========================
        feature["production_anomaly"] = 0
        feature["breakdown_anomaly"] = 0
        feature["rainfall_anomaly"] = 0
        feature["fuel_anomaly"] = 0

        anomalies = []

        # =========================
        # RULE 1: PRODUKSI TURUN > 30%
        # =========================
        if feature.loc[0, "production_drop_pct"] < -0.3:
            anomalies.append({
                "date": data.date,
                "type_anomaly": "Production Anomaly",
                "trigger_value": round(feature.loc[0, "production_drop_pct"], 3),
                "description": "Produksi turun lebih dari 30% dari target"
            })

        # =========================
        # RULE 2: BREAKDOWN TINGGI + UTILIZATION RENDAH
        # =========================
        if (
            feature.loc[0, "breakdown_hours"] > 4 and
            feature.loc[0, "utilization_percent"] < 50
        ):
            anomalies.append({
                "date": data.date,
                "type_anomaly": "Breakdown Anomaly",
                "trigger_value": feature.loc[0, "breakdown_hours"],
                "description": "Jam breakdown tinggi tapi utilisasi rendah"
            })

        # =========================
        # RULE 3: HUJAN TINGGI + PRODUKSI TIDAK REALISTIS
        # =========================
        if (
            feature.loc[0, "precip_mm"] > 40 and
            feature.loc[0, "actual_tons"] > feature.loc[0, "target_tons"] * 1.1
        ):
            anomalies.append({
                "date": data.date,
                "type_anomaly": "Rainfall Anomaly",
                "trigger_value": feature.loc[0, "precip_mm"],
                "description": "Curah hujan tinggi namun produksi sangat tinggi (tidak realistis)"
            })

        # =========================
        # RULE 4: FUEL CONSUMPTION SPIKE
        # =========================
        FUEL_THRESHOLD_DEFAULT = 300  # Bisa kamu sesuaikan dari data historis

        if feature.loc[0, "fuel_consumption_l"] > FUEL_THRESHOLD_DEFAULT:
            anomalies.append({
                "date": data.date,
                "type_anomaly": "Fuel Anomaly",
                "trigger_value": feature.loc[0, "fuel_consumption_l"],
                "description": "Konsumsi BBM melebihi batas wajar"
            })

        # =========================
        # JIKA TIDAK ADA ANOMALI
        # =========================
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
