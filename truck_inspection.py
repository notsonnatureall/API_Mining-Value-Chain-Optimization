import joblib
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

TRUCK_LABEL_COLS = [
    "Overheating",
    "Oil Leak",
    "Knocking Noise",
    "Fuel Low",
    "Brake Issue",
    "Radiator Leak",
    "Dirty Air Filter",
    "Excessive Smoke",
    "Low Hydraulic Oil"
]

pipeline = joblib.load("model/model_Truck_Inspection.pkl")
label_cols = TRUCK_LABEL_COLS

class PredictionRequest(BaseModel):
    Brand: str
    Type: str 
    Model: str
    PKM1: int = Field(..., ge=0, le=1)
    PKM2: int = Field(..., ge=0, le=1)
    PKM3: int = Field(..., ge=0, le=1)
    PKM4: int = Field(..., ge=0, le=1)
    PKM5: int = Field(..., ge=0, le=1)
    PKM6: int = Field(..., ge=0, le=1)
    PKM7: int = Field(..., ge=0, le=1)
    PKM8: int = Field(..., ge=0, le=1)
    PKM9: int = Field(..., ge=0, le=1)
    PKM10: int = Field(..., ge=0, le=1)
    PKM11: int = Field(..., ge=0, le=1)
    PKM12: int = Field(..., ge=0, le=1)
    PKM13: int = Field(..., ge=0, le=1)
    PKM14: int = Field(..., ge=0, le=1)
    PKM15: int = Field(..., ge=0, le=1)

def predict_inspection(data):
    sample = pd.DataFrame([{
            "Brand": data.Brand,
            "Type": data.Type,
            "Model": data.Model,

            "PKM1": data.PKM1,
            "PKM2": data.PKM2,
            "PKM3": data.PKM3,
            "PKM4": data.PKM4,
            "PKM5": data.PKM5,
            "PKM6": data.PKM6,
            "PKM7": data.PKM7,
            "PKM8": data.PKM8,
            "PKM9": data.PKM9,
            "PKM10": data.PKM10,
            "PKM11": data.PKM11,
            "PKM12": data.PKM12,
            "PKM13": data.PKM13,
            "PKM14": data.PKM14,
            "PKM15": data.PKM15}])
    
    prediction = pipeline.predict(sample)
    hasil = pd.DataFrame(prediction, columns=label_cols)

    return hasil.to_dict(orient="records")[0]
