import pandas as pd 
import joblib
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

model_sales = joblib.load("model/delivery_risk_pipeline.pkl")
model_capacity = joblib.load("model/system_capacity.pkl")

class ContractRequest(BaseModel):
    customer_id: str        # Contoh: "D"
    loading_port: str       # Contoh: "Port North"
    required_tons: int      # Contoh: 25000
    deadline_date: str

def predict_sales_risk(data):
    input_data = pd.DataFrame([{
        'customer': data.customer_id,
        'loading_port': data.loading_port,
        'required_tons': data.required_tons,
        'weekly_capacity_est': model_capacity
    }])

    prob_delay = model_sales.predict_proba(input_data)[0][1]
    return prob_delay

