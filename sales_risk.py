import pandas as pd
import joblib
import os
from pydantic import BaseModel
from datetime import datetime

# --- LOAD MODEL ---
# Pastikan path folder "model/" sesuai dengan struktur folder Anda
try:
    model_path = "model/delivery_risk_pipeline.pkl" 
    capacity_path = "model/system_capacity.pkl"
    
    if os.path.exists(model_path):
        model_sales = joblib.load(model_path)
    else:
        model_sales = None
        print(f"Warning: {model_path} tidak ditemukan.")

    if os.path.exists(capacity_path):
        model_capacity = joblib.load(capacity_path)
    else:
        model_capacity = 42000.0 # Default fallback

except Exception as e:
    print(f"Error loading model: {e}")
    model_sales = None
    model_capacity = 42000.0

# --- DATA STRUCTURE ---
class ContractRequest(BaseModel):
    customer_id: str        # Contoh: "D"
    loading_port: str       # Contoh: "Port North"
    required_tons: int      # Contoh: 25000
    deadline_date: str      # Contoh: "2025-12-12"


def predict_sales_risk(data: ContractRequest):
    if not model_sales:
        return {"error": "Model belum siap"}
    input_df = pd.DataFrame([{
        'customer': data.customer_id,
        'loading_port': data.loading_port,
        'required_tons': data.required_tons,
        'weekly_capacity_est': model_capacity
    }])

    try:
        prob_delay = model_sales.predict_proba(input_df)[0][1]
    except Exception as e:
        return 0.0, "Error pada model AI"
    try:
        deadline = datetime.strptime(data.deadline_date, "%Y-%m-%d")
        today = datetime.now()
        days_left = (deadline - today).days

        if days_left < 3:
            prob_delay = max(prob_delay, 0.99) # Paksa risiko jadi 99%
        elif days_left < 7:
            prob_delay = max(prob_delay, 0.85) # Paksa risiko jadi 85%
            
    except ValueError:
        days_left = -1 # Format tanggal salah

    return float(prob_delay), days_left