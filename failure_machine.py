import pandas as pd
import joblib

from pydantic import BaseModel
from datetime import datetime

import numpy as np
import pandas as pd

model_path = "model/model_failure.pkl"

def simulate_single_point_per_machine(seed=None):

    if seed is not None:
        np.random.seed(seed)

    machines = {
        "EX-204": "Excavator",
        "CB-112": "Conveyor",
        "DR-221": "Drill",
        "HT-405": "Haul Truck",
        "CR-08":  "Crusher",
        "LD-315": "Loader"
    }

    base_profiles = {
        "EX-204":  {"rpm": 1400, "vib": 0.32, "hyd": 150},
        "CB-112":  {"rpm": 900,  "vib": 0.20, "hyd": 80},
        "DR-221":  {"rpm": 1600, "vib": 0.45, "hyd": 140},
        "HT-405":  {"rpm": 1300, "vib": 0.28, "hyd": 120},
        "CR-08":   {"rpm": 1700, "vib": 0.55, "hyd": 160},
        "LD-315":  {"rpm": 1250, "vib": 0.30, "hyd": 110}
    }

    noise = {
        "engine_hours": 5,
        "ambient_temp": 1,
        "engine_temp": 1,
        "oil_pressure": 0.1,
        "hydraulic_pressure": 3,
        "rpm": 30,
        "vibration": 0.03,
        "fuel_rate": 0.1
    }

    rows = []

    for code, mtype in machines.items():

        profile = base_profiles[code]

        point = {
            "engine_hours"      : 4000 + np.random.uniform(0, 500) + np.random.normal(0, noise["engine_hours"]),
            "ambient_temp"      : 25 + np.random.normal(0, noise["ambient_temp"]),
            "engine_temp"       : 70 + np.random.uniform(0, 5) + np.random.normal(0, noise["engine_temp"]),
            "oil_pressure"      : 2.8 + np.random.normal(0, noise["oil_pressure"]),
            "hydraulic_pressure": profile["hyd"] + np.random.normal(0, noise["hydraulic_pressure"]),
            "rpm"               : profile["rpm"] + np.random.normal(0, noise["rpm"]),
            "vibration"         : profile["vib"] + np.random.normal(0, noise["vibration"]),
            "fuel_rate"         : 3.0 + np.random.normal(0, noise["fuel_rate"]),
            "machine_id"        : code,
            "machine_type"      : mtype
        }

        rows.append(point)

    return pd.DataFrame(rows)

def predict_from_simulated_data(df_sim):
    model = joblib.load(model_path)

    feature_cols = [
        "engine_hours", "ambient_temp", "engine_temp",
        "oil_pressure", "hydraulic_pressure", "rpm",
        "vibration", "fuel_rate"
    ]

    X = df_sim[feature_cols]

    df_sim["prediction"] = model.predict(X).astype(int)

    try:
        df_sim["prediction_prob"] = model.predict_proba(X)[:, 1].astype(float)
    except:
        df_sim["prediction_prob"] = None

    return df_sim[[
        "machine_id",
        "machine_type",
        "prediction",
        "prediction_prob"
    ]].to_dict(orient="records")


