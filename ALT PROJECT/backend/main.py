from contextlib import asynccontextmanager
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Dynamic Path Resolution ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pdm_framework_artifacts.joblib')

artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    artifacts.update(joblib.load(MODEL_PATH))
    yield
    artifacts.clear()

app = FastAPI(title="Industrial PdM Core Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TelemetryBatch(BaseModel):
    machine_type: str
    air_temp: float
    process_temp: float
    rpm: float
    torque: float
    tool_wear: float

@app.post("/predict")
def predict_machine_health(telemetry: TelemetryBatch):
    clf = artifacts.get("classifier")
    rul_model = artifacts.get("rul_model")

    if not clf or not rul_model:
        raise HTTPException(status_code=503, detail="Models not loaded")

    temp_diff = telemetry.process_temp - telemetry.air_temp
    mech_power = telemetry.torque * (telemetry.rpm * (2 * np.pi / 60))

    row = pd.DataFrame([{
        "Air temperature [K]": telemetry.air_temp,
        "Process temperature [K]": telemetry.process_temp,
        "Rotational speed [rpm]": telemetry.rpm,
        "Torque [Nm]": telemetry.torque,
        "Tool wear [min]": telemetry.tool_wear,
        "Temp_Difference": temp_diff,
        "Mechanical_Power": mech_power,
        "Type": telemetry.machine_type,
    }])

    failure_prob = float(clf.predict_proba(row)[0, 1])
    est_rul = float(max(0.0, rul_model.predict(row)[0]))

    risk_level = "CRITICAL" if failure_prob >= 0.75 else "WARNING" if failure_prob >= 0.40 else "NOMINAL"

    return {
        "failure_predicted": failure_prob >= 0.5,
        "failure_probability": round(failure_prob, 4),
        "risk_level": risk_level,
        "estimated_rul_minutes": round(est_rul, 1),
        "mechanical_power_w": round(mech_power, 2),
        "temp_difference": round(temp_diff, 2),
        "feature_importance": artifacts["importance"],
    }