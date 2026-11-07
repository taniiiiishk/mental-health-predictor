from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mental Health Treatment Predictor")

# Load the model and column names saved from the notebook
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")


class PredictRequest(BaseModel):
    data: Dict[str, Any]


@app.get("/")
def home():
    return {"message": "API is running. Open /docs to test it."}


@app.get("/features")
def features():
    """Shows which input fields the model expects."""
    return {"columns": columns}


@app.post("/predict")
def predict(request: PredictRequest):
    # Any missing feature becomes NaN (the pipeline imputes it)
    row = {col: request.data.get(col) for col in columns}
    df = pd.DataFrame([row], columns=columns)

    try:
        pred = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "treatment": "Yes" if pred == 1 else "No",
        "probability_yes": round(proba, 4),
    }
