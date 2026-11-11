
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from fastapi.responses import FileResponse
from pydantic import BaseModel


BASE_DIR = Path(__file__).parent

app = FastAPI(title="Mental Health Treatment Predictor")

# Load the model and column names saved from the notebook
model = joblib.load(BASE_DIR / "model.pkl")
columns = joblib.load(BASE_DIR / "columns.pkl")

# If a RandomizedSearchCV object was saved, use the pipeline inside it
if hasattr(model, "best_estimator_"):
    model = model.best_estimator_


def build_schema():
    """Reads the fitted pipeline and works out each input field automatically."""
    fields = {}
    preprocessor = model.named_steps["preprocessor"]
    for name, pipe, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if name == "categorical":
            cats = pipe.named_steps["encode"].categories_
            for col, options in zip(cols, cats):
                fields[col] = {
                    "type": "categorical",
                    "options": [str(o) for o in options],
                }
        else:
            for col in cols:
                fields[col] = {"type": "number"}
    # keep the same order as the training columns
    return [{"name": c, **fields.get(c, {"type": "number"})} for c in columns]



schema = build_schema()


class PredictRequest(BaseModel):
    data: Dict[str, Any]


@app.get("/")
def home():
    return {"message": "API is running. Open /docs to test it."}


@app.get("/features")
def features():
    """Shows which input fields the model expects."""
    return {"columns": columns}


@app.get("/schema")
def get_schema():
    """Field types and dropdown options, used by the UI."""
    return {"fields": schema}


@app.post("/predict")
def predict(request: PredictRequest):
    # Blank / missing features become NaN (the pipeline imputes them)
    row = {}
    for col in columns:
        value = request.data.get(col)
        row[col] = None if value in ("", None) else value
    df = pd.DataFrame([row], columns=columns)

    # make numeric columns numeric
    for field in schema:
        if field["type"] == "number":
            df[field["name"]] = pd.to_numeric(df[field["name"]], errors="coerce")


    try:
        pred = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "treatment": "Yes" if pred == 1 else "No",
        "probability_yes": round(proba, 4),


    }