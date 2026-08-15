"""FastAPI backend: a thin HTTP wrapper over `schema.py`, `inference.py`,
`customer_lookup.py`, and `monitoring.py`. Contains no business logic of its
own.

Run with: uvicorn backend.api:app --reload
"""
import math

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend import customer_lookup, monitoring
from backend.api_schemas import ClientInput, PredictionResult
from backend.inference import ChurnPredictor
from backend.schema import INPUT_SCHEMA

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

predictor = ChurnPredictor()


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """Catches malformed/out-of-bound request bodies rejected automatically
    by Pydantic (before the route function even runs — e.g. csat_score=9)
    and logs them to monitoring too, so invalid attempts aren't invisible.
    """
    monitoring.log_event(
        source="api",
        status="error",
        detail=f"{request.url.path}: {exc.errors()}",
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/schema")
def get_schema():
    """Numeric ranges/medians and valid values/defaults per category, so the
    frontend can build the form without importing `backend/` directly.
    """
    return INPUT_SCHEMA


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "modelo": "xgboost_refinado.joblib",
        "espacio_shap": predictor.shap_space,
    }


@app.post("/predict", response_model=PredictionResult)
def predict(client: ClientInput):
    try:
        result = predictor.predict(client.model_dump())
    except ValueError as error:
        monitoring.log_event(source="form", status="error", detail=str(error))
        raise HTTPException(status_code=422, detail=str(error))

    monitoring.log_event(
        source="form",
        status="success",
        probability=result["probability"],
        risk_band=result["risk_band"],
        decision_threshold=result["decision_threshold"],
        response_time_ms=result["response_time_ms"],
    )
    return result


@app.get("/customers")
def list_customers():
    return {"customer_ids": customer_lookup.list_validation_customer_ids()}


@app.get("/customers/{customer_id}", response_model=PredictionResult)
def get_customer_profile(customer_id: str):
    raw_data = customer_lookup.get_validation_customer(customer_id)
    if raw_data is None:
        detail = f"No existe el cliente '{customer_id}' en el conjunto de validación."
        monitoring.log_event(source="customer_profile", status="error", detail=detail)
        raise HTTPException(status_code=404, detail=detail)

    result = predictor.predict(raw_data)
    result["actual_churn"] = customer_lookup.get_validation_customer_actual_churn(customer_id)

    monitoring.log_event(
        source="customer_profile",
        status="success",
        probability=result["probability"],
        risk_band=result["risk_band"],
        decision_threshold=result["decision_threshold"],
        response_time_ms=result["response_time_ms"],
    )
    return result


@app.get("/monitoring")
def get_monitoring():
    # Error events leave the prediction fields empty, which pandas reads
    # back as float NaN (not valid JSON) — converted to null before returning.
    # (DataFrame.where(..., None) doesn't work here: pandas coerces None
    # right back to NaN on a float64 column, so the replacement has to
    # happen after to_dict(), on plain Python values.)
    records = monitoring.read_events().to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
    return records
