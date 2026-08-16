"""Batch prediction: validates and predicts each row independently, so a
malformed row is isolated into `errors` instead of failing the whole batch.

Each row is run through the exact same `ClientInput` model and
`ChurnPredictor.predict` used by `/predict` (see `api_schemas.py` for why
`features` arrives untyped and gets validated here, per row, instead of at
the request-parsing boundary). That reuse is what guarantees `probability`
and `risk_band` are identical between `/predict` and `/predict/batch` for
the same customer.
"""
import uuid
from datetime import datetime, timezone

from pydantic import ValidationError

from backend.api_schemas import (
    BatchClientInput,
    BatchErrorItem,
    BatchPredictionResult,
    BatchResultItem,
    BatchSummary,
    ClientInput,
)
from backend.inference import ChurnPredictor

MODEL_NAME = "xgboost_refinado.joblib"
RISK_BANDS = ("Bajo", "Medio", "Alto")


def _validate_row(row: BatchClientInput) -> tuple[dict | None, str | None]:
    """Returns (raw_data, None) on success or (None, error_detail) on
    failure. `churn` is checked explicitly (rather than relying on
    `ClientInput` silently ignoring unknown keys) so a client passing it is
    caught and rejected, not just harmlessly dropped.
    """
    if "churn" in row.features:
        return None, "'churn' no puede enviarse como variable de entrada."

    try:
        validated = ClientInput(**row.features)
    except ValidationError as error:
        messages = [f"{e['loc'][-1]}: {e['msg']}" for e in error.errors()]
        return None, "; ".join(messages)

    return validated.model_dump(), None


def run_batch(customers: list[BatchClientInput], predictor: ChurnPredictor) -> BatchPredictionResult:
    results: list[BatchResultItem] = []
    errors: list[BatchErrorItem] = []
    band_counts = {band: 0 for band in RISK_BANDS}
    high_risk_monthly_fee = 0.0

    for row_index, row in enumerate(customers, start=1):
        raw_data, error_detail = _validate_row(row)
        if raw_data is None:
            errors.append(
                BatchErrorItem(customer_id=row.customer_id, row_index=row_index, detail=error_detail)
            )
            continue

        try:
            result = predictor.predict(raw_data)
        except ValueError as error:
            errors.append(
                BatchErrorItem(customer_id=row.customer_id, row_index=row_index, detail=str(error))
            )
            continue

        band_counts[result["risk_band"]] += 1
        if result["risk_band"] == "Alto":
            high_risk_monthly_fee += raw_data["monthly_fee"]

        results.append(BatchResultItem(customer_id=row.customer_id, row_index=row_index, **result))

    summary = BatchSummary(
        total=len(customers),
        valid=len(results),
        invalid=len(errors),
        band_counts=band_counts,
        high_risk_monthly_fee=high_risk_monthly_fee,
    )

    return BatchPredictionResult(
        run_id=str(uuid.uuid4()),
        processed_at=datetime.now(timezone.utc),
        model=MODEL_NAME,
        summary=summary,
        results=results,
        errors=errors,
    )
