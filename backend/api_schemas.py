"""Pydantic request/response models for the API (the HTTP boundary). Not to
be confused with `schema.py`, which describes the 30 raw dataset variables —
this file only defines the input/output shapes that `api.py` exposes.

Numeric fields are bounded here (not just in the Streamlit widgets) so the
API rejects objectively impossible values — e.g. csat_score=9 or
nps_score=250 — regardless of which client calls it. See
`backend.schema.HARD_BOUNDS` / `NON_NEGATIVE_VARIABLES` for the reasoning
behind each limit.
"""
from datetime import datetime

from pydantic import BaseModel, Field, create_model

from backend.schema import (
    CATEGORICAL_VARIABLES,
    HARD_BOUNDS,
    NON_NEGATIVE_VARIABLES,
    NUMERIC_VARIABLES,
)


def _numeric_field(var: str):
    if var in HARD_BOUNDS:
        low, high = HARD_BOUNDS[var]
        return (float, Field(..., ge=low, le=high))
    if var in NON_NEGATIVE_VARIABLES:
        return (float, Field(..., ge=0))
    return (float, ...)


_client_input_fields = {var: _numeric_field(var) for var in NUMERIC_VARIABLES}
_client_input_fields.update({var: (str, ...) for var in CATEGORICAL_VARIABLES})
ClientInput = create_model("ClientInput", **_client_input_fields)


class SHAPContribution(BaseModel):
    variable: str
    shap: float
    direction: str


class PredictionResult(BaseModel):
    probability: float
    prediction: int
    risk_band: str
    decision_threshold: float
    low_band: float
    shap_contributions: list[SHAPContribution]
    shap_space: str
    response_time_ms: float
    input_values: dict
    range_warnings: list[str]
    actual_churn: int | None = None


class BatchClientInput(BaseModel):
    """One row of a batch request. `features` is deliberately untyped here
    (not the nested `ClientInput` model): if it were, a single malformed row
    would fail Pydantic parsing for the *entire* request body before batch.py
    ever runs, turning "one bad row" into "the whole batch gets rejected".
    Each row's features are instead validated against `ClientInput`
    individually inside `batch.run_batch`, so one bad row becomes an entry
    in `errors` and the rest of the batch still succeeds.
    """
    customer_id: str = Field(..., min_length=1)
    features: dict[str, float | str]


class BatchPredictRequest(BaseModel):
    customers: list[BatchClientInput] = Field(..., min_length=1, max_length=100)


class BatchResultItem(PredictionResult):
    customer_id: str
    row_index: int


class BatchErrorItem(BaseModel):
    customer_id: str | None = None
    row_index: int
    detail: str


class BatchSummary(BaseModel):
    total: int
    valid: int
    invalid: int
    band_counts: dict[str, int]
    high_risk_monthly_fee: float


class BatchPredictionResult(BaseModel):
    run_id: str
    processed_at: datetime
    model: str
    summary: BatchSummary
    results: list[BatchResultItem]
    errors: list[BatchErrorItem]
