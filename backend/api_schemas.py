"""Pydantic request/response models for the API (the HTTP boundary). Not to
be confused with `schema.py`, which describes the 30 raw dataset variables —
this file only defines the input/output shapes that `api.py` exposes.

Numeric fields are bounded here (not just in the Streamlit widgets) so the
API rejects objectively impossible values — e.g. csat_score=9 or
nps_score=250 — regardless of which client calls it. See
`backend.schema.HARD_BOUNDS` / `NON_NEGATIVE_VARIABLES` for the reasoning
behind each limit.
"""
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
