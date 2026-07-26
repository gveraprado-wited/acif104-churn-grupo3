"""Schema of the 30 raw variables the model expects.

Single source of truth: `artifacts/esquema_entrada.json`, generated in
`notebook/06_refinamiento_modelo.ipynb` from
`data/processed/customer_churn_clean.csv`.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "artifacts" / "esquema_entrada.json", encoding="utf-8") as f:
    INPUT_SCHEMA = json.load(f)

NUMERIC_VARIABLES = list(INPUT_SCHEMA["numericas"].keys())
CATEGORICAL_VARIABLES = list(INPUT_SCHEMA["categoricas"].keys())
RAW_COLUMNS = NUMERIC_VARIABLES + CATEGORICAL_VARIABLES

# Sorted by descending length to avoid prefix collisions
# (e.g. "country_" matching before "city_" when both could fit).
_CATEGORICAL_VARIABLES_BY_LENGTH = sorted(
    CATEGORICAL_VARIABLES, key=len, reverse=True
)


def default_values() -> dict:
    """A dict with the 30 raw columns set to their default values (median/mode)."""
    values = {var: data["mediana"] for var, data in INPUT_SCHEMA["numericas"].items()}
    values.update(
        {var: data["default"] for var, data in INPUT_SCHEMA["categoricas"].items()}
    )
    return values


def readable_name(technical_name: str) -> str:
    """Translates a transformed column name (with the ColumnTransformer prefix)
    back to its raw variable, e.g. 'categoricas__country_Germany' -> 'country = Germany'.
    Same logic used in notebook/07_explicabilidad_shap.ipynb.
    """
    if technical_name.startswith("numericas__"):
        return technical_name.replace("numericas__", "")
    if technical_name.startswith("categoricas__"):
        rest = technical_name.replace("categoricas__", "")
        for variable in _CATEGORICAL_VARIABLES_BY_LENGTH:
            prefix = variable + "_"
            if rest.startswith(prefix):
                value = rest[len(prefix):]
                return f"{variable} = {value}"
        return rest
    return technical_name


def raw_variable_of(readable_feature_name: str) -> str:
    """Maps a readable (already-translated) feature name back to its ORIGINAL
    raw variable, e.g. 'city = Dhaka' -> 'city', 'age' -> 'age'. Used to group
    one-hot dummy columns of the same categorical variable back together
    before aggregating their SHAP contributions (see inference.py) — a
    one-hot-encoded categorical produces one transformed column per category,
    and SHAP assigns a value to each of them independently, including the
    ones that are 0 for a given customer.
    """
    if " = " in readable_feature_name:
        return readable_feature_name.split(" = ", 1)[0]
    return readable_feature_name


# Hard bounds for variables with a real, definitional limit — values outside
# these are objectively impossible (not just historically unusual), so the
# API rejects them via Pydantic instead of accepting them silently.
HARD_BOUNDS = {
    "age": (0, 120),
    "tenure_months": (0, 600),
    "weekly_active_days": (0, 7),
    "csat_score": (1, 5),
    "nps_score": (-100, 100),
    "email_open_rate": (0.0, 1.0),
    "marketing_click_rate": (0.0, 1.0),
}

# Non-negative counts/amounts with no natural ceiling: the model can
# extrapolate above the historical maximum (a customer can plausibly exceed
# every value seen in training), but they still can't be negative.
NON_NEGATIVE_VARIABLES = [
    "monthly_logins", "avg_session_time", "features_used",
    "last_login_days_ago", "monthly_fee", "total_revenue",
    "payment_failures", "support_tickets", "avg_resolution_time",
    "escalations", "referral_count",
]
