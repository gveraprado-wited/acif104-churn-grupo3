"""Retrieves real validation-set customers (with customer_id) for the app's
customer profile lookup.

`artifacts/splits_preprocesados.joblib` only stores already-transformed
arrays (no customer_id, no raw columns), so this reproduces the exact same
train/val/test split from notebook/02_preprocesamiento.ipynb (same
parameters: test_size, random_state=42, stratify) over
`data/processed/customer_churn_clean.csv`, which does keep customer_id. The
resulting index was verified to match the official X_val/y_val 1:1.

Validation (not test) is used deliberately here: the test set is reserved
exclusively for notebook/08_evaluacion_final_test.ipynb.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.schema import RAW_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_churn_clean.csv"
SPLITS_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "splits_preprocesados.joblib"


def _verify_matches_official_split(val_df: pd.DataFrame) -> None:
    """Guards against this reproduced split silently drifting from the
    official one persisted by notebook/02_preprocesamiento.ipynb (e.g. a
    scikit-learn version change, or an edited CSV) by checking it against
    the actual y_val saved in SPLITS_ARTIFACT_PATH, instead of just
    trusting that the split parameters stay in sync forever.
    """
    official_y_val = joblib.load(SPLITS_ARTIFACT_PATH)["y_val"]

    if set(val_df.index) != set(official_y_val.index):
        raise RuntimeError(
            "customer_lookup: the reproduced validation split no longer "
            f"matches the official one persisted in {SPLITS_ARTIFACT_PATH.name}. "
            "Re-check the split parameters against "
            "notebook/02_preprocesamiento.ipynb."
        )
    if not (val_df.loc[official_y_val.index, "churn"] == official_y_val).all():
        raise RuntimeError(
            "customer_lookup: churn labels for the reproduced validation "
            f"split don't match {SPLITS_ARTIFACT_PATH.name}'s y_val."
        )


@lru_cache(maxsize=1)
def load_validation_customers() -> pd.DataFrame:
    """DataFrame indexed by customer_id with the 30 raw columns + actual churn,
    restricted to the same 1,500 customers from the official validation set.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["churn"]
    )
    val_df, _ = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["churn"]
    )

    _verify_matches_official_split(val_df)

    return val_df.set_index("customer_id")[RAW_COLUMNS + ["churn"]]


def list_validation_customer_ids() -> list[str]:
    return sorted(load_validation_customers().index.tolist())


def get_validation_customer(customer_id: str) -> dict | None:
    """Returns the 30 raw variables (without 'churn') for a validation-set
    customer, or None if the id doesn't exist. The actual churn label is
    returned separately via `get_validation_customer_actual_churn`, so it
    never gets mixed into the input passed to the model.
    """
    customers = load_validation_customers()
    if customer_id not in customers.index:
        return None
    row = customers.loc[customer_id]
    return row[RAW_COLUMNS].to_dict()


def get_validation_customer_actual_churn(customer_id: str) -> int | None:
    customers = load_validation_customers()
    if customer_id not in customers.index:
        return None
    return int(customers.loc[customer_id, "churn"])
