"""Retrieves real test-set customers (with customer_id) for the app's
customer profile lookup.

`artifacts/splits_preprocesados.joblib` only stores already-transformed
arrays (no customer_id, no raw columns), so this reproduces the exact same
train/val/test split from notebook/02_preprocesamiento.ipynb (same
parameters: test_size, random_state=42, stratify) over
`data/processed/customer_churn_clean.csv`, which does keep customer_id. The
resulting index was verified to match the official X_test/y_test 1:1.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from backend.schema import RAW_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_churn_clean.csv"


@lru_cache(maxsize=1)
def load_test_customers() -> pd.DataFrame:
    """DataFrame indexed by customer_id with the 30 raw columns + actual churn,
    restricted to the same 1,500 customers from the official test set.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["churn"]
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["churn"]
    )

    return test_df.set_index("customer_id")[RAW_COLUMNS + ["churn"]]


def list_test_customer_ids() -> list[str]:
    return sorted(load_test_customers().index.tolist())


def get_test_customer(customer_id: str) -> dict | None:
    """Returns the 30 raw variables (without 'churn') for a test-set
    customer, or None if the id doesn't exist. The actual churn label is
    returned separately via `get_test_customer_actual_churn`, so it never
    gets mixed into the input passed to the model.
    """
    customers = load_test_customers()
    if customer_id not in customers.index:
        return None
    row = customers.loc[customer_id]
    return row[RAW_COLUMNS].to_dict()


def get_test_customer_actual_churn(customer_id: str) -> int | None:
    customers = load_test_customers()
    if customer_id not in customers.index:
        return None
    return int(customers.loc[customer_id, "churn"])
