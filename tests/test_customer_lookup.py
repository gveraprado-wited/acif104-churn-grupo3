import joblib
import pandas as pd

from backend import customer_lookup


def test_validation_customers_match_official_y_val():
    """Matching churn-label distributions alone doesn't prove this is the
    validation split — another stratified split (e.g. test) would pass that
    check too, since both have the same class balance. Instead, this
    reconstructs the actual expected customer_id set for y_val's rows
    (via their shared index into the raw CSV) and compares it directly
    against what customer_lookup exposes.
    """
    official_y_val = joblib.load(customer_lookup.SPLITS_ARTIFACT_PATH)["y_val"]

    df = pd.read_csv(customer_lookup.PROCESSED_DATA_PATH)
    expected_ids = set(df.loc[official_y_val.index, "customer_id"].astype(str))

    ids = set(customer_lookup.list_validation_customer_ids())
    assert ids == expected_ids
