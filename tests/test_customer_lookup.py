import joblib

from backend import customer_lookup


def test_validation_customers_match_official_y_val():
    """Count alone doesn't prove this is the validation split — test also
    has 1,500 rows. Compares the actual churn labels exposed by
    customer_lookup against y_val from the official split artifact.
    """
    official_y_val = joblib.load(customer_lookup.SPLITS_ARTIFACT_PATH)["y_val"]

    ids = customer_lookup.list_validation_customer_ids()
    assert len(ids) == len(official_y_val)

    churns = [customer_lookup.get_validation_customer_actual_churn(cid) for cid in ids]
    assert sorted(churns) == sorted(official_y_val.tolist())
