import pytest
from fastapi.testclient import TestClient

from backend import customer_lookup, monitoring
from backend.api import app
from backend.schema import RAW_COLUMNS


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, "LOG_PATH", tmp_path / "registro_predicciones.csv")
    return TestClient(app)


def _row(customer_id, features):
    return {"customer_id": customer_id, "features": features}


def _batch_of(n):
    ids = customer_lookup.list_validation_customer_ids()[:n]
    return [_row(cid, customer_lookup.get_validation_customer(cid)) for cid in ids]


def test_batch_fully_valid(client):
    response = client.post("/predict/batch", json={"customers": _batch_of(3)})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total"] == 3
    assert body["summary"]["valid"] == 3
    assert body["summary"]["invalid"] == 0
    assert len(body["results"]) == 3
    assert body["errors"] == []
    assert sum(body["summary"]["band_counts"].values()) == 3
    assert set(body["summary"]["band_counts"]) == {"Bajo", "Medio", "Alto"}


def test_batch_mixed_success_and_error(client):
    customers = _batch_of(2)
    bad_features = customer_lookup.get_validation_customer(
        customer_lookup.list_validation_customer_ids()[0]
    )
    bad_features["gender"] = "Nonexistent_Category"
    customers.append(_row("BAD-1", bad_features))

    response = client.post("/predict/batch", json={"customers": customers})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total"] == 3
    assert body["summary"]["valid"] == 2
    assert body["summary"]["invalid"] == 1
    assert len(body["errors"]) == 1
    assert body["errors"][0]["customer_id"] == "BAD-1"
    assert body["errors"][0]["row_index"] == 3


def test_batch_empty_returns_422(client):
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422


def test_batch_over_100_rows_returns_422(client):
    ids = (customer_lookup.list_validation_customer_ids() * 2)[:101]
    customers = [_row(f"{cid}-{i}", customer_lookup.get_validation_customer(cid)) for i, cid in enumerate(ids)]
    response = client.post("/predict/batch", json={"customers": customers})
    assert response.status_code == 422


def test_batch_duplicate_customer_id_processed_independently(client):
    cid = customer_lookup.list_validation_customer_ids()[0]
    features = customer_lookup.get_validation_customer(cid)
    payload = {"customers": [_row(cid, features), _row(cid, features)]}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["valid"] == 2
    assert [r["customer_id"] for r in body["results"]] == [cid, cid]


def test_batch_probability_and_band_match_individual_predict(client):
    cid = customer_lookup.list_validation_customer_ids()[0]
    features = customer_lookup.get_validation_customer(cid)

    single = client.post("/predict", json=features).json()
    batch_response = client.post("/predict/batch", json={"customers": [_row(cid, features)]}).json()
    batch_result = batch_response["results"][0]

    assert batch_result["probability"] == single["probability"]
    assert batch_result["risk_band"] == single["risk_band"]
    assert batch_result["decision_threshold"] == single["decision_threshold"]


def test_batch_never_returns_real_actual_churn(client):
    """Locks the guarantee from the plan (sections 5.4 and 7.3): the batch
    path must never expose a real churn label, since `batch.py` calls
    `predictor.predict()` directly and never touches `customer_lookup` (the
    only place that fetches `actual_churn`). Every row's `actual_churn`
    must be `None`, not just "nobody happened to set it".
    """
    response = client.post("/predict/batch", json={"customers": _batch_of(3)})
    body = response.json()
    assert len(body["results"]) == 3
    assert all(result["actual_churn"] is None for result in body["results"])


def test_batch_missing_column_only_that_row_fails(client):
    ids = customer_lookup.list_validation_customer_ids()[:2]
    good = _row(ids[0], customer_lookup.get_validation_customer(ids[0]))
    incomplete_features = customer_lookup.get_validation_customer(ids[1])
    del incomplete_features["age"]
    bad = _row(ids[1], incomplete_features)

    response = client.post("/predict/batch", json={"customers": [good, bad]})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["valid"] == 1
    assert body["summary"]["invalid"] == 1
    assert body["errors"][0]["customer_id"] == ids[1]


def test_batch_unknown_category_only_that_row_fails(client):
    ids = customer_lookup.list_validation_customer_ids()[:2]
    good = _row(ids[0], customer_lookup.get_validation_customer(ids[0]))
    bad_features = customer_lookup.get_validation_customer(ids[1])
    bad_features["gender"] = "Nonexistent_Category"
    bad = _row(ids[1], bad_features)

    response = client.post("/predict/batch", json={"customers": [good, bad]})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["valid"] == 1
    assert body["summary"]["invalid"] == 1
    assert body["errors"][0]["customer_id"] == ids[1]


def test_batch_rejects_churn_as_input(client):
    cid = customer_lookup.list_validation_customer_ids()[0]
    features = customer_lookup.get_validation_customer(cid)
    features["churn"] = 1

    response = client.post("/predict/batch", json={"customers": [_row(cid, features)]})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["invalid"] == 1
    assert "churn" in body["errors"][0]["detail"]


def test_batch_all_rows_invalid_logs_error_status(client):
    features = customer_lookup.get_validation_customer(customer_lookup.list_validation_customer_ids()[0])
    features["gender"] = "Nonexistent_Category"

    response = client.post("/predict/batch", json={"customers": [_row("BAD-1", features)]})
    assert response.status_code == 200
    assert response.json()["summary"]["valid"] == 0

    events = client.get("/monitoring").json()
    assert events[-1]["source"] == "batch"
    assert events[-1]["status"] == "error"


def test_batch_logs_single_aggregate_monitoring_event(client):
    response = client.post("/predict/batch", json={"customers": _batch_of(2)})
    run_id = response.json()["run_id"]

    events = client.get("/monitoring").json()
    assert len(events) == 1
    assert events[0]["source"] == "batch"
    assert events[0]["status"] == "success"
    assert run_id in events[0]["detail"]


def test_batch_monitoring_event_does_not_store_raw_features(client):
    client.post("/predict/batch", json={"customers": _batch_of(2)})

    events = client.get("/monitoring").json()
    detail = events[0]["detail"]
    assert not any(var in detail for var in RAW_COLUMNS)
