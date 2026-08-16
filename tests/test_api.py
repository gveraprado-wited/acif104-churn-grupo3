import pytest
from fastapi.testclient import TestClient

from backend import customer_lookup, monitoring
from backend.api import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient against the real app (no server is actually started),
    redirecting the monitoring log to a temp file so test events don't
    pollute the repo's real CSV.
    """
    monkeypatch.setattr(monitoring, "LOG_PATH", tmp_path / "registro_predicciones.csv")
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["estado"] == "ok"
    assert body["espacio_shap"] in ("probabilidad", "margen (log-odds)")


def test_get_schema(client):
    response = client.get("/schema")
    assert response.status_code == 200
    body = response.json()
    assert "numericas" in body and "categoricas" in body
    assert "age" in body["numericas"]
    assert "gender" in body["categoricas"]


def test_list_customers(client):
    response = client.get("/customers")
    assert response.status_code == 200
    ids = response.json()["customer_ids"]
    assert len(ids) == 1500
    assert all(isinstance(cid, str) for cid in ids)


def test_get_existing_customer_profile(client):
    customer_id = customer_lookup.list_validation_customer_ids()[0]
    response = client.get(f"/customers/{customer_id}")
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["actual_churn"] in (0, 1)
    assert 0 < len(body["shap_contributions"]) <= 6


def test_get_nonexistent_customer_profile_returns_404(client):
    response = client.get("/customers/CUST_DOES_NOT_EXIST")
    assert response.status_code == 404


def test_predict_with_valid_data(client):
    customer_id = customer_lookup.list_validation_customer_ids()[0]
    data = customer_lookup.get_validation_customer(customer_id)
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "probability" in response.json()


def test_predict_with_invalid_category_returns_422(client):
    customer_id = customer_lookup.list_validation_customer_ids()[0]
    data = customer_lookup.get_validation_customer(customer_id)
    data["gender"] = "Nonexistent_Category"
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_predict_with_missing_key_returns_422(client):
    customer_id = customer_lookup.list_validation_customer_ids()[0]
    data = customer_lookup.get_validation_customer(customer_id)
    del data["age"]
    response = client.post("/predict", json=data)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field, value",
    [
        ("weekly_active_days", 99),
        ("csat_score", 9),
        ("email_open_rate", 1.7),
        ("nps_score", 250),
        ("tenure_months", 999),
    ],
)
def test_predict_rejects_objectively_impossible_values(client, field, value):
    """The API must not depend solely on the Streamlit widgets' limits: it
    should reject these on its own, regardless of which client calls it.
    """
    customer_id = customer_lookup.list_validation_customer_ids()[0]
    data = customer_lookup.get_validation_customer(customer_id)
    data[field] = value
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_predict_warns_but_accepts_value_outside_historical_range(client):
    customer_id = customer_lookup.list_validation_customer_ids()[0]
    data = customer_lookup.get_validation_customer(customer_id)
    data["total_revenue"] = 50000  # historical max is 5900, no hard bound
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert any("total_revenue" in w for w in response.json()["range_warnings"])


def test_monitoring_logs_successful_predictions(client):
    customer_id = customer_lookup.list_validation_customer_ids()[0]
    client.get(f"/customers/{customer_id}")

    response = client.get("/monitoring")
    assert response.status_code == 200
    events = response.json()
    assert len(events) == 1
    assert events[0]["source"] == "customer_profile"
    assert events[0]["status"] == "success"


def test_monitoring_logs_pydantic_validation_errors(client):
    customer_id = customer_lookup.list_validation_customer_ids()[0]
    data = customer_lookup.get_validation_customer(customer_id)
    data["csat_score"] = 9
    client.post("/predict", json=data)

    events = client.get("/monitoring").json()
    assert len(events) == 1
    assert events[0]["status"] == "error"
    assert events[0]["source"] == "api"


def test_monitoring_logs_business_validation_errors(client):
    customer_id = customer_lookup.list_validation_customer_ids()[0]
    data = customer_lookup.get_validation_customer(customer_id)
    data["gender"] = "Nonexistent_Category"
    client.post("/predict", json=data)

    events = client.get("/monitoring").json()
    assert len(events) == 1
    assert events[0]["status"] == "error"
    assert events[0]["source"] == "form"


def test_get_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["decision_threshold"] == 0.56
    assert body["low_band"] == 0.24
    assert body["shap_space"] in ("probabilidad", "margen (log-odds)")


def test_get_global_explainability(client):
    response = client.get("/explainability/global")
    assert response.status_code == 200
    body = response.json()
    assert len(body) > 0
    assert all("variable" in row and "importancia_media_abs" in row for row in body)
    importances = [row["importancia_media_abs"] for row in body]
    assert importances == sorted(importances, reverse=True)


def test_get_model_metrics(client):
    response = client.get("/model/metrics")
    assert response.status_code == 200
    body = response.json()
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert metric in body
        assert 0.0 <= body[metric] <= 1.0


def test_monitoring_logs_customer_not_found_errors(client):
    client.get("/customers/CUST_DOES_NOT_EXIST")

    events = client.get("/monitoring").json()
    assert len(events) == 1
    assert events[0]["status"] == "error"
    assert events[0]["source"] == "customer_profile"
