from backend import monitoring


def test_read_events_without_file_returns_empty(tmp_path):
    path = tmp_path / "does_not_exist.csv"
    df = monitoring.read_events(log_path=path)
    assert df.empty
    assert list(df.columns) == monitoring.LOG_COLUMNS


def test_log_and_read_event_roundtrip(tmp_path):
    path = tmp_path / "registro_predicciones.csv"

    monitoring.log_event(
        source="form",
        status="success",
        probability=0.73,
        risk_band="Alto",
        decision_threshold=0.56,
        response_time_ms=12.5,
        log_path=path,
    )
    monitoring.log_event(
        source="customer_profile",
        status="success",
        probability=0.10,
        risk_band="Bajo",
        decision_threshold=0.56,
        response_time_ms=8.1,
        log_path=path,
    )

    df = monitoring.read_events(log_path=path)
    assert len(df) == 2
    assert list(df.columns) == monitoring.LOG_COLUMNS
    assert set(df["source"]) == {"form", "customer_profile"}
    assert set(df["status"]) == {"success"}
    assert df.loc[0, "risk_band"] == "Alto"


def test_log_error_event(tmp_path):
    path = tmp_path / "registro_predicciones.csv"
    monitoring.log_event(
        source="form",
        status="error",
        detail="'csat_score' = 9 no es un valor válido.",
        log_path=path,
    )

    df = monitoring.read_events(log_path=path)
    assert len(df) == 1
    assert df.loc[0, "status"] == "error"
    assert "csat_score" in df.loc[0, "detail"]
    assert df.loc[0, "probability"] != df.loc[0, "probability"]  # NaN, no prediction happened


def test_no_raw_customer_data_is_leaked(tmp_path):
    """Privacy requirement: only prediction metadata gets stored, never the
    customer's raw variables (e.g. csat_score, age, etc.).
    """
    path = tmp_path / "registro_predicciones.csv"
    monitoring.log_event(
        source="form",
        status="success",
        probability=0.5,
        risk_band="Medio",
        decision_threshold=0.56,
        response_time_ms=5.0,
        log_path=path,
    )

    df = monitoring.read_events(log_path=path)
    forbidden_columns = {"age", "csat_score", "gender", "country", "tenure_months"}
    assert forbidden_columns.isdisjoint(df.columns)
