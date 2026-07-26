import pytest

from backend.inference import assign_risk_band


class TestAssignRiskBand:
    """Pure function: tested in isolation, without going through the full model."""

    THRESHOLD = 0.56
    LOW_BAND = 0.24

    @pytest.mark.parametrize(
        "probability, expected_band",
        [
            (0.0, "Bajo"),
            (0.10, "Bajo"),
            (0.23, "Bajo"),
            (0.24, "Medio"),  # inclusive lower bound
            (0.40, "Medio"),
            (0.55, "Medio"),
            (0.56, "Alto"),  # inclusive decision boundary
            (0.80, "Alto"),
            (1.0, "Alto"),
        ],
    )
    def test_bands(self, probability, expected_band):
        assert assign_risk_band(probability, self.THRESHOLD, self.LOW_BAND) == expected_band


class TestValidate:
    def test_valid_input_has_no_errors(self, predictor, valid_customer):
        assert predictor.validate(valid_customer) == []

    def test_rejects_missing_key(self, predictor, valid_customer):
        data = dict(valid_customer)
        del data["age"]
        errors = predictor.validate(data)
        assert any("age" in e for e in errors)

    def test_rejects_invalid_numeric_type(self, predictor, valid_customer):
        data = dict(valid_customer)
        data["age"] = "not-a-number"
        errors = predictor.validate(data)
        assert any("age" in e for e in errors)

    def test_rejects_unknown_category(self, predictor, valid_customer):
        data = dict(valid_customer)
        data["gender"] = "Nonexistent_Category"
        errors = predictor.validate(data)
        assert any("gender" in e for e in errors)

    def test_accepts_numeric_value_outside_historical_range(self, predictor, valid_customer):
        """Design decision: a value outside the historical min/max isn't
        rejected (the model can extrapolate); only unknown categories and
        structural problems (missing key, invalid type) are rejected. Truly
        impossible values (csat_score=9, nps_score=250, etc.) are rejected
        one layer up, by Pydantic in api_schemas.ClientInput — see
        test_api.py.
        """
        data = dict(valid_customer)
        data["tenure_months"] = 999
        assert predictor.validate(data) == []


class TestRangeWarnings:
    def test_no_warnings_for_typical_customer(self, predictor, valid_customer):
        assert predictor.range_warnings(valid_customer) == []

    def test_warns_when_outside_historical_range(self, predictor, valid_customer):
        data = dict(valid_customer)
        data["total_revenue"] = 50000  # historical max is 5900
        warnings = predictor.range_warnings(data)
        assert any("total_revenue" in w for w in warnings)


class TestPredict:
    def test_returns_expected_shape(self, predictor, valid_customer):
        result = predictor.predict(valid_customer)

        assert 0.0 <= result["probability"] <= 1.0
        assert result["prediction"] in (0, 1)
        assert result["risk_band"] in ("Bajo", "Medio", "Alto")
        assert result["decision_threshold"] == predictor.decision_threshold
        assert result["low_band"] == predictor.low_band
        assert result["shap_space"] in ("probabilidad", "margen (log-odds)")
        assert result["response_time_ms"] > 0
        assert result["range_warnings"] == []
        assert 0 < len(result["shap_contributions"]) <= 6
        for contribution in result["shap_contributions"]:
            assert set(contribution.keys()) == {"variable", "shap", "direction"}
            assert contribution["direction"] in ("increases_risk", "decreases_risk")

    def test_categorical_contributions_reflect_the_customers_actual_value(
        self, predictor, valid_customer
    ):
        """Regression test: one-hot encoding produces one transformed column
        per category, and SHAP assigns a value to every one of them,
        including categories the customer does NOT belong to. Before
        aggregating by raw variable, a contribution could show e.g.
        'city = Dhaka' for a customer who actually lives in Sydney, just
        because that unrelated dummy had a non-zero SHAP value.
        """
        result = predictor.predict(valid_customer)
        for contribution in result["shap_contributions"]:
            if " = " not in contribution["variable"]:
                continue
            raw_var, shown_value = contribution["variable"].split(" = ", 1)
            assert shown_value == str(valid_customer[raw_var]), (
                f"'{contribution['variable']}' no coincide con el valor real "
                f"del cliente para '{raw_var}': {valid_customer[raw_var]!r}"
            )

    def test_shap_contributions_include_both_directions_when_available(
        self, predictor, valid_customer
    ):
        """With 58 transformed features it's virtually certain some push risk
        up and some push it down — the explanation should show both sides,
        not just whichever direction happens to dominate by raw magnitude.
        """
        result = predictor.predict(valid_customer)
        directions = {c["direction"] for c in result["shap_contributions"]}
        assert directions == {"increases_risk", "decreases_risk"}

    def test_prediction_consistent_with_band_and_threshold(self, predictor, valid_customer):
        result = predictor.predict(valid_customer)
        expected_prediction = int(result["probability"] >= result["decision_threshold"])
        assert result["prediction"] == expected_prediction
        if result["risk_band"] == "Alto":
            assert result["prediction"] == 1

    def test_raises_valueerror_on_invalid_input(self, predictor, valid_customer):
        data = dict(valid_customer)
        data["gender"] = "Nonexistent_Category"
        with pytest.raises(ValueError):
            predictor.predict(data)
