"""Inference logic: input validation, prediction, and local SHAP explanation
for a customer. No dependency on FastAPI or Streamlit — usable (and
testable) as a plain Python library; `backend/api.py` only exposes it over
HTTP.
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import shap

from backend.schema import (
    CATEGORICAL_VARIABLES,
    INPUT_SCHEMA,
    NUMERIC_VARIABLES,
    RAW_COLUMNS,
    raw_variable_of,
    readable_name,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOP_N_PER_DIRECTION = 3
SHAP_BACKGROUND_SIZE = 100
RANDOM_STATE = 42


def assign_risk_band(probability: float, decision_threshold: float, low_band: float) -> str:
    if probability >= decision_threshold:
        return "Alto"
    if probability >= low_band:
        return "Medio"
    return "Bajo"


class ChurnPredictor:
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self._root = project_root

        self.preprocessor = joblib.load(self._root / "artifacts" / "preprocesador.joblib")
        self.model = joblib.load(self._root / "models" / "xgboost_refinado.joblib")

        with open(self._root / "models" / "config_umbral.json", encoding="utf-8") as f:
            self.threshold_config = json.load(f)
        self.decision_threshold = self.threshold_config["umbral_decision"]
        self.low_band = self.threshold_config["banda_bajo"]

        self.readable_feature_names = [
            readable_name(n) for n in self.preprocessor.get_feature_names_out()
        ]

        splits_artifact = joblib.load(
            self._root / "artifacts" / "splits_preprocesados.joblib"
        )
        train_df = pd.DataFrame(splits_artifact["X_train"], columns=self.readable_feature_names)
        background = shap.sample(train_df, SHAP_BACKGROUND_SIZE, random_state=RANDOM_STATE)

        try:
            self.explainer = shap.TreeExplainer(
                self.model,
                data=background,
                model_output="probability",
                feature_perturbation="interventional",
            )
            _ = self.explainer(train_df.iloc[:1])
            self.shap_space = "probabilidad"
        except Exception:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_space = "margen (log-odds)"

    def validate(self, raw_data: dict) -> list[str]:
        errors = []
        for var in RAW_COLUMNS:
            if var not in raw_data:
                errors.append(f"Falta la variable obligatoria '{var}'.")
        if errors:
            return errors

        for var in NUMERIC_VARIABLES:
            value = raw_data[var]
            try:
                float(value)
            except (TypeError, ValueError):
                errors.append(
                    f"'{var}' debe ser un valor numérico, se recibió: {value!r}."
                )

        for var in CATEGORICAL_VARIABLES:
            value = raw_data[var]
            valid_values = INPUT_SCHEMA["categoricas"][var]["valores"]
            if value not in valid_values:
                errors.append(
                    f"'{var}' = {value!r} no es una categoría reconocida. "
                    f"Valores válidos: {', '.join(valid_values)}."
                )

        return errors

    def range_warnings(self, raw_data: dict) -> list[str]:
        """Values that pass the hard Pydantic bounds (so they're not
        rejected) but still fall outside the historical range observed in
        training — the model can extrapolate, but the caller should know
        it's doing so.
        """
        warnings = []
        for var in NUMERIC_VARIABLES:
            info = INPUT_SCHEMA["numericas"][var]
            value = float(raw_data[var])
            if value < info["min"] or value > info["max"]:
                warnings.append(
                    f"'{var}' = {value:g} está fuera del rango histórico "
                    f"observado en el entrenamiento ({info['min']:g} a "
                    f"{info['max']:g}); el modelo está extrapolando."
                )
        return warnings

    def predict(self, raw_data: dict) -> dict:
        start = time.perf_counter()

        errors = self.validate(raw_data)
        if errors:
            raise ValueError("; ".join(errors))

        raw_row = pd.DataFrame([{var: raw_data[var] for var in RAW_COLUMNS}])
        transformed_row = self.preprocessor.transform(raw_row)

        probability = float(self.model.predict_proba(transformed_row)[0, 1])
        risk_band = assign_risk_band(
            probability, self.decision_threshold, self.low_band
        )
        prediction = int(probability >= self.decision_threshold)

        readable_row_df = pd.DataFrame(transformed_row, columns=self.readable_feature_names)
        explanation = self.explainer(readable_row_df)[0]

        # A one-hot-encoded categorical produces one transformed column per
        # category, and SHAP assigns a value to every one of them for this
        # row — including the categories the customer does NOT belong to.
        # Showing the raw per-column values directly can misattribute risk
        # to the wrong category (e.g. "city = Dhaka" for a customer who
        # actually lives in Sydney, just because that dummy had a non-zero
        # SHAP value). Summing the SHAP values of all dummies belonging to
        # the same original variable is mathematically valid (SHAP values
        # are additive across features by construction) and gives one
        # combined contribution per raw input variable, labeled with the
        # customer's real value.
        aggregated_shap = {}
        for i, feature_name in enumerate(self.readable_feature_names):
            raw_var = raw_variable_of(feature_name)
            aggregated_shap[raw_var] = aggregated_shap.get(raw_var, 0.0) + float(explanation.values[i])

        def display_label(raw_var):
            if raw_var in CATEGORICAL_VARIABLES:
                return f"{raw_var} = {raw_data[raw_var]}"
            return raw_var

        contributions_by_var = [
            {
                "variable": display_label(raw_var),
                "shap": value,
                "direction": "increases_risk" if value > 0 else "decreases_risk",
            }
            for raw_var, value in aggregated_shap.items()
        ]

        increasing = sorted(
            (c for c in contributions_by_var if c["shap"] > 0),
            key=lambda c: c["shap"], reverse=True,
        )
        decreasing = sorted(
            (c for c in contributions_by_var if c["shap"] <= 0),
            key=lambda c: c["shap"],
        )
        shap_contributions = increasing[:TOP_N_PER_DIRECTION] + decreasing[:TOP_N_PER_DIRECTION]

        response_time_ms = (time.perf_counter() - start) * 1000

        return {
            "probability": probability,
            "prediction": prediction,
            "risk_band": risk_band,
            "decision_threshold": self.decision_threshold,
            "low_band": self.low_band,
            "range_warnings": self.range_warnings(raw_data),
            "shap_contributions": shap_contributions,
            "shap_space": self.shap_space,
            "response_time_ms": response_time_ms,
            "input_values": {var: raw_data[var] for var in RAW_COLUMNS},
        }
