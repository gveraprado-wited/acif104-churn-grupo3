"""Read-only accessors over the artifacts in `models/`, so the frontend
never needs local filesystem access to them for explainability or metrics —
it only talks to `backend/api.py` over HTTP (see plan section 6.3,
"Cambios de desacoplamiento").
"""
from pathlib import Path

import pandas as pd

from backend.inference import ChurnPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GLOBAL_IMPORTANCE_PATH = PROJECT_ROOT / "models" / "shap_importancia_global.csv"
TEST_METRICS_PATH = PROJECT_ROOT / "models" / "metricas_test_modelo_refinado.csv"


def get_model_info(predictor: ChurnPredictor) -> dict:
    return {
        "model": "xgboost_refinado.joblib",
        "decision_threshold": predictor.decision_threshold,
        "low_band": predictor.low_band,
        "shap_space": predictor.shap_space,
    }


def get_global_explainability() -> list[dict]:
    """Global SHAP feature importance (mean |SHAP| over the validation set),
    computed once in notebook/07_explicabilidad_shap.ipynb — never
    recalculated here.
    """
    importance = pd.read_csv(GLOBAL_IMPORTANCE_PATH)
    importance = importance.sort_values("importancia_media_abs", ascending=False)
    return importance.to_dict(orient="records")


def get_model_metrics() -> dict:
    """Final held-out test-set metrics from
    notebook/08_evaluacion_final_test.ipynb — the only place the test set is
    touched; never recomputed here.
    """
    metrics = pd.read_csv(TEST_METRICS_PATH, index_col=0)
    return metrics.iloc[:, 0].to_dict()
