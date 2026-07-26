"""Monitoring log: prediction metadata for both successful and failed
requests (probability, risk band, threshold, response time, or the error
detail). Raw customer variables are never stored, so no potentially
sensitive information gets persisted.

A single append-only CSV is adequate for this prototype (low request
volume, one process, no concurrent writers). It does not scale to multiple
backend instances or high write concurrency — appending to the same file
from several processes risks interleaved/corrupted rows, and there's no
indexing for querying by date range or status at volume. Migrating to a
real database (e.g. SQLite for a single instance, or Postgres for several)
is the natural next step if the app moves past prototype stage.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "monitoring" / "registro_predicciones.csv"

LOG_COLUMNS = [
    "timestamp",
    "source",
    "status",
    "probability",
    "risk_band",
    "decision_threshold",
    "response_time_ms",
    "detail",
]


def log_event(
    source: str,
    status: str = "success",
    probability: float | None = None,
    risk_band: str | None = None,
    decision_threshold: float | None = None,
    response_time_ms: float | None = None,
    detail: str = "",
    log_path: Path | None = None,
) -> None:
    """Appends a row to the monitoring CSV (append-only). `source`
    distinguishes predictions from the manual form ('formulario') from
    customer-profile lookups ('ficha_cliente'). `status` is 'success' or
    'error': error events (invalid input, unknown customer id) log `detail`
    with the reason instead of the prediction fields, which stay empty.
    If `log_path` isn't given, `LOG_PATH` is read at call time (not at
    function-definition time), so tests can override the module with
    `monkeypatch`.
    """
    log_path = log_path or LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            source,
            status,
            probability,
            risk_band,
            decision_threshold,
            response_time_ms,
            detail,
        ])


def read_events(log_path: Path | None = None) -> pd.DataFrame:
    log_path = log_path or LOG_PATH
    if not log_path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(log_path)
