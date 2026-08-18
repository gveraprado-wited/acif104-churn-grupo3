"""Streamlit frontend: Dashboard de Cartera de Clientes (ACIF104 - Grupo 3)
Talks to the FastAPI backend over HTTP (never imports `backend/` directly and
never reads local data or model artifacts from the filesystem).

All Python identifiers in this file are in English; the string literals
shown to the user are in Spanish (Chile-based audience).

Run (with the backend already running in another terminal):
    uvicorn backend.api:app --reload
    streamlit run frontend/streamlit_app.py
"""
import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
DISPLAY_TIMEZONE = "America/Santiago"

# Per-variable form-widget config: (min, max, step, is_integer).
NUMERIC_FIELD_CONFIG = {
    "age": (0, 120, 1, True),
    "tenure_months": (0, 600, 1, True),
    "monthly_logins": (0, None, 1, True),
    "weekly_active_days": (0, 7, 1, True),
    "avg_session_time": (0.0, None, 0.1, False),
    "features_used": (0, None, 1, True),
    "usage_growth_rate": (None, None, 0.01, False),
    "last_login_days_ago": (0, None, 1, True),
    "monthly_fee": (0, None, 1, True),
    "total_revenue": (0, None, 1, True),
    "payment_failures": (0, None, 1, True),
    "support_tickets": (0, None, 1, True),
    "avg_resolution_time": (0.0, None, 0.1, False),
    "csat_score": (1, 5, 1, True),
    "escalations": (0, None, 1, True),
    "email_open_rate": (0.0, 1.0, 0.01, False),
    "marketing_click_rate": (0.0, 1.0, 0.01, False),
    "nps_score": (-100, 100, 1, True),
    "referral_count": (0, None, 1, True),
}

RISK_BAND_COLOR = {"Bajo": "green", "Medio": "orange", "Alto": "red"}
RISK_BAND_HEX = {"Bajo": "#2ecc71", "Medio": "#f1c40f", "Alto": "#e74c3c"}
RISK_BAND_EMOJI = {"Bajo": "🙂", "Medio": "😐", "Alto": "🚨"}

# How to render each raw variable's real value with its natural unit.
VALUE_FORMATTERS = {
    "age": lambda v: f"{v:.0f} años",
    "tenure_months": lambda v: f"{v:.0f} meses",
    "monthly_logins": lambda v: f"{v:.0f} al mes",
    "weekly_active_days": lambda v: f"{v:.0f} días",
    "avg_session_time": lambda v: f"{v:.1f} min",
    "features_used": lambda v: f"{v:.0f}",
    "usage_growth_rate": lambda v: f"{v:+.0%}",
    "last_login_days_ago": lambda v: f"hace {v:.0f} días",
    "monthly_fee": lambda v: f"${v:,.0f}",
    "total_revenue": lambda v: f"${v:,.0f}",
    "payment_failures": lambda v: f"{v:.0f}",
    "support_tickets": lambda v: f"{v:.0f}",
    "avg_resolution_time": lambda v: f"{v:.1f} hrs",
    "csat_score": lambda v: f"{v:.0f}/5",
    "escalations": lambda v: f"{v:.0f}",
    "email_open_rate": lambda v: f"{v:.0%}",
    "marketing_click_rate": lambda v: f"{v:.0%}",
    "nps_score": lambda v: f"{v:.0f} puntos",
    "referral_count": lambda v: f"{v:.0f}",
}

# Human-friendly Spanish names, for the form, charts, and explanatory text.
VARIABLE_LABELS = {
    "gender": "Género",
    "age": "Edad",
    "country": "País",
    "city": "Ciudad",
    "customer_segment": "Segmento de cliente",
    "signup_channel": "Canal de registro",
    "contract_type": "Tipo de contrato",
    "tenure_months": "Antigüedad (meses)",
    "monthly_logins": "Inicios de sesión mensuales",
    "weekly_active_days": "Días activos por semana",
    "avg_session_time": "Duración promedio de sesión (min)",
    "features_used": "Funciones utilizadas",
    "usage_growth_rate": "Crecimiento de uso reciente",
    "last_login_days_ago": "Días desde el último acceso",
    "monthly_fee": "Tarifa mensual",
    "total_revenue": "Ingresos totales del cliente",
    "payment_method": "Método de pago",
    "payment_failures": "Fallos de pago",
    "discount_applied": "Descuento aplicado",
    "price_increase_last_3m": "Alza de precio (últimos 3 meses)",
    "support_tickets": "Tickets de soporte",
    "avg_resolution_time": "Tiempo promedio de resolución (hrs)",
    "complaint_type": "Tipo de reclamo",
    "csat_score": "Puntaje de satisfacción (1 a 5)",
    "escalations": "Escalamientos de soporte",
    "email_open_rate": "Tasa de apertura de correos",
    "marketing_click_rate": "Tasa de clics en marketing",
    "nps_score": "Puntaje de recomendación (NPS)",
    "survey_response": "Respuesta de encuesta",
    "referral_count": "Cantidad de referidos",
}

# Categorical values translated for UI display.
VALUE_LABELS = {
    "Yes": "Sí",
    "No": "No",
    "Female": "Femenino",
    "Male": "Masculino",
    "Australia": "Australia",
    "Bangladesh": "Bangladés",
    "Canada": "Canadá",
    "Germany": "Alemania",
    "India": "India",
    "UK": "Reino Unido",
    "USA": "Estados Unidos",
    "Berlin": "Berlín",
    "Delhi": "Delhi",
    "Dhaka": "Daca",
    "London": "Londres",
    "New York": "Nueva York",
    "Sydney": "Sídney",
    "Toronto": "Toronto",
    "Enterprise": "Empresarial",
    "Individual": "Individual",
    "SME": "Pyme",
    "Mobile": "Móvil",
    "Referral": "Referido",
    "Web": "Sitio web",
    "Monthly": "Mensual",
    "Quarterly": "Trimestral",
    "Yearly": "Anual",
    "Bank Transfer": "Transferencia bancaria",
    "Card": "Tarjeta",
    "PayPal": "PayPal",
    "Billing": "Facturación",
    "No complaint": "Sin reclamo",
    "Service": "Servicio",
    "Technical": "Técnico",
    "Neutral": "Neutral",
    "Satisfied": "Satisfecho",
    "Unsatisfied": "Insatisfecho",
}

METRIC_LABELS = {
    "accuracy": "Exactitud (Accuracy)",
    "precision": "Precisión (Precision)",
    "recall": "Exhaustividad (Recall)",
    "f1": "Puntaje F1 (F1-Score)",
    "roc_auc": "Área ROC (ROC-AUC)",
    "pr_auc": "Área PR (PR-AUC)",
}

REQUEST_TIMEOUT_S = 15


class BackendUnavailableError(Exception):
    """Raised when the backend cannot be reached or times out."""


def _request(method: str, path: str, **kwargs) -> tuple[bool, dict]:
    """Shared HTTP client plumbing."""
    try:
        response = requests.request(
            method, f"{API_URL}{path}", timeout=REQUEST_TIMEOUT_S, **kwargs
        )
    except requests.exceptions.Timeout:
        raise BackendUnavailableError(
            f"El backend no respondió en {REQUEST_TIMEOUT_S} segundos ({path})."
        )
    except requests.exceptions.RequestException as error:
        raise BackendUnavailableError(f"No se pudo conectar al backend ({path}): {error}")

    if response.status_code >= 500:
        raise BackendUnavailableError(
            f"El backend respondió con un error interno ({response.status_code}) en {path}."
        )
    return response.ok, response.json()


def api_get(path: str) -> tuple[bool, dict]:
    return _request("GET", path)


def api_post(path: str, json_body: dict) -> tuple[bool, dict]:
    return _request("POST", path, json=json_body)


def api_get_required(path: str) -> dict:
    ok, body = api_get(path)
    if not ok:
        raise BackendUnavailableError(f"El backend devolvió un error inesperado en {path}.")
    return body


@st.cache_data(ttl=60)
def get_schema() -> dict:
    return api_get_required("/schema")


@st.cache_data(ttl=60)
def get_model_info() -> dict:
    return api_get_required("/model/info")


@st.cache_data(ttl=60)
def get_global_explainability() -> list[dict]:
    ok, body = api_get("/explainability/global")
    if not ok:
        raise BackendUnavailableError("Error al obtener la explicabilidad global.")
    return body


@st.cache_data(ttl=60)
def get_model_metrics() -> dict:
    return api_get_required("/model/metrics")


@st.cache_data(ttl=60)
def get_synthetic_demo_cartera() -> dict:
    return api_get_required("/demo/cartera-sintetica")


def shap_variable_label(technical_variable: str) -> str:
    """Translates technical variable or 'var = val' into natural language."""
    if " = " in technical_variable:
        raw_variable, value = technical_variable.split(" = ", 1)
        label = VARIABLE_LABELS.get(raw_variable, raw_variable)
        return f"{label}: {VALUE_LABELS.get(value, value)}"
    return VARIABLE_LABELS.get(technical_variable, technical_variable)


def format_factor(technical_variable: str, input_values: dict) -> str:
    """Formats a SHAP factor with its real value and natural unit."""
    if " = " in technical_variable:
        return shap_variable_label(technical_variable)
    label = VARIABLE_LABELS.get(technical_variable, technical_variable)
    value = input_values.get(technical_variable)
    if value is None:
        return label
    formatter = VALUE_FORMATTERS.get(technical_variable, lambda v: f"{v:g}")
    return f"{label}: {formatter(value)}"


def build_reason_sentence(result: dict) -> str:
    probability_pct = result["probability"] * 100
    threshold_pct = result["decision_threshold"] * 100
    comparison = "inferior" if result["probability"] < result["decision_threshold"] else "igual o superior"
    return (
        f"El modelo estimó una probabilidad de abandono de "
        f"**{probability_pct:.1f}%**. Como este valor es {comparison} al "
        f"umbral de **{threshold_pct:.0f}%**, el cliente se clasifica con "
        f"**riesgo {result['risk_band'].lower()}**."
    )


def build_synthesis_sentence(result: dict) -> str | None:
    increasing = [c for c in result["shap_contributions"] if c["direction"] == "increases_risk"]
    decreasing = [c for c in result["shap_contributions"] if c["direction"] == "decreases_risk"]
    if not increasing or not decreasing:
        return None

    def top_labels(contributions, n=2):
        names = [
            format_factor(c["variable"], result["input_values"]).split(":")[0].strip()
            for c in contributions[:n]
        ]
        return " y ".join(names)

    band = result["risk_band"]
    if band == "Alto":
        return (
            f"Aunque {top_labels(decreasing)} redujeron el riesgo, "
            f"{top_labels(increasing)} tuvieron más peso, por lo que el "
            f"modelo clasificó a este cliente con riesgo alto."
        )
    return (
        f"Aunque {top_labels(increasing)} aumentaron el riesgo, "
        f"{top_labels(decreasing)} tuvieron más peso, por lo que el modelo "
        f"clasificó a este cliente con riesgo {band.lower()}."
    )


def build_threshold_distance_caption(result: dict) -> str:
    probability = result["probability"]
    threshold = result["decision_threshold"]
    diff_points = abs(probability - threshold) * 100
    direction = "bajo" if probability < threshold else "sobre"
    return (
        f"Este cliente se encuentra **{diff_points:.1f} puntos porcentuales "
        f"{direction}** el umbral de riesgo."
    )


def draw_probability_bar(result: dict):
    probability = result["probability"]
    threshold = result["decision_threshold"]
    low_band = result["low_band"]

    fig, ax = plt.subplots(figsize=(6.5, 1.4))
    zones = [
        (0, low_band, RISK_BAND_HEX["Bajo"]),
        (low_band, threshold, RISK_BAND_HEX["Medio"]),
        (threshold, 1.0, RISK_BAND_HEX["Alto"]),
    ]
    for start, end, color in zones:
        ax.barh(0, end - start, left=start, color=color, height=0.5, edgecolor="white")

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, zorder=4)
    ax.annotate(
        f"Umbral {threshold:.0%}", xy=(threshold, 0.38), ha="center",
        fontsize=8, fontweight="bold",
    )

    ax.plot(probability, 0.26, marker="v", color="black", markersize=12, zorder=5)
    ax.annotate(
        f"Este cliente: {probability:.1%}", xy=(probability, -0.42), ha="center",
        fontsize=9, fontweight="bold",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, low_band, threshold, 1.0])
    ax.set_xticklabels([f"{0:.0%}", f"{low_band:.0%}", f"{threshold:.0%}", "100%"], fontsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def centered_pyplot(fig, width: int = 650):
    left, center, right = st.columns([1, 6, 1])
    with center:
        st.pyplot(fig)


def build_template_csv(schema: dict) -> str:
    """Creates a sample CSV template with all 30 required columns and 2 valid rows."""
    row1 = {}
    row2 = {}
    
    # Fill numeric columns
    for var, info in schema["numericas"].items():
        median = info["mediana"]
        min_v = info["min"]
        max_v = info["max"]
        row1[var] = median
        row2[var] = min(max_v, median * 1.2) if isinstance(median, (int, float)) else median
        
    # Fill categorical columns
    for var, info in schema["categoricas"].items():
        row1[var] = info["default"]
        vals = info["valores"]
        row2[var] = vals[1] if len(vals) > 1 else info["default"]
        
    df = pd.DataFrame([row1, row2])
    df.insert(0, "customer_id", ["CUST_EJEMPLO_001", "CUST_EJEMPLO_002"])
    return df.to_csv(index=False)


def validate_csv_structure(df: pd.DataFrame, schema: dict) -> tuple[list[str], list[str]]:
    """Validates CSV structure against the schema before any processing.

    Returns (errors, warnings). A non-empty `errors` means the caller must
    block processing (no "Procesar cartera" button); `warnings` are shown
    but don't block. Cell-level type/range validation is intentionally not
    duplicated here — that's already the backend's job per row via
    `ClientInput`, surfaced in the batch response's `errors`.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if len(df) == 0:
        errors.append("❌ El archivo no contiene registros de clientes para procesar.")
        return errors, warnings

    if "customer_id" not in df.columns:
        errors.append(
            "❌ El archivo no contiene la columna 'customer_id'. Todas las filas deben "
            "tener un identificador único de cliente en una columna llamada exactamente "
            "'customer_id'."
        )
    else:
        dupes = df["customer_id"][df["customer_id"].duplicated(keep=False)].unique().tolist()
        if dupes:
            shown = ", ".join(str(d) for d in dupes[:20])
            suffix = " y más" if len(dupes) > 20 else ""
            errors.append(
                f"❌ Se encontraron valores duplicados de 'customer_id': {shown}{suffix}. "
                "Cada cliente debe tener un identificador único."
            )

    required = sorted(set(schema["numericas"]) | set(schema["categoricas"]))
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(
            f"❌ Faltan {len(missing)} columna(s) requerida(s) por el modelo: "
            f"{', '.join(missing)}. Descargue la plantilla CSV para ver todas las "
            "variables necesarias."
        )

    found_churn_cols = [c for c in ("churn", "actual_churn") if c in df.columns]
    if found_churn_cols:
        errors.append(
            f"❌ El archivo contiene la(s) columna(s) {', '.join(found_churn_cols)}, que "
            "no debe(n) incluirse como variable de entrada. Elimínela(s) del archivo y "
            "vuelva a cargarlo."
        )

    if len(df) > 100:
        errors.append(
            f"❌ El archivo contiene {len(df)} registros, superando el máximo permitido "
            "de 100 por carga. Divida el archivo en lotes de 100 clientes o menos."
        )

    known = {"customer_id", "churn", "actual_churn"} | set(required)
    extra = [c for c in df.columns if c not in known]
    if extra:
        warnings.append(
            f"⚠️ Se detectaron {len(extra)} columna(s) no reconocida(s) que serán "
            f"ignoradas: {', '.join(extra)}."
        )

    return errors, warnings


def send_batch_request(customers_payload: list[dict]) -> dict:
    """Sends customers to POST /predict/batch in a single request (the
    100-row cap is already enforced upstream by validate_csv_structure)."""
    ok, res = api_post("/predict/batch", {"customers": customers_payload})
    if not ok:
        err_detail = res.get("detail", "Error en solicitud batch")
        if isinstance(err_detail, list):
            err_detail = "; ".join(f"{e.get('loc', '')}: {e.get('msg', '')}" for e in err_detail)
        raise BackendUnavailableError(f"El backend rechazó la solicitud batch: {err_detail}")

    return {
        "summary": res.get("summary", {}),
        "results": res.get("results", []),
        "errors": res.get("errors", []),
    }


def parse_dataframe_to_payload(df: pd.DataFrame) -> list[dict]:
    """Converts an already-validated DataFrame (see validate_csv_structure)
    to the POST /predict/batch payload."""
    customers_payload = []
    for _, row in df.iterrows():
        cid = str(row["customer_id"])
        features = {}
        for col in df.columns:
            if col == "customer_id" or col.lower() in ("churn", "actual_churn"):
                continue
            val = row[col]
            if pd.isna(val):
                features[col] = None
            elif isinstance(val, (int, float)):
                features[col] = float(val) if not float(val).is_integer() else int(val)
            else:
                features[col] = str(val)
        customers_payload.append({"customer_id": cid, "features": features})

    return customers_payload


# -----------------------------------------------------------------------------
# Streamlit Application Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard de Cartera — Predicción de Churn", layout="wide")

# Custom CSS for container width and dashboard styling
st.markdown(
    """
    <style>
    @media (min-width: 992px) {
        [data-testid="stMainBlockContainer"] {
            max-width: 85%;
            margin-left: auto;
            margin-right: auto;
        }
    }
    @media (max-width: 991px) {
        [data-testid="stMainBlockContainer"] {
            max-width: 96%;
            margin-left: auto;
            margin-right: auto;
        }
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Dashboard de Cartera — Riesgo de Abandono")

try:
    health = api_get_required("/health")
    st.caption(f"🟢 Backend conectado | Modelo productivo: `{health['modelo']}` | Espacio SHAP: `{health['espacio_shap']}`")
except BackendUnavailableError:
    st.error(
        f"❌ No se pudo conectar al backend en {API_URL}. "
        f"Verifica que el servicio esté corriendo con: `uvicorn backend.api:app --reload`"
    )
    st.stop()

# Initialize session state for batch results
if "batch_result" not in st.session_state:
    st.session_state["batch_result"] = None
if "batch_df" not in st.session_state:
    st.session_state["batch_df"] = None

# Tab definitions
tab_upload, tab_summary, tab_priorities, tab_model = st.tabs([
    "📥 Cargar cartera",
    "📈 Resumen",
    "🎯 Clientes prioritarios",
    "🧠 Modelo",
])

# -----------------------------------------------------------------------------
# TAB 1: Cargar cartera
# -----------------------------------------------------------------------------
with tab_upload:
    st.subheader("Carga y Validación de Cartera de Clientes")
    st.markdown(
        "Cargue un archivo CSV con la información de los clientes para evaluar su probabilidad de abandono "
        "en lote mediante el modelo de aprendizaje automático. La validación se realiza fila por fila, "
        "permitiendo procesar registros válidos incluso si otros contienen inconsistencias."
    )

    try:
        schema = get_schema()
    except BackendUnavailableError as error:
        st.error(str(error))
        st.stop()

    col_actions1, col_actions2, col_actions3 = st.columns([1, 1, 1])
    with col_actions1:
        template_csv = build_template_csv(schema)
        st.download_button(
            label="📄 Descargar plantilla CSV",
            data=template_csv,
            file_name="plantilla_cartera_churn.csv",
            mime="text/csv",
            help="Descarga un archivo CSV con las 30 columnas requeridas y valores de ejemplo válidos.",
        )

    with col_actions2:
        try:
            demo_cartera = get_synthetic_demo_cartera()
            st.download_button(
                label="📥 Descargar cartera_demo.csv",
                data=demo_cartera["csv_content"],
                file_name="cartera_demo.csv",
                mime="text/csv",
                help="Descarga el dataset sintético de demostración con 25 clientes operacionales.",
            )
        except BackendUnavailableError:
            pass

    with col_actions3:
        if st.button("🧪 Cargar lote demo (cartera_demo.csv)", help="Carga y procesa automáticamente el dataset sintético servido por el backend."):
            try:
                demo_cartera = get_synthetic_demo_cartera()
                demo_df = pd.read_csv(io.StringIO(demo_cartera["csv_content"]))
                st.session_state["batch_df"] = demo_df

                structure_errors, structure_warnings = validate_csv_structure(demo_df, schema)
                for w in structure_warnings:
                    st.warning(w)
                if structure_errors:
                    for e in structure_errors:
                        st.error(e)
                else:
                    customers_payload = parse_dataframe_to_payload(demo_df)
                    with st.spinner(f"Analizando {len(customers_payload)} clientes sintéticos con `POST /predict/batch`..."):
                        batch_res = send_batch_request(customers_payload)
                        st.session_state["batch_result"] = batch_res
                    summary = batch_res["summary"]
                    st.success(
                        f"¡Lote sintético cargado! **{summary['valid']} clientes válidos** analizados "
                        f"({summary['invalid']} errores). Diríjase a las pestañas **Resumen** y **Clientes prioritarios**."
                    )
            except BackendUnavailableError as e:
                st.error(f"Error al conectar con el backend: {e}")
            except Exception as e:
                st.error(f"Error al procesar cartera_demo.csv: {e}")

    uploaded_file = st.file_uploader(
        "Seleccione o arrastre su archivo CSV de cartera",
        type=["csv"],
        help="El archivo debe contener las variables descriptivas de cada cliente.",
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.session_state["batch_df"] = uploaded_df
            st.markdown(f"**Vista previa del archivo cargado:** `{len(uploaded_df)} registros` y `{len(uploaded_df.columns)} columnas`")
            st.dataframe(uploaded_df.head(10), use_container_width=True)

            structure_errors, structure_warnings = validate_csv_structure(uploaded_df, schema)
            for w in structure_warnings:
                st.warning(w)

            if structure_errors:
                for e in structure_errors:
                    st.error(e)
            else:
                customers_payload = parse_dataframe_to_payload(uploaded_df)

                if st.button("🚀 Procesar cartera con el modelo", type="primary"):
                    with st.spinner(f"Analizando {len(customers_payload)} clientes mediante `POST /predict/batch`..."):
                        batch_res = send_batch_request(customers_payload)
                        st.session_state["batch_result"] = batch_res

                    summary = batch_res["summary"]
                    if summary["valid"] > 0:
                        st.success(
                            f"✅ Procesamiento completado: **{summary['valid']} clientes válidos** analizados "
                            f"({summary['invalid']} errores aislados). Diríjase a las pestañas **Resumen** y **Clientes prioritarios** para ver los resultados."
                        )
                    else:
                        st.error(
                            f"⚠️ Se procesaron {summary['total']} registros pero todos presentaron errores de validación. "
                            "Revise el detalle de inconsistencias en la pestaña Resumen."
                        )
        except Exception as err:
            st.error(f"Error al leer el archivo CSV: {err}")

# -----------------------------------------------------------------------------
# TAB 2: Resumen
# -----------------------------------------------------------------------------
with tab_summary:
    st.subheader("Resumen Ejecutivo y Distribución de Riesgo")
    batch_data = st.session_state.get("batch_result")

    if not batch_data or not batch_data.get("results"):
        st.info("ℹ️ No hay resultados de cartera para mostrar. Por favor cargue y procese un archivo CSV en la pestaña **Cargar cartera**.")
    else:
        summary = batch_data["summary"]
        results = batch_data["results"]
        errors = batch_data.get("errors", [])

        # KPI row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Clientes Procesados", f"{summary['total']}")
        valid_rate = (summary["valid"] / summary["total"]) if summary["total"] > 0 else 0
        kpi2.metric("Registros Válidos", f"{summary['valid']}", f"{valid_rate:.0%} éxito")
        high_risk_count = summary["band_counts"].get("Alto", 0)
        high_risk_pct = (high_risk_count / summary["valid"]) if summary["valid"] > 0 else 0
        kpi3.metric("🚨 En Riesgo Alto", f"{high_risk_count}", f"{high_risk_pct:.1%} de válidos", delta_color="inverse")
        kpi4.metric("💰 Facturación Mensual en Riesgo", f"${summary['high_risk_monthly_fee']:,.0f}")

        st.divider()

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("#### Distribución por Banda de Riesgo")
            band_counts_data = summary["band_counts"]
            band_names = ["Bajo", "Medio", "Alto"]
            counts = [band_counts_data.get(b, 0) for b in band_names]
            colors = [RISK_BAND_HEX[b] for b in band_names]

            fig_band, ax_band = plt.subplots(figsize=(5, 3.2))
            bars = ax_band.bar(band_names, counts, color=colors, edgecolor="#333333", linewidth=0.8)
            ax_band.set_ylabel("Cantidad de Clientes", fontsize=9)
            ax_band.tick_params(labelsize=9)
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax_band.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{count} ({(count / max(summary['valid'], 1)):.0%})",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )
            for spine in ("top", "right"):
                ax_band.spines[spine].set_visible(False)
            fig_band.tight_layout()
            st.pyplot(fig_band)
            plt.close(fig_band)

        with col_chart2:
            st.markdown("#### Distribución de Probabilidades Estimadas")
            probs = [r["probability"] for r in results]
            if probs:
                fig_prob, ax_prob = plt.subplots(figsize=(5, 3.2))
                ax_prob.hist(probs, bins=10, range=(0, 1), color="#3498db", edgecolor="white", linewidth=0.8)
                model_info = get_model_info()
                thresh = model_info.get("decision_threshold", 0.56)
                ax_prob.axvline(thresh, color="red", linestyle="--", linewidth=1.5, label=f"Umbral ({thresh:.0%})")
                ax_prob.set_xlabel("Probabilidad de Abandono", fontsize=9)
                ax_prob.set_ylabel("Frecuencia", fontsize=9)
                ax_prob.tick_params(labelsize=9)
                ax_prob.legend(fontsize=8)
                for spine in ("top", "right"):
                    ax_prob.spines[spine].set_visible(False)
                fig_prob.tight_layout()
                st.pyplot(fig_prob)
                plt.close(fig_prob)

        if errors:
            st.divider()
            with st.expander(f"⚠️ Detalle de Registros Inválidos o con Error ({len(errors)})", expanded=False):
                st.caption(
                    "Estos registros presentaron inconsistencias (valores fuera de rango o categorías desconocidas) "
                    "y fueron aislados sin interrumpir el análisis del resto del lote."
                )
                errors_df = pd.DataFrame(errors)
                errors_df = errors_df.rename(columns={"row_index": "Fila", "customer_id": "ID Cliente", "detail": "Detalle del Error"})
                st.dataframe(errors_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Clientes prioritarios
# -----------------------------------------------------------------------------
with tab_priorities:
    st.subheader("Priorización Operativa y Ficha Individual con SHAP")
    st.markdown(
        "Lista de clientes clasificados por nivel de riesgo y ordenados descendentemente por probabilidad "
        "de abandono para focalizar las acciones de retención inmediata."
    )

    batch_data = st.session_state.get("batch_result")
    if not batch_data or not batch_data.get("results"):
        st.info("ℹ️ Cargue una cartera en la pestaña **Cargar cartera** para habilitar la priorización y fichas de clientes.")
    else:
        results = batch_data["results"]

        # Build interactive table DataFrame
        table_rows = []
        for r in results:
            inp = r.get("input_values", {})
            table_rows.append({
                "ID": r["customer_id"],
                "Probabilidad": r["probability"],
                "Banda de Riesgo": r["risk_band"],
                "Tarifa Mensual": inp.get("monthly_fee", 0),
                "Antigüedad (meses)": inp.get("tenure_months", 0),
                "Tickets Soporte": inp.get("support_tickets", 0),
                "Satisfacción (CSAT)": inp.get("csat_score", 0),
                "Tipo de Contrato": VALUE_LABELS.get(inp.get("contract_type"), inp.get("contract_type", "-")),
                "Segmento": VALUE_LABELS.get(inp.get("customer_segment"), inp.get("customer_segment", "-")),
                "_raw_result": r,
            })

        df_results = pd.DataFrame(table_rows)

        # Filters
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            band_filter = st.selectbox("Filtrar por Banda de Riesgo", ["Todas", "Alto", "Medio", "Bajo"], index=0)
        with f_col2:
            search_query = st.text_input("🔍 Buscar por ID de Cliente", "")

        filtered_df = df_results.copy()
        if band_filter != "Todas":
            filtered_df = filtered_df[filtered_df["Banda de Riesgo"] == band_filter]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df["ID"].astype(str).str.contains(search_query.strip(), case=False)]

        filtered_df = filtered_df.sort_values("Probabilidad", ascending=False)

        # Display table with formatting
        display_df = filtered_df.drop(columns=["_raw_result"]).copy()
        display_df["Probabilidad"] = display_df["Probabilidad"].map(lambda p: f"{p:.1%}")
        display_df["Tarifa Mensual"] = display_df["Tarifa Mensual"].map(lambda f: f"${f:,.0f}")
        display_df["Satisfacción (CSAT)"] = display_df["Satisfacción (CSAT)"].map(lambda s: f"{s:.0f}/5")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()

        # Individual SHAP Profile
        st.markdown("### 🔍 Ficha Explicativa Individual (SHAP)")
        available_ids = list(filtered_df["ID"].unique())

        if not available_ids:
            st.warning("No hay clientes que coincidan con los filtros seleccionados.")
        else:
            selected_id = st.selectbox("Seleccione un cliente para inspeccionar sus factores:", available_ids)
            selected_row = filtered_df[filtered_df["ID"] == selected_id].iloc[0]
            selected_result = selected_row["_raw_result"]

            risk_band = selected_result["risk_band"]
            color = RISK_BAND_COLOR.get(risk_band, "gray")
            emoji = RISK_BAND_EMOJI.get(risk_band, "")

            st.markdown(f"#### {emoji} Cliente `{selected_id}` — :{color}[Riesgo {risk_band.lower()}] de abandono")
            
            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.metric("Probabilidad Estimada", f"{selected_result['probability']:.1%}")
            with p_col2:
                st.caption("Ubicación frente al umbral de decisión:")
                bar_fig = draw_probability_bar(selected_result)
                st.pyplot(bar_fig)
                plt.close(bar_fig)
                st.caption(build_threshold_distance_caption(selected_result))

            for warning in selected_result.get("range_warnings", []):
                st.warning(warning)

            st.markdown("#### Factores que explicaron la predicción")
            st.markdown(f"**¿Por qué obtuvo este resultado?** {build_reason_sentence(selected_result)}")

            increasing = [c for c in selected_result["shap_contributions"] if c["direction"] == "increases_risk"]
            decreasing = [c for c in selected_result["shap_contributions"] if c["direction"] == "decreases_risk"]

            col_up, col_down = st.columns(2)
            with col_up:
                st.markdown("**🔺 Factores que aumentaron el riesgo**")
                if increasing:
                    for contribution in increasing:
                        st.markdown(f"- {format_factor(contribution['variable'], selected_result['input_values'])}")
                else:
                    st.caption("Ningún factor relevante empujó el riesgo hacia arriba.")
            with col_down:
                st.markdown("**🔻 Factores que redujeron el riesgo**")
                if decreasing:
                    for contribution in decreasing:
                        st.markdown(f"- {format_factor(contribution['variable'], selected_result['input_values'])}")
                else:
                    st.caption("Ningún factor relevante empujó el riesgo hacia abajo.")

            synthesis = build_synthesis_sentence(selected_result)
            if synthesis:
                st.markdown(synthesis)

            contributions = pd.DataFrame(selected_result["shap_contributions"])
            if not contributions.empty:
                contributions["label"] = contributions["variable"].apply(shap_variable_label)
                contributions = contributions.sort_values("shap")
                colors = ["#d62728" if v > 0 else "#2ca02c" for v in contributions["shap"]]

                fig, ax = plt.subplots(figsize=(6.5, 3))
                ax.barh(contributions["label"], contributions["shap"], color=colors)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Fuerza del efecto (rojo = aumenta el riesgo, verde = lo reduce)", fontsize=9)
                ax.tick_params(labelsize=9)
                fig.tight_layout()
                centered_pyplot(fig, width=650)
                plt.close(fig)

                st.caption(
                    "Valores calculados mediante explicaciones locales SHAP (TreeExplainer) expresados en espacio de "
                    f"`{selected_result['shap_space']}`. La magnitud y dirección representan la contribución marginal de cada variable."
                )

# -----------------------------------------------------------------------------
# TAB 4: Modelo
# -----------------------------------------------------------------------------
with tab_model:
    st.subheader("Gobernanza, Umbrales y Métricas del Modelo")
    st.markdown(
        "Información técnica y métricas de desempeño validadas sobre el conjunto de prueba independiente. "
        "Todos los datos de esta vista se consultan en tiempo real vía HTTP desde los endpoints del backend."
    )

    try:
        model_info = get_model_info()
        metrics = get_model_metrics()
        global_shap = get_global_explainability()
    except BackendUnavailableError as error:
        st.error(str(error))
        st.stop()

    # Model configuration cards
    st.markdown("#### Configuración Operativa")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelo Activo", model_info.get("model", "xgboost_refinado.joblib"))
    c2.metric("Umbral de Decisión", f"{model_info.get('decision_threshold', 0.56):.0%}")
    c3.metric("Límite Banda Baja", f"{model_info.get('low_band', 0.24):.0%}")
    c4.metric("Espacio SHAP", model_info.get("shap_space", "margen"))

    st.divider()

    # Test metrics
    st.markdown("#### Desempeño Validado (Conjunto de Prueba Independiente)")
    st.caption(
        "Métricas offline congeladas en la evaluación final. Reflejan el comportamiento real del modelo "
        "ante clientes no vistos durante el entrenamiento ni la selección de hiperparámetros."
    )

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Exhaustividad (Recall)", f"{metrics.get('recall', 0):.1%}")
    m_col2.metric("Precisión (Precision)", f"{metrics.get('precision', 0):.1%}")
    m_col3.metric("F1-Score", f"{metrics.get('f1', 0):.1%}")
    m_col4.metric("Área ROC (ROC-AUC)", f"{metrics.get('roc_auc', 0):.3f}")

    metrics_display = []
    for k, val in metrics.items():
        metrics_display.append({
            "Métrica": METRIC_LABELS.get(k, k.upper()),
            "Valor": f"{val:.1%}" if val <= 1.0 else f"{val:.3f}",
        })
    st.dataframe(pd.DataFrame(metrics_display), use_container_width=True, hide_index=True)

    st.divider()

    # Global SHAP feature importance
    st.markdown("#### Importancia Global de Variables (SHAP)")
    st.caption(
        "Importancia media absoluta de cada variable sobre el conjunto de validación, consumida vía `GET /explainability/global`."
    )

    if global_shap:
        df_global = pd.DataFrame(global_shap).head(15).copy()
        df_global["label"] = df_global["variable"].apply(shap_variable_label)
        df_global = df_global.sort_values("importancia_media_abs", ascending=True)

        fig_glob, ax_glob = plt.subplots(figsize=(8, 5))
        ax_glob.barh(df_global["label"], df_global["importancia_media_abs"], color="#2b5c8f")
        ax_glob.set_xlabel("Impacto medio absoluto en la probabilidad de abandono (|SHAP|)", fontsize=9)
        ax_glob.tick_params(labelsize=9)
        for spine in ("top", "right"):
            ax_glob.spines[spine].set_visible(False)
        fig_glob.tight_layout()
        centered_pyplot(fig_glob, width=800)
        plt.close(fig_glob)
    else:
        st.info("No se pudieron cargar los datos de importancia global.")
