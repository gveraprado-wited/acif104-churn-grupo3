"""Streamlit frontend: talks to the FastAPI backend over HTTP (never imports
`backend/` directly), so the front end / back end separation is real.

All Python identifiers in this file are in English; the string literals
shown to the user are in Spanish (Chile-based audience).

Run (with the backend already running in another terminal):
    uvicorn backend.api:app --reload
    streamlit run frontend/streamlit_app.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISPLAY_TIMEZONE = "America/Santiago"

# Per-variable form-widget config: (min, max, step, is_integer). `None`
# means "no bound" (the model can still extrapolate above the historical
# max — only variables with a real definitional limit get a max here, e.g.
# csat_score can't exceed 5 by definition). Integer variables render without
# decimals; counts get a non-negative floor even without a hard ceiling.
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

FORM_GROUPS = {
    "Demografía": ["gender", "age", "country", "city"],
    "Contrato y uso": [
        "customer_segment", "signup_channel", "contract_type", "tenure_months",
        "monthly_logins", "weekly_active_days", "avg_session_time",
        "features_used", "usage_growth_rate", "last_login_days_ago",
    ],
    "Facturación": [
        "monthly_fee", "total_revenue", "payment_method", "payment_failures",
        "discount_applied", "price_increase_last_3m",
    ],
    "Soporte": [
        "support_tickets", "avg_resolution_time", "complaint_type",
        "csat_score", "escalations",
    ],
    "Interacción y marketing": [
        "email_open_rate", "marketing_click_rate", "nps_score",
        "survey_response", "referral_count",
    ],
}

RISK_BAND_COLOR = {"Bajo": "green", "Medio": "orange", "Alto": "red"}
RISK_BAND_HEX = {"Bajo": "#2ecc71", "Medio": "#f1c40f", "Alto": "#e74c3c"}
RISK_BAND_EMOJI = {"Bajo": "🙂", "Medio": "😐", "Alto": "🚨"}

# How to render each raw variable's real value with its natural unit, for
# the "factores que aumentaron/redujeron el riesgo" lists. Falls back to a
# plain number when a variable has no specific formatter.
VALUE_FORMATTERS = {
    "age": lambda v: f"{v:.0f} años",
    "tenure_months": lambda v: f"{v:.0f} meses",
    "monthly_logins": lambda v: f"{v:.0f} al mes",
    "weekly_active_days": lambda v: f"{v:.0f} días",
    "avg_session_time": lambda v: f"{v:.1f} min",
    "features_used": lambda v: f"{v:.0f}",
    "usage_growth_rate": lambda v: f"{v:+.0%}",
    "last_login_days_ago": lambda v: f"hace {v:.0f} días",
    "monthly_fee": lambda v: f"${v:.0f}",
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

# Categorical values, translated for the form, charts, and explanatory text
# (the actual value sent to the API is still the dataset's original one;
# this is only for what the user sees).
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

# Column headers for the raw monitoring events table (English internally,
# translated only for display so the UI stays in Spanish).
EVENT_COLUMN_LABELS = {
    "timestamp": "Fecha y hora",
    "source": "Origen",
    "status": "Resultado",
    "probability": "Probabilidad",
    "risk_band": "Banda de riesgo",
    "decision_threshold": "Umbral",
    "response_time_ms": "Tiempo de respuesta (ms)",
    "detail": "Detalle",
}
EVENT_SOURCE_LABELS = {
    "form": "Formulario",
    "customer_profile": "Ficha de cliente",
    "api": "Solicitud rechazada antes de procesar",
}
EVENT_STATUS_LABELS = {"success": "Éxito", "error": "Error"}

REQUEST_TIMEOUT_S = 10


class BackendUnavailableError(Exception):
    """Raised when the backend can't be reached or times out — every caller
    turns this into a clear st.error() instead of letting the app crash or
    hang indefinitely.
    """


def _request(method: str, path: str, **kwargs) -> tuple[bool, dict]:
    """Shared plumbing for api_get/api_post. Returns (ok, body): `ok` is
    False for an ordinary 4xx (invalid input, not found — the caller decides
    what that means for its endpoint). A network failure, timeout, or 5xx is
    NOT something a caller's success/failure branch can meaningfully handle,
    so those raise BackendUnavailableError instead.
    """
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
    """For endpoints that should never legitimately return a 4xx (schema,
    customer list, health, monitoring log) — any failure here is treated as
    the backend being unavailable, not a normal business outcome.
    """
    ok, body = api_get(path)
    if not ok:
        raise BackendUnavailableError(f"El backend devolvió un error inesperado en {path}.")
    return body


@st.cache_data(ttl=60)
def get_schema() -> dict:
    return api_get_required("/schema")


@st.cache_data(ttl=60)
def get_customer_ids() -> list[str]:
    return api_get_required("/customers")["customer_ids"]


def form_field(variable: str, schema: dict, key_prefix: str):
    label = VARIABLE_LABELS.get(variable, variable)
    if variable in schema["numericas"]:
        info = schema["numericas"][variable]
        min_value, max_value, step, is_integer = NUMERIC_FIELD_CONFIG.get(
            variable, (None, None, 1.0, False)
        )
        cast = int if is_integer else float
        default_value = cast(round(info["mediana"])) if is_integer else float(info["mediana"])
        return st.number_input(
            label,
            value=default_value,
            min_value=cast(min_value) if min_value is not None else None,
            max_value=cast(max_value) if max_value is not None else None,
            step=cast(step),
            help=f"Rango histórico observado: {info['min']:g} a {info['max']:g}.",
            key=f"{key_prefix}_{variable}",
        )
    info = schema["categoricas"][variable]
    return st.selectbox(
        label,
        options=info["valores"],
        index=info["valores"].index(info["default"]),
        format_func=lambda v: VALUE_LABELS.get(v, v),
        key=f"{key_prefix}_{variable}",
    )


def shap_variable_label(technical_variable: str) -> str:
    """Translates a variable name (or 'variable = value' for categoricals)
    into natural language, for charts and explanatory text.
    """
    if " = " in technical_variable:
        raw_variable, value = technical_variable.split(" = ", 1)
        label = VARIABLE_LABELS.get(raw_variable, raw_variable)
        return f"{label}: {VALUE_LABELS.get(value, value)}"
    return VARIABLE_LABELS.get(technical_variable, technical_variable)


def format_factor(technical_variable: str, input_values: dict) -> str:
    """Formats a SHAP-contributing variable with its real value and unit,
    e.g. 'Puntaje de satisfacción (1 a 5): 4/5' or 'Respuesta de encuesta:
    Insatisfecho' — never a vague 'nivel alto/bajo'.
    """
    if " = " in technical_variable:
        return shap_variable_label(technical_variable)
    label = VARIABLE_LABELS.get(technical_variable, technical_variable)
    value = input_values.get(technical_variable)
    if value is None:
        return label
    formatter = VALUE_FORMATTERS.get(technical_variable, lambda v: f"{v:g}")
    return f"{label}: {formatter(value)}"


def build_reason_sentence(result: dict) -> str:
    """States why the client landed in that risk band: probability vs.
    threshold, both on the same percentage scale.
    """
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
    """A short closing sentence naming the actual top factors on each side —
    only when both directions are present, and only naming the real SHAP
    factors already shown (nothing invented beyond them).
    """
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
    """Segmented horizontal bar (not a gauge) showing the client's
    probability against the model's real risk bands and decision threshold —
    easier to compare at a glance than a lone percentage.
    """
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


def centered_pyplot(fig, width: int):
    """Renders a matplotlib figure centered in the available width, instead
    of left-aligned (Streamlit's default for a fixed-width image)."""
    left, center, right = st.columns([1, 4, 1])
    with center:
        st.pyplot(fig, width=width)


def show_result(result: dict):
    risk_band = result["risk_band"]
    color = RISK_BAND_COLOR.get(risk_band, "gray")
    emoji = RISK_BAND_EMOJI.get(risk_band, "")

    st.markdown(f"### {emoji} :{color}[Riesgo {risk_band.lower()}] de abandono")
    st.metric("Probabilidad estimada", f"{result['probability']:.1%}")

    for warning in result.get("range_warnings", []):
        st.warning(warning)

    st.caption("Nivel estimado de riesgo de abandono (churn)")
    bar_fig = draw_probability_bar(result)
    centered_pyplot(bar_fig, width=650)
    plt.close(bar_fig)
    st.caption(build_threshold_distance_caption(result))

    if result.get("actual_churn") is not None:
        actual_label = "Abandonó" if result["actual_churn"] == 1 else "Permaneció"
        st.caption(
            f"Resultado observado del conjunto de validación: este cliente "
            f"**{actual_label.lower()}** realmente. Esta información se muestra "
            f"solo con fines de demostración y no forma parte de las variables "
            f"utilizadas por el modelo."
        )

    st.markdown("#### Factores que influyeron en esta predicción")
    st.markdown(f"**¿Por qué obtuvo este resultado?** {build_reason_sentence(result)}")

    increasing = [c for c in result["shap_contributions"] if c["direction"] == "increases_risk"]
    decreasing = [c for c in result["shap_contributions"] if c["direction"] == "decreases_risk"]

    col_up, col_down = st.columns(2)
    with col_up:
        st.markdown("**🔺 Factores que aumentaron el riesgo**")
        if increasing:
            for contribution in increasing:
                st.markdown(f"- {format_factor(contribution['variable'], result['input_values'])}")
        else:
            st.caption("Ningún factor relevante empujó el riesgo hacia arriba.")
    with col_down:
        st.markdown("**🔻 Factores que redujeron el riesgo**")
        if decreasing:
            for contribution in decreasing:
                st.markdown(f"- {format_factor(contribution['variable'], result['input_values'])}")
        else:
            st.caption("Ningún factor relevante empujó el riesgo hacia abajo.")

    synthesis = build_synthesis_sentence(result)
    if synthesis:
        st.markdown(synthesis)

    contributions = pd.DataFrame(result["shap_contributions"])
    contributions["label"] = contributions["variable"].apply(shap_variable_label)
    contributions = contributions.sort_values("shap")
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in contributions["shap"]]

    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.barh(contributions["label"], contributions["shap"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        "Fuerza del efecto (rojo = aumenta el riesgo, verde = lo reduce)", fontsize=9
    )
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    centered_pyplot(fig, width=650)
    plt.close(fig)

    st.caption(
        "Estas son asociaciones que aprendió el modelo a partir de datos "
        "históricos, no causas comprobadas: no implican que cambiar una sola "
        "variable vaya a cambiar el resultado por sí sola. Los valores "
        f"numéricos del gráfico están expresados en espacio de "
        f"{result['shap_space']}: el signo indica la dirección del efecto y "
        "la magnitud qué tan fuerte es en relación con las demás variables "
        "de este caso — no se leen como puntos de probabilidad."
    )
    st.caption(f"Tiempo de respuesta: {result['response_time_ms']:.1f} ms")


@st.dialog("Resultado de la predicción", width="large")
def show_result_modal(result: dict):
    show_result(result)


st.set_page_config(page_title="Predicción de churn", layout="wide")

# On desktop the content is capped at ~70% of the window (full-bleed "wide"
# layout isn't necessary for this app); on narrow/mobile screens it uses
# close to the full width, with small margins instead of a fixed cap.
st.markdown(
    """
    <style>
    @media (min-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            max-width: 70%;
            margin-left: auto;
            margin-right: auto;
        }
    }
    @media (max-width: 767px) {
        [data-testid="stMainBlockContainer"] {
            max-width: 96%;
            margin-left: auto;
            margin-right: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Predicción de abandono de clientes")

try:
    health = api_get_required("/health")
    st.caption(f"Backend conectado — modelo: {health['modelo']}")
except BackendUnavailableError:
    st.error(
        f"No se pudo conectar al backend en {API_URL}. "
        f"¿Está corriendo `uvicorn backend.api:app --reload`?"
    )
    st.stop()

tab_predict, tab_profile, tab_explainability, tab_monitoring = st.tabs(
    ["Predicción individual", "Ficha de cliente", "Explicabilidad global", "Monitoreo"]
)

with tab_predict:
    st.subheader("Ingresar un cliente nuevo")
    try:
        schema = get_schema()
    except BackendUnavailableError as error:
        st.error(str(error))
        st.stop()

    with st.form("prediction_form"):
        form_data = {}
        for group_name, variables in FORM_GROUPS.items():
            st.markdown(f"**{group_name}**")
            columns = st.columns(3)
            for i, variable in enumerate(variables):
                with columns[i % 3]:
                    form_data[variable] = form_field(variable, schema, "form")
        submitted = st.form_submit_button("Predecir")

    if submitted:
        try:
            with st.spinner("Calculando predicción y explicación del modelo...", show_time=True):
                ok, body = api_post("/predict", form_data)
        except BackendUnavailableError as error:
            st.error(str(error))
        else:
            if ok:
                show_result_modal(body)
            else:
                st.error(body.get("detail", "Error al predecir."))

with tab_profile:
    st.subheader("Buscar un cliente del conjunto de validación")
    try:
        customer_ids = get_customer_ids()
    except BackendUnavailableError as error:
        st.error(str(error))
        st.stop()

    customer_id = st.selectbox("Cliente", options=customer_ids, key="profile_customer_id")

    if st.button("Ver ficha"):
        try:
            with st.spinner("Calculando predicción y explicación del modelo...", show_time=True):
                ok, body = api_get(f"/customers/{customer_id}")
        except BackendUnavailableError as error:
            st.error(str(error))
        else:
            if ok:
                show_result_modal(body)
            else:
                st.error(body.get("detail", "Cliente no encontrado."))

with tab_explainability:
    st.subheader("Importancia global de variables (SHAP)")
    st.caption(
        "Calculada una sola vez sobre el conjunto de validación en "
        "notebook/07_explicabilidad_shap.ipynb — no se recalcula en la app."
    )
    importance_path = PROJECT_ROOT / "models" / "shap_importancia_global.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(15).copy()
        importance["label"] = importance["variable"].apply(shap_variable_label)
        importance = importance.sort_values("importancia_media_abs")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(importance["label"], importance["importancia_media_abs"])
        ax.set_xlabel(
            "Importancia media (cuánto influye, en promedio, en las predicciones)",
            fontsize=10,
        )
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        centered_pyplot(fig, width=850)
        plt.close(fig)
    else:
        st.warning("No se encontró models/shap_importancia_global.csv.")

with tab_monitoring:
    st.subheader("Actividad de la aplicación")
    try:
        events = api_get_required("/monitoring")
    except BackendUnavailableError as error:
        st.error(str(error))
        events = []

    if events:
        events_df = pd.DataFrame(events)
        successes = events_df[events_df["status"] == "success"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Solicitudes registradas", len(events_df))
        col2.metric(
            "Tasa de éxito",
            f"{(events_df['status'] == 'success').mean():.0%}",
        )
        col3.metric(
            "Tiempo de respuesta promedio",
            f"{successes['response_time_ms'].mean():.1f} ms" if len(successes) else "—",
        )
        col4.metric(
            "% en banda Alto (éxitos)",
            f"{(successes['risk_band'] == 'Alto').mean():.0%}" if len(successes) else "—",
        )
        if len(successes):
            st.bar_chart(successes["risk_band"].value_counts(), height=220)
        with st.expander("Ver eventos registrados"):
            display_df = events_df.copy()
            display_df["timestamp"] = (
                pd.to_datetime(display_df["timestamp"])
                .dt.tz_convert(DISPLAY_TIMEZONE)
                .dt.strftime("%d/%m/%Y %H:%M")
            )
            # .fillna(original) so an unmapped future value shows as-is
            # instead of silently turning into "None".
            display_df["source"] = display_df["source"].map(EVENT_SOURCE_LABELS).fillna(display_df["source"])
            display_df["status"] = display_df["status"].map(EVENT_STATUS_LABELS).fillna(display_df["status"])
            display_df = display_df.fillna("")  # empty prediction fields on error rows, not "None"
            display_df = display_df.rename(columns=EVENT_COLUMN_LABELS)
            st.dataframe(display_df, width="stretch")
    else:
        st.info("Todavía no se registran predicciones en esta sesión.")

    st.divider()
    st.subheader("Desempeño validado en el conjunto de prueba")
    st.caption(
        "Métricas offline del modelo (notebook/08_evaluacion_final_test.ipynb), "
        "no de esta ejecución de la app: no existe todavía churn real posterior "
        "para los clientes ingresados por la app, así que estas métricas no se "
        "recalculan aquí."
    )
    metrics_path = PROJECT_ROOT / "models" / "metricas_test_modelo_refinado.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics.columns = ["métrica", "valor"]
        st.dataframe(metrics, width="stretch", hide_index=True)
    else:
        st.warning("No se encontró models/metricas_test_modelo_refinado.csv.")
