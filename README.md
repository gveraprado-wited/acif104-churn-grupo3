# Predicción de abandono de clientes (churn) — ACIF104, Grupo 3

Proyecto académico de la asignatura ACIF104 - Aprendizaje de Máquinas (UNAB).

Analiza un dataset de clientes de una empresa de suscripción y desarrolla un
modelo de aprendizaje automático para predecir el riesgo de abandono
(*churn*). Incluye el análisis exploratorio, la comparación de técnicas de
ML/DL, el refinamiento del modelo final y un prototipo de aplicación
(backend + frontend) que permite ingresar un cliente nuevo o consultar uno
del conjunto de prueba, ver su probabilidad de abandono, entender por qué el
modelo llegó a ese resultado (SHAP) y revisar el uso de la aplicación
(monitoreo).

## Integrantes

- Isabel Vera
- Gabriel Vera
- Diego Mallea

## Estructura del repositorio

| Carpeta / archivo | Contenido |
|---|---|
| `data/raw/`, `data/processed/` | Dataset original y datos depurados. |
| `notebook/` | Análisis exploratorio, modelamiento, balanceo, refinamiento y explicabilidad (orden de ejecución más abajo). |
| `artifacts/` | Preprocesador ajustado y particiones train/val/test ya transformadas, generados por `notebook/02_preprocesamiento.ipynb`. |
| `models/` | Modelos entrenados (`.joblib`, `.pt`), umbral de decisión, métricas e importancia global de SHAP. |
| `figures/` | Gráficos generados por los notebooks y capturas de la aplicación. |
| `backend/` | API de inferencia (FastAPI): esquema de entrada, validación, predicción, explicabilidad y monitoreo. |
| `frontend/` | Aplicación Streamlit que consume el backend por HTTP. |
| `tests/` | Pruebas automatizadas del backend (`pytest`). |
| `monitoring/` | Registro de predicciones de la app, generado en tiempo de ejecución (no se versiona). |
| `reports/` | Informes de las entregas del curso (`.docx`). |
| `configuracion_entorno.md` | Detalle del entorno de desarrollo, hardware y verificaciones de compatibilidad. |

## Instalación

Requisitos: Python 3.14 y Git (detalle del entorno probado, hardware y
compatibilidad de versiones en `configuracion_entorno.md`).

```powershell
git clone https://github.com/gveraprado-wited/acif104-churn-grupo3.git
cd acif104-churn-grupo3
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS/Linux, activar el entorno con `source .venv/bin/activate` en vez de
`.venv\Scripts\activate`.

## Cómo ejecutar

### 1. Notebooks (análisis y modelamiento)

Se ejecutan en orden desde `notebook/` — cada uno depende de artefactos que
genera el anterior:

1. `01_calidad_y_eda.ipynb` — análisis exploratorio de datos.
2. `02_preprocesamiento.ipynb` — limpieza, partición train/val/test y
   preprocesador (genera todo lo que hay en `artifacts/`).
3. `03_modelamiento_ml.ipynb` — regresión logística, Random Forest, XGBoost.
4. `04_balanceo_modelos.ipynb` — comparación de estrategias de balanceo de
   clases.
5. `05_modelamiento_dl.ipynb` — arquitecturas de deep learning y selección
   del modelo final (baseline).
6. `06_refinamiento_modelo.ipynb` — búsqueda de hiperparámetros y umbral de
   decisión de XGBoost + ROS (genera el modelo productivo,
   `models/xgboost_refinado.joblib`).
7. `06b_refinamiento_mlp_profunda.ipynb` — refinamiento de la arquitectura
   de red neuronal (comparación de configuraciones, no reemplaza al modelo
   final).
8. `07_explicabilidad_shap.ipynb` — importancia global y explicaciones
   locales con SHAP (genera `models/shap_importancia_global.csv`, que usa
   la app).

### 2. Aplicación (backend + frontend)

Requiere dos terminales abiertas al mismo tiempo, ambas con el entorno
virtual activado:

```powershell
# Terminal 1 — backend
uvicorn backend.api:app --reload
```

Verificar en <http://127.0.0.1:8000/health> o revisar la documentación
interactiva de la API en <http://127.0.0.1:8000/docs>.

```powershell
# Terminal 2 — frontend
streamlit run frontend/streamlit_app.py
```

Se abre automáticamente en <http://localhost:8501>. Si el backend no está
corriendo, la app lo indica con un mensaje de error en vez de fallar en
silencio.

### 3. Pruebas automatizadas

```powershell
pytest tests/
```

43 pruebas: validación de entradas (incluyendo los límites objetivamente
imposibles rechazados por la API, ej. csat_score fuera de 1-5), cálculo de
la predicción, agregación correcta de SHAP para variables categóricas,
clasificación del riesgo, registro de monitoreo (éxitos y errores) y los
seis endpoints de la API. No requieren que el backend esté corriendo — usan
el `TestClient` de FastAPI directamente.

## Modelo final

XGBoost + Random Over-Sampling (ROS), refinado en
`notebook/06_refinamiento_modelo.ipynb`. Umbral de decisión: 0,56 (banda de
riesgo baja bajo 0,24), elegido para mantener recall ≥ 0,80 en validación —
la justificación completa está en la sección 4.5.3 del informe.

## Modelos considerados

- Regresión logística
- Random Forest
- XGBoost (modelo final, refinado)
- Redes neuronales (MLP superficial y MLP profunda)
