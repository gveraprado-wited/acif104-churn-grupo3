# Predicción de abandono de clientes (churn) — ACIF104, Grupo 3

Proyecto académico de la asignatura ACIF104 - Aprendizaje de Máquinas (UNAB).

Analiza un dataset de clientes de una empresa de suscripción y desarrolla un
modelo de aprendizaje automático para predecir el riesgo de abandono
(*churn*). Incluye el análisis exploratorio, la comparación de técnicas de
ML/DL, el refinamiento del modelo final y un prototipo de aplicación
(backend + frontend) que permite ingresar un cliente nuevo o consultar uno
del conjunto de validación, ver su probabilidad de abandono, entender por qué
el modelo llegó a ese resultado (SHAP) y revisar el uso de la aplicación
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
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si `Activate.ps1` falla con un error de política de ejecución ("no se puede
cargar el archivo ... porque la ejecución de scripts está deshabilitada en
este sistema"), es una restricción por defecto de PowerShell en Windows —
corre esto una vez en esa misma terminal y vuelve a intentar activar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Ese cambio de política solo aplica a la ventana de terminal actual, no
requiere permisos de administrador y no modifica la configuración del
equipo. Si se abre una terminal nueva (por ejemplo, para correr el backend
y el frontend en paralelo), hay que repetirlo ahí también.

En cmd.exe, activar con `.venv\Scripts\activate.bat` (sin el problema de
política de ejecución, que es específico de PowerShell). En macOS/Linux,
activar con `source .venv/bin/activate`.

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
8. `07_explicabilidad_shap.ipynb` — importancia global con SHAP sobre el
   conjunto de validación (genera `models/shap_importancia_global.csv`, que
   usa la app).
9. `08_evaluacion_final_test.ipynb` — única evaluación de todo el proyecto
   sobre el conjunto de prueba: antes de abrirlo, verifica contra
   `models/manifiesto_modelo_final.json` que el modelo y el umbral no
   cambiaron desde que se congelaron en el notebook 06; luego calcula las
   métricas finales, la matriz de confusión y las explicaciones locales SHAP
   para 3 clientes de prueba.

Los notebooks 03 a 07 no abren en ningún momento el conjunto de prueba
(`X_test`/`y_test`) — solo entrenamiento y validación. El conjunto de prueba
se reserva íntegramente para el notebook 08, después de que modelo,
hiperparámetros, balanceo y umbral ya están congelados.

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

44 pruebas: validación de entradas (incluyendo los límites objetivamente
imposibles rechazados por la API, ej. csat_score fuera de 1-5), cálculo de
la predicción, agregación correcta de SHAP para variables categóricas,
clasificación del riesgo, registro de monitoreo (éxitos y errores), los
seis endpoints de la API, y que los clientes que expone `customer_lookup`
correspondan efectivamente al conjunto de validación (comparando sus
etiquetas de churn contra `y_val`, no solo el conteo). No requieren que el
backend esté corriendo — usan el `TestClient` de FastAPI directamente.

## Modelo final

XGBoost + Random Over-Sampling (ROS), refinado en
`notebook/06_refinamiento_modelo.ipynb`. Umbral de decisión: 0,56 (banda de
riesgo baja bajo 0,24), elegido para mantener recall ≥ 0,80 en validación —
la justificación completa está en la sección 4.5.3 del informe (esa sección
del informe describe el estado previo a la reestructuración que separó la
evaluación de prueba en el notebook 08; se corrige en la rama documental).

El modelo y el umbral quedan congelados al final del notebook 06, con
`models/manifiesto_modelo_final.json` como evidencia verificable de ese
punto (hashes del `.joblib` y del `config_umbral.json`). Sus métricas sobre
el conjunto de prueba — la única vez que se abre en todo el proyecto — están
en `notebook/08_evaluacion_final_test.ipynb` y en
`models/metricas_test_modelo_refinado.csv`.

## Modelos considerados

- Regresión logística
- Random Forest
- XGBoost (modelo final, refinado)
- Redes neuronales (MLP superficial y MLP profunda)
