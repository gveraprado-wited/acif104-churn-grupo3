# Configuración del entorno

## Requisitos previos

- Python 3.14 (el proyecto se desarrolló y probó con 3.14.6).
- Git.

## Crear y activar el entorno virtual

Desde la raíz del repositorio:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS/Linux, reemplazar la activación por `source .venv/bin/activate`.

## Dependencias actuales

Declaradas en `requirements.txt` (versiones fijadas: son las que efectivamente se probaron juntas en este entorno):

```
pandas==3.0.5
numpy==2.4.6
matplotlib==3.11.1
seaborn==0.13.2
scikit-learn==1.9.0
imbalanced-learn==0.14.2
xgboost==3.3.0
torch==2.13.0
jupyter==1.1.1
joblib==1.5.3
shap==0.52.0
fastapi==0.140.0
uvicorn==0.51.0
pydantic==2.13.4
streamlit==1.60.0
requests==2.34.2
pytest==9.1.1
httpx2==2.9.1
```

`python-docx` se usó para editar los informes `.docx` de forma programática durante el desarrollo, pero no es una dependencia del proyecto (no lo importa ningún notebook ni el código de `backend/`/`frontend/`), así que no está en `requirements.txt`.

## Verificación de compatibilidad (2026-07-24, actualizado 2026-07-26)

El entorno usa versiones muy recientes (Python 3.14.6, numpy 2.x, pandas 3.0), por lo que antes de construir cada etapa se verificó que las librerías nuevas instalaran y funcionaran en conjunto con el resto del stack:

- `pip install shap` instala sin problemas (versión 0.52.0), aunque baja `numpy` de 2.5.1 a 2.4.6 para satisfacer sus dependencias (compatible con el resto del proyecto, sin romper nada).
- `pip install streamlit fastapi uvicorn pytest` instala sin problemas (streamlit 1.60.0, incluye `pyarrow` con wheel disponible para esta versión de Python; `fastapi` trae `pydantic` 2.13.4 como dependencia).
- Prueba combinada: cargar `artifacts/preprocesador.joblib` y `models/xgboost_final.joblib`, predecir sobre `X_val`, calcular valores SHAP con `TreeExplainer` y pasar los mismos datos por una capa de PyTorch — todo funciona correctamente en conjunto.
- Suite de pruebas del backend: `pytest tests/` — **43 pruebas, todas en verde**, sin mocks (cargan los artefactos reales).
- Verificación de equivalencia: se comparó, para 20 clientes reales del conjunto de prueba, la probabilidad que entrega la API contra la que se obtiene cargando `preprocesador.joblib` y `xgboost_refinado.joblib` directamente (como en los notebooks) — diferencia 0,0 en los 20 casos.
- Prueba de extremo a extremo con navegador real (Playwright, herramienta de verificación puntual, no es dependencia del proyecto): formulario, ficha de cliente, explicabilidad global y monitoreo probados en Chromium headless, sin errores de consola.

**Nota técnica:** `artifacts/splits_preprocesados.joblib` contiene `X_train`/`X_val`/`X_test` como arrays **densos** de NumPy, no matrices dispersas. `ColumnTransformer` decide automáticamente el formato de salida según qué tan dispersa resulte la combinación de columnas numéricas (densas) y categóricas codificadas (`sparse_threshold` por defecto en scikit-learn); con las proporciones de este dataset, el resultado quedó denso. El código de los notebooks de modelamiento contempla ambos casos (`hasattr(matriz, "toarray")`) por seguridad, pero no es necesario convertir manualmente.

## Estructura del proyecto

- `data/raw`: datos originales.
- `data/processed`: datos depurados.
- `notebook/`: notebooks de análisis y modelamiento (orden de ejecución: 01 → 02 → 03 → 04 → 05 → 06/06b → 07).
- `artifacts/`: preprocesador y particiones ya transformadas, generados por el notebook 02.
- `models/`: modelos entrenados y resultados tabulados.
- `figures/`: gráficos generados por los notebooks y por la app.
- `backend/`: API de inferencia (FastAPI) — lógica de predicción, validación, explicabilidad y monitoreo.
- `frontend/`: aplicación Streamlit que consume el backend por HTTP.
- `tests/`: pruebas automatizadas del backend (`pytest`).
- `monitoring/`: registro de predicciones de la app (`registro_predicciones.csv`, generado en tiempo de ejecución, no versionado).
- `reports/`: informes del proyecto.

Ver el `README.md` en la raíz del repositorio para el detalle de cada módulo y cómo ejecutar la aplicación y los tests.


## Entorno de desarrollo y hardware

- Sistema operativo: Windows 11 Home.
- IDE: Visual Studio Code con Jupyter Notebook.
- Hardware: Intel Core i9-12900H (14 núcleos / 20 procesadores lógicos), 64 GB RAM.
- GPU física: NVIDIA GeForce RTX 3070 Ti Laptop GPU (no utilizada por el proyecto; ver nota siguiente).
- Dispositivo utilizado por PyTorch: CPU. Se instaló la build genérica de `torch` (`2.13.0+cpu`), que no incluye soporte CUDA, por lo que `torch.cuda.is_available()` devuelve `False` aunque el equipo tenga GPU NVIDIA. Es consistente con lo indicado en el informe: el dataset es pequeño (10.000 filas, 58 características) y todos los modelos, incluidas las redes, entrenan en segundos/minutos en CPU sin necesidad de GPU.
- CUDA disponible para el proyecto: No (por elección de build de PyTorch, no por limitación de hardware).
