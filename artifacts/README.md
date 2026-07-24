# Artifacts

Esta carpeta contiene artefactos generados por `notebook/02_preprocesamiento.ipynb`.

- `preprocesador.joblib`: ColumnTransformer ajustado únicamente con el conjunto de entrenamiento.
- `splits_preprocesados.joblib`: conjuntos train, validation y test ya transformados, junto con etiquetas, pesos de clase y nombres de variables.

Estos archivos permiten que los notebooks de modelamiento reutilicen exactamente la misma partición y transformación, evitando reajustar el preprocesador sobre validación o prueba.