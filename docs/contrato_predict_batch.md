# Contrato de `POST /predict/batch`

Congelado el 2026-08-16 por Gabriel (`feature/backend-batch`), para que Diego
pueda desarrollar `feature/frontend-dashboard` contra una respuesta JSON
simulada que respete exactamente esta forma, sin esperar a que el backend
esté fusionado (ver plan de trabajo, sección 8.1).

Los archivos [contrato_predict_batch_request.json](contrato_predict_batch_request.json)
y [contrato_predict_batch_response.json](contrato_predict_batch_response.json)
son un ejemplo real, generado ejecutando el endpoint contra dos clientes del
conjunto de validación (uno válido y uno con una categoría inválida en
`gender`) — no están escritos a mano.

## Reglas del contrato (no cambian sin volver a congelar)

- `customers`: lista de 1 a 100 elementos. Menos de 1 o más de 100 responde
  `422` a nivel de request completo (no es un error de fila).
- Cada elemento: `{"customer_id": str, "features": {30 variables crudas}}`.
  `features` no se valida al parsear el request — se valida fila por fila
  dentro del backend, por eso una fila inválida no tumba las demás.
- `results` solo contiene las filas válidas, en el mismo orden de llegada.
  `errors` contiene las inválidas, con `row_index` en base 1 (posición en
  `customers`, contando también las filas válidas).
- `probability`, `risk_band` y `decision_threshold` de una fila del lote son
  idénticos a lo que devolvería `POST /predict` para ese mismo cliente.
- `customer_id` duplicado no se rechaza: cada fila se procesa de forma
  independiente.
- `summary.high_risk_monthly_fee` es la suma de `monthly_fee` únicamente de
  las filas en banda `Alto`.
- El frontend nunca debe leer `models/`, `artifacts/` ni `data/`
  directamente — todo lo que necesita (umbral, espacio SHAP, importancia
  global, métricas de test) también está disponible por HTTP en
  `GET /model/info`, `GET /explainability/global` y `GET /model/metrics`.

## Otros endpoints nuevos

| Endpoint | Devuelve |
|---|---|
| `GET /model/info` | `{"model", "decision_threshold", "low_band", "shap_space"}` |
| `GET /explainability/global` | Lista `[{"variable", "importancia_media_abs"}]` ordenada descendente |
| `GET /model/metrics` | `{"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}` |
