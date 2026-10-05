# Model evaluation and comparison API

The AI Models page reads evaluation reports produced by the ML training pipeline.

## Endpoints

### `GET /api/models/evaluations`

Returns measured Random Forest and XGBoost results for the supported datasets and tasks.

Optional query parameters:

- `dataset=cicids2017|cicids2018|nsl-kdd`
- `task=binary|multiclass`

The response includes accuracy, precision, recall, F1, balanced accuracy, ROC-AUC, average precision (PR-AUC for binary runs), test sample count, classification-report data, and confusion-matrix data.

### `GET /api/models/evaluations/{dataset}/{task}/{model}`

Returns the detailed evaluation for one model, including:

- confusion matrix
- classification report
- measured scalar metrics
- ROC curve points when ROC is defined
- precision-recall curve points for binary evaluation
- available report files

## Comparison policy

The API intentionally does not rank Random Forest and XGBoost or declare a universal best model. The frontend exposes a production-candidate selection control, but selecting a candidate does not deploy or automatically promote the model.

The project administrator should document the production criteria appropriate to the deployment, such as recall requirements, precision/false-positive tolerance, F1, ROC/PR performance, latency, resource cost, and operational constraints.

## SHAP explainability API

Tree-based SHAP explanations are calculated with `shap.TreeExplainer` from the actual loaded Random Forest or XGBoost artifact. The application never fabricates SHAP values.

### Global explanation

`GET /api/explain/shap/global?dataset=cicids2017&task=binary&model=random_forest`

Returns measured global mean absolute and mean signed SHAP contributions from held-out test rows used during training (sampled to at most 1,000 rows).

### Individual explanation

`GET /api/explain/shap/{detection_id}`

The detection registry at `reports/detections/<detection_id>.json` must contain:

```json
{
  "dataset": "cicids2017",
  "task": "binary",
  "model": "random_forest",
  "transformed_features": [0.1, -0.4],
  "original_feature_values": [123.0, "TCP"]
}
```

`transformed_features` must be the exact model-ready feature vector in the same order as `feature_names.json`. `original_feature_values` is persisted by the detection/inference layer so the explanation can show the actual input value rather than a fabricated or reconstructed value. The endpoint calculates prediction/probability itself from the loaded model and then calculates SHAP contributions for the predicted class.
