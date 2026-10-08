"""Real SHAP explanations for tree-based IDS models.

No explanation values are fabricated: every value returned here is calculated by
SHAP TreeExplainer against the loaded Random Forest or XGBoost model.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_TREE_MODELS = {"random_forest", "xgboost"}


@lru_cache(maxsize=1)
def _cached_tree_explainer(estimator: Any):
    import shap
    # Use training path counts already saved in the trees; no background sweep.
    return shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _tree_estimator(model: Any, model_name: str) -> Any:
    if model_name == "random_forest":
        return model.estimator
    if model_name == "xgboost":
        return model.booster
    raise ValueError(f"SHAP TreeExplainer is not supported for model {model_name!r}")


def _values_for_class(shap_values: Any, class_index: int, n_features: int) -> np.ndarray:
    """Normalize SHAP's several output layouts to (rows, features) for one class."""
    values = np.asarray(shap_values)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        # Newer SHAP commonly returns (rows, features, outputs).
        if values.shape[2] == n_features:
            return values[:, :, class_index]
        return values[:, :, class_index]
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])
    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def _all_class_values(shap_values: Any, n_classes: int, n_features: int) -> np.ndarray:
    values = np.asarray(shap_values)
    if isinstance(shap_values, list):
        return np.stack([np.asarray(v) for v in shap_values], axis=-1)
    if values.ndim == 2:
        return values[:, :, None]
    if values.ndim == 3:
        # Expected (rows, features, outputs).
        if values.shape[1] == n_features:
            return values
        # Defensive support for (rows, outputs, features).
        return np.transpose(values, (0, 2, 1))
    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def _base_for_class(base_values: Any, class_index: int) -> float | None:
    if base_values is None:
        return None
    values = np.asarray(base_values)
    if values.ndim == 0:
        return float(values)
    if values.ndim == 1:
        return float(values[class_index] if len(values) > class_index else values[0])
    return float(values[0, class_index] if values.shape[1] > class_index else values[0, 0])


def explain_local(model: Any, model_name: str, X: np.ndarray, feature_names: list[str], class_names: list[str], original_values: list[Any] | None = None) -> dict[str, Any]:
    """Explain one transformed row and return a browser-safe structured result."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != 1:
        raise ValueError("Local SHAP explanation requires exactly one feature row")
    if X.shape[1] != len(feature_names):
        raise ValueError("Feature count does not match feature_names")

    estimator = _tree_estimator(model, model_name)
    explainer = _cached_tree_explainer(estimator)
    explanation = explainer(X)
    probabilities = model.predict_proba(X)[0]
    prediction_index = int(np.argmax(probabilities))
    values = _values_for_class(explanation.values, prediction_index, len(feature_names))[0]
    base = _base_for_class(explanation.base_values, prediction_index)

    if original_values is None:
        original_values = X[0].tolist()
    if len(original_values) != len(feature_names):
        raise ValueError("original_values must align one-to-one with feature_names")

    rows = []
    for name, original, contribution in zip(feature_names, original_values, values):
        contribution = float(contribution)
        rows.append({
            "feature_name": name,
            "original_feature_value": original.item() if isinstance(original, np.generic) else original,
            "transformed_feature_value": float(X[0, len(rows)]),
            "shap_contribution": contribution,
            "contribution_direction": "toward_prediction" if contribution >= 0 else "away_from_prediction",
        })
    rows.sort(key=lambda r: abs(r["shap_contribution"]), reverse=True)

    return {
        "prediction": class_names[prediction_index],
        "prediction_index": prediction_index,
        "probability": float(probabilities[prediction_index]),
        "confidence": float(probabilities[prediction_index]),
        "base_value": base,
        "features": rows,
        "explainer": "shap.TreeExplainer",
        "model": model_name,
    }


def explain_global(model: Any, model_name: str, X: np.ndarray, feature_names: list[str], class_names: list[str], max_samples: int = 1000) -> dict[str, Any]:
    """Calculate global SHAP importance from real model predictions on supplied rows."""
    import shap

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError("X shape does not match feature_names")
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        X = X[rng.choice(len(X), size=max_samples, replace=False)]

    estimator = _tree_estimator(model, model_name)
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(X)
    all_values = _all_class_values(explanation.values, len(class_names), len(feature_names))
    importance = np.mean(np.abs(all_values), axis=(0, 2))
    mean_signed = np.mean(all_values, axis=(0, 2))
    ranking = [
        {"feature_name": feature_names[i], "mean_abs_shap": float(importance[i]), "mean_shap": float(mean_signed[i])}
        for i in np.argsort(importance)[::-1]
    ]
    return {
        "model": model_name,
        "explainer": "shap.TreeExplainer",
        "n_samples": int(len(X)),
        "feature_importance": ranking,
        "class_names": class_names,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
