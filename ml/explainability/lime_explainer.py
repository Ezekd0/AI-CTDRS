"""Real, bounded LIME tabular explanations for individual predictions."""
from __future__ import annotations
from typing import Any
import numpy as np

SUPPORTED_MODELS = {'random_forest', 'xgboost'}
MAX_BACKGROUND_ROWS = 256
NUM_SAMPLES = 512
MAX_FEATURES = 20


def _explainer(model, background, feature_names, class_names):
    from lime.lime_tabular import LimeTabularExplainer
    # The cached bundle owns this explainer; eviction releases both together.
    key = (id(background), tuple(feature_names), tuple(class_names))
    cached = getattr(model, '_local_lime_explainer', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if background.ndim != 2 or background.shape[1] != len(feature_names) or len(background) < 2:
        raise ValueError('LIME requires a saved background with at least two matching rows')
    indices = np.linspace(0, len(background) - 1, min(len(background), MAX_BACKGROUND_ROWS), dtype=int)
    bounded = np.asarray(background[indices], dtype=np.float32)
    explainer = LimeTabularExplainer(
        bounded, feature_names=feature_names, class_names=class_names,
        mode='classification', discretize_continuous=True, random_state=42,
        feature_selection='highest_weights',
    )
    model._local_lime_explainer = (key, explainer)
    return explainer


def explain_local(model: Any, X: np.ndarray, feature_names: list[str], class_names: list[str], training_data: np.ndarray | None = None, original_values: list[Any] | None = None) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float32)
    if X.shape != (1, len(feature_names)):
        raise ValueError('LIME requires exactly one feature row matching feature_names')
    if training_data is None:
        raise ValueError('LIME requires the saved training background')
    if original_values is not None and len(original_values) != len(feature_names):
        raise ValueError('original_values must align one-to-one with feature_names')
    explainer = _explainer(model, training_data, feature_names, class_names)
    # Serialized by the explanation worker; deterministic across cached requests.
    explainer.random_state.seed(42)
    probs = np.asarray(model.predict_proba(X))[0]
    pred = int(np.argmax(probs))
    exp = explainer.explain_instance(
        X[0], model.predict_proba, labels=(pred,), top_labels=None,
        num_features=min(MAX_FEATURES, len(feature_names)), num_samples=NUM_SAMPLES,
    )
    # Feature indices are authoritative; parsing text rules confuses names such
    # as src_bytes and log_src_bytes and can assign weights to the wrong feature.
    weights = dict(exp.local_exp[pred])
    rows = []
    for i, name in enumerate(feature_names):
        contribution = float(weights.get(i, 0.0))
        original = original_values[i] if original_values is not None else float(X[0, i])
        rows.append({'feature_name': name, 'feature_value': original,
                     'transformed_feature_value': float(X[0, i]),
                     'lime_contribution': contribution,
                     'contribution_direction': 'positive' if contribution >= 0 else 'negative'})
    rows.sort(key=lambda r: abs(r['lime_contribution']), reverse=True)
    return {'prediction': class_names[pred], 'prediction_index': pred,
            'probability': float(probs[pred]), 'confidence': float(probs[pred]),
            'features': rows,
            'positive_contributions': [r for r in rows if r['lime_contribution'] > 0],
            'negative_contributions': [r for r in rows if r['lime_contribution'] < 0],
            'explainer': 'lime.lime_tabular.LimeTabularExplainer',
            'num_samples': NUM_SAMPLES, 'surrogate_score': float(exp.score),
            'local_prediction': float(np.asarray(exp.local_pred).ravel()[0])}
