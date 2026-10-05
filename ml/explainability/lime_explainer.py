"""Real LIME tabular explanations for individual network-security predictions."""
from __future__ import annotations
from typing import Any
import numpy as np

SUPPORTED_MODELS = {'random_forest', 'xgboost'}

def explain_local(model: Any, X: np.ndarray, feature_names: list[str], class_names: list[str], training_data: np.ndarray | None = None, original_values: list[Any] | None = None) -> dict[str, Any]:
    from lime.lime_tabular import LimeTabularExplainer
    X = np.asarray(X, dtype=np.float32)
    if X.shape != (1, len(feature_names)): raise ValueError('LIME requires exactly one feature row matching feature_names')
    background = np.asarray(training_data, dtype=np.float32) if training_data is not None else X
    if background.ndim != 2 or background.shape[1] != len(feature_names): raise ValueError('LIME background data shape mismatch')
    explainer = LimeTabularExplainer(background, feature_names=feature_names, class_names=class_names, mode='classification', discretize_continuous=True, random_state=42)
    exp = explainer.explain_instance(X[0], model.predict_proba, num_features=len(feature_names), top_labels=1)
    probs = np.asarray(model.predict_proba(X))[0]
    pred = int(np.argmax(probs))
    pairs = dict(exp.as_list(label=pred))
    rows=[]
    for i,name in enumerate(feature_names):
        contribution = 0.0
        # LIME returns human-readable rules; match the feature name prefix conservatively.
        for rule, weight in exp.as_list(label=pred):
            token = rule.split()[0]
            if token == name or name in rule:
                contribution += float(weight)
        original = original_values[i] if original_values is not None else float(X[0,i])
        rows.append({'feature_name':name,'feature_value':original,'transformed_feature_value':float(X[0,i]),'lime_contribution':contribution,'contribution_direction':'positive' if contribution >= 0 else 'negative'})
    rows.sort(key=lambda r: abs(r['lime_contribution']), reverse=True)
    return {'prediction':class_names[pred],'prediction_index':pred,'probability':float(probs[pred]),'confidence':float(probs[pred]),'features':rows,'positive_contributions':[r for r in rows if r['lime_contribution']>0],'negative_contributions':[r for r in rows if r['lime_contribution']<0],'explainer':'lime.lime_tabular.LimeTabularExplainer'}
