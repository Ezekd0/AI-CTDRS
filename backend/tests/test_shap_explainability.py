import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.explainability.shap_explainer import explain_global, explain_local


def test_shap_local_is_real_and_structured():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=float)
    y = np.array([0, 1, 1, 1, 1])
    est = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)

    class Wrapper:
        estimator = est
        def predict_proba(self, x): return self.estimator.predict_proba(x)

    result = explain_local(Wrapper(), "random_forest", X[:1], ["a", "b"], ["BENIGN", "ATTACK"], [0.0, 0.0])
    assert result["explainer"] == "shap.TreeExplainer"
    assert len(result["features"]) == 2
    assert all("shap_contribution" in row for row in result["features"])
    assert any(abs(row["shap_contribution"]) > 0 for row in result["features"])


def test_shap_global_contains_measured_importance():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=float)
    y = np.array([0, 1, 1, 1, 1])
    est = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)

    class Wrapper:
        estimator = est

    result = explain_global(Wrapper(), "random_forest", X, ["a", "b"], ["BENIGN", "ATTACK"])
    assert result["explainer"] == "shap.TreeExplainer"
    assert result["n_samples"] == len(X)
    assert len(result["feature_importance"]) == 2
    assert all(row["mean_abs_shap"] >= 0 for row in result["feature_importance"])
