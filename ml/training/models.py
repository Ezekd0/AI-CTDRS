"""Model wrappers with one interface: ``fit`` / ``predict_proba`` / ``save``.

Random Forest (scikit-learn), XGBoost and the LSTM (PyTorch) are imported lazily, so a missing optional
library only affects the model that needs it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL_NAMES = ("random_forest", "xgboost", "lstm")
_MODEL_ALIASES = {"randomforest": "random_forest", "rf": "random_forest", "xgboost": "xgboost", "xgb": "xgboost", "lstm": "lstm"}

RF_DEFAULTS: dict[str, Any] = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}
XGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 1, "early_stopping_rounds": 20,
}
LSTM_DEFAULTS: dict[str, Any] = {
    "hidden_size": 64, "num_layers": 1, "dropout": 0.2, "features_per_step": 8, "epochs": 20,
    "batch_size": 512, "learning_rate": 1e-3, "weight_decay": 0.0, "patience": 3, "grad_clip": 1.0, "device": "auto",
}
DEFAULTS = {"random_forest": RF_DEFAULTS, "xgboost": XGB_DEFAULTS, "lstm": LSTM_DEFAULTS}


def resolve_model(name: str) -> str:
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    if key not in _MODEL_ALIASES:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODEL_NAMES)}")
    return _MODEL_ALIASES[key]


def merge_params(model: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(DEFAULTS[model])
    for k, v in (overrides or {}).items():
        if k not in params:
            raise ValueError(f"Unknown parameter {k!r} for {model}. Known: {sorted(params)}")
        params[k] = v
    return params


def align_proba(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Return an (n, n_classes) matrix even if the estimator saw fewer classes."""
    if proba.shape[1] == n_classes:
        return proba
    out = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    out[:, classes.astype(int)] = proba
    return out


class RandomForestModel:
    name = "random_forest"
    model_file = "model.joblib"

    def __init__(self, n_classes: int, seed: int, params: dict[str, Any]) -> None:
        self.n_classes, self.params = n_classes, params
        self.estimator = RandomForestClassifier(n_jobs=-1, random_state=seed, **params)

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, class_weights=None) -> dict[str, Any]:
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return {}

    def predict_proba(self, X) -> np.ndarray:
        return align_proba(self.estimator.predict_proba(X), self.estimator.classes_, self.n_classes)

    def save(self, directory: Path) -> Path:
        path = Path(directory) / self.model_file
        joblib.dump(self.estimator, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path, n_classes: int) -> "RandomForestModel":
        obj = cls.__new__(cls)
        obj.n_classes, obj.params = n_classes, {}
        obj.estimator = joblib.load(path)
        return obj


def booster_predict_proba(booster, X: np.ndarray, n_classes: int) -> np.ndarray:
    import xgboost as xgb

    raw = booster.predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)))
    return np.column_stack([1.0 - raw, raw]) if n_classes == 2 else raw


class XGBoostModel:
    """Trains with XGBClassifier, then keeps only the best-iteration trees as a native Booster. The same
    Booster is used for evaluation here and for loading later, so both give identical probabilities."""

    name = "xgboost"
    model_file = "model.json"

    def __init__(self, n_classes: int, seed: int, params: dict[str, Any]) -> None:
        self.n_classes, self.seed, self.params = n_classes, seed, params
        self.booster = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, class_weights=None) -> dict[str, Any]:
        from xgboost import XGBClassifier

        params = dict(self.params)
        has_val = X_val is not None and len(X_val) > 0
        if not has_val:
            params["early_stopping_rounds"] = None
        binary = self.n_classes == 2
        clf = XGBClassifier(
            objective="binary:logistic" if binary else "multi:softprob",
            eval_metric="logloss" if binary else "mlogloss",
            tree_method="hist", n_jobs=os.cpu_count() or 1, random_state=self.seed, **params,
        )
        fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight, "verbose": False}
        if has_val:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        clf.fit(X, y, **fit_kwargs)
        booster = clf.get_booster()
        best = getattr(clf, "best_iteration", None) if has_val and params.get("early_stopping_rounds") else None
        if best is not None:
            booster = booster[: int(best) + 1]
        self.booster = booster
        return {"best_iteration": None if best is None else int(best),
                "n_trees_kept": int(booster.num_boosted_rounds()), "evals_result": clf.evals_result() if has_val else {}}

    def predict_proba(self, X) -> np.ndarray:
        return booster_predict_proba(self.booster, X, self.n_classes)

    def save(self, directory: Path) -> Path:
        path = Path(directory) / self.model_file
        self.booster.save_model(str(path))
        return path

    @classmethod
    def load(cls, path: Path, n_classes: int) -> "XGBoostModel":
        import xgboost as xgb

        obj = cls.__new__(cls)
        obj.n_classes, obj.seed, obj.params = n_classes, 0, {}
        obj.booster = xgb.Booster()
        obj.booster.load_model(str(path))
        return obj


def build_model(name: str, n_classes: int, n_features: int, seed: int, params: dict[str, Any]):
    name = resolve_model(name)
    if name == "random_forest":
        return RandomForestModel(n_classes, seed, params)
    if name == "xgboost":
        return XGBoostModel(n_classes, seed, params)
    from ml.training.lstm_model import LSTMModel

    return LSTMModel(n_classes, seed, params, n_features=n_features)


def load_model(name: str, path: Path, n_classes: int, n_features: int | None = None):
    name = resolve_model(name)
    loaders: dict[str, Callable[[], Any]] = {
        "random_forest": lambda: RandomForestModel.load(path, n_classes),
        "xgboost": lambda: XGBoostModel.load(path, n_classes),
    }
    if name in loaders:
        return loaders[name]()
    from ml.training.lstm_model import LSTMModel

    return LSTMModel.load(path)
