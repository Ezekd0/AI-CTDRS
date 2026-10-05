"""Server-side SHAP/LIME generation and persistence for detections."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.models import Detection, Explanation
from ml.explainability.shap_explainer import explain_local as explain_shap_local
from ml.training.artifacts import load_bundle


def _safe_detection_id(detection_id: str) -> str:
    if not detection_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in detection_id):
        raise ValueError('Invalid detection_id')
    return detection_id


def detection_path(detection_id: str) -> Path:
    return get_settings().resolved_reports_root / 'detections' / f'{_safe_detection_id(detection_id)}.json'


def load_detection(detection_id: str) -> dict[str, Any]:
    path = detection_path(detection_id)
    if not path.exists():
        raise FileNotFoundError(detection_id)
    return json.loads(path.read_text())


def save_detection(detection_id: str, payload: dict[str, Any]) -> Path:
    path = detection_path(detection_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x))
    return path


def _bundle(detection: Detection):
    settings = get_settings()
    return load_bundle(settings.resolved_artifacts_root, detection.dataset_id, detection.model.task, detection.model.name)


def _persist(db: Session, detection: Detection, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    row = db.execute(select(Explanation).where(Explanation.detection_id == detection.id, Explanation.method == method)).scalar_one_or_none()
    if row is None:
        db.add(Explanation(detection_id=detection.id, method=method, payload=payload))
    else:
        row.payload = payload
    status = dict(detection.explanation_status or {})
    status[method] = 'generated'
    detection.explanation_status = status
    return payload


def generate_shap_for_detection(db: Session, detection: Detection) -> dict[str, Any]:
    if detection.model.name not in {'random_forest', 'xgboost'}:
        raise ValueError('SHAP supports Random Forest and XGBoost only')
    bundle = _bundle(detection)
    X = np.asarray(detection.transformed_features, dtype=np.float32).reshape(1, -1)
    result = explain_shap_local(
        bundle.model, detection.model.name, X, detection.feature_names or bundle.feature_names,
        bundle.class_names, detection.original_feature_values,
    )
    result.update({'detection_id': detection.id, 'dataset': detection.dataset_id, 'task': detection.model.task, 'model': detection.model.name})
    _persist(db, detection, 'shap', result)
    return result


def generate_lime_for_detection(db: Session, detection: Detection) -> dict[str, Any]:
    if detection.model.name not in {'random_forest', 'xgboost'}:
        raise ValueError('LIME supports Random Forest and XGBoost only')
    try:
        from ml.explainability.lime_explainer import explain_local as explain_lime_local
    except ImportError as exc:
        raise RuntimeError("LIME dependency is not installed; install backend/requirements.txt") from exc
    bundle = _bundle(detection)
    X = np.asarray(detection.transformed_features, dtype=np.float32).reshape(1, -1)
    if bundle.lime_background is None:
        raise RuntimeError('LIME background artifact is missing for this model run; retrain the model bundle.')
    result = explain_lime_local(
        bundle.model, X, detection.feature_names or bundle.feature_names, bundle.class_names,
        training_data=bundle.lime_background, original_values=detection.original_feature_values,
    )
    result.update({'detection_id': detection.id, 'dataset': detection.dataset_id, 'task': detection.model.task, 'model': detection.model.name})
    _persist(db, detection, 'lime', result)
    return result


def explain_detection(detection_id: str) -> dict[str, Any]:
    """Backward-compatible file/API helper; DB persistence is preferred."""
    from app.db.session import get_engine
    from sqlalchemy.orm import joinedload
    with Session(get_engine()) as db:
        detection = db.execute(
            select(Detection).options(joinedload(Detection.model)).where(Detection.id == detection_id)
        ).scalar_one_or_none()
        if detection is None:
            raise FileNotFoundError(detection_id)
        result = generate_shap_for_detection(db, detection)
        db.commit()
        return result
